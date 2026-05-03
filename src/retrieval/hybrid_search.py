"""Hybrid retrieval: dense (ChromaDB) + sparse (BM25) fused via RRF.

FIX — ChromaDB single-document filter crash
────────────────────────────────────────────
ChromaDB requires $and/$or to have ≥ 2 expressions. When filtering by a
single document the old code built {"$or": [one_item]} which ChromaDB
rejected with:
  "Expected where value for $and or $or to be a list with at least two
   where expressions, got [{'source': {'$eq': '...'}}]"

Fix: if len(documents) == 1, use a plain equality filter directly.
     if len(documents) >= 2, use $or as before.
     if documents is None / empty, send no where clause.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.retrieval.embeddings import get_embedding_service
from src.retrieval.sparse_index import get_bm25_index
from src.utils.config import get_settings
from src.utils.exceptions import RetrievalError
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)
    rank: int = 0


def _build_where_clause(documents: list[str] | None) -> dict | None:
    """Build a ChromaDB-compatible where clause for document filtering.

    ChromaDB rules:
      - No filter      → pass where=None (omit the parameter)
      - Single doc     → {"source": {"$eq": "path"}}          ← plain equality
      - Multiple docs  → {"$or": [{"source": {"$eq": "a"}},   ← needs ≥ 2 items
                                  {"source": {"$eq": "b"}}]}
    """
    if not documents:
        return None
    if len(documents) == 1:
        # FIX: plain equality — no $or wrapper needed (and not allowed)
        return {"source": {"$eq": documents[0]}}
    return {"$or": [{"source": {"$eq": doc}} for doc in documents]}


class HybridSearch:
    """Retrieve via dense + sparse, fuse with Reciprocal Rank Fusion."""

    def __init__(
        self,
        top_k_dense: int | None = None,
        top_k_sparse: int | None = None,
        rrf_k: int | None = None,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
    ) -> None:
        self.top_k_dense  = top_k_dense  or _settings.top_k_dense
        self.top_k_sparse = top_k_sparse or _settings.top_k_sparse
        self.rrf_k        = rrf_k        or _settings.rrf_k
        self.dense_weight  = dense_weight  or _settings.dense_weight
        self.sparse_weight = sparse_weight or _settings.sparse_weight
        self._emb = get_embedding_service()

    async def search(
        self,
        query: str,
        top_k: int | None = None,
        documents: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        tk = top_k or self.top_k_dense
        query_emb    = await self._emb.embed_query(query)
        dense_results  = self._dense_search(query_emb, self.top_k_dense, documents)
        sparse_results = self._sparse_search(query, self.top_k_sparse, documents)
        fused = self._rrf_fusion(dense_results, sparse_results)
        return fused[:tk]

    # ── Dense ────────────────────────────────────────────────────────────────────

    def _dense_search(
        self,
        query_emb: list[float],
        top_k: int,
        documents: list[str] | None = None,
    ) -> list[tuple[str, float, str, dict]]:
        """Return list of (chunk_id, score, text, metadata)."""
        from src.db.chroma_client import get_chroma_client
        from src.utils.config import get_settings as _cfg

        try:
            collection = get_chroma_client().get_or_create_collection()

            # FIX: use helper that handles 0/1/N doc cases correctly
            where_clause = _build_where_clause(documents)

            query_kwargs: dict = dict(
                query_embeddings=[query_emb],
                n_results=min(top_k, max(1, collection.count())),
                include=["documents", "metadatas", "distances"],
            )
            if where_clause is not None:
                query_kwargs["where"] = where_clause

            results = collection.query(**query_kwargs)

            ids   = results.get("ids",       [[]])[0]
            docs  = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            cfg = _cfg()
            if dists:
                logger.debug(
                    "Dense search | embed_model=%s raw_distances=%s",
                    cfg.embed_model_primary,
                    [round(d, 4) for d in dists[:5]],
                )
                converted_scores = [max(0.0, min(1.0, 1.0 - float(d))) for d in dists]
                near_zero = sum(1 for s in converted_scores if s < 0.01)
                if near_zero == len(dists) and len(dists) > 0:
                    logger.warning(
                        "ALL dense scores near-zero after distance conversion "
                        "(embed_model=%s). Query embedding model may not match "
                        "ingestion model. Raw distances: %s",
                        cfg.embed_model_primary,
                        [round(d, 4) for d in dists[:5]],
                    )

            items = []
            for cid, doc, meta, dist in zip(ids, docs, metas, dists):
                score = max(0.0, min(1.0, 1.0 - float(dist)))
                items.append((cid, score, doc, meta or {}))
            return items

        except Exception as exc:
            logger.warning("Dense search failed: %s", exc)
            return []

    # ── Sparse ───────────────────────────────────────────────────────────────────

    def _sparse_search(
        self,
        query: str,
        top_k: int,
        documents: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        bm25 = get_bm25_index()
        return bm25.search(query, top_k)

    # ── RRF fusion ───────────────────────────────────────────────────────────────

    def _rrf_fusion(
        self,
        dense: list[tuple[str, float, str, dict]],
        sparse: list[tuple[str, float]],
    ) -> list[RetrievedChunk]:
        k = self.rrf_k
        scores: dict[str, float] = {}

        for rank, (cid, _score, _text, _meta) in enumerate(dense, 1):
            scores[cid] = scores.get(cid, 0.0) + self.dense_weight / (k + rank)

        for rank, (cid, _score) in enumerate(sparse, 1):
            scores[cid] = scores.get(cid, 0.0) + self.sparse_weight / (k + rank)

        dense_lookup: dict[str, tuple[str, dict]] = {
            cid: (text, meta) for cid, _s, text, meta in dense
        }

        sparse_only = [cid for cid in scores if cid not in dense_lookup]
        if sparse_only:
            extra = self._fetch_by_ids(sparse_only)
            dense_lookup.update(extra)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        results: list[RetrievedChunk] = []
        for i, (cid, rrf_score) in enumerate(ranked):
            text, meta = dense_lookup.get(cid, ("", {}))
            results.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=text,
                    source=meta.get("source", ""),
                    score=rrf_score,
                    metadata=meta,
                    rank=i + 1,
                )
            )
        return results

    def _fetch_by_ids(self, ids: list[str]) -> dict[str, tuple[str, dict]]:
        from src.db.chroma_client import get_chroma_client

        try:
            collection = get_chroma_client().get_or_create_collection()
            results = collection.get(ids=ids, include=["documents", "metadatas"])
            out = {}
            for cid, doc, meta in zip(
                results.get("ids",       []),
                results.get("documents", []),
                results.get("metadatas", []),
            ):
                out[cid] = (doc or "", meta or {})
            return out
        except Exception:
            return {}