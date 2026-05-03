"""Retriever agent: tier-aware, self-correcting retrieval with query transform.

LATENCY FIXES
─────────────
FIX L8 — Skip query transform for TIER_1 queries
    The query transformer makes an LLM call (another load+infer cycle).
    For simple TIER_1 queries the transform rarely helps — it just rewrites
    "tell me X" to "Determine X." which retrieves nearly identical chunks.
    With the warm TTL cache the second LLM call in a session is fast, but
    the first query still pays the cold-start cost twice.

    Fix: skip transform entirely for TIER_1. Use the raw query directly.
    TIER_2 and TIER_3 still transform (multi-hop needs it).

FIX L9 — Reduce sub-query cap from 3 to 2 for TIER_2
    Each sub-query is a separate embedding + ChromaDB call. Reducing from 3
    to 2 saves ~30% of TIER_2 retrieval time with minimal quality loss.
"""
from __future__ import annotations

import time
from typing import Any

from src.agents.state import RAGState, Tier
from src.retrieval.hybrid_search import HybridSearch, RetrievedChunk
from src.retrieval.query_transform import get_query_transformer
from src.retrieval.reranker import get_reranker
from src.utils.config import get_settings
from src.utils.exceptions import RetrievalError
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()

_SELF_CORRECT_THRESHOLD = 0.5


class RetrieverAgent:
    """Retrieves context based on the assigned tier."""

    def __init__(self) -> None:
        self._hybrid = HybridSearch()
        self._reranker = get_reranker()
        self._transformer = get_query_transformer()

    async def retrieve(self, state: RAGState, documents: list[str] | None = None) -> RAGState:
        t0 = time.perf_counter()
        state.retrieval_iteration += 1

        try:
            # FIX L8: only transform for TIER_2 / TIER_3
            if state.retrieval_iteration == 1:
                if state.tier == Tier.TIER_1:
                    # Skip LLM transform — use raw query directly
                    state.transformed_query = state.query
                    state.sub_queries = [state.query]
                    logger.info(
                        "Query transform SKIPPED (TIER_1): %r", state.query[:60]
                    )
                else:
                    transformed = await self._transformer.transform(state.query)
                    state.transformed_query = transformed.rewritten
                    state.sub_queries = transformed.sub_queries or [state.query]
                    logger.info(
                        "Query transformed: %r → %r | sub_queries=%d",
                        state.query[:50],
                        state.transformed_query[:50],
                        len(state.sub_queries),
                    )

            if state.tier == Tier.TIER_1:
                chunks = await self._tier1_retrieve(state, documents)
            elif state.tier == Tier.TIER_2:
                chunks = await self._tier2_retrieve(state, documents)
            else:
                chunks = await self._tier3_retrieve(state)

            # Self-correction: if top score too low, fallback to raw query search
            if chunks and chunks[0].score < _SELF_CORRECT_THRESHOLD:
                logger.info(
                    "Low retrieval score (%.3f); self-correcting with raw query…",
                    chunks[0].score,
                )
                fallback = await self._hybrid.search(
                    state.query, top_k=_settings.top_k_dense, documents=documents
                )
                chunks = self._merge_dedupe(chunks, fallback)

            reranked = self._reranker.rerank(
                state.transformed_query, chunks, top_k=_settings.top_k_rerank
            )
            if reranked and reranked[0].score < 0.01:
                logger.warning(
                    "Reranked top chunk score very low (%.6f). Returning anyway.",
                    reranked[0].score,
                )

            state.context = [self._chunk_to_dict(c) for c in reranked]
            state.retrieval_scores = [c.score for c in reranked]
            state.sources = [
                {"chunk_id": c.chunk_id, "source": c.source, "score": c.score}
                for c in reranked
            ]
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            state.error = f"Retrieval error: {exc}"
            state.context = []

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "RETRIEVE tier=%s iter=%d chunks=%d elapsed_ms=%.1f",
            state.tier.value,
            state.retrieval_iteration,
            len(state.context),
            elapsed,
        )
        return state

    # ── Tier strategies ──────────────────────────────────────────────────────────

    async def _tier1_retrieve(
        self, state: RAGState, documents: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """Single-pass hybrid search — no LLM involvement."""
        return await self._hybrid.search(
            state.transformed_query,
            top_k=_settings.top_k_dense,
            documents=documents,
        )

    async def _tier2_retrieve(
        self, state: RAGState, documents: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """Iterative retrieval with sub-question expansion.

        FIX L9: cap at 2 sub-queries (was 3) — each is an embed + ChromaDB call.
        """
        all_chunks: list[RetrievedChunk] = []
        queries = state.sub_queries or [state.transformed_query]
        for sq in queries[:2]:  # FIX L9: was 3
            results = await self._hybrid.search(sq, top_k=8, documents=documents)
            all_chunks.extend(results)
        return self._merge_dedupe(all_chunks, [])

    async def _tier3_retrieve(
        self, state: RAGState, documents: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """Graph-first then vector search."""
        graph_chunk_ids: list[str] = []
        try:
            from src.db.neo4j_client import get_neo4j_client
            import re

            neo4j = get_neo4j_client()
            entity_candidates = re.findall(
                r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)*\b",
                state.query,
            )
            if entity_candidates:
                graph_chunk_ids = neo4j.find_related_chunks(entity_candidates, max_hops=2)
                logger.info("Graph returned %d related chunks", len(graph_chunk_ids))
        except Exception as exc:
            logger.warning("Graph retrieval skipped: %s", exc)

        graph_chunks: list[RetrievedChunk] = []
        if graph_chunk_ids:
            from src.retrieval.vector_store import VectorStore
            vs = VectorStore()
            graph_chunks = vs.fetch_by_ids(graph_chunk_ids[:20])

        hybrid_chunks = await self._hybrid.search(
            state.transformed_query,
            top_k=_settings.top_k_dense,
            documents=documents,
        )
        return self._merge_dedupe(graph_chunks, hybrid_chunks)

    # ── Helpers ──────────────────────────────────────────────────────────────────

    def _merge_dedupe(
        self,
        primary: list[RetrievedChunk],
        secondary: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        seen: set[str] = set()
        result: list[RetrievedChunk] = []
        for c in primary + secondary:
            if c.chunk_id not in seen and c.text.strip():
                seen.add(c.chunk_id)
                result.append(c)
        return result

    def _chunk_to_dict(self, c: RetrievedChunk) -> dict[str, Any]:
        return {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "source": c.source,
            "score": c.score,
            **c.metadata,
        }