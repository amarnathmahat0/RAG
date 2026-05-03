"""FlashRank cross-encoder reranker for retrieved chunks."""
from __future__ import annotations

from src.retrieval.hybrid_search import RetrievedChunk
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()


class Reranker:
    """Rerank retrieved chunks using FlashRank cross-encoder."""

    def __init__(self) -> None:
        self._ranker = None

    def _load(self) -> bool:
        if self._ranker is not None:
            return True
        try:
            from flashrank import Ranker

            self._ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="./data/models")
            logger.info("FlashRank reranker loaded.")
            return True
        except Exception as exc:
            logger.warning("FlashRank not available (%s); skipping rerank.", exc)
            return False

    def rerank(
        self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None
    ) -> list[RetrievedChunk]:
        top_k = top_k or _settings.top_k_rerank
        if not chunks:
            return []
        if not self._load():
            return chunks[:top_k]

        try:
            from flashrank import RerankRequest

            passages = [{"id": c.chunk_id, "text": c.text} for c in chunks]
            req = RerankRequest(query=query, passages=passages)
            results = self._ranker.rerank(req)
            id_to_chunk = {c.chunk_id: c for c in chunks}
            reranked: list[RetrievedChunk] = []
            for i, r in enumerate(results[:top_k]):
                cid = r.get("id", "")
                chunk = id_to_chunk.get(cid)
                if chunk:
                    chunk.score = float(r.get("score", chunk.score))
                    chunk.rank = i + 1
                    reranked.append(chunk)
            logger.debug("Reranked %d → %d chunks", len(chunks), len(reranked))
            return reranked
        except Exception as exc:
            logger.warning("Reranking failed: %s", exc)
            return chunks[:top_k]


_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _reranker
    if _reranker is None:
        _reranker = Reranker()
    return _reranker
