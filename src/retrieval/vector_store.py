"""Vector store operations wrapper for ChromaDB."""
from __future__ import annotations

from typing import Any

from src.db.chroma_client import get_chroma_client
from src.retrieval.hybrid_search import RetrievedChunk
from src.utils.logger import get_logger

logger = get_logger(__name__)


class VectorStore:
    """High-level vector store operations."""

    def __init__(self) -> None:
        self._chroma = get_chroma_client()

    def get_collection_count(self) -> int:
        try:
            return self._chroma.get_or_create_collection().count()
        except Exception:
            return 0

    def fetch_by_ids(self, ids: list[str]) -> list[RetrievedChunk]:
        try:
            coll = self._chroma.get_or_create_collection()
            results = coll.get(ids=ids, include=["documents", "metadatas"])
            chunks = []
            for cid, doc, meta in zip(
                results.get("ids", []),
                results.get("documents", []),
                results.get("metadatas", []),
            ):
                chunks.append(
                    RetrievedChunk(
                        chunk_id=cid,
                        text=doc or "",
                        source=(meta or {}).get("source", ""),
                        score=1.0,
                        metadata=meta or {},
                    )
                )
            return chunks
        except Exception as exc:
            logger.warning("fetch_by_ids failed: %s", exc)
            return []

    def health(self) -> dict[str, Any]:
        return self._chroma.health()
