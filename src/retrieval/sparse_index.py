"""BM25-based sparse retrieval index."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()

_INDEX_PATH = Path(_settings.cache_dir) / "bm25_index.pkl"


class BM25Index:
    """Persistent BM25 index backed by rank-bm25."""

    def __init__(self) -> None:
        self._corpus: list[str] = []
        self._ids: list[str] = []
        self._bm25: Any = None

    def add_documents(self, docs: list[str], ids: list[str]) -> None:
        """Add documents and rebuild index."""
        self._corpus.extend(docs)
        self._ids.extend(ids)
        self._rebuild()

    def _rebuild(self) -> None:
        try:
            from rank_bm25 import BM25Okapi

            tokenized = [self._tokenize(d) for d in self._corpus]
            self._bm25 = BM25Okapi(tokenized)
            logger.debug("BM25 index rebuilt: %d docs", len(self._corpus))
        except ImportError:
            logger.warning("rank-bm25 not installed; sparse retrieval disabled.")

    def _tokenize(self, text: str) -> list[str]:
        import re

        return re.findall(r"\b[a-z0-9]+\b", text.lower())

    def search(self, query: str, top_k: int = 15) -> list[tuple[str, float]]:
        """Return list of (doc_id, score) sorted by relevance."""
        if self._bm25 is None or not self._corpus:
            return []
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = sorted(zip(self._ids, scores), key=lambda x: x[1], reverse=True)
        return [(doc_id, float(score)) for doc_id, score in ranked[:top_k] if score > 0]

    def save(self) -> None:
        _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_INDEX_PATH, "wb") as f:
            pickle.dump({"corpus": self._corpus, "ids": self._ids}, f)
        logger.info("BM25 index saved to %s", _INDEX_PATH)

    def load(self) -> bool:
        if not _INDEX_PATH.exists():
            return False
        try:
            with open(_INDEX_PATH, "rb") as f:
                data = pickle.load(f)
            self._corpus = data["corpus"]
            self._ids = data["ids"]
            self._rebuild()
            logger.info("BM25 index loaded: %d docs", len(self._corpus))
            return True
        except Exception as exc:
            logger.warning("Failed to load BM25 index: %s", exc)
            return False

    def count(self) -> int:
        return len(self._corpus)


_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _index
    if _index is None:
        _index = BM25Index()
        _index.load()
    return _index
