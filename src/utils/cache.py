"""Query and embedding cache for latency optimization."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()


class CacheManager:
    """Simple file-based cache for queries and embeddings."""

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or _settings.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl_sec = _settings.cache_ttl_sec

    def _get_key_path(self, key: str) -> Path:
        """Generate cache file path for key."""
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.json"

    def _get_cache_key(
        self, query: str | None = None, embedding_text: str | None = None, documents: list[str] | None = None
    ) -> str:
        """Generate cache key from query + documents."""
        parts = []
        if query:
            parts.append(f"q:{query}")
        if embedding_text:
            parts.append(f"e:{embedding_text}")
        if documents:
            parts.append(f"d:{','.join(sorted(documents))}")
        return "|".join(parts)

    def get_query_result(self, query: str, documents: list[str] | None = None) -> dict[str, Any] | None:
        """Retrieve cached query result if valid."""
        if not _settings.enable_query_cache:
            return None

        key = self._get_cache_key(query=query, documents=documents)
        cache_file = self._get_key_path(key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            
            # Check TTL
            if time.time() - data.get("timestamp", 0) > self.ttl_sec:
                cache_file.unlink()  # Delete expired
                return None

            logger.debug("Query cache HIT for: %s", query[:50])
            return data.get("result")
        except Exception as e:
            logger.debug("Cache read error: %s", e)
            return None

    def set_query_result(self, query: str, result: dict[str, Any], documents: list[str] | None = None) -> None:
        """Cache query result."""
        if not _settings.enable_query_cache:
            return

        key = self._get_cache_key(query=query, documents=documents)
        cache_file = self._get_key_path(key)

        try:
            cache_data = {"timestamp": time.time(), "result": result}
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
            logger.debug("Query cached: %s", query[:50])
        except Exception as e:
            logger.warning("Cache write error: %s", e)

    def get_embedding(self, text: str) -> list[float] | None:
        """Retrieve cached embedding if valid."""
        if not _settings.enable_embedding_cache:
            return None

        key = self._get_cache_key(embedding_text=text)
        cache_file = self._get_key_path(key)

        if not cache_file.exists():
            return None

        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
            
            # Check TTL
            if time.time() - data.get("timestamp", 0) > self.ttl_sec:
                cache_file.unlink()
                return None

            logger.debug("Embedding cache HIT for text length: %d", len(text))
            return data.get("embedding")
        except Exception as e:
            logger.debug("Embedding cache read error: %s", e)
            return None

    def set_embedding(self, text: str, embedding: list[float]) -> None:
        """Cache embedding."""
        if not _settings.enable_embedding_cache:
            return

        key = self._get_cache_key(embedding_text=text)
        cache_file = self._get_key_path(key)

        try:
            cache_data = {"timestamp": time.time(), "embedding": embedding}
            with open(cache_file, "w") as f:
                json.dump(cache_data, f)
            logger.debug("Embedding cached for text length: %d", len(text))
        except Exception as e:
            logger.warning("Embedding cache write error: %s", e)

    def clear_cache(self) -> None:
        """Clear all cache files."""
        try:
            for f in self.cache_dir.glob("*.json"):
                f.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.warning("Cache clear error: %s", e)


_cache_manager: CacheManager | None = None


def get_cache_manager() -> CacheManager:
    """Get singleton cache manager."""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
