"""ChromaDB client wrapper (SQLite-backed, no external server needed)."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.utils.config import get_settings
from src.utils.exceptions import VectorDBError
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()


class ChromaClient:
    def __init__(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            # Try to disable telemetry completely
            try:
                # Monkey patch PostHog to prevent telemetry errors
                import chromadb.telemetry.product.posthog as posthog_module
                if hasattr(posthog_module, 'Posthog'):
                    original_capture = posthog_module.Posthog.capture
                    def safe_capture(self, *args, **kwargs):
                        try:
                            return original_capture(*args, **kwargs)
                        except TypeError:
                            # Ignore telemetry errors
                            pass
                    posthog_module.Posthog.capture = safe_capture
            except Exception:
                # If monkey patching fails, continue anyway
                pass

            self._client = chromadb.PersistentClient(
                path=_settings.chroma_persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info("ChromaDB initialised at %s", _settings.chroma_persist_dir)
        except ImportError as exc:
            raise VectorDBError("chromadb not installed") from exc

    def get_or_create_collection(self, name: str | None = None):
        cname = name or _settings.chroma_collection
        try:
            return self._client.get_or_create_collection(
                name=cname,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            raise VectorDBError(f"Failed to get/create collection {cname}: {exc}") from exc

    def health(self) -> dict[str, Any]:
        try:
            collections = self._client.list_collections()
            return {"status": "ok", "collections": [c.name for c in collections]}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    def delete_collection(self, name: str | None = None) -> None:
        cname = name or _settings.chroma_collection
        self._client.delete_collection(cname)
        logger.info("Deleted ChromaDB collection: %s", cname)


@lru_cache(maxsize=1)
def get_chroma_client() -> ChromaClient:
    return ChromaClient()
