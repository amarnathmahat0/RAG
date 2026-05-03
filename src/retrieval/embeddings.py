"""Embedding helpers using Ollama models."""
from __future__ import annotations

import asyncio
from typing import Any

from src.utils.config import get_settings
from src.utils.exceptions import EmbeddingError
from src.utils.logger import get_logger
from src.utils.model_manager import get_model_manager

logger = get_logger(__name__)
_settings = get_settings()


class EmbeddingService:
    """Thin async embedding interface backed by Ollama."""

    def __init__(self) -> None:
        self._manager = get_model_manager()
        # FIX 4: Log active embedding models at startup for mismatch diagnosis
        logger.info(
            "EmbeddingService initialised | primary_model=%s | fallback_model=%s",
            _settings.embed_model_primary,
            _settings.embed_model_fallback,
        )

    async def embed_query(self, text: str) -> list[float]:
        return (await self.embed_batch([text]))[0]

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        
        # Try cache first
        from src.utils.cache import get_cache_manager
        cache = get_cache_manager()
        cached_embeddings = []
        texts_to_embed = []
        text_indices = []
        
        for i, text in enumerate(texts):
            cached = cache.get_embedding(text)
            if cached:
                cached_embeddings.append((i, cached))
            else:
                texts_to_embed.append(text)
                text_indices.append(i)
        
        logger.info(
            "EmbeddingService using primary model=%s for batch size=%d (cached=%d)",
            _settings.embed_model_primary,
            len(texts),
            len(cached_embeddings),
        )
        
        # Embed uncached texts
        new_embeddings = {}
        if texts_to_embed:
            try:
                embedded = await self._manager.embed(_settings.embed_model_primary, texts_to_embed)
                for text, embedding in zip(texts_to_embed, embedded):
                    cache.set_embedding(text, embedding)
                    new_embeddings[text] = embedding
            except Exception as exc:
                logger.warning("Primary embedding failed (%s), trying fallback…", exc)
                try:
                    logger.info("EmbeddingService fallback to model=%s", _settings.embed_model_fallback)
                    embedded = await self._manager.embed(_settings.embed_model_fallback, texts_to_embed)
                    for text, embedding in zip(texts_to_embed, embedded):
                        cache.set_embedding(text, embedding)
                        new_embeddings[text] = embedding
                except Exception as exc2:
                    raise EmbeddingError(f"All embedding models failed: {exc2}") from exc2
        
        # Recombine in original order
        result = [None] * len(texts)
        for i, embedding in cached_embeddings:
            result[i] = embedding
        for idx, text in zip(text_indices, texts_to_embed):
            result[idx] = new_embeddings[text]
        
        return result


_service: EmbeddingService | None = None


def get_embedding_service() -> EmbeddingService:
    global _service
    if _service is None:
        _service = EmbeddingService()
    return _service
