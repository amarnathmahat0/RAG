"""Context compressor: trim irrelevant sentences if context > token budget."""
from __future__ import annotations

import re

from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.model_manager import get_model_manager

logger = get_logger(__name__)
_settings = get_settings()

_DEFAULT_BUDGET_CHARS = 3000  # ~750 tokens


class ContextCompressor:
    """Compress context to fit within token budget using gemma3:1b."""

    def __init__(self, budget_chars: int = _DEFAULT_BUDGET_CHARS) -> None:
        self.budget_chars = budget_chars
        self._manager = get_model_manager()

    async def compress(self, query: str, context_chunks: list[dict]) -> list[dict]:
        """Return compressed context list. Modifies .text in-place for large chunks."""
        total_chars = sum(len(c.get("text", "")) for c in context_chunks)
        if total_chars <= self.budget_chars:
            return context_chunks

        logger.info(
            "Context compression: total_chars=%d > budget=%d; compressing…",
            total_chars,
            self.budget_chars,
        )
        # Simple strategy: truncate long chunks proportionally
        budget_per_chunk = self.budget_chars // max(len(context_chunks), 1)
        compressed = []
        for chunk in context_chunks:
            text = chunk.get("text", "")
            if len(text) > budget_per_chunk:
                # LLM-based sentence filtering for large chunks
                text = await self._llm_compress(query, text, budget_per_chunk)
            compressed.append({**chunk, "text": text})
        return compressed

    async def _llm_compress(self, query: str, text: str, target_chars: int) -> str:
        """Use gemma3:1b to keep only the most relevant sentences."""
        prompt = (
            f"Query: {query}\n\n"
            f"Text (keep only sentences directly relevant to the query, "
            f"output compressed text ≤{target_chars} chars):\n\n{text[:2000]}"
        )
        try:
            result = await self._manager.generate(
                _settings.llm_router,
                prompt,
                temperature=0.0,
                max_tokens=target_chars // 4,
            )
            return result.strip() or text[:target_chars]
        except Exception as exc:
            logger.warning("LLM compression failed: %s", exc)
            return text[:target_chars]
