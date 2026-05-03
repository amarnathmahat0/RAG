"""Response builder: assemble prompt → call phi3:mini → stream tokens."""
from __future__ import annotations

import time
from typing import AsyncIterator

from src.agents.state import RAGState
from src.generation.context_compressor import ContextCompressor
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.model_manager import get_model_manager

logger = get_logger(__name__)
_settings = get_settings()

_SYSTEM_PROMPT = """You are a precise, helpful assistant that answers questions based ONLY on the provided context.

Rules:
1. Answer ONLY from the provided context. Never add outside knowledge.
2. If the context doesn't contain the answer, say: "The provided documents don't contain enough information to answer this question."
3. Cite sources as inline numbers [1], [2], etc. matching the context chunk numbers.
4. Be concise. Lead with the direct answer, then supporting details.
5. Never make up facts. Never extrapolate beyond the context.
6. Assume terms like "user", "author", "he", or "she" refer to the subject of the provided documents."""


class ResponseBuilder:
    """Build and optionally stream answers using phi3:mini (with qwen fallback)."""

    def __init__(self) -> None:
        self._manager = get_model_manager()
        self._compressor = ContextCompressor()

    async def build(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        prompt = await self._build_prompt(state)

        # Try phi3:mini first, fall back to qwen:1.8b
        for model in [_settings.llm_generator, _settings.llm_judge]:
            try:
                answer = await self._manager.generate(
                    model,
                    prompt,
                    system=_SYSTEM_PROMPT,
                    temperature=0.1,
                    max_tokens=1024,
                )
                state.raw_answer = answer.strip()
                elapsed = (time.perf_counter() - t0) * 1000
                logger.info(
                    "GENERATE model=%s len=%d elapsed_ms=%.1f",
                    model,
                    len(state.raw_answer),
                    elapsed,
                )
                return state
            except Exception as exc:
                logger.warning("Generation with %s failed: %s; trying fallback…", model, exc)

        state.raw_answer = (
            "I was unable to generate a response at this time. "
            "Please try again or rephrase your question."
        )
        state.error = "All generation models failed."
        return state

    async def stream(self, state: RAGState) -> AsyncIterator[str]:
        """Yield answer tokens."""
        prompt = await self._build_prompt(state)
        try:
            async for token in self._manager.generate_stream(
                _settings.llm_generator,
                prompt,
                system=_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=1024,
            ):
                yield token
        except Exception as exc:
            logger.error("Streaming generation failed: %s", exc)
            yield "Error: unable to generate response."

    async def _build_prompt(self, state: RAGState) -> str:
        # Compress context
        compressed = await self._compressor.compress(state.query, state.context)

        ctx_parts = []
        for i, chunk in enumerate(compressed[:6], 1):
            text = chunk.get("text", "").strip()
            src = chunk.get("source", "")
            if text:
                ctx_parts.append(f"[{i}] Source: {src}\n{text}")

        context_str = "\n\n---\n\n".join(ctx_parts) if ctx_parts else "No context available."

        return (
            f"Context:\n{context_str}\n\n"
            f"Question: {state.query}\n\n"
            "Answer (cite sources as [1], [2], etc.):"
        )
