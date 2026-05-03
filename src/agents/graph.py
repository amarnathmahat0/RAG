"""LangGraph orchestration: router → tier → retriever → critic → compliance → formatter."""
from __future__ import annotations

import time
import uuid
from typing import AsyncIterator

from src.agents.compliance_agent import ComplianceAgent
from src.agents.critic_agent import CriticAgent
from src.agents.formatter_agent import FormatterAgent
from src.agents.retriever_agent import RetrieverAgent
from src.agents.router import Router
from src.agents.state import RAGState, Tier
from src.generation.response_builder import ResponseBuilder
from src.utils.logger import get_logger
from src.utils.model_manager import get_model_manager

logger = get_logger(__name__)


class RAGPipeline:
    """Full RAG pipeline orchestrator (replaces LangGraph for M2 RAM budget)."""

    def __init__(self) -> None:
        self._router = Router()
        self._retriever = RetrieverAgent()
        self._critic = CriticAgent()
        self._compliance = ComplianceAgent()
        self._formatter = FormatterAgent()
        self._builder = ResponseBuilder()

    async def run(self, query: str, request_id: str | None = None, documents: list[str] | None = None) -> RAGState:
        """Execute the full pipeline and return final state."""
        state = RAGState(
            query=query,
            request_id=request_id or str(uuid.uuid4()),
            start_time_ms=time.perf_counter() * 1000,
            documents=documents,
        )

        # ── Step 1: Compliance pre-check (query only) ──────────────────────────
        from src.agents.compliance_agent import ComplianceAgent

        pre_comp = ComplianceAgent()
        state.raw_answer = ""  # empty answer for pre-check
        state = pre_comp.check(state)
        if not state.compliance_passed:
            state = self._formatter.format(state)
            return state
        # Reset for next use
        state.compliance_passed = True
        state.compliance_violations = []

        # ── Step 2: Route ──────────────────────────────────────────────────────
        state = await self._router.route(state)

        # ── Step 3: Retrieve (with possible re-retrieval loop) ─────────────────
        state = await self._retriever.retrieve(state, state.documents)

        # ── Step 4: Generate ───────────────────────────────────────────────────
        state = await self._builder.build(state)

        # ── Step 5: Critique + optional re-retrieve loop ──────────────────────
        state = await self._critic.critique(state)
        if (
            state.critique
            and state.critique.faithfulness_score < 0.3
            and state.critic_loop_count < state.max_critic_loops
        ):
            logger.info(
                "Critic failed (score=%.3f); re-retrieving with stricter filters…",
                state.critique.faithfulness_score,
            )
            # Tighten retrieval and regenerate
            state.retrieval_iteration = 0
            state = await self._retriever.retrieve(state, state.documents)
            state = await self._builder.build(state)
            state = await self._critic.critique(state)

        # ── Step 6: Compliance post-check (answer) ────────────────────────────
        state = self._compliance.check(state)

        # ── Step 7: Format ────────────────────────────────────────────────────
        state = self._formatter.format(state)
        return state

    async def stream(self, query: str, request_id: str | None = None, documents: list[str] | None = None) -> AsyncIterator[str]:
        """Stream tokens from generator; yield final metadata at end."""
        state = RAGState(
            query=query,
            request_id=request_id or str(uuid.uuid4()),
            start_time_ms=time.perf_counter() * 1000,
            documents=documents,
        )
        # Pre-checks
        from src.agents.compliance_agent import ComplianceAgent

        state = ComplianceAgent().check(state)
        if not state.compliance_passed:
            yield state.final_answer
            return
        state.compliance_passed = True
        state.compliance_violations = []

        state = await self._router.route(state)
        state = await self._retriever.retrieve(state, documents)

        # Stream generation
        async for token in self._builder.stream(state):
            yield token

        # No critic/compliance on streamed content — post-process separately if needed


_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
