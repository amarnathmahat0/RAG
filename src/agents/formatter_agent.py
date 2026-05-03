"""Formatter agent: structure answer with inline citations and confidence score."""
from __future__ import annotations

import time

from src.agents.state import RAGState
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FormatterAgent:
    """Format the raw answer into the standard response structure."""

    def format(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        if not state.compliance_passed:
            # Already set by compliance agent
            return state

        if state.error and not state.raw_answer:
            state.final_answer = (
                "I encountered an error processing your request. "
                f"Details: {state.error}"
            )
            state.confidence = 0.0
            return state

        # Build citation map
        sources = state.sources[:10]
        citation_map: dict[str, int] = {}
        for i, s in enumerate(sources, 1):
            cid = s.get("chunk_id", "")
            if cid:
                citation_map[cid] = i

        # Inject citations into answer if not already present
        answer = state.raw_answer.strip()
        if sources and "[1]" not in answer and "[Source" not in answer:
            answer = self._inject_citations(answer, citation_map)

        # Compute confidence
        retrieval_score = (
            sum(state.retrieval_scores) / len(state.retrieval_scores)
            if state.retrieval_scores
            else 0.5
        )
        faithfulness = (
            state.critique.faithfulness_score if state.critique else 0.5
        )
        confidence = round(0.5 * retrieval_score + 0.5 * faithfulness, 3)

        # Source lines
        source_lines = []
        for i, s in enumerate(sources, 1):
            src = s.get("source", "unknown")
            score = s.get("score", 0.0)
            source_lines.append(f"[{i}] {src} (score: {score:.3f})")

        # Structured output
        parts = [answer]
        if source_lines:
            parts.append("\n**Sources:**\n" + "\n".join(source_lines))

        hallucination_warning = ""
        if state.critique and state.critique.hallucinated_claims:
            hallucination_warning = (
                "\n\n⚠️ *Note: Some claims could not be fully verified against source documents.*"
            )
            parts.append(hallucination_warning)

        state.final_answer = "\n\n".join(parts)
        state.confidence = confidence
        state.end_time_ms = time.perf_counter() * 1000

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "FORMATTER confidence=%.3f sources=%d elapsed_ms=%.1f",
            confidence,
            len(sources),
            elapsed,
        )
        return state

    def _inject_citations(self, text: str, citation_map: dict[str, int]) -> str:
        """Append citation numbers to sentences that likely reference sources."""
        # Simple heuristic: append [1][2]... at the end of the first paragraph
        if citation_map:
            nums = sorted(set(citation_map.values()))[:3]
            cite_str = "".join(f"[{n}]" for n in nums)
            lines = text.split("\n\n", 1)
            lines[0] = lines[0].rstrip() + " " + cite_str
            return "\n\n".join(lines)
        return text
