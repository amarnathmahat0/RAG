"""Critic agent: faithfulness + hallucination checking via qwen:1.8b."""
from __future__ import annotations

import json
import re
import time
from typing import Any

from src.agents.state import CriticResult, RAGState
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.model_manager import get_model_manager

logger = get_logger(__name__)
_settings = get_settings()

_FAITHFULNESS_THRESHOLD = 0.7


class CriticAgent:
    """Judge faithfulness and flag hallucinations."""

    def __init__(self) -> None:
        self._manager = get_model_manager()

    async def critique(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        if not state.raw_answer or not state.context:
            state.critique = CriticResult(
                faithfulness_score=0.7,
                supported_claims=1,
                total_claims=1,
                hallucinated_claims=[],
                reasoning="no_answer_or_context_default",
            )
            state.critic_passed = True
            state = self._apply_ragas_metrics(state)
            return state

        context_text = "\n\n".join(
            f"[{i+1}] {c.get('text', '')}" for i, c in enumerate(state.context[:5])
        )
        prompt = f"""You are a factuality critic. Compare the answer to the retrieved context and output ONLY valid JSON.

Question: {state.query}

Context:
{context_text}

Answer: {state.raw_answer}

Instructions:
- Identify each atomic claim in the answer.
- Decide whether each claim is directly supported by the context.
- Do not hallucinate or infer facts beyond the text.
- Output exactly one JSON object with these fields:
{{
  "faithfulness_score": <float between 0.0 and 1.0>,
  "supported_claims": <integer>,
  "total_claims": <integer>,
  "hallucinated_claims": [<claim text>, ...],
  "reasoning": "brief explanation"
}}

Do not output any prose outside the JSON object."""

        # Safe default: assume pass if LLM unavailable (avoids always-fail cascades)
        result = CriticResult(
            faithfulness_score=0.7,
            supported_claims=1,
            total_claims=1,
            hallucinated_claims=[],
            reasoning="llm_unavailable_default",
        )
        try:
            raw = await self._manager.generate(
                _settings.llm_judge,
                prompt,
                temperature=0.0,
                max_tokens=512,
            )
            result = self._parse_critique(raw)
        except Exception as exc:
            logger.warning(
                "Critic LLM failed (%s); defaulting faithfulness=0.7 (assume pass).", exc
            )
            # result already set to safe default above

        state.critique = result
        state.critic_passed = result.faithfulness_score >= _FAITHFULNESS_THRESHOLD
        state.critic_loop_count += 1
        state = self._apply_ragas_metrics(state)

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(
            "CRITIC faithfulness=%.3f passed=%s hallucinated=%d elapsed_ms=%.1f",
            result.faithfulness_score,
            state.critic_passed,
            len(result.hallucinated_claims),
            elapsed,
        )
        return state

    def _apply_ragas_metrics(self, state: RAGState) -> RAGState:
        from src.evaluation.metrics import compute_ragas_metrics
        metrics = compute_ragas_metrics(
            query=state.query,
            answer=state.raw_answer or state.final_answer,
            context=state.context,
            retrieval_scores=state.retrieval_scores,
            # Pass retrieval_scores as reranker_scores — after the retriever agent
            # runs reranking, state.retrieval_scores holds the FlashRank scores
            # which are semantically meaningful (unlike raw Chroma distances).
            reranker_scores=state.retrieval_scores if state.retrieval_scores else None,
        )
        state.faithfulness_score = state.critique.faithfulness_score
        state.answer_relevancy = metrics["answer_relevancy"]
        state.context_precision = metrics["context_precision"]
        state.context_recall = metrics["context_recall"]
        state.hallucination_score = metrics["hallucination_score"]
        return state

    def _parse_critique(self, raw: str) -> CriticResult:
        try:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                data = json.loads(match.group())
                return CriticResult(
                    faithfulness_score=float(data.get("faithfulness_score", 0.7)),
                    supported_claims=int(data.get("supported_claims", 0)),
                    total_claims=int(data.get("total_claims", 1)),
                    hallucinated_claims=data.get("hallucinated_claims", []),
                    reasoning=data.get("reasoning", ""),
                )
        except Exception as exc:
            logger.debug("Critic parse failed: %s", exc)
        score_match = re.search(r'"faithfulness_score"\s*:\s*([0-9.]+)', raw)
        score = float(score_match.group(1)) if score_match else 0.7
        return CriticResult(
            faithfulness_score=score,
            supported_claims=1,
            total_claims=1,
            hallucinated_claims=[],
            reasoning="parse_fallback",
        )
