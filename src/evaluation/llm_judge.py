"""Layer 2: LLM-as-judge evaluation using qwen:1.8b."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.model_manager import get_model_manager

logger = get_logger(__name__)
_settings = get_settings()


@dataclass
class LLMJudgeResult:
    faithfulness_score: float = 0.0
    relevancy_score: float = 0.0
    completeness_score: float = 0.0
    supported_claims: int = 0
    total_claims: int = 0
    hallucinated_claims: list[str] = None
    covered_aspects: list[str] = None
    missing_aspects: list[str] = None
    faithfulness_reasoning: str = ""
    relevancy_reasoning: str = ""
    completeness_reasoning: str = ""

    def __post_init__(self) -> None:
        if self.hallucinated_claims is None:
            self.hallucinated_claims = []
        if self.covered_aspects is None:
            self.covered_aspects = []
        if self.missing_aspects is None:
            self.missing_aspects = []


class LLMJudge:
    """Run all three LLM evaluation prompts."""

    def __init__(self) -> None:
        self._manager = get_model_manager()

    async def evaluate(
        self, query: str, context: str, answer: str, ground_truth: str = ""
    ) -> LLMJudgeResult:
        faith = await self._faithfulness(query, context, answer)
        relev = await self._relevancy(query, answer)
        comp = await self._completeness(query, answer, ground_truth)
        return LLMJudgeResult(
            faithfulness_score=faith.get("faithfulness_score", 0.5),
            relevancy_score=relev.get("relevancy_score", 0.5),
            completeness_score=comp.get("completeness_score", 0.5),
            supported_claims=faith.get("supported_claims", 0),
            total_claims=faith.get("total_claims", 1),
            hallucinated_claims=faith.get("hallucinated_claims", []),
            covered_aspects=comp.get("covered_aspects", []),
            missing_aspects=comp.get("missing_aspects", []),
            faithfulness_reasoning=faith.get("reasoning", ""),
            relevancy_reasoning=relev.get("reasoning", ""),
            completeness_reasoning=comp.get("reasoning", ""),
        )

    async def _faithfulness(self, query: str, context: str, answer: str) -> dict:
        prompt = f"""You are a strict fact-checker. Given question, context, answer:
1. Break answer into atomic claims
2. Verify each claim against context
3. Output JSON: {{"faithfulness_score": 0.0-1.0, "supported_claims": int, "total_claims": int, "hallucinated_claims": [], "reasoning": "..."}}
Rules: unsupported info = hallucinated. Contradiction = score 0. Be STRICT.

Question: {query}
Context: {context[:1500]}
Answer: {answer}"""
        return await self._call_llm(prompt)

    async def _relevancy(self, query: str, answer: str) -> dict:
        prompt = f"""Score how well the answer addresses the question.
Output JSON: {{"relevancy_score": 0.0-1.0, "reasoning": "..."}}
1.0 = perfectly addresses all aspects. 0.0 = completely off-topic.

Question: {query}
Answer: {answer}"""
        return await self._call_llm(prompt)

    async def _completeness(self, query: str, answer: str, ground_truth: str) -> dict:
        aspects_hint = f"\nGround truth for reference: {ground_truth[:500]}" if ground_truth else ""
        prompt = f"""Check if the answer covers all important aspects of the question.
Output JSON: {{"completeness_score": 0.0-1.0, "covered_aspects": [], "missing_aspects": [], "reasoning": "..."}}

Question: {query}{aspects_hint}
Answer: {answer}"""
        return await self._call_llm(prompt)

    async def _call_llm(self, prompt: str) -> dict:
        try:
            raw = await self._manager.generate(
                _settings.llm_judge,
                prompt,
                temperature=0.0,
                max_tokens=512,
            )
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception as exc:
            logger.warning("LLM judge call failed: %s", exc)
        return {}
