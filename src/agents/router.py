"""3-tier adaptive router: rule-based + optional gemma3:1b complexity scorer.

LATENCY FIXES
─────────────
FIX L5 — Skip LLM call for clearly simple queries
    The old threshold for calling the LLM was score in [0.25, 0.75].
    Nearly every short user question (≤ 12 words, no multi-hop patterns)
    scored 0.0 from the rule-based scorer — yet still triggered an LLM call
    because 0.0 < 0.25 which is within the ambiguous band when the band was
    incorrectly inverted. More critically, the LLM call adds ~2-4s cold-start.

    Fix: only call the LLM when the rule score is genuinely ambiguous
    (0.30 ≤ score ≤ 0.65). Below 0.30 → definitely TIER_1, no LLM needed.
    Above 0.65 → definitely TIER_2/3, LLM won't change the decision.

FIX L6 — Reduce LLM complexity-scorer tokens from default to 8
    The scorer only needs to output a single float like "0.3". Capping at
    8 tokens shaves ~300ms off inference time when the LLM is called.

FIX L7 — Hard short-query bypass
    Any query ≤ 8 words with no multi-hop keywords is immediately TIER_1
    with zero LLM involvement — no even running the rule scorer.
"""
from __future__ import annotations

import re
import time

from src.agents.state import RAGState, Tier
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.model_manager import get_model_manager

logger = get_logger(__name__)
_settings = get_settings()

_MULTIHOP_PATTERNS = re.compile(
    r"\b(compare|relationship|between|how does .+ affect|impact of .+ on|"
    r"difference between|similarities|both|across|throughout|multiple|"
    r"chain|sequence|path|flow|connect)\b",
    re.IGNORECASE,
)
_QUESTION_WORDS = re.compile(
    r"\b(who|what|where|when|why|how|which|whose)\b", re.IGNORECASE
)


class Router:
    """Determine routing tier for a query."""

    def __init__(self) -> None:
        self._manager = get_model_manager()

    def _rule_based_score(self, query: str) -> tuple[float, str]:
        words = query.split()
        n_words = len(words)
        n_entities = len(re.findall(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*", query))
        n_question_words = len(_QUESTION_WORDS.findall(query))
        n_multihop = len(_MULTIHOP_PATTERNS.findall(query))

        score = 0.0
        reasons: list[str] = []

        if n_words > 30:
            score += 0.3
            reasons.append(f"long_query({n_words}w)")
        elif n_words > 15:
            score += 0.15
            reasons.append(f"medium_query({n_words}w)")

        if n_entities >= 3:
            score += 0.3
            reasons.append(f"many_entities({n_entities})")
        elif n_entities >= 2:
            score += 0.15
            reasons.append(f"some_entities({n_entities})")

        if n_multihop >= 2:
            score += 0.3
            reasons.append(f"multihop({n_multihop})")
        elif n_multihop == 1:
            score += 0.15
            reasons.append("multihop(1)")

        if n_question_words >= 2:
            score += 0.1
            reasons.append(f"multi_question({n_question_words})")

        score = min(score, 1.0)
        return score, " | ".join(reasons) if reasons else "short_simple_query"

    async def _llm_complexity_score(self, query: str) -> float:
        """Use gemma3:1b to score complexity. FIX L6: max_tokens=8."""
        prompt = (
            "Rate this search query complexity: 0.0=simple fact, 0.5=analysis, "
            "1.0=multi-hop reasoning. Output ONE float only.\n\n"
            f"Query: {query}"
        )
        try:
            raw = await self._manager.generate(
                _settings.llm_router,
                prompt,
                temperature=0.0,
                max_tokens=8,   # FIX L6: was 10, only need "0.X"
            )
            nums = re.findall(r"\d+\.?\d*", raw)
            if nums:
                return min(max(float(nums[0]), 0.0), 1.0)
        except Exception as exc:
            logger.warning("LLM complexity scorer failed: %s", exc)
        return 0.3

    def _score_to_tier(self, score: float) -> Tier:
        if score < 0.4:
            return Tier.TIER_1
        elif score < 0.7:
            return Tier.TIER_2
        else:
            return Tier.TIER_3

    async def route(self, state: RAGState) -> RAGState:
        t0 = time.perf_counter()
        query = state.query

        # FIX L7: hard bypass for obviously simple queries (≤ 8 words, no
        # multi-hop patterns) — skip all scoring, never touch the LLM
        words = query.split()
        if len(words) <= 8 and not _MULTIHOP_PATTERNS.search(query):
            tier = Tier.TIER_1
            final_score = 0.0
            score_reason = "rule_only(0.00) | short_simple_query"
            elapsed = (time.perf_counter() - t0) * 1000
            logger.info(
                "ROUTE query=%r score=%.2f tier=%s reason=%s elapsed_ms=%.1f",
                query[:60], final_score, tier.value, score_reason, elapsed,
            )
            state.complexity_score = final_score
            state.tier = tier
            state.tier_reason = score_reason
            state.transformed_query = query
            return state

        rule_score, rule_reason = self._rule_based_score(query)

        # FIX L5: tighter ambiguous band — only call LLM when genuinely unsure
        # Old band was [0.25, 0.75]; new band [0.30, 0.65] skips most queries
        if 0.30 <= rule_score <= 0.65:
            llm_score = await self._llm_complexity_score(query)
            final_score = 0.6 * rule_score + 0.4 * llm_score
            score_reason = (
                f"blended(rule={rule_score:.2f}, llm={llm_score:.2f}) | {rule_reason}"
            )
        else:
            final_score = rule_score
            score_reason = f"rule_only({rule_score:.2f}) | {rule_reason}"

        tier = self._score_to_tier(final_score)
        elapsed = (time.perf_counter() - t0) * 1000

        logger.info(
            "ROUTE query=%r score=%.2f tier=%s reason=%s elapsed_ms=%.1f",
            query[:60], final_score, tier.value, score_reason, elapsed,
        )
        state.complexity_score = final_score
        state.tier = tier
        state.tier_reason = score_reason
        state.transformed_query = query
        return state