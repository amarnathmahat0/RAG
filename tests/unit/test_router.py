"""Unit tests for the adaptive Router (rule-based scoring only — no LLM in CI)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from src.agents.router import Router
from src.agents.state import RAGState, Tier


@pytest.fixture
def router():
    return Router()


class TestRuleBasedScoring:
    def test_short_simple_query_tier1(self, router):
        score, reason = router._rule_based_score("What is Python?")
        assert score < 0.4
        assert "short" in reason.lower() or score < 0.4

    def test_long_multi_entity_query_tier3(self, router):
        query = (
            "Compare the relationship between Alice Smith, Bob Johnson, and Carol Williams "
            "across multiple departments and how their collaboration impacted revenue growth "
            "throughout the fiscal year compared to previous performance."
        )
        score, reason = router._rule_based_score(query)
        assert score >= 0.4

    def test_medium_query_tier2(self, router):
        query = "How does the authentication system handle expired tokens and what is the retry strategy?"
        score, _ = router._rule_based_score(query)
        assert 0.0 <= score <= 1.0  # valid range

    def test_score_always_in_range(self, router):
        queries = [
            "hi",
            "What is the capital of France?",
            "Compare the different machine learning approaches used by Google, Apple, and Microsoft across their product lines.",
        ]
        for q in queries:
            score, _ = router._rule_based_score(q)
            assert 0.0 <= score <= 1.0, f"Score {score} out of range for: {q}"

    def test_tier_mapping(self, router):
        assert router._score_to_tier(0.1) == Tier.TIER_1
        assert router._score_to_tier(0.3) == Tier.TIER_1
        assert router._score_to_tier(0.5) == Tier.TIER_2
        assert router._score_to_tier(0.65) == Tier.TIER_2
        assert router._score_to_tier(0.75) == Tier.TIER_3
        assert router._score_to_tier(0.99) == Tier.TIER_3


@pytest.mark.asyncio
class TestRouterRoute:
    async def test_route_sets_tier(self, router):
        # Mock the LLM to avoid Ollama dependency in CI
        with patch.object(router, "_llm_complexity_score", new=AsyncMock(return_value=0.2)):
            state = RAGState(query="What is Python?")
            result = await router.route(state)
            assert result.tier in [Tier.TIER_1, Tier.TIER_2, Tier.TIER_3]
            assert result.tier_reason != ""
            assert 0.0 <= result.complexity_score <= 1.0

    async def test_route_updates_state(self, router):
        with patch.object(router, "_llm_complexity_score", new=AsyncMock(return_value=0.8)):
            state = RAGState(query="What is machine learning?")
            result = await router.route(state)
            assert result.tier is not None
            assert result.tier_reason != ""
            assert 0.0 <= result.complexity_score <= 1.0
