"""Unit tests for FormatterAgent."""
from __future__ import annotations

import pytest
from src.agents.formatter_agent import FormatterAgent
from src.agents.state import RAGState, CriticResult


@pytest.fixture
def agent():
    return FormatterAgent()


def make_state(**kwargs) -> RAGState:
    defaults = {
        "query": "What is Python?",
        "raw_answer": "Python is a programming language.",
        "sources": [{"chunk_id": "c1", "source": "doc.pdf", "score": 0.9}],
        "retrieval_scores": [0.9],
        "compliance_passed": True,
    }
    defaults.update(kwargs)
    return RAGState(**defaults)


class TestFormatterAgent:
    def test_formats_answer(self, agent):
        state = make_state()
        result = agent.format(state)
        assert result.final_answer != ""
        assert "Python" in result.final_answer

    def test_includes_sources(self, agent):
        state = make_state()
        result = agent.format(state)
        assert "doc.pdf" in result.final_answer or "Sources" in result.final_answer

    def test_confidence_in_range(self, agent):
        state = make_state()
        result = agent.format(state)
        assert 0.0 <= result.confidence <= 1.0

    def test_compliance_blocked_passthrough(self, agent):
        state = make_state(
            compliance_passed=False,
            final_answer="⚠️ Blocked",
        )
        result = agent.format(state)
        assert "Blocked" in result.final_answer

    def test_error_state_handled(self, agent):
        state = make_state(raw_answer="", error="Something went wrong")
        result = agent.format(state)
        assert result.final_answer != ""

    def test_critique_incorporated(self, agent):
        critique = CriticResult(
            faithfulness_score=0.4,
            hallucinated_claims=["This claim is unsupported"],
        )
        state = make_state(critique=critique)
        result = agent.format(state)
        # Should include hallucination warning
        assert "⚠️" in result.final_answer or result.confidence < 0.8

    def test_high_faithfulness_no_warning(self, agent):
        critique = CriticResult(faithfulness_score=0.95, hallucinated_claims=[])
        state = make_state(critique=critique)
        result = agent.format(state)
        assert "hallucinated" not in result.final_answer.lower()
