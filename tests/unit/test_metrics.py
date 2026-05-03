"""Unit tests for deterministic evaluation metrics."""
from __future__ import annotations

import math
import pytest

from src.evaluation.metrics import (
    context_precision,
    context_recall,
    cosine_similarity,
    context_relevance,
    mrr,
    ndcg_at_k,
    citation_accuracy,
    latency_stats,
    compute_all_deterministic,
)


class TestContextPrecision:
    def test_perfect_precision(self):
        assert context_precision(["a", "b"], {"a", "b"}) == 1.0

    def test_zero_precision(self):
        assert context_precision(["x", "y"], {"a", "b"}) == 0.0

    def test_partial_precision(self):
        assert context_precision(["a", "x"], {"a", "b"}) == 0.5

    def test_empty_retrieved(self):
        assert context_precision([], {"a"}) == 0.0


class TestContextRecall:
    def test_perfect_recall(self):
        assert context_recall(["a", "b", "c"], {"a", "b"}) == 1.0

    def test_zero_recall(self):
        assert context_recall(["x"], {"a", "b"}) == 0.0

    def test_partial_recall(self):
        assert context_recall(["a"], {"a", "b"}) == 0.5

    def test_empty_relevant(self):
        assert context_recall(["a"], set()) == 1.0


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 0.0]) == 0.0

    def test_negative_cosine(self):
        result = cosine_similarity([1.0, 0.0], [-1.0, 0.0])
        assert result == pytest.approx(-1.0)


class TestMRR:
    def test_first_result_relevant(self):
        assert mrr(["a", "b", "c"], {"a"}) == pytest.approx(1.0)

    def test_second_result_relevant(self):
        assert mrr(["x", "a", "b"], {"a"}) == pytest.approx(0.5)

    def test_no_relevant(self):
        assert mrr(["x", "y"], {"a"}) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        score = ndcg_at_k(["a", "b"], {"a", "b"}, k=2)
        assert score == pytest.approx(1.0)

    def test_zero_ndcg(self):
        score = ndcg_at_k(["x", "y"], {"a", "b"}, k=2)
        assert score == 0.0


class TestCitationAccuracy:
    def test_all_valid(self):
        assert citation_accuracy("answer [1] [2]", {1, 2}) == 1.0

    def test_invalid_citation(self):
        assert citation_accuracy("answer [99]", {1, 2}) == 0.0

    def test_no_citations(self):
        assert citation_accuracy("answer without citations", {1, 2}) == 1.0


class TestLatencyStats:
    def test_empty(self):
        stats = latency_stats([])
        assert stats.count == 0

    def test_single_value(self):
        stats = latency_stats([100.0])
        assert stats.mean == 100.0
        assert stats.p50 == 100.0

    def test_percentiles(self):
        data = list(range(1, 101))  # 1 to 100
        stats = latency_stats(data)
        assert stats.p50 == pytest.approx(50.5, abs=1.0)
        assert stats.p95 >= 95.0
        assert stats.count == 100


class TestComputeAll:
    def test_runs_without_error(self):
        emb = [0.1, 0.2, 0.3]
        result = compute_all_deterministic(
            retrieved_ids=["a", "b"],
            relevant_ids={"a"},
            query_emb=emb,
            chunk_embs=[emb, [0.4, 0.5, 0.6]],
            answer="answer [1]",
            latency_ms=250.0,
        )
        assert result.context_precision == pytest.approx(0.5)
        assert result.context_recall == pytest.approx(1.0)
        assert result.latency_ms == 250.0
        assert 0.0 <= result.context_relevance <= 1.0
