"""Unit tests for RRF fusion in HybridSearch."""
from __future__ import annotations

import pytest
from src.retrieval.hybrid_search import HybridSearch, RetrievedChunk


class TestRRFFusion:
    def setup_method(self):
        self.hs = HybridSearch(top_k_dense=10, top_k_sparse=10, rrf_k=60)

    def test_fusion_combines_results(self):
        dense = [
            ("chunk_1", 0.9, "text1", {"source": "doc1"}),
            ("chunk_2", 0.8, "text2", {"source": "doc2"}),
        ]
        sparse = [
            ("chunk_2", 5.0),
            ("chunk_3", 3.0),
        ]
        fused = self.hs._rrf_fusion(dense, sparse)
        ids = [c.chunk_id for c in fused]
        # chunk_2 appears in both → should rank high
        assert "chunk_2" in ids
        assert ids.index("chunk_2") <= 1

    def test_fusion_deduplicates(self):
        dense = [("chunk_1", 0.9, "text1", {}), ("chunk_1", 0.9, "text1", {})]
        sparse = [("chunk_1", 2.0)]
        fused = self.hs._rrf_fusion(dense, sparse)
        chunk_ids = [c.chunk_id for c in fused]
        assert chunk_ids.count("chunk_1") == 1

    def test_empty_inputs(self):
        fused = self.hs._rrf_fusion([], [])
        assert fused == []

    def test_scores_decrease(self):
        dense = [
            ("a", 0.9, "text_a", {}),
            ("b", 0.8, "text_b", {}),
            ("c", 0.7, "text_c", {}),
        ]
        fused = self.hs._rrf_fusion(dense, [])
        scores = [c.score for c in fused]
        assert scores == sorted(scores, reverse=True)

    def test_rrf_formula_correctness(self):
        """Verify RRF score = weight / (k + rank)."""
        dense = [("a", 1.0, "text_a", {})]
        fused = self.hs._rrf_fusion(dense, [])
        expected = self.hs.dense_weight / (60 + 1)
        assert abs(fused[0].score - expected) < 1e-9
