"""Unit tests for BM25Index."""
from __future__ import annotations

import pytest
from src.retrieval.sparse_index import BM25Index


class TestBM25Index:
    def setup_method(self):
        self.idx = BM25Index()
        docs = [
            "Python is a high-level programming language",
            "Machine learning uses algorithms to learn from data",
            "Deep learning is a subset of machine learning",
            "FastAPI is a modern web framework for Python",
        ]
        ids = [f"doc_{i}" for i in range(len(docs))]
        self.idx.add_documents(docs, ids)

    def test_search_returns_results(self):
        results = self.idx.search("Python programming", top_k=3)
        assert len(results) > 0

    def test_search_returns_tuple(self):
        results = self.idx.search("machine learning", top_k=2)
        for item in results:
            assert len(item) == 2
            assert isinstance(item[0], str)
            assert isinstance(item[1], float)

    def test_relevant_doc_ranks_high(self):
        results = self.idx.search("machine learning algorithms", top_k=4)
        ids = [r[0] for r in results]
        assert "doc_1" in ids[:2] or "doc_2" in ids[:2]

    def test_empty_query(self):
        results = self.idx.search("", top_k=5)
        # BM25 with empty query returns no positive scores
        assert isinstance(results, list)

    def test_irrelevant_query_low_score(self):
        results = self.idx.search("zxqwerty nonexistent term", top_k=5)
        # All scores should be 0 (filtered out)
        assert len(results) == 0

    def test_count(self):
        assert self.idx.count() == 4

    def test_empty_index_search(self):
        idx = BM25Index()
        results = idx.search("test", top_k=5)
        assert results == []
