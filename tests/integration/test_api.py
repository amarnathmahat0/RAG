"""Integration tests for FastAPI endpoints (pipeline mocked)."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient

from src.api.main import app
from src.agents.state import RAGState, Tier, CriticResult


@pytest.fixture
def client():
    return TestClient(app)


def _mock_state(query: str = "test query") -> RAGState:
    return RAGState(
        query=query,
        final_answer="This is a test answer.",
        tier=Tier.TIER_1,
        tier_reason="short_simple_query",
        confidence=0.85,
        sources=[{"chunk_id": "c1", "source": "doc.pdf", "score": 0.9}],
        critique=CriticResult(faithfulness_score=0.9),
        compliance_passed=True,
    )


class TestHealthEndpoint:
    def test_health_returns_200_or_503(self, client):
        with patch("src.api.routes.health._check_chroma", return_value={"status": "ok"}):
            with patch("src.api.routes.health._check_neo4j", return_value={"status": "ok"}):
                resp = client.get("/health")
                assert resp.status_code in [200, 503]

    def test_health_has_components(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "components" in data
        assert "status" in data


class TestQueryEndpoint:
    def test_missing_body_returns_422(self, client):
        resp = client.post("/query", json={})
        assert resp.status_code == 422

    def test_valid_query_returns_200(self, client):
        mock_state = _mock_state()
        with patch("src.api.routes.query.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.run = AsyncMock(return_value=mock_state)
            mock_get.return_value = mock_pipeline
            resp = client.post("/query", json={"query": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "tier" in data
        assert "confidence" in data

    def test_response_has_required_fields(self, client):
        mock_state = _mock_state()
        with patch("src.api.routes.query.get_pipeline") as mock_get:
            mock_pipeline = MagicMock()
            mock_pipeline.run = AsyncMock(return_value=mock_state)
            mock_get.return_value = mock_pipeline
            resp = client.post("/query", json={"query": "Test query"})
        data = resp.json()
        required = ["request_id", "query", "answer", "tier", "confidence", "sources", "latency_ms"]
        for field in required:
            assert field in data, f"Missing field: {field}"

    def test_empty_query_rejected(self, client):
        resp = client.post("/query", json={"query": ""})
        assert resp.status_code == 422

    def test_too_long_query_rejected(self, client):
        resp = client.post("/query", json={"query": "x" * 3000})
        assert resp.status_code == 422


class TestMetricsEndpoint:
    def test_metrics_endpoint_exists(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "nexus_rag" in resp.text or "python" in resp.text.lower()


class TestIngestEndpoint:
    def test_ingest_accepts_sources(self, client):
        with patch("src.api.routes.ingest._run_ingestion", new=AsyncMock()):
            resp = client.post(
                "/ingest",
                json={"sources": ["/tmp/test.pdf"]},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] == 1
