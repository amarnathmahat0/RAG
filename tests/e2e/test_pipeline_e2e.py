"""End-to-end pipeline tests (requires Ollama running — skipped in CI)."""
from __future__ import annotations

import pytest
import os

# Skip all E2E tests if OLLAMA_AVAILABLE env var not set
pytestmark = pytest.mark.skipif(
    os.environ.get("OLLAMA_AVAILABLE") != "1",
    reason="E2E tests require Ollama running (set OLLAMA_AVAILABLE=1)",
)


@pytest.mark.asyncio
async def test_pipeline_simple_query():
    from src.agents.graph import RAGPipeline
    from src.agents.state import Tier

    pipeline = RAGPipeline()
    state = await pipeline.run("What is machine learning?")
    assert state.query == "What is machine learning?"
    assert state.tier in [Tier.TIER_1, Tier.TIER_2, Tier.TIER_3]
    assert state.final_answer != ""


@pytest.mark.asyncio
async def test_pipeline_compliance_blocks_pii():
    from src.agents.graph import RAGPipeline

    pipeline = RAGPipeline()
    state = await pipeline.run("Send email to test@example.com with SSN 123-45-6789")
    assert not state.compliance_passed or "⚠️" in state.final_answer


@pytest.mark.asyncio
async def test_tier1_latency():
    import time
    from src.agents.graph import RAGPipeline

    pipeline = RAGPipeline()
    t0 = time.perf_counter()
    state = await pipeline.run("What is Python?")
    elapsed_ms = (time.perf_counter() - t0) * 1000
    # TIER_1 should be < 5000ms
    assert elapsed_ms < 5000, f"TIER_1 latency {elapsed_ms:.0f}ms exceeds 5000ms"
