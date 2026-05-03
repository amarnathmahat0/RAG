"""POST /query — full RAG pipeline with optional streaming.

SSE streaming protocol
----------------------
Each SSE frame is:  data: <json>\n\n

Frame types (field "type"):
  {"type": "step",  "icon": "🔍", "text": "...", "done": false}
  {"type": "step",  "icon": "✓",  "text": "...", "done": true}
  {"type": "token", "text": "..."}
  {"type": "meta",  ...QueryResponse fields}
  {"type": "error", "text": "..."}
  {"type": "done"}
"""
from __future__ import annotations

import json
import time
from typing import AsyncIterator

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.agents.graph import RAGPipeline, get_pipeline
from src.agents.state import RAGState
from src.api.dependencies import get_request_id
from src.monitoring.prometheus_metrics import record_query, record_critique
from src.monitoring.tracer import get_tracer
from src.utils.cache import get_cache_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    stream: bool = False
    documents: list[str] | None = None


class QueryResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    tier: str
    tier_reason: str
    confidence: float
    sources: list[dict]
    latency_ms: float
    faithfulness_score: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    hallucination_score: float | None = None


# ── SSE frame helpers ─────────────────────────────────────────────────────────

def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"

def _step(text: str, icon: str, done: bool = False) -> str:
    return _sse({"type": "step", "icon": icon, "text": text, "done": done})

def _token(text: str) -> str:
    return _sse({"type": "token", "text": text})

def _meta(state: RAGState, latency_ms: float, request_id: str) -> str:
    tier_str = state.tier.value if state.tier else "TIER_1"
    return _sse({
        "type": "meta",
        "request_id": request_id,
        "tier": tier_str,
        "tier_reason": state.tier_reason,
        "confidence": round(state.confidence, 3),
        "sources": state.sources[:10],
        "latency_ms": round(latency_ms, 1),
        "faithfulness_score": state.faithfulness_score,
        "answer_relevancy": state.answer_relevancy,
        "context_precision": state.context_precision,
        "context_recall": state.context_recall,
        "hallucination_score": state.hallucination_score,
    })


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def query_endpoint(
    body: QueryRequest,
    request: Request,
    request_id: str = Depends(get_request_id),
) -> JSONResponse | StreamingResponse:
    pipeline = get_pipeline()
    tracer = get_tracer()
    trace = tracer.new_trace(body.query, request_id)
    cache = get_cache_manager()

    if not body.stream:
        cached = cache.get_query_result(body.query, body.documents)
        if cached:
            logger.info("Cache hit: %s", body.query[:50])
            return JSONResponse(content=cached)

    if body.documents:
        logger.info("Query with document filter: %s", body.documents)

    if body.stream:
        return StreamingResponse(
            _stream_with_steps(pipeline, body.query, request_id, trace, body.documents),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Access-Control-Allow-Origin": "*",
            },
        )

    # Non-streaming
    t0 = time.perf_counter()
    with tracer.span(trace, "full_pipeline"):
        state = await pipeline.run(body.query, request_id=request_id, documents=body.documents)

    latency_ms = (time.perf_counter() - t0) * 1000
    tier_str = state.tier.value if state.tier else "TIER_1"

    record_query(tier_str, "ok" if not state.error else "error", latency_ms / 1000)
    if state.critique:
        record_critique(
            state.critique.faithfulness_score,
            state.critique.total_claims,
            len(state.critique.hallucinated_claims),
        )

    trace.finish(status="ok" if not state.error else "error", tier=tier_str, latency_ms=round(latency_ms, 1))
    tracer.save(trace)

    response_data = QueryResponse(
        request_id=request_id,
        query=body.query,
        answer=state.final_answer,
        tier=tier_str,
        tier_reason=state.tier_reason,
        confidence=round(state.confidence, 3),
        sources=state.sources[:10],
        latency_ms=round(latency_ms, 1),
        faithfulness_score=state.faithfulness_score,
        answer_relevancy=state.answer_relevancy,
        context_precision=state.context_precision,
        context_recall=state.context_recall,
        hallucination_score=state.hallucination_score,
    ).model_dump()

    cache.set_query_result(body.query, response_data, body.documents)
    return JSONResponse(content=response_data)


# ── SSE streaming generator ───────────────────────────────────────────────────

async def _stream_with_steps(
    pipeline: RAGPipeline,
    query: str,
    request_id: str,
    trace,
    documents: list[str] | None,
) -> AsyncIterator[str]:
    """
    Emits frames in order:
      step (pending) → step (done) × N
      token × N
      meta (scores + sources)
      done
    """
    t0 = time.perf_counter()

    try:
        import uuid
        from src.agents.compliance_agent import ComplianceAgent
        from src.agents.router import Router
        from src.agents.retriever_agent import RetrieverAgent
        from src.agents.critic_agent import CriticAgent
        from src.agents.formatter_agent import FormatterAgent
        from src.generation.response_builder import ResponseBuilder

        state = RAGState(
            query=query,
            request_id=request_id,
            start_time_ms=t0 * 1000,
            documents=documents,
        )

        # ── 1. Compliance pre-check ───────────────────────────────────────────
        yield _step("Checking query compliance…", "🛡️")
        state = ComplianceAgent().check(state)
        if not state.compliance_passed:
            yield _step("Query blocked by compliance filter", "⛔", done=True)
            yield _token(state.final_answer or "This query is not allowed.")
            yield _sse({"type": "done"})
            return
        state.compliance_passed = True
        state.compliance_violations = []
        yield _step("Compliance check passed", "✓", done=True)

        # ── 2. Routing ────────────────────────────────────────────────────────
        yield _step("Analysing complexity and selecting retrieval tier…", "🔍")
        state = await Router().route(state)
        tier_str = state.tier.value if state.tier else "TIER_1"
        tier_labels = {
            "TIER_1": "⚡ TIER 1 — simple factual",
            "TIER_2": "🔄 TIER 2 — multi-hop",
            "TIER_3": "🕸️ TIER 3 — GraphRAG",
        }
        yield _step(
            f"Routed to {tier_labels.get(tier_str, tier_str)}: {state.tier_reason}",
            "✓", done=True,
        )

        # ── 3. Retrieval ──────────────────────────────────────────────────────
        yield _step("Hybrid search (dense + BM25 sparse) → ChromaDB…", "🗄️")
        retriever = RetrieverAgent()
        state = await retriever.retrieve(state, documents)
        n_chunks = len(state.context)
        yield _step(
            f"Retrieved {n_chunks} chunks · cross-encoder reranking complete",
            "✓", done=True,
        )

        # ── 4. Generation (stream tokens) ─────────────────────────────────────
        yield _step("Generating answer with retrieved context…", "🧠")
        builder = ResponseBuilder()
        collected: list[str] = []
        async for token in builder.stream(state):
            collected.append(token)
            yield _token(token)
        state.final_answer = "".join(collected)
        yield _step("Answer generated", "✓", done=True)

        # ── 5. Critic ─────────────────────────────────────────────────────────
        yield _step("Running faithfulness critic…", "🔬")
        critic = CriticAgent()
        state = await critic.critique(state)
        if state.critique:
            fs = state.critique.faithfulness_score
            if fs < 0.3 and state.critic_loop_count < state.max_critic_loops:
                yield _step(
                    f"Low faithfulness ({fs:.2f}) — tightening retrieval and regenerating…",
                    "⚠️",
                )
                state.retrieval_iteration = 0
                state = await retriever.retrieve(state, documents)
                regen: list[str] = []
                async for token in builder.stream(state):
                    regen.append(token)
                    yield _token(token)
                state.final_answer = "".join(regen)
                state = await critic.critique(state)
            yield _step(
                f"Critic passed · faithfulness={state.critique.faithfulness_score:.2f}",
                "✓", done=True,
            )

        # ── 6. Compliance post-check ──────────────────────────────────────────
        yield _step("Post-generation compliance check…", "🛡️")
        state = ComplianceAgent().check(state)
        yield _step("Compliance OK", "✓", done=True)

        # ── 7. Format ─────────────────────────────────────────────────────────
        state = FormatterAgent().format(state)

        # ── Metadata frame ────────────────────────────────────────────────────
        latency_ms = (time.perf_counter() - t0) * 1000
        yield _meta(state, latency_ms, request_id)

        record_query(tier_str, "ok" if not state.error else "error", latency_ms / 1000)
        if state.critique:
            record_critique(
                state.critique.faithfulness_score,
                state.critique.total_claims,
                len(state.critique.hallucinated_claims),
            )

        trace.finish(status="streamed", tier=tier_str, latency_ms=round(latency_ms, 1))
        get_tracer().save(trace)

        yield _sse({"type": "done"})

    except Exception as exc:
        logger.error("Stream error: %s", exc, exc_info=True)
        yield _sse({"type": "error", "text": str(exc)})
        yield _sse({"type": "done"})
        trace.finish(status="error")
        get_tracer().save(trace)