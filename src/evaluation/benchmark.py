"""Benchmark runner: execute test cases, collect metrics, produce report."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.agents.graph import get_pipeline
from src.evaluation.llm_judge import LLMJudge, LLMJudgeResult
from src.evaluation.metrics import (
    RetrievalMetrics,
    compute_all_deterministic,
    latency_stats,
)
from src.retrieval.embeddings import get_embedding_service
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()


@dataclass
class TestCase:
    query: str
    ground_truth_answer: str
    ground_truth_context: list[str]  # chunk IDs
    all_relevant_chunks: list[str]   # superset with distractors
    difficulty: str  # easy / medium / hard
    expected_tier: str  # TIER_1 / TIER_2 / TIER_3


@dataclass
class CaseResult:
    case_index: int
    query: str
    difficulty: str
    expected_tier: str
    actual_tier: str
    deterministic: RetrievalMetrics
    llm_judge: LLMJudgeResult | None
    latency_ms: float
    answer: str
    error: str | None


@dataclass
class BenchmarkReport:
    run_id: str
    total_cases: int
    mean_context_precision: float
    mean_context_recall: float
    mean_faithfulness: float
    mean_relevancy: float
    mean_completeness: float
    latency: dict
    tier_distribution: dict[str, int]
    per_case: list[dict]


class BenchmarkRunner:
    """Load test cases → run pipeline → compute metrics → return report."""

    def __init__(self, use_llm_judge: bool = True) -> None:
        self._use_llm_judge = use_llm_judge
        self._emb = get_embedding_service()
        self._judge = LLMJudge() if use_llm_judge else None

    def load_test_cases(self, path: str | None = None) -> list[TestCase]:
        p = Path(path or _settings.benchmark_path)
        if not p.exists():
            logger.warning("Benchmark file not found: %s — using empty set", p)
            return []
        with open(p) as f:
            raw = json.load(f)
        return [
            TestCase(
                query=item["query"],
                ground_truth_answer=item.get("ground_truth_answer", ""),
                ground_truth_context=item.get("ground_truth_context", []),
                all_relevant_chunks=item.get("all_relevant_chunks", []),
                difficulty=item.get("difficulty", "medium"),
                expected_tier=item.get("expected_tier", "TIER_1"),
            )
            for item in raw
        ]

    async def run(
        self, test_cases: list[TestCase] | None = None, run_id: str | None = None
    ) -> BenchmarkReport:
        run_id = run_id or str(uuid.uuid4())[:8]
        cases = test_cases or self.load_test_cases()
        if not cases:
            logger.error("No test cases to run.")
            return BenchmarkReport(
                run_id=run_id, total_cases=0,
                mean_context_precision=0, mean_context_recall=0,
                mean_faithfulness=0, mean_relevancy=0, mean_completeness=0,
                latency={}, tier_distribution={}, per_case=[],
            )

        pipeline = get_pipeline()
        results: list[CaseResult] = []

        for i, case in enumerate(cases):
            logger.info("Running benchmark case %d/%d: %r", i + 1, len(cases), case.query[:50])
            t0 = time.perf_counter()
            try:
                state = await pipeline.run(case.query, request_id=f"{run_id}_{i}")
                latency_ms = (time.perf_counter() - t0) * 1000

                # Deterministic metrics
                retrieved_ids = [s.get("chunk_id", "") for s in state.sources]
                relevant_set = set(case.all_relevant_chunks)
                query_emb = await self._emb.embed_query(case.query)
                chunk_embs: list[list[float]] = []
                det_metrics = compute_all_deterministic(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_set,
                    query_emb=query_emb,
                    chunk_embs=chunk_embs,
                    answer=state.final_answer,
                    latency_ms=latency_ms,
                )

                # LLM judge
                llm_result: LLMJudgeResult | None = None
                if self._use_llm_judge and self._judge:
                    context_text = "\n\n".join(
                        c.get("text", "") for c in state.context[:4]
                    )
                    llm_result = await self._judge.evaluate(
                        query=case.query,
                        context=context_text,
                        answer=state.final_answer,
                        ground_truth=case.ground_truth_answer,
                    )

                results.append(CaseResult(
                    case_index=i,
                    query=case.query,
                    difficulty=case.difficulty,
                    expected_tier=case.expected_tier,
                    actual_tier=state.tier.value,
                    deterministic=det_metrics,
                    llm_judge=llm_result,
                    latency_ms=latency_ms,
                    answer=state.final_answer[:500],
                    error=state.error,
                ))
            except Exception as exc:
                logger.error("Case %d failed: %s", i, exc)
                results.append(CaseResult(
                    case_index=i, query=case.query, difficulty=case.difficulty,
                    expected_tier=case.expected_tier, actual_tier="ERROR",
                    deterministic=RetrievalMetrics(), llm_judge=None,
                    latency_ms=(time.perf_counter() - t0) * 1000,
                    answer="", error=str(exc),
                ))

        return self._aggregate(run_id, results)

    def _aggregate(self, run_id: str, results: list[CaseResult]) -> BenchmarkReport:
        n = len(results)
        latencies = [r.latency_ms for r in results]

        def _mean(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        prec = _mean([r.deterministic.context_precision for r in results])
        rec = _mean([r.deterministic.context_recall for r in results])
        faith = _mean([
            r.llm_judge.faithfulness_score for r in results
            if r.llm_judge is not None
        ])
        relev = _mean([
            r.llm_judge.relevancy_score for r in results
            if r.llm_judge is not None
        ])
        comp = _mean([
            r.llm_judge.completeness_score for r in results
            if r.llm_judge is not None
        ])
        lat = latency_stats(latencies)
        tiers: dict[str, int] = {}
        for r in results:
            tiers[r.actual_tier] = tiers.get(r.actual_tier, 0) + 1

        per_case = []
        for r in results:
            d = asdict(r.deterministic)
            lj = asdict(r.llm_judge) if r.llm_judge else {}
            per_case.append({
                "index": r.case_index,
                "query": r.query,
                "difficulty": r.difficulty,
                "expected_tier": r.expected_tier,
                "actual_tier": r.actual_tier,
                "latency_ms": round(r.latency_ms, 1),
                "answer": r.answer,
                "error": r.error,
                **{f"det_{k}": round(v, 4) if isinstance(v, float) else v for k, v in d.items()},
                **{f"llm_{k}": round(v, 4) if isinstance(v, float) else v for k, v in lj.items()},
            })

        report = BenchmarkReport(
            run_id=run_id,
            total_cases=n,
            mean_context_precision=round(prec, 4),
            mean_context_recall=round(rec, 4),
            mean_faithfulness=round(faith, 4),
            mean_relevancy=round(relev, 4),
            mean_completeness=round(comp, 4),
            latency={
                "p50": round(lat.p50, 1),
                "p95": round(lat.p95, 1),
                "p99": round(lat.p99, 1),
                "mean": round(lat.mean, 1),
            },
            tier_distribution=tiers,
            per_case=per_case,
        )
        self._save(report)
        return report

    def _save(self, report: BenchmarkReport) -> None:
        Path(_settings.eval_runs_dir).mkdir(parents=True, exist_ok=True)
        out = Path(_settings.eval_runs_dir) / f"run_{report.run_id}.json"
        with open(out, "w") as f:
            json.dump(asdict(report), f, indent=2, default=str)
        logger.info("Benchmark report saved to %s", out)
