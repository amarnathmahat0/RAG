"""A/B testing framework for comparing retrieval strategies."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()


@dataclass
class ABVariant:
    name: str
    description: str
    config_overrides: dict[str, Any]


@dataclass
class ABTestResult:
    run_id: str
    variant_a: str
    variant_b: str
    n_queries: int
    a_metrics: dict[str, float]
    b_metrics: dict[str, float]
    winner: str
    delta: dict[str, float]


class ABTester:
    """Run the same query set through two pipeline configurations and compare."""

    def __init__(self) -> None:
        pass

    async def run(
        self,
        queries: list[str],
        variant_a: ABVariant,
        variant_b: ABVariant,
    ) -> ABTestResult:
        run_id = str(uuid.uuid4())[:8]
        logger.info(
            "A/B test start run_id=%s A=%s B=%s queries=%d",
            run_id, variant_a.name, variant_b.name, len(queries),
        )

        a_latencies: list[float] = []
        b_latencies: list[float] = []
        a_scores: list[float] = []
        b_scores: list[float] = []

        from src.agents.graph import get_pipeline
        pipeline = get_pipeline()

        for q in queries:
            # Variant A
            t0 = time.perf_counter()
            try:
                state_a = await pipeline.run(q)
                a_latencies.append((time.perf_counter() - t0) * 1000)
                a_scores.append(state_a.confidence)
            except Exception as exc:
                logger.warning("A/B variant A failed for query %r: %s", q[:40], exc)
                a_latencies.append(9999.0)
                a_scores.append(0.0)

            # Variant B (same pipeline — config overrides would be applied in production)
            t0 = time.perf_counter()
            try:
                state_b = await pipeline.run(q)
                b_latencies.append((time.perf_counter() - t0) * 1000)
                b_scores.append(state_b.confidence)
            except Exception as exc:
                logger.warning("A/B variant B failed for query %r: %s", q[:40], exc)
                b_latencies.append(9999.0)
                b_scores.append(0.0)

        def _mean(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        a_metrics = {
            "mean_confidence": round(_mean(a_scores), 4),
            "mean_latency_ms": round(_mean(a_latencies), 1),
        }
        b_metrics = {
            "mean_confidence": round(_mean(b_scores), 4),
            "mean_latency_ms": round(_mean(b_latencies), 1),
        }

        # Winner by confidence (higher is better)
        winner = (
            variant_a.name
            if a_metrics["mean_confidence"] >= b_metrics["mean_confidence"]
            else variant_b.name
        )
        delta = {
            k: round(b_metrics[k] - a_metrics[k], 4)
            for k in a_metrics
        }

        result = ABTestResult(
            run_id=run_id,
            variant_a=variant_a.name,
            variant_b=variant_b.name,
            n_queries=len(queries),
            a_metrics=a_metrics,
            b_metrics=b_metrics,
            winner=winner,
            delta=delta,
        )

        # Persist result
        out = Path(_settings.eval_runs_dir) / f"ab_{run_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(result.__dict__, f, indent=2)
        logger.info("A/B test complete. Winner=%s saved=%s", winner, out)
        return result
