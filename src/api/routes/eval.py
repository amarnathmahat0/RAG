"""Evaluation endpoints: POST /eval/run, GET /eval/results/{run_id}."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.evaluation.benchmark import BenchmarkRunner
from src.evaluation.report_generator import ReportGenerator
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
_settings = get_settings()

_run_status: dict[str, str] = {}


class EvalRunRequest(BaseModel):
    use_llm_judge: bool = True
    max_cases: int = 50


@router.post("/eval/run")
async def trigger_eval(body: EvalRunRequest, background_tasks: BackgroundTasks) -> dict:
    import uuid
    run_id = str(uuid.uuid4())[:8]
    _run_status[run_id] = "running"
    background_tasks.add_task(_run_eval, run_id, body.use_llm_judge, body.max_cases)
    return {"run_id": run_id, "status": "started"}


@router.get("/eval/results/{run_id}")
async def get_eval_results(run_id: str) -> JSONResponse:
    status = _run_status.get(run_id, "not_found")
    if status == "running":
        return JSONResponse({"run_id": run_id, "status": "running"})
    if status == "not_found":
        return JSONResponse({"error": "run not found"}, status_code=404)

    # Load saved results
    result_file = Path(_settings.eval_runs_dir) / f"run_{run_id}.json"
    if not result_file.exists():
        return JSONResponse({"error": "results file not found"}, status_code=404)

    with open(result_file) as f:
        data = json.load(f)

    # Also load report paths
    report_file = Path(_settings.eval_runs_dir) / f"report_{run_id}.md"
    data["report_md"] = str(report_file) if report_file.exists() else None

    chart_files = list(Path(_settings.metrics_dir).glob(f"*_{run_id}.png"))
    data["charts"] = [str(p) for p in chart_files]

    return JSONResponse(data)


async def _run_eval(run_id: str, use_llm_judge: bool, max_cases: int) -> None:
    try:
        runner = BenchmarkRunner(use_llm_judge=use_llm_judge)
        cases = runner.load_test_cases()[:max_cases]
        report = await runner.run(cases, run_id=run_id)
        gen = ReportGenerator()
        gen.generate(report.__dict__, run_id)
        _run_status[run_id] = "complete"
        logger.info("Eval run %s complete: %d cases", run_id, report.total_cases)
    except Exception as exc:
        logger.error("Eval run %s failed: %s", run_id, exc)
        _run_status[run_id] = f"failed: {exc}"
