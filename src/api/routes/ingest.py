"""POST /ingest — trigger document ingestion and file upload."""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, File, UploadFile
from pydantic import BaseModel, Field

from src.ingestion.pipeline import IngestionPipeline
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()
_settings = get_settings()


class IngestRequest(BaseModel):
    sources: list[str] = Field(..., min_length=1, description="File paths or URLs")
    metadata: dict = Field(default_factory=dict)


class IngestResponse(BaseModel):
    accepted: int
    sources: list[str]
    message: str


class IngestStatusResponse(BaseModel):
    results: list[dict]


_results_store: dict[str, list[dict]] = {}


@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(
    body: IngestRequest,
    background_tasks: BackgroundTasks,
) -> IngestResponse:
    import uuid
    job_id = str(uuid.uuid4())[:8]
    background_tasks.add_task(
        _run_ingestion, job_id, body.sources, body.metadata
    )
    return IngestResponse(
        accepted=len(body.sources),
        sources=body.sources,
        message=f"Ingestion started (job_id={job_id}). Poll /ingest/status/{job_id}.",
    )


@router.get("/ingest/status/{job_id}")
async def ingest_status(job_id: str) -> dict:
    if job_id not in _results_store:
        return {"status": "pending_or_not_found", "job_id": job_id}
    return {"status": "complete", "job_id": job_id, "results": _results_store[job_id]}


async def _run_ingestion(job_id: str, sources: list[str], metadata: dict) -> None:
    pipeline = IngestionPipeline()
    results = []
    for src in sources:
        try:
            r = await pipeline.ingest(src, metadata)
            results.append({
                "source": r.source,
                "chunks": r.chunks_created,
                "entities": r.entities_extracted,
                "elapsed_sec": round(r.elapsed_sec, 2),
                "status": "ok",
            })
        except Exception as exc:
            logger.error("Ingestion failed for %s: %s", src, exc)
            results.append({"source": src, "status": "error", "error": str(exc)})
    _results_store[job_id] = results
    logger.info("Ingestion job %s complete: %d sources", job_id, len(results))


@router.post("/ingest/upload", response_model=IngestResponse)
async def upload_files(
    files: list[UploadFile] = File(...),
    background_tasks: BackgroundTasks = None,
) -> IngestResponse:
    """Handle file uploads and trigger ingestion."""
    job_id = str(uuid.uuid4())[:8]
    upload_dir = Path(_settings.chroma_persist_dir).parent / "raw"
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved_paths = []
    for file in files:
        try:
            # Save uploaded file
            file_path = upload_dir / file.filename
            content = await file.read()
            file_path.write_bytes(content)
            saved_paths.append(str(file_path))
            logger.info("Uploaded file saved: %s", file_path)
        except Exception as exc:
            logger.error("Failed to save uploaded file %s: %s", file.filename, exc)

    if saved_paths and background_tasks:
        background_tasks.add_task(_run_ingestion, job_id, saved_paths, {})

    return IngestResponse(
        accepted=len(saved_paths),
        sources=saved_paths,
        message=f"Upload complete ({len(saved_paths)} files). Ingestion started (job_id={job_id}). Poll /ingest/status/{job_id}.",
    )
