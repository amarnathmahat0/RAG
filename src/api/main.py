"""NexusRAG FastAPI application."""
from __future__ import annotations

import os
# Disable telemetry before any imports
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["CHROMA_SERVER_NO_TELEMETRY"] = "true"

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from src.api.middleware import register_middleware
from src.api.routes import query, ingest, health, eval as eval_routes
from src.api.routes import documents as docs_routes
from src.monitoring.logger_config import configure_logging
from src.monitoring.prometheus_metrics import (
    get_metrics_output,
    update_system_gauges,
    CHROMA_DOCS,
)
from src.utils.config import get_settings
from src.utils.logger import get_logger

_settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging(_settings.api_log_level.upper())
    logger.info("NexusRAG starting up…")

    # Background RAM/metrics updater
    async def _gauge_loop() -> None:
        while True:
            try:
                update_system_gauges()
                # Update ChromaDB doc count
                from src.db.chroma_client import get_chroma_client
                try:
                    cnt = get_chroma_client().get_or_create_collection().count()
                    CHROMA_DOCS.set(cnt)
                except Exception:
                    pass
            except Exception:
                pass
            await asyncio.sleep(_settings.ram_check_interval_sec)

    task = asyncio.create_task(_gauge_loop())
    logger.info("NexusRAG ready. API at http://%s:%d", _settings.api_host, _settings.api_port)
    yield
    task.cancel()
    logger.info("NexusRAG shutting down.")


app = FastAPI(
    title="NexusRAG",
    description="Production-grade multi-agent RAG system with 3-tier adaptive routing",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ────────────────────────────────────────────────────────────────────
register_middleware(app)

# ── Routes ────────────────────────────────────────────────────────────────────────
app.include_router(query.router, tags=["Query"])
app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(health.router, tags=["Health"])
app.include_router(eval_routes.router, tags=["Evaluation"])
app.include_router(docs_routes.router, tags=["Documents"])



@app.get("/metrics", include_in_schema=False)
async def metrics() -> Response:
    body, content_type = get_metrics_output()
    return Response(content=body, media_type=content_type)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": str(exc),
            "path": str(request.url.path),
        },
    )
