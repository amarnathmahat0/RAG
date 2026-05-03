"""GET /health — system health check."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.utils.model_manager import get_model_manager
from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/health")
async def health() -> JSONResponse:
    mm = get_model_manager()
    model_status = await mm.status()

    chroma_status = _check_chroma()
    neo4j_status = _check_neo4j()

    overall = (
        "ok"
        if chroma_status.get("status") == "ok"
        else "degraded"
    )

    return JSONResponse(
        status_code=200 if overall == "ok" else 503,
        content={
            "status": overall,
            "components": {
                "chroma": chroma_status,
                "neo4j": neo4j_status,
                "models": model_status,
            },
        },
    )


def _check_chroma() -> dict:
    try:
        from src.db.chroma_client import get_chroma_client
        return get_chroma_client().health()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def _check_neo4j() -> dict:
    try:
        from src.db.neo4j_client import get_neo4j_client
        return get_neo4j_client().health()
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}
