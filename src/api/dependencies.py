"""FastAPI dependency injection."""
from __future__ import annotations

from fastapi import Request

from src.agents.graph import get_pipeline, RAGPipeline
from src.ingestion.pipeline import IngestionPipeline
from src.utils.model_manager import get_model_manager, ModelManager


def get_rag_pipeline(request: Request) -> RAGPipeline:
    return get_pipeline()


def get_ingestion_pipeline() -> IngestionPipeline:
    return IngestionPipeline()


def get_mm() -> ModelManager:
    return get_model_manager()


def get_request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")
