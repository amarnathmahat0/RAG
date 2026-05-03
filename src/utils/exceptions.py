"""Custom exceptions for NexusRAG."""
from __future__ import annotations


class NexusRAGError(Exception):
    """Base exception."""


class RAMGuardrailError(NexusRAGError):
    """Raised when available RAM is below the minimum threshold."""


class ModelLoadError(NexusRAGError):
    """Raised when Ollama fails to load a model."""


class ModelUnloadError(NexusRAGError):
    """Raised when Ollama fails to unload a model."""


class EmbeddingError(NexusRAGError):
    """Raised on embedding failure."""


class RetrievalError(NexusRAGError):
    """Raised on retrieval failure."""


class IngestionError(NexusRAGError):
    """Raised when document ingestion fails."""


class ComplianceViolationError(NexusRAGError):
    """Raised when a query or answer violates compliance rules."""

    def __init__(self, message: str, violation_type: str) -> None:
        super().__init__(message)
        self.violation_type = violation_type


class GraphDBError(NexusRAGError):
    """Raised on Neo4j errors."""


class VectorDBError(NexusRAGError):
    """Raised on ChromaDB errors."""


class GenerationError(NexusRAGError):
    """Raised when LLM generation fails."""


class EvaluationError(NexusRAGError):
    """Raised on evaluation pipeline errors."""
