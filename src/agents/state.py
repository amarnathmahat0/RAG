"""LangGraph state schema for the RAG pipeline."""
from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Tier(str, Enum):
    TIER_1 = "TIER_1"
    TIER_2 = "TIER_2"
    TIER_3 = "TIER_3"


class CriticResult(BaseModel):
    faithfulness_score: float = 0.0
    supported_claims: int = 0
    total_claims: int = 0
    hallucinated_claims: list[str] = Field(default_factory=list)
    reasoning: str = ""


class RAGState(BaseModel):
    """Full pipeline state passed through the LangGraph."""

    # Input
    query: str = ""
    request_id: str = ""
    documents: list[str] | None = None  # Optional list of document filenames to query against

    # Routing
    transformed_query: str = ""
    sub_queries: list[str] = Field(default_factory=list)
    complexity_score: float = 0.0
    tier: Tier = Tier.TIER_1
    tier_reason: str = ""

    # Retrieval
    context: list[dict[str, Any]] = Field(default_factory=list)  # list of chunk dicts
    retrieval_scores: list[float] = Field(default_factory=list)
    retrieval_iteration: int = 0
    max_retrieval_iterations: int = 2

    # Generation
    intermediate_steps: list[str] = Field(default_factory=list)
    raw_answer: str = ""
    final_answer: str = ""
    sources: list[dict[str, Any]] = Field(default_factory=list)

    # Critique
    critique: Optional[CriticResult] = None
    critic_loop_count: int = 0
    max_critic_loops: int = 1  # Cap at 1 re-retrieval to avoid double latency
    critic_passed: bool = False

    # Compliance
    compliance_passed: bool = True
    compliance_label: str = "OTHER"
    compliance_violations: list[str] = Field(default_factory=list)

    # RAGAS metrics
    faithfulness_score: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    hallucination_score: float = 0.0

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    confidence: float = 0.0

    # Timing
    start_time_ms: float = 0.0
    end_time_ms: float = 0.0
