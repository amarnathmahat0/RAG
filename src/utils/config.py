"""Central configuration loaded from .env / environment variables."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Ollama ──────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://localhost:11434"

    # ── ChromaDB ────────────────────────────────────────────────────────────────
    chroma_persist_dir: str = "./data/chroma"
    chroma_collection: str = "nexus_rag"

    # ── Neo4j ───────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "nexusrag123"

    # ── API ─────────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 1
    api_log_level: Literal["debug", "info", "warning", "error"] = "info"

    # ── Rate limiting ────────────────────────────────────────────────────────────
    rate_limit_requests: int = 10
    rate_limit_window: int = 60

    # ── RAM guardrails ───────────────────────────────────────────────────────────
    min_free_ram_mb: float = 1500.0
    ram_check_interval_sec: int = 5

    # ── Models ───────────────────────────────────────────────────────────────────
    embed_model_primary: str = "all-minilm:latest"
    embed_model_fallback: str = "all-minilm:latest"
    llm_router: str = "gemma3:1b"
    llm_judge: str = "qwen:1.8b"
    llm_generator: str = "gemma3:1b"

    # Model size registry (MB) — used by ModelManager
    model_sizes_mb: dict[str, float] = Field(
        default={
            "all-minilm:latest": 45.0,
            "mxbai-embed-large:latest": 669.0,
            "gemma3:1b": 815.0,
            "qwen:1.8b": 1100.0,
            "phi3:mini": 2200.0,
        }
    )

    # ── Retrieval ────────────────────────────────────────────────────────────────
    # Optimized for latency: reduced top-k values
    top_k_dense: int = 8
    top_k_sparse: int = 8
    top_k_rerank: int = 3
    rrf_k: int = 60
    dense_weight: float = 0.7
    sparse_weight: float = 0.3

    # ── Chunking ─────────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 2  # sentences

    # ── Evaluation ───────────────────────────────────────────────────────────────
    eval_runs_dir: str = "./reports/eval_runs"
    benchmark_path: str = "./data/benchmarks/test.json"

    # ── Tracing ──────────────────────────────────────────────────────────────────
    trace_dir: str = "./reports/traces"
    metrics_dir: str = "./reports/charts"

    # ── Cache ────────────────────────────────────────────────────────────────────
    cache_dir: str = "./data/cache"
    cache_ttl_sec: int = 3600
    enable_query_cache: bool = True
    enable_embedding_cache: bool = True

    # ── Documents ────────────────────────────────────────────────────────────────
    # Support for document filtering and multi-doc queries
    enable_doc_filtering: bool = True
    enable_multi_doc_query: bool = True

    @field_validator("dense_weight", "sparse_weight")
    @classmethod
    def weight_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("weights must be in [0, 1]")
        return v

    def ensure_dirs(self) -> None:
        """Create all required directories."""
        dirs = [
            self.chroma_persist_dir,
            self.eval_runs_dir,
            self.trace_dir,
            self.metrics_dir,
            self.cache_dir,
            "./data/raw",
            "./data/processed",
            "./data/benchmarks",
            "./reports/charts",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.ensure_dirs()
    return s
