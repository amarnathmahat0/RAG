"""Prometheus metrics definitions for NexusRAG."""
from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Use a custom registry to avoid conflicts in tests
REGISTRY = CollectorRegistry(auto_describe=True)

# ── Counters ─────────────────────────────────────────────────────────────────────
QUERIES_TOTAL = Counter(
    "nexus_rag_queries_total",
    "Total number of RAG queries processed",
    ["tier", "status"],
    registry=REGISTRY,
)

# ── Histograms ───────────────────────────────────────────────────────────────────
LATENCY_SECONDS = Histogram(
    "nexus_rag_latency_seconds",
    "End-to-end query latency in seconds",
    ["tier"],
    buckets=(0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0),
    registry=REGISTRY,
)

FAITHFULNESS_SCORE = Histogram(
    "nexus_rag_faithfulness_score",
    "Faithfulness score from critic agent",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
    registry=REGISTRY,
)

HALLUCINATION_RATE = Histogram(
    "nexus_rag_hallucination_rate",
    "Fraction of hallucinated claims per query",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0),
    registry=REGISTRY,
)

# ── Gauges ───────────────────────────────────────────────────────────────────────
CACHE_HIT_RATE = Gauge(
    "nexus_rag_cache_hit_rate",
    "Current cache hit rate (rolling)",
    registry=REGISTRY,
)

RAM_USAGE_MB = Gauge(
    "nexus_rag_ram_usage_mb",
    "Current RAM usage in MB",
    registry=REGISTRY,
)

FREE_RAM_MB = Gauge(
    "nexus_rag_free_ram_mb",
    "Available RAM in MB",
    registry=REGISTRY,
)

CHROMA_DOCS = Gauge(
    "nexus_rag_chroma_docs_total",
    "Total documents in ChromaDB",
    registry=REGISTRY,
)


def get_metrics_output() -> tuple[bytes, str]:
    """Return (body, content_type) for /metrics endpoint."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def record_query(tier: str, status: str, latency_sec: float) -> None:
    QUERIES_TOTAL.labels(tier=tier, status=status).inc()
    LATENCY_SECONDS.labels(tier=tier).observe(latency_sec)


def record_critique(faithfulness: float, total_claims: int, hallucinated: int) -> None:
    FAITHFULNESS_SCORE.observe(faithfulness)
    if total_claims > 0:
        HALLUCINATION_RATE.observe(hallucinated / total_claims)


def update_system_gauges() -> None:
    """Update RAM gauges — call periodically."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        RAM_USAGE_MB.set(vm.used / 1024 / 1024)
        FREE_RAM_MB.set(vm.available / 1024 / 1024)
    except Exception:
        pass
