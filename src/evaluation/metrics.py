"""Layer 1: Deterministic evaluation metrics (no LLM, instant, reproducible)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RetrievalMetrics:
    context_precision: float = 0.0  # relevant_retrieved / total_retrieved
    context_recall: float = 0.0     # relevant_retrieved / total_relevant
    context_relevance: float = 0.0  # mean cosine(query_emb, chunk_embs)
    mrr: float = 0.0                # 1 / rank_of_first_relevant
    ndcg_at_k: float = 0.0
    citation_accuracy: float = 0.0  # valid_citations / total_citations
    latency_ms: float = 0.0


@dataclass
class LatencyStats:
    p50: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    count: int = 0


def context_precision(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not retrieved_ids:
        return 0.0
    hits = sum(1 for cid in retrieved_ids if cid in relevant_ids)
    return hits / len(retrieved_ids)


def context_recall(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 1.0
    hits = sum(1 for cid in retrieved_ids if cid in relevant_ids)
    return hits / len(relevant_ids)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(y * y for y in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def context_relevance(
    query_emb: list[float], chunk_embs: list[list[float]]
) -> float:
    if not chunk_embs:
        return 0.0
    sims = [cosine_similarity(query_emb, e) for e in chunk_embs]
    return sum(sims) / len(sims)


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    for i, cid in enumerate(retrieved_ids, 1):
        if cid in relevant_ids:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int = 5) -> float:
    def dcg(ids: list[str], n: int) -> float:
        return sum(
            (1.0 if ids[i] in relevant_ids else 0.0) / math.log2(i + 2)
            for i in range(min(n, len(ids)))
        )

    actual = dcg(retrieved_ids, k)
    ideal_ids = [cid for cid in retrieved_ids if cid in relevant_ids]
    ideal_ids += [cid for cid in retrieved_ids if cid not in relevant_ids]
    ideal = dcg(ideal_ids, k)
    return actual / ideal if ideal > 0 else 0.0


def citation_accuracy(answer: str, valid_source_indices: set[int]) -> float:
    """Count inline [n] citations that map to real sources."""
    import re

    cited = re.findall(r"\[(\d+)\]", answer)
    if not cited:
        return 1.0  # no citations → neutral
    valid = sum(1 for n in cited if int(n) in valid_source_indices)
    return valid / len(cited)


def latency_stats(latencies_ms: list[float]) -> LatencyStats:
    if not latencies_ms:
        return LatencyStats()
    s = sorted(latencies_ms)
    n = len(s)
    return LatencyStats(
        p50=_percentile(s, 50),
        p95=_percentile(s, 95),
        p99=_percentile(s, 99),
        mean=sum(s) / n,
        count=n,
    )


def _percentile(sorted_data: list[float], pct: float) -> float:
    if not sorted_data:
        return 0.0
    idx = (pct / 100) * (len(sorted_data) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


def compute_all_deterministic(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    query_emb: list[float],
    chunk_embs: list[list[float]],
    answer: str,
    latency_ms: float,
) -> RetrievalMetrics:
    n_retrieved = len(retrieved_ids)
    valid_indices = set(range(1, n_retrieved + 1))
    return RetrievalMetrics(
        context_precision=context_precision(retrieved_ids, relevant_ids),
        context_recall=context_recall(retrieved_ids, relevant_ids),
        context_relevance=context_relevance(query_emb, chunk_embs),
        mrr=mrr(retrieved_ids, relevant_ids),
        ndcg_at_k=ndcg_at_k(retrieved_ids, relevant_ids, k=5),
        citation_accuracy=citation_accuracy(answer, valid_indices),
        latency_ms=latency_ms,
    )


def _tokenize(text: str) -> set[str]:
    tokens = [
        tok.strip(".,!?;:\"'()[]{}<>")
        for tok in text.lower().split()
        if tok.strip(".,!?;:\"'()[]{}<>")
    ]
    return {tok for tok in tokens if tok}


def answer_relevancy(query: str, answer: str) -> float:
    q = _tokenize(query)
    a = _tokenize(answer)
    if not q or not a:
        return 0.0
    overlap = q.intersection(a)
    return min(1.0, len(overlap) / len(q))


def faithfulness_score(answer: str, context: list[dict[str, Any]]) -> float:
    a = _tokenize(answer)
    c = _tokenize(" ".join(item.get("text", "") for item in context))
    if not a or not c:
        return 0.0
    overlap = a.intersection(c)
    return min(1.0, len(overlap) / len(a))


def context_precision_from_scores(scores: list[float]) -> float:
    """Mean of retrieval scores; treats scores >= 0.0 as valid."""
    if not scores:
        return 0.0
    valid = [s for s in scores if s >= 0.0]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def context_recall_from_scores(scores: list[float], max_k: int = 5) -> float:
    if max_k <= 0:
        return 0.0
    good = sum(1 for s in scores if s >= 0.01)
    return min(1.0, good / max_k)


def compute_ragas_metrics(
    query: str,
    answer: str,
    context: list[dict[Any, Any]],
    retrieval_scores: list[float],
    reranker_scores: list[float] | None = None,
    top_k: int = 5,
) -> dict[str, float]:
    """Compute heuristic RAGAS metrics.

    Args:
        query: The user's original query.
        answer: The generated answer string.
        context: List of retrieved chunk dicts (each has a 'text' key).
        retrieval_scores: Raw Chroma/RRF scores (may be near-zero for cosine collections).
        reranker_scores: FlashRank cross-encoder scores (preferred for context_precision
            because they are semantically meaningful regardless of embedding model).
        top_k: Number of chunks expected for recall computation.
    """
    faith = faithfulness_score(answer, context)
    relevance = answer_relevancy(query, answer)

    # Prefer reranker scores for context_precision — they are always meaningful
    # (FlashRank ms-marco scores), unlike raw Chroma distances which depend on
    # whether the collection was created with the same embedding model.
    scores_for_precision = reranker_scores if reranker_scores else retrieval_scores
    precision = context_precision_from_scores(scores_for_precision)
    recall = context_recall_from_scores(scores_for_precision, top_k)

    return {
        "faithfulness": faith,
        "answer_relevancy": relevance,
        "context_precision": precision,
        "context_recall": recall,
        "hallucination_score": max(0.0, 1.0 - faith),
    }
