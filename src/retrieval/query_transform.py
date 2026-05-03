"""Query transformation: grammar fix, acronym expansion, multi-part decomposition.

LATENCY FIXES
─────────────
FIX L10 — Reduce max_tokens 256 → 120
    The JSON output for rewritten + sub_queries + expansions rarely exceeds
    120 tokens. Capping at 256 was burning inference time on nothing.

FIX L11 — Shorter, tighter prompt
    Fewer prompt tokens = faster prefill. Removed verbose rules section;
    gemma3:1b follows the example format reliably without them.

FIX L12 — Short-query threshold raised 5 → 8 words
    Queries under 8 words almost never need LLM transformation. This aligns
    with the router's short-query bypass (FIX L7) so both are consistent.

FIX L13 — Robust JSON extraction with fallback sub-query splitting
    Previously any malformed JSON silently fell back to the original query
    with sub_queries=[original], meaning the LLM call was wasted. Now we
    try multiple extraction strategies before giving up.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.model_manager import get_model_manager

logger = get_logger(__name__)
_settings = get_settings()


@dataclass
class TransformedQuery:
    original: str
    rewritten: str
    sub_queries: list[str]
    expansions: list[str]


class QueryTransformer:
    """Transform queries for better retrieval using gemma3:1b."""

    def __init__(self) -> None:
        self._manager = get_model_manager()

    async def transform(self, query: str) -> TransformedQuery:
        """Rewrite, expand acronyms, decompose multi-part questions.

        FIX L12: raised short-query bypass from 5 → 8 words to match router.
        """
        if len(query.split()) < 8:
            return TransformedQuery(
                original=query,
                rewritten=query,
                sub_queries=[query],
                expansions=[],
            )
        try:
            return await self._llm_transform(query)
        except Exception as exc:
            logger.warning("Query transform failed (%s); using original.", exc)
            return TransformedQuery(
                original=query,
                rewritten=query,
                sub_queries=[query],
                expansions=[],
            )

    async def _llm_transform(self, query: str) -> TransformedQuery:
        # FIX L11: shorter prompt — same instructions, less prefill cost
        prompt = (
            "Rewrite this search query for document retrieval. "
            'Output ONLY valid JSON, no explanation:\n'
            '{"rewritten":"fixed grammar + expanded acronyms",'
            '"sub_queries":["focused sub-question 1","sub-question 2"],'
            '"expansions":["acronym or synonym"]}\n\n'
            f"Query: {query}"
        )

        raw = await self._manager.generate(
            _settings.llm_router,
            prompt,
            temperature=0.0,
            max_tokens=120,     # FIX L10: was 256
        )
        return self._parse_response(query, raw)

    def _parse_response(self, original: str, raw: str) -> TransformedQuery:
        """FIX L13: multi-strategy JSON extraction, never wastes the LLM call."""
        import json

        # Strategy 1: find outermost {...} block
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
                rewritten   = data.get("rewritten", original)
                sub_queries = data.get("sub_queries", [original])
                expansions  = data.get("expansions", [])
                # Validate types — gemma3:1b sometimes returns strings for lists
                if isinstance(sub_queries, str):
                    sub_queries = [sq.strip() for sq in sub_queries.split(";") if sq.strip()]
                if not sub_queries:
                    sub_queries = [rewritten or original]
                return TransformedQuery(
                    original=original,
                    rewritten=rewritten,
                    sub_queries=sub_queries[:3],  # cap at 3 to match retriever
                    expansions=expansions if isinstance(expansions, list) else [],
                )
            except json.JSONDecodeError:
                pass

        # Strategy 2: extract individual fields with regex when JSON is malformed
        rewritten_match = re.search(r'"rewritten"\s*:\s*"([^"]+)"', raw)
        sub_match = re.findall(r'"([^"]{10,})"', raw)  # any quoted string ≥10 chars

        if rewritten_match:
            rewritten = rewritten_match.group(1)
            # Use longer quoted strings as sub-queries (skip the rewritten itself)
            subs = [s for s in sub_match if s != rewritten and len(s) > 10][:2]
            if not subs:
                subs = [rewritten]
            logger.debug("Query transform: partial parse succeeded via regex")
            return TransformedQuery(
                original=original,
                rewritten=rewritten,
                sub_queries=subs,
                expansions=[],
            )

        # Strategy 3: use the raw text as a single rewritten query if it looks
        # like a sentence (not JSON debris)
        raw_clean = raw.strip().strip('"').strip("'")
        if (
            20 < len(raw_clean) < 300
            and not raw_clean.startswith("{")
            and "\n" not in raw_clean
        ):
            logger.debug("Query transform: using raw LLM output as rewritten query")
            return TransformedQuery(
                original=original,
                rewritten=raw_clean,
                sub_queries=[raw_clean],
                expansions=[],
            )

        # Complete fallback — LLM produced unusable output
        logger.warning("Query transform: all parse strategies failed; using original")
        return TransformedQuery(
            original=original,
            rewritten=original,
            sub_queries=[original],
            expansions=[],
        )


_transformer: QueryTransformer | None = None


def get_query_transformer() -> QueryTransformer:
    global _transformer
    if _transformer is None:
        _transformer = QueryTransformer()
    return _transformer