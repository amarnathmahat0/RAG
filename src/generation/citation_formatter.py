"""Citation formatting utilities."""
from __future__ import annotations

import re


def format_inline_citations(text: str, sources: list[dict]) -> str:
    """Ensure citation numbers in text match the sources list."""
    # Already has citations → clean up duplicates
    text = re.sub(r"\[(\d+)\]\s*\[(\d+)\]", r"[\1][\2]", text)
    return text


def extract_cited_sources(text: str, sources: list[dict]) -> list[dict]:
    """Return only sources actually cited in the text."""
    cited_nums = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
    return [s for i, s in enumerate(sources, 1) if i in cited_nums]


def build_source_block(sources: list[dict]) -> str:
    lines = []
    for i, s in enumerate(sources, 1):
        src = s.get("source", "unknown")
        score = s.get("score", 0.0)
        lines.append(f"[{i}] {src} (relevance: {score:.3f})")
    return "\n".join(lines)
