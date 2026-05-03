"""Header-aware semantic chunker with sentence-level overlap."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    header_path: str  # e.g. "H1: Intro > H2: Section A"
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "source": self.source,
            "header_path": self.header_path,
            "chunk_index": self.chunk_index,
            **self.metadata,
        }


class SemanticChunker:
    """Split documents on heading boundaries with sentence overlap."""

    def __init__(
        self,
        chunk_size: int | None = None,
        overlap_sentences: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or _settings.chunk_size
        self.overlap_sentences = overlap_sentences or _settings.chunk_overlap

    def chunk(self, content: str, source: str, metadata: dict | None = None) -> list[Chunk]:
        metadata = metadata or {}
        sections = self._split_by_headings(content)
        chunks: list[Chunk] = []
        for section_header, section_text in sections:
            section_chunks = self._chunk_section(section_text, section_header, source, metadata)
            chunks.extend(section_chunks)
        # Deduplicate and assign final IDs
        seen: set[str] = set()
        final: list[Chunk] = []
        for i, c in enumerate(chunks):
            if c.text not in seen:
                seen.add(c.text)
                c.chunk_index = i
                c.chunk_id = f"{_safe_id(source)}_{i:04d}"
                final.append(c)
        logger.info("Chunked %s → %d chunks", source, len(final))
        return final

    def _split_by_headings(self, content: str) -> list[tuple[str, str]]:
        """Return list of (header_path, section_text) pairs."""
        lines = content.split("\n")
        sections: list[tuple[str, str]] = []
        current_headers: list[tuple[int, str]] = []  # (level, title)
        current_lines: list[str] = []

        for line in lines:
            m = _HEADING_RE.match(line.strip())
            if m:
                # Flush current section
                if current_lines:
                    sections.append((
                        _header_path(current_headers),
                        "\n".join(current_lines).strip(),
                    ))
                    current_lines = []
                level = len(m.group(1))
                title = m.group(2).strip()
                # Pop headers at same or deeper level
                current_headers = [(l, t) for l, t in current_headers if l < level]
                current_headers.append((level, title))
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((_header_path(current_headers), "\n".join(current_lines).strip()))

        if not sections:
            sections = [("", content)]
        return sections

    def _chunk_section(
        self,
        text: str,
        header_path: str,
        source: str,
        metadata: dict,
    ) -> list[Chunk]:
        if not text.strip():
            return []
        sentences = _split_sentences(text)
        if not sentences:
            return []

        prefix = f"[{header_path}]\n" if header_path else ""
        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_len = 0

        def flush(buf: list[str], idx: int) -> Chunk:
            body = " ".join(buf)
            return Chunk(
                chunk_id=f"{_safe_id(source)}_{idx:04d}",
                text=f"{prefix}{body}".strip(),
                source=source,
                header_path=header_path,
                chunk_index=idx,
                metadata=metadata,
            )

        for i, sent in enumerate(sentences):
            sent_len = len(sent)
            if buf_len + sent_len > self.chunk_size and buf:
                chunks.append(flush(buf, len(chunks)))
                # Keep overlap: last N sentences
                buf = buf[-self.overlap_sentences :] if self.overlap_sentences else []
                buf_len = sum(len(s) for s in buf)
            buf.append(sent)
            buf_len += sent_len

        if buf:
            chunks.append(flush(buf, len(chunks)))
        return chunks


def _split_sentences(text: str) -> list[str]:
    raw = _SENTENCE_END.split(text.strip())
    return [s.strip() for s in raw if s.strip()]


def _header_path(headers: list[tuple[int, str]]) -> str:
    if not headers:
        return ""
    return " > ".join(f"H{l}: {t}" for l, t in headers)


def _safe_id(source: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", source)[-40:]
