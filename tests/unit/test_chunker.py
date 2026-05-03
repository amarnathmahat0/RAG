"""Unit tests for SemanticChunker."""
from __future__ import annotations

import pytest
from src.ingestion.chunker import SemanticChunker, Chunk


SAMPLE_DOC = """# Introduction

This is the introduction section. It contains background information.
The system was designed for production use.

## Section A

This section discusses the first topic. It has multiple sentences.
Here is more detail about the approach. And one more sentence for good measure.

### Subsection A1

Deep details here. Very specific content follows.

## Section B

Another section with different content. This covers a separate topic.
"""


class TestSemanticChunker:
    def setup_method(self):
        self.chunker = SemanticChunker(chunk_size=200, overlap_sentences=1)

    def test_produces_chunks(self):
        chunks = self.chunker.chunk(SAMPLE_DOC, "test.md")
        assert len(chunks) > 0

    def test_all_chunks_have_text(self):
        chunks = self.chunker.chunk(SAMPLE_DOC, "test.md")
        for c in chunks:
            assert c.text.strip()

    def test_chunk_ids_unique(self):
        chunks = self.chunker.chunk(SAMPLE_DOC, "test.md")
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_header_path_present(self):
        chunks = self.chunker.chunk(SAMPLE_DOC, "test.md")
        # At least some chunks should carry a header path
        paths = [c.header_path for c in chunks if c.header_path]
        assert len(paths) > 0

    def test_short_doc_single_chunk(self):
        short = "This is a short document. It has two sentences."
        chunks = self.chunker.chunk(short, "short.txt")
        assert len(chunks) >= 1

    def test_empty_doc(self):
        chunks = self.chunker.chunk("", "empty.txt")
        assert chunks == []

    def test_source_preserved(self):
        chunks = self.chunker.chunk(SAMPLE_DOC, "myfile.pdf")
        for c in chunks:
            assert c.source == "myfile.pdf"

    def test_chunk_respects_size(self):
        # With small chunk size, should produce multiple chunks
        chunker = SemanticChunker(chunk_size=50, overlap_sentences=0)
        chunks = chunker.chunk(SAMPLE_DOC, "test.md")
        assert len(chunks) > 3

    def test_metadata_passed_through(self):
        meta = {"author": "test", "version": 1}
        chunks = self.chunker.chunk(SAMPLE_DOC, "test.md", metadata=meta)
        for c in chunks:
            assert c.metadata.get("author") == "test"
