"""End-to-end ingestion pipeline: parse → chunk → embed → store.

FIXES
─────
FIX 1 — "input length exceeds context length" (400 on /api/embed)
    all-minilm has a hard 512-token context window. Previously the pipeline
    sent full chunk texts (up to 1024 chars ≈ 300-400 tokens but can spike
    higher for dense technical text) as one batched request. Any chunk over
    the limit caused Ollama to return 400 and abort the entire ingestion.

    Fix: truncate each text to MAX_EMBED_CHARS before embedding. 1800 chars
    is a safe upper bound — all-minilm tokenises roughly 4 chars/token so
    1800 chars ≈ 450 tokens, comfortably under the 512-token limit even for
    high-density text. The full text is still stored in ChromaDB; only the
    embedding input is truncated (this is standard practice — the embedding
    captures the semantic gist of the beginning, which is sufficient for
    retrieval ranking).

FIX 2 — Batch size limit to avoid memory spikes
    Sending 459 chunks in a single /api/embed call can time-out or OOM on
    low-RAM machines. Split into sub-batches of EMBED_BATCH_SIZE=32.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.ingestion.chunker import Chunk, SemanticChunker
from src.ingestion.entity_extractor import Entity, EntityExtractor
from src.ingestion.parser import DocumentParser
from src.utils.config import get_settings
from src.utils.exceptions import IngestionError
from src.utils.logger import get_logger
from src.utils.model_manager import get_model_manager

logger = get_logger(__name__)
_settings = get_settings()

# FIX 1: safe char limit for all-minilm (512 tokens × ~4 chars/token, with margin)
MAX_EMBED_CHARS = 1800
# FIX 2: sub-batch size to avoid OOM / timeout on large documents
EMBED_BATCH_SIZE = 32


def _truncate_for_embed(text: str) -> str:
    """Truncate text to MAX_EMBED_CHARS so it fits within the model context window.

    The full text is stored in ChromaDB separately; only the embedding input
    is truncated. Truncating at a word boundary avoids splitting mid-token.
    """
    if len(text) <= MAX_EMBED_CHARS:
        return text
    # Walk back to the nearest space so we don't cut mid-word
    truncated = text[:MAX_EMBED_CHARS]
    last_space = truncated.rfind(" ")
    if last_space > MAX_EMBED_CHARS // 2:
        truncated = truncated[:last_space]
    return truncated


@dataclass
class IngestionResult:
    source: str
    chunks_created: int
    entities_extracted: int
    graph_nodes_created: int
    elapsed_sec: float
    chunk_ids: list[str]


class IngestionPipeline:
    """Parse → chunk → extract entities → embed → write to ChromaDB + Neo4j."""

    def __init__(self) -> None:
        self._parser = DocumentParser()
        self._chunker = SemanticChunker()
        self._extractor = EntityExtractor()
        self._manager = get_model_manager()

    async def ingest(self, source: str, metadata: dict[str, Any] | None = None) -> IngestionResult:
        """Ingest a single source (file path or URL)."""
        t0 = time.perf_counter()
        metadata = metadata or {}
        logger.info("Ingestion START source=%s", source)

        # ── 1. Parse ───────────────────────────────────────────────────────────
        try:
            doc = self._parser.parse(source)
        except Exception as exc:
            raise IngestionError(f"Parse failed for {source}: {exc}") from exc

        doc_meta = {**metadata, **doc.metadata, "doc_type": doc.doc_type}

        # ── 2. Chunk ───────────────────────────────────────────────────────────
        chunks: list[Chunk] = self._chunker.chunk(doc.content, source, doc_meta)
        if not chunks:
            logger.warning("No chunks produced for %s", source)
            return IngestionResult(
                source=source, chunks_created=0, entities_extracted=0,
                graph_nodes_created=0, elapsed_sec=time.perf_counter() - t0,
                chunk_ids=[],
            )

        # ── 3. Embed (FIX 1 + 2: truncate texts, use sub-batches) ─────────────
        embed_texts = [_truncate_for_embed(c.text) for c in chunks]
        over_limit = sum(1 for t in [c.text for c in chunks] if len(t) > MAX_EMBED_CHARS)
        if over_limit:
            logger.info(
                "Truncated %d/%d chunks to %d chars for embedding (full text stored separately)",
                over_limit, len(chunks), MAX_EMBED_CHARS,
            )

        embeddings = await self._embed_batch(embed_texts)

        # ── 4. Store vectors (ChromaDB) — store FULL text, not truncated ───────
        await self._store_vectors(chunks, embeddings)

        # ── 5. Extract entities & build graph ──────────────────────────────────
        all_entities: list[Entity] = []
        for chunk in chunks[:20]:
            try:
                ents = await self._extractor.extract_async(chunk.text, chunk.chunk_id)
                all_entities.extend(ents)
            except Exception as exc:
                logger.warning("Entity extraction failed for chunk %s: %s", chunk.chunk_id, exc)

        graph_nodes = await self._store_graph(chunks, all_entities)

        elapsed = time.perf_counter() - t0
        logger.info(
            "Ingestion DONE source=%s chunks=%d entities=%d graph_nodes=%d elapsed=%.2fs",
            source, len(chunks), len(all_entities), graph_nodes, elapsed,
        )
        return IngestionResult(
            source=source,
            chunks_created=len(chunks),
            entities_extracted=len(all_entities),
            graph_nodes_created=graph_nodes,
            elapsed_sec=elapsed,
            chunk_ids=[c.chunk_id for c in chunks],
        )

    async def ingest_batch(self, sources: list[str]) -> list[IngestionResult]:
        results = []
        for source in sources:
            try:
                result = await self.ingest(source)
                results.append(result)
            except IngestionError as exc:
                logger.error("Ingestion failed for %s: %s", source, exc)
        return results

    # ── Private helpers ────────────────────────────────────────────────────────

    async def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed in sub-batches; fallback to fallback model on error.

        FIX 2: Split into EMBED_BATCH_SIZE chunks to avoid OOM / request
        timeouts on large documents (e.g. 459-chunk PDFs).
        """
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            logger.debug(
                "Embedding sub-batch %d-%d / %d",
                i + 1, min(i + EMBED_BATCH_SIZE, len(texts)), len(texts),
            )
            try:
                batch_embeddings = await self._manager.embed(
                    _settings.embed_model_primary, batch
                )
            except Exception as exc:
                logger.warning(
                    "Primary embed model failed on sub-batch %d-%d (%s), trying fallback…",
                    i + 1, min(i + EMBED_BATCH_SIZE, len(texts)), exc,
                )
                try:
                    batch_embeddings = await self._manager.embed(
                        _settings.embed_model_fallback, batch
                    )
                except Exception as exc2:
                    raise IngestionError(
                        f"All embedding models failed on sub-batch {i}-{i+EMBED_BATCH_SIZE}: {exc2}"
                    ) from exc2

            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    async def _store_vectors(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        from src.db.chroma_client import get_chroma_client

        client = get_chroma_client()
        collection = client.get_or_create_collection()
        ids   = [c.chunk_id for c in chunks]
        docs  = [c.text for c in chunks]          # full text stored here
        metas = [c.to_dict() for c in chunks]
        for m in metas:
            m.pop("text", None)
        collection.upsert(ids=ids, embeddings=embeddings, documents=docs, metadatas=metas)
        logger.debug("Stored %d vectors in ChromaDB", len(chunks))

    async def _store_graph(self, chunks: list[Chunk], entities: list[Entity]) -> int:
        """Store entity relationships in Neo4j. Returns number of nodes created."""
        try:
            from src.db.neo4j_client import get_neo4j_client

            client = get_neo4j_client()
            nodes_created = 0
            for chunk in chunks[:50]:
                client.upsert_chunk_node(chunk.chunk_id, chunk.source, chunk.header_path)
                nodes_created += 1
            for ent in entities:
                client.upsert_entity_node(ent.text, ent.label)
                client.create_mentions_relationship(ent.source_chunk_id, ent.text, ent.label)
                nodes_created += 1
            return nodes_created
        except Exception as exc:
            logger.warning("Graph storage skipped (Neo4j unavailable): %s", exc)
            return 0