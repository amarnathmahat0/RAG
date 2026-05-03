"""DB schema constants and shared types."""
from __future__ import annotations

CHUNK_NODE_LABEL = "Chunk"
ENTITY_NODE_LABEL = "Entity"
MENTIONS_REL = "MENTIONS"
RELATED_REL = "RELATED_TO"

CHROMA_METADATA_FIELDS = [
    "source",
    "header_path",
    "chunk_index",
    "doc_type",
]
