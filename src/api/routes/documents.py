"""GET /documents — list all ingested document sources from ChromaDB."""
from __future__ import annotations

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter()


@router.get("/documents")
async def list_documents() -> JSONResponse:
    """Return all unique source filenames currently stored in ChromaDB."""
    try:
        from src.db.chroma_client import get_chroma_client
        collection = get_chroma_client().get_or_create_collection()
        total = collection.count()
        if total == 0:
            return JSONResponse(content={"documents": [], "total_chunks": 0})

        # Fetch all metadata to extract unique sources
        # ChromaDB get() with no ids returns all records
        results = collection.get(include=["metadatas"], limit=total)
        metas = results.get("metadatas") or []

        # Collect unique sources with chunk counts
        source_counts: dict[str, int] = {}
        for meta in metas:
            if meta:
                src = meta.get("source") or meta.get("file_path") or ""
                if src:
                    source_counts[src] = source_counts.get(src, 0) + 1

        docs = [
            {"source": src, "chunks": cnt, "name": src.split("/")[-1]}
            for src, cnt in sorted(source_counts.items())
        ]
        return JSONResponse(content={
            "documents": docs,
            "total_chunks": total,
            "total_documents": len(docs),
        })
    except Exception as exc:
        logger.error("Failed to list documents: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": str(exc), "documents": [], "total_chunks": 0},
        )
