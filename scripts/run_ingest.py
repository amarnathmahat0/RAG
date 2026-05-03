import asyncio
from pathlib import Path
from src.ingestion.pipeline import IngestionPipeline


async def main():
    p = IngestionPipeline()
    docs = list(Path("data/raw").rglob("*.pdf")) + list(Path("data/raw").rglob("*.txt"))

    results = await p.ingest_batch([str(d) for d in docs])

    for r in results:
        print(f"{r.source}: {r.chunks_created} chunks")


if __name__ == "__main__":
    asyncio.run(main())