#!/usr/bin/env python3
"""
Helper script to build or extend the benchmark test.json from real documents.

Usage:
    python scripts/create_benchmark.py --docs data/raw/ --output data/benchmarks/test.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def build_benchmark(docs_dir: str, output: str, n_questions: int = 10) -> None:
    from src.ingestion.pipeline import IngestionPipeline
    from src.ingestion.parser import DocumentParser
    from src.utils.model_manager import get_model_manager
    from src.utils.config import get_settings

    settings = get_settings()
    parser = DocumentParser()
    pipeline = IngestionPipeline()
    mm = get_model_manager()

    docs_path = Path(docs_dir)
    sources = list(docs_path.rglob("*.pdf")) + list(docs_path.rglob("*.txt")) + \
              list(docs_path.rglob("*.md")) + list(docs_path.rglob("*.docx"))

    if not sources:
        print(f"No documents found in {docs_dir}")
        return

    print(f"Found {len(sources)} documents. Ingesting…")
    results = await pipeline.ingest_batch([str(s) for s in sources])

    # Load existing benchmark if any
    output_path = Path(output)
    existing: list[dict] = []
    if output_path.exists():
        with open(output_path) as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing test cases.")

    # Generate questions for each document using qwen:1.8b
    new_cases: list[dict] = []
    for src in sources[:5]:  # limit to 5 docs
        print(f"Generating questions for {src.name}…")
        parsed = parser.parse(str(src))
        snippet = parsed.content[:2000]

        prompt = f"""Given the document excerpt below, generate {n_questions} diverse questions
that can ONLY be answered from this document. For each question, also provide:
- The exact answer from the document (2-3 sentences)
- Difficulty: easy/medium/hard
- Expected tier: TIER_1 for simple facts, TIER_2 for analysis, TIER_3 for multi-hop

Output ONLY a JSON array:
[{{"query": "...", "ground_truth_answer": "...", "difficulty": "easy", "expected_tier": "TIER_1"}}]

Document:
{snippet}"""

        try:
            raw = await mm.generate(settings.llm_judge, prompt, temperature=0.2, max_tokens=1500)
            import re
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if match:
                cases = json.loads(match.group())
                for c in cases:
                    c["ground_truth_context"] = []
                    c["all_relevant_chunks"] = []
                    new_cases.append(c)
                print(f"  Generated {len(cases)} questions from {src.name}")
        except Exception as e:
            print(f"  Failed for {src.name}: {e}")

    all_cases = existing + new_cases
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_cases, f, indent=2)
    print(f"\nSaved {len(all_cases)} test cases to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build NexusRAG benchmark dataset")
    parser.add_argument("--docs", default="data/raw", help="Directory with source documents")
    parser.add_argument("--output", default="data/benchmarks/test.json", help="Output JSON path")
    parser.add_argument("--questions", type=int, default=10, help="Questions per document")
    args = parser.parse_args()
    asyncio.run(build_benchmark(args.docs, args.output, args.questions))


if __name__ == "__main__":
    main()
