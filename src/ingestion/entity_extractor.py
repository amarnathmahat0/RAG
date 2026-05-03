"""Entity extraction: spaCy primary, qwen:1.8b LLM fallback for domain terms."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from src.utils.logger import get_logger
from src.utils.model_manager import get_model_manager

logger = get_logger(__name__)


@dataclass
class Entity:
    text: str
    label: str  # PERSON, ORG, GPE, PRODUCT, etc.
    start_char: int
    end_char: int
    source_chunk_id: str


class EntityExtractor:
    """Extract named entities from chunks using spaCy + optional LLM fallback."""

    def __init__(self, use_llm_fallback: bool = True) -> None:
        self._nlp = None
        self._use_llm_fallback = use_llm_fallback
        self._manager = get_model_manager()

    def _load_spacy(self) -> bool:
        if self._nlp is not None:
            return True
        try:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")
            return True
        except (ImportError, OSError):
            logger.warning("spaCy en_core_web_sm not available.")
            return False

    def extract(self, text: str, chunk_id: str) -> list[Entity]:
        """Extract entities from text, returning a deduplicated list."""
        entities: list[Entity] = []
        if self._load_spacy():
            entities = self._extract_spacy(text, chunk_id)
        if not entities and self._use_llm_fallback:
            import asyncio

            entities = asyncio.get_event_loop().run_until_complete(
                self._extract_llm(text, chunk_id)
            )
        return entities

    async def extract_async(self, text: str, chunk_id: str) -> list[Entity]:
        entities: list[Entity] = []
        if self._load_spacy():
            entities = self._extract_spacy(text, chunk_id)
        if not entities and self._use_llm_fallback:
            entities = await self._extract_llm(text, chunk_id)
        return entities

    def _extract_spacy(self, text: str, chunk_id: str) -> list[Entity]:
        doc = self._nlp(text[:5000])  # truncate for speed
        seen: set[str] = set()
        entities: list[Entity] = []
        for ent in doc.ents:
            key = (ent.text.lower(), ent.label_)
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                Entity(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                    source_chunk_id=chunk_id,
                )
            )
        return entities

    async def _extract_llm(self, text: str, chunk_id: str) -> list[Entity]:
        """Use qwen:1.8b to extract domain-specific entities as JSON."""
        from src.utils.config import get_settings

        settings = get_settings()
        prompt = (
            "Extract named entities from the text below.\n"
            "Return ONLY a JSON array: [{\"text\": \"...\", \"label\": \"PERSON|ORG|GPE|PRODUCT|EVENT|OTHER\"}]\n"
            "Text:\n" + text[:2000]
        )
        try:
            raw = await self._manager.generate(
                settings.llm_judge,
                prompt,
                temperature=0.0,
                max_tokens=512,
            )
            # Extract JSON array
            match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not match:
                return []
            items = json.loads(match.group())
            return [
                Entity(
                    text=item["text"],
                    label=item.get("label", "OTHER"),
                    start_char=0,
                    end_char=0,
                    source_chunk_id=chunk_id,
                )
                for item in items
                if isinstance(item, dict) and "text" in item
            ]
        except Exception as exc:
            logger.warning("LLM entity extraction failed: %s", exc)
            return []
