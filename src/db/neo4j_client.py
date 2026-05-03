"""Neo4j client for entity graph operations."""
from __future__ import annotations

from functools import lru_cache
from typing import Any

from src.utils.config import get_settings
from src.utils.exceptions import GraphDBError
from src.utils.logger import get_logger

logger = get_logger(__name__)
_settings = get_settings()


class Neo4jClient:
    def __init__(self) -> None:
        try:
            from neo4j import GraphDatabase

            self._driver = GraphDatabase.driver(
                _settings.neo4j_uri,
                auth=(_settings.neo4j_user, _settings.neo4j_password),
            )
            self._driver.verify_connectivity()
            self._init_schema()
            logger.info("Neo4j connected: %s", _settings.neo4j_uri)
        except Exception as exc:
            raise GraphDBError(f"Neo4j connection failed: {exc}") from exc

    def _init_schema(self) -> None:
        """Create indexes and constraints."""
        with self._driver.session() as s:
            s.run("CREATE INDEX chunk_id IF NOT EXISTS FOR (c:Chunk) ON (c.chunk_id)")
            s.run("CREATE INDEX entity_text IF NOT EXISTS FOR (e:Entity) ON (e.text, e.label)")

    def upsert_chunk_node(self, chunk_id: str, source: str, header_path: str) -> None:
        with self._driver.session() as s:
            s.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.source = $source, c.header_path = $header_path
                """,
                chunk_id=chunk_id,
                source=source,
                header_path=header_path,
            )

    def upsert_entity_node(self, text: str, label: str) -> None:
        with self._driver.session() as s:
            s.run(
                "MERGE (e:Entity {text: $text, label: $label})",
                text=text,
                label=label,
            )

    def create_mentions_relationship(
        self, chunk_id: str, entity_text: str, entity_label: str
    ) -> None:
        with self._driver.session() as s:
            s.run(
                """
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MATCH (e:Entity {text: $entity_text, label: $entity_label})
                MERGE (c)-[:MENTIONS]->(e)
                """,
                chunk_id=chunk_id,
                entity_text=entity_text,
                entity_label=entity_label,
            )

    def find_related_chunks(self, entity_texts: list[str], max_hops: int = 2) -> list[str]:
        """Return chunk_ids related to any of the given entities (multi-hop)."""
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (e:Entity) WHERE e.text IN $entity_texts
                MATCH (c:Chunk)-[:MENTIONS]->(e)
                WITH c
                OPTIONAL MATCH (c2:Chunk)-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(c)
                RETURN DISTINCT coalesce(c2.chunk_id, c.chunk_id) AS chunk_id
                LIMIT 50
                """,
                entity_texts=entity_texts,
            )
            return [r["chunk_id"] for r in result if r["chunk_id"]]

    def health(self) -> dict[str, Any]:
        try:
            with self._driver.session() as s:
                res = s.run("RETURN 1 AS ok")
                res.single()
            return {"status": "ok"}
        except Exception as exc:
            return {"status": "error", "detail": str(exc)}

    def close(self) -> None:
        self._driver.close()


@lru_cache(maxsize=1)
def get_neo4j_client() -> Neo4jClient:
    return Neo4jClient()
