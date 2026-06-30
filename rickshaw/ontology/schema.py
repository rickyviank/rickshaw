"""Symbolic ontology graph with typed entities and relationships.

This is the *structural* layer — no embeddings, just an explicit typed
schema/graph supporting exact lookups, validation, and traversal.

The default store is in-memory (optionally file-backed via JSON); swap in a
vector DB or other persistence by subclassing :class:`OntologyGraph`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass
class Entity:
    """A node in the ontology graph."""

    id: str
    entity_type: str
    fields: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "entity_type": self.entity_type,
            "fields": self.fields,
            "tags": list(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Entity:
        return cls(
            id=data["id"],
            entity_type=data["entity_type"],
            fields=data.get("fields", {}),
            tags=data.get("tags", []),
        )


@dataclass
class Relationship:
    """A directed edge between two entities."""

    source_id: str
    target_id: str
    relation_type: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Relationship:
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation_type=data["relation_type"],
            metadata=data.get("metadata", {}),
        )


class OntologyGraph:
    """In-memory typed graph with optional file persistence.

    Extension point: subclass and override the persistence methods to back
    this with a vector database or other storage engine.
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []
        self._path = Path(path) if path else None
        if self._path and self._path.is_file():
            self._load()

    # -- Entity operations ---------------------------------------------------

    def add_entity(self, entity: Entity) -> None:
        if entity.id in self._entities:
            raise ValueError(f"Entity {entity.id!r} already exists")
        self._entities[entity.id] = entity

    def get_entity(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def remove_entity(self, entity_id: str) -> None:
        self._entities.pop(entity_id, None)
        self._relationships = [
            r
            for r in self._relationships
            if r.source_id != entity_id and r.target_id != entity_id
        ]

    def list_entities(self, entity_type: str | None = None) -> list[Entity]:
        entities = list(self._entities.values())
        if entity_type is not None:
            entities = [e for e in entities if e.entity_type == entity_type]
        return entities

    # -- Relationship operations ---------------------------------------------

    def add_relationship(self, rel: Relationship) -> None:
        if rel.source_id not in self._entities:
            raise ValueError(f"Source entity {rel.source_id!r} not found")
        if rel.target_id not in self._entities:
            raise ValueError(f"Target entity {rel.target_id!r} not found")
        self._relationships.append(rel)

    def get_relationships(
        self,
        entity_id: str | None = None,
        relation_type: str | None = None,
    ) -> list[Relationship]:
        rels = self._relationships
        if entity_id is not None:
            rels = [
                r for r in rels if r.source_id == entity_id or r.target_id == entity_id
            ]
        if relation_type is not None:
            rels = [r for r in rels if r.relation_type == relation_type]
        return rels

    # -- Traversal -----------------------------------------------------------

    def neighbors(self, entity_id: str) -> Iterator[Entity]:
        """Yield entities directly connected to *entity_id*."""
        seen: set[str] = set()
        for rel in self._relationships:
            other: str | None = None
            if rel.source_id == entity_id:
                other = rel.target_id
            elif rel.target_id == entity_id:
                other = rel.source_id
            if other and other not in seen:
                seen.add(other)
                entity = self._entities.get(other)
                if entity:
                    yield entity

    # -- Validation ----------------------------------------------------------

    def validate(self) -> list[str]:
        """Return a list of issues (empty if the graph is consistent)."""
        issues: list[str] = []
        for rel in self._relationships:
            if rel.source_id not in self._entities:
                issues.append(
                    f"Relationship references missing source {rel.source_id!r}"
                )
            if rel.target_id not in self._entities:
                issues.append(
                    f"Relationship references missing target {rel.target_id!r}"
                )
        return issues

    # -- Persistence ---------------------------------------------------------

    def save(self, path: str | Path | None = None) -> None:
        dest = Path(path) if path else self._path
        if dest is None:
            raise ValueError("No path specified for saving")
        data = {
            "entities": [e.to_dict() for e in self._entities.values()],
            "relationships": [r.to_dict() for r in self._relationships],
        }
        dest.write_text(json.dumps(data, indent=2))

    def _load(self) -> None:
        assert self._path is not None
        data = json.loads(self._path.read_text())
        for edata in data.get("entities", []):
            entity = Entity.from_dict(edata)
            self._entities[entity.id] = entity
        for rdata in data.get("relationships", []):
            self._relationships.append(Relationship.from_dict(rdata))
