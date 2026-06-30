"""Tests for the ontology layer and concept matcher."""

import json
import math
from unittest.mock import MagicMock

import pytest

from rickshaw.ontology.schema import Entity, OntologyGraph, Relationship
from rickshaw.ontology.concept_matcher import ConceptMatcher, _cosine_similarity
from rickshaw.providers.base import Capabilities, EmbeddingMixin, Effort, LLMProvider, Message, Response


# ---------------------------------------------------------------------------
# OntologyGraph
# ---------------------------------------------------------------------------

def test_add_and_get_entity():
    g = OntologyGraph()
    e = Entity(id="py", entity_type="language", fields={"label": "Python"})
    g.add_entity(e)
    assert g.get_entity("py") is e


def test_add_duplicate_entity_raises():
    g = OntologyGraph()
    g.add_entity(Entity(id="py", entity_type="language"))
    with pytest.raises(ValueError, match="already exists"):
        g.add_entity(Entity(id="py", entity_type="language"))


def test_remove_entity_cleans_relationships():
    g = OntologyGraph()
    g.add_entity(Entity(id="a", entity_type="t"))
    g.add_entity(Entity(id="b", entity_type="t"))
    g.add_relationship(Relationship(source_id="a", target_id="b", relation_type="r"))
    g.remove_entity("a")
    assert g.get_entity("a") is None
    assert g.get_relationships() == []


def test_list_entities_by_type():
    g = OntologyGraph()
    g.add_entity(Entity(id="py", entity_type="language"))
    g.add_entity(Entity(id="ml", entity_type="concept"))
    assert len(g.list_entities("language")) == 1
    assert len(g.list_entities()) == 2


def test_add_relationship_missing_entity():
    g = OntologyGraph()
    g.add_entity(Entity(id="a", entity_type="t"))
    with pytest.raises(ValueError, match="not found"):
        g.add_relationship(
            Relationship(source_id="a", target_id="missing", relation_type="r")
        )


def test_neighbors():
    g = OntologyGraph()
    g.add_entity(Entity(id="a", entity_type="t"))
    g.add_entity(Entity(id="b", entity_type="t"))
    g.add_entity(Entity(id="c", entity_type="t"))
    g.add_relationship(Relationship(source_id="a", target_id="b", relation_type="r"))
    g.add_relationship(Relationship(source_id="c", target_id="a", relation_type="r"))
    neighbors = list(g.neighbors("a"))
    ids = {n.id for n in neighbors}
    assert ids == {"b", "c"}


def test_validate_clean():
    g = OntologyGraph()
    g.add_entity(Entity(id="a", entity_type="t"))
    assert g.validate() == []


def test_save_and_load(tmp_path):
    path = tmp_path / "graph.json"
    g = OntologyGraph()
    g.add_entity(Entity(id="a", entity_type="t", fields={"label": "A"}))
    g.add_entity(Entity(id="b", entity_type="t", fields={"label": "B"}))
    g.add_relationship(Relationship(source_id="a", target_id="b", relation_type="r"))
    g.save(path)

    g2 = OntologyGraph(path)
    assert g2.get_entity("a") is not None
    assert len(g2.get_relationships()) == 1


# ---------------------------------------------------------------------------
# ConceptMatcher
# ---------------------------------------------------------------------------

class _FakeEmbeddingProvider(EmbeddingMixin, LLMProvider):
    """Test provider that supports embeddings."""

    _vectors: dict[str, list[float]] = {}

    @property
    def name(self) -> str:
        return "fake"

    def complete(self, messages, effort=Effort.MEDIUM, **kw) -> Response:
        raise NotImplementedError

    def available_models(self) -> list[str]:
        return ["fake"]

    def validate(self) -> None:
        pass

    def capabilities(self) -> Capabilities:
        return Capabilities(embeddings=True)

    def embed(self, text: str) -> list[float]:
        return self._vectors.get(text, [0.0] * 3)


class _FakeNoEmbeddingProvider(LLMProvider):
    """Test provider without embeddings."""

    @property
    def name(self) -> str:
        return "nonembed"

    def complete(self, messages, effort=Effort.MEDIUM, **kw) -> Response:
        raise NotImplementedError

    def available_models(self) -> list[str]:
        return []

    def validate(self) -> None:
        pass

    def capabilities(self) -> Capabilities:
        return Capabilities(embeddings=False)


def test_concept_matcher_not_available_without_embeddings():
    g = OntologyGraph()
    provider = _FakeNoEmbeddingProvider()
    matcher = ConceptMatcher(provider=provider, graph=g)
    assert matcher.available is False


def test_concept_matcher_raises_without_embeddings():
    g = OntologyGraph()
    provider = _FakeNoEmbeddingProvider()
    matcher = ConceptMatcher(provider=provider, graph=g)
    with pytest.raises(RuntimeError, match="not available"):
        matcher.classify("test")


def test_concept_matcher_classify():
    g = OntologyGraph()
    g.add_entity(Entity(id="py", entity_type="lang", fields={"label": "Python"}))
    g.add_entity(Entity(id="js", entity_type="lang", fields={"label": "JavaScript"}))

    provider = _FakeEmbeddingProvider()
    # Make "Python programming" very similar to "Python"
    provider._vectors = {
        "Python programming": [1.0, 0.0, 0.0],
        "Python": [0.99, 0.1, 0.0],
        "JavaScript": [0.0, 1.0, 0.0],
    }

    matcher = ConceptMatcher(provider=provider, graph=g, similarity_threshold=0.5)
    results = matcher.classify("Python programming")
    assert len(results) >= 1
    assert results[0].entity.id == "py"


def test_concept_matcher_detect_duplicates():
    g = OntologyGraph()
    g.add_entity(Entity(id="a", entity_type="t", fields={"label": "dog"}))
    g.add_entity(Entity(id="b", entity_type="t", fields={"label": "canine"}))
    g.add_entity(Entity(id="c", entity_type="t", fields={"label": "cat"}))

    provider = _FakeEmbeddingProvider()
    provider._vectors = {
        "dog": [1.0, 0.0, 0.0],
        "canine": [0.98, 0.1, 0.0],
        "cat": [0.0, 1.0, 0.0],
    }

    matcher = ConceptMatcher(provider=provider, graph=g, similarity_threshold=0.5)
    dupes = matcher.detect_duplicates()
    assert len(dupes) >= 1
    ids = {dupes[0][0].id, dupes[0][1].id}
    assert ids == {"a", "b"}


def test_cosine_similarity_identical():
    assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal():
    assert _cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert _cosine_similarity([0, 0, 0], [1, 0, 0]) == 0.0
