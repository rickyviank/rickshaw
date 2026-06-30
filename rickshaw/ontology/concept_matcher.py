"""Optional embedding-backed concept matching.

All operations here SUGGEST mappings — they never silently decide.
They are disabled gracefully when no embedding-capable provider is configured
(checked via ``capabilities().embeddings``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rickshaw.ontology.schema import Entity, OntologyGraph
from rickshaw.providers.base import EmbeddingMixin, LLMProvider


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass
class MatchResult:
    """A suggested mapping from embedding similarity."""

    entity: Entity
    score: float


class ConceptMatcher:
    """Embedding-backed helper for fuzzy concept operations.

    Requires a provider that implements :class:`EmbeddingMixin` and reports
    ``capabilities().embeddings == True``.  If the provider does not support
    embeddings, all methods raise :class:`RuntimeError` with a clear message.
    """

    def __init__(
        self,
        provider: LLMProvider,
        graph: OntologyGraph,
        similarity_threshold: float = 0.75,
    ) -> None:
        self._provider = provider
        self._graph = graph
        self._threshold = similarity_threshold
        self._cache: dict[str, list[float]] = {}

    @property
    def available(self) -> bool:
        return (
            isinstance(self._provider, EmbeddingMixin)
            and self._provider.capabilities().embeddings
        )

    def _require_embeddings(self) -> None:
        if not self.available:
            raise RuntimeError(
                f"Embedding features are not available: provider "
                f"{self._provider.name!r} does not support embeddings. "
                f"Configure an embedding-capable provider to use concept matching."
            )

    def _embed(self, text: str) -> list[float]:
        if text not in self._cache:
            assert isinstance(self._provider, EmbeddingMixin)
            self._cache[text] = self._provider.embed(text)
        return self._cache[text]

    def classify(
        self,
        text: str,
        entity_type: str | None = None,
    ) -> list[MatchResult]:
        """Suggest entities that match *text* by embedding similarity.

        Returns matches above the similarity threshold, sorted by score
        descending.  These are SUGGESTIONS — the caller should confirm.
        """
        self._require_embeddings()
        text_vec = self._embed(text)
        results: list[MatchResult] = []
        for entity in self._graph.list_entities(entity_type=entity_type):
            label = entity.fields.get("label", entity.id)
            entity_vec = self._embed(str(label))
            score = _cosine_similarity(text_vec, entity_vec)
            if score >= self._threshold:
                results.append(MatchResult(entity=entity, score=score))
        results.sort(key=lambda m: m.score, reverse=True)
        return results

    def detect_duplicates(
        self,
        entity_type: str | None = None,
    ) -> list[tuple[Entity, Entity, float]]:
        """Find entity pairs that may be synonyms/duplicates.

        Returns triples ``(entity_a, entity_b, similarity)`` above the
        threshold.  These are SUGGESTIONS for human review.
        """
        self._require_embeddings()
        entities = self._graph.list_entities(entity_type=entity_type)
        pairs: list[tuple[Entity, Entity, float]] = []
        for i, a in enumerate(entities):
            vec_a = self._embed(a.fields.get("label", a.id))
            for b in entities[i + 1 :]:
                vec_b = self._embed(b.fields.get("label", b.id))
                score = _cosine_similarity(vec_a, vec_b)
                if score >= self._threshold:
                    pairs.append((a, b, score))
        pairs.sort(key=lambda t: t[2], reverse=True)
        return pairs

    def suggest_links(
        self,
        text: str,
        entity_type: str | None = None,
    ) -> list[MatchResult]:
        """Suggest fuzzy links between *text* and existing entities.

        Identical to :meth:`classify` — provided as a distinct semantic
        entry point for callers building relationship suggestions.
        """
        return self.classify(text, entity_type=entity_type)
