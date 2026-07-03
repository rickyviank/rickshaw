"""Embedder protocol and implementations — tiered local embeddings.

Tiered design behind the stable :class:`Embedder` interface:

- **Model2VecEmbedder** (DEFAULT): static distilled sentence embeddings via
  ``model2vec``. The default model (``potion-base-4M``, 128-dim) is **vendored**
  in ``rickshaw/memory/models/`` so a fresh install works fully air-gapped with
  no model download. No inference cost (a static token-lookup table), beats
  TF-IDF semantically.
- **FastEmbedEmbedder** (opt-in upgrade): ``fastembed`` (ONNX runtime, no
  PyTorch) with ``bge-small-en-v1.5`` (384-dim). One-time model fetch then
  offline. Higher quality than Model2Vec at higher latency.
- **ProviderEmbedder**: adapter wrapping an ``EmbeddingMixin``-capable LLM
  provider — the not-offline-constrained option.
- **TFIDFEmbedder**: lexical fallback, kept for tests and as the graceful
  degradation path when no neural embedder is available.

Select the tier via :func:`make_embedder` (``tier="model2vec"`` default).
The interface and ``dimension`` contract stay stable across tiers.
"""

from __future__ import annotations

import logging
import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rickshaw.providers.base import EmbeddingMixin

logger = logging.getLogger(__name__)

_DEFAULT_DIM = 128

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Path to the vendored Model2Vec default model (air-gapped default).
_VENDORED_MODEL_DIR = Path(__file__).parent / "models" / "potion-base-4M"


def _tokenize(text: str) -> list[str]:
    """Lowercase and split on non-alphanumeric boundaries."""
    return _TOKEN_RE.findall(text.lower())


class Embedder(ABC):
    """Protocol for producing embedding vectors from text."""

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Return a fixed-dimension embedding vector for *text*."""

    @property
    def dimension(self) -> int:
        """Return the dimensionality of produced vectors."""
        return _DEFAULT_DIM


class TFIDFEmbedder(Embedder):
    """Offline, semantically-meaningful TF-IDF embedder with feature hashing.

    Maintains an in-memory vocabulary and document-frequency table that grows
    as new texts are embedded ("fit-on-the-fly"). Each token is hashed into a
    fixed-dimension vector; its weight is ``tf * idf``. The output is
    L2-normalized so cosine similarity reflects lexical overlap: texts that
    share words score higher than texts that do not.

    This is a **stepping stone**. It captures lexical (word-overlap) similarity
    only — "apple" and "fruit" are related only if they co-occur in observed
    text. Future iterations should replace this with a learned sentence encoder
    (e.g. ``all-MiniLM-L6-v2`` via ``sentence-transformers``) for true semantic
    similarity. See FUTURE.md ("Embeddings") for the roadmap.

    Determinism: for a given vocabulary state, ``embed`` is deterministic. The
    vocabulary only affects IDF weights; the hash bucket for a token is fixed.
    """

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        self._dim = dim
        self._doc_freq: Counter[str] = Counter()
        self._n_docs = 0

    @property
    def dimension(self) -> int:
        return self._dim

    def _bucket(self, token: str) -> int:
        """Deterministically map a token to a feature bucket in [0, dim)."""
        # Python's built-in hash is salted per-process; use a stable hash.
        h = 0
        for ch in token:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return h % self._dim

    def _fit(self, tokens: list[str]) -> None:
        """Update document-frequency stats with a newly-seen document."""
        self._n_docs += 1
        for token in set(tokens):
            self._doc_freq[token] += 1

    def _idf(self, token: str) -> float:
        """Smoothed inverse document frequency for *token*."""
        df = self._doc_freq.get(token, 0)
        return math.log((1 + self._n_docs) / (1 + df)) + 1.0

    def embed(self, text: str) -> list[float]:
        tokens = _tokenize(text)
        # Fit-on-the-fly: incorporate this text into the vocabulary/IDF table.
        self._fit(tokens)

        vec = [0.0] * self._dim
        if not tokens:
            return vec

        counts = Counter(tokens)
        total = len(tokens)
        for token, count in counts.items():
            tf = count / total
            weight = tf * self._idf(token)
            vec[self._bucket(token)] += weight

        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec


class ProviderEmbedder(Embedder):
    """Adapter wrapping an EmbeddingMixin-capable LLMProvider."""

    def __init__(self, provider: EmbeddingMixin, dim: int = _DEFAULT_DIM) -> None:
        self._provider = provider
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        return self._provider.embed(text)


class Model2VecEmbedder(Embedder):
    """Static distilled sentence embeddings via ``model2vec`` — the DEFAULT tier.

    Uses a vendored ``potion-base-4M`` model (128-dim, ~16MB) bundled in
    ``rickshaw/memory/models/`` so a fresh install works fully air-gapped with
    no model download. Embedding is a static token-lookup + average: no
    inference cost, deterministic, fully offline.

    If ``model2vec`` is not installed or the vendored model is missing, this
    falls back to :class:`TFIDFEmbedder` (graceful degradation) — callers
    should prefer :func:`make_embedder` which handles this automatically.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        self._model = None
        self._dim = 0
        self._fallback: TFIDFEmbedder | None = None
        path = Path(model_path) if model_path else _VENDORED_MODEL_DIR
        try:
            from model2vec import StaticModel

            self._model = StaticModel.from_pretrained(str(path))
            self._dim = self._model.dim
        except Exception as exc:
            logger.warning(
                "Model2Vec unavailable (%s); falling back to TFIDFEmbedder. "
                "Install model2vec and ensure the vendored model is present "
                "for semantic embeddings.",
                exc,
            )
            self._model = None
            self._fallback = TFIDFEmbedder(dim=_DEFAULT_DIM)
            self._dim = self._fallback.dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        if self._model is not None:
            vec = self._model.encode([text])[0]
            return [float(x) for x in vec]
        return self._fallback.embed(text)


class FastEmbedEmbedder(Embedder):
    """Opt-in upgrade tier: ``fastembed`` (ONNX, no PyTorch) sentence embeddings.

    Defaults to ``bge-small-en-v1.5`` (384-dim). One-time model fetch from the
    network on first use, then cached and fully offline. Higher quality than
    Model2Vec at higher latency. Requires the ``fastembed`` extra::

        pip install rickshaw[embed]
    """

    DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or self.DEFAULT_MODEL
        self._model = None
        self._dim = 0
        self._fallback: TFIDFEmbedder | None = None
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(model_name=self._model_name)
            # Infer dimension from a dummy embedding.
            dim_vec = next(self._model.embed(["dimension probe"]))
            self._dim = len(dim_vec)
        except Exception as exc:
            logger.warning(
                "fastembed unavailable (%s); falling back to TFIDFEmbedder. "
                "Install with: pip install fastembed",
                exc,
            )
            self._model = None
            self._fallback = TFIDFEmbedder(dim=_DEFAULT_DIM)
            self._dim = self._fallback.dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        if self._model is not None:
            vec = next(self._model.embed([text]))
            return [float(x) for x in vec]
        return self._fallback.embed(text)


def make_embedder(tier: str = "model2vec", **kwargs) -> Embedder:
    """Construct an embedder by tier name.

    Tiers:
      - ``"model2vec"`` (default): bundled static embeddings, fully offline.
      - ``"fastembed"``: ONNX bge-small, one-time fetch then offline.
      - ``"tfidf"``: lexical fallback (no deps).
      - ``"provider"``: wraps an ``EmbeddingMixin`` provider (pass ``provider=``).
    """
    tier = (tier or "model2vec").lower()
    if tier == "model2vec":
        return Model2VecEmbedder(**kwargs)
    if tier == "fastembed":
        return FastEmbedEmbedder(**kwargs)
    if tier == "tfidf":
        return TFIDFEmbedder(**kwargs)
    if tier == "provider":
        return ProviderEmbedder(**kwargs)
    raise ValueError(f"Unknown embedder tier: {tier!r}")
