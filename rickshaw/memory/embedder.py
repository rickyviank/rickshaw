"""Embedder protocol and implementations — local default + provider adapter."""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rickshaw.providers.base import EmbeddingMixin

_DEFAULT_DIM = 128

_TOKEN_RE = re.compile(r"[a-z0-9]+")


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
