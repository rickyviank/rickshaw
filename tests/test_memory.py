"""Tests for the memory subsystem — store, ranker, dedupe, embedder, tools."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from rickshaw.memory._math import cosine_similarity
from rickshaw.memory.embedder import LocalEmbedder, ProviderEmbedder
from rickshaw.memory.ranker import Ranker
from rickshaw.memory.record import MemoryRecord, MemoryScope, MemoryType
from rickshaw.memory.service import MemoryService
from rickshaw.memory.store import MemoryStore
from rickshaw.memory.tools import (
    FORGET_SPEC,
    MEMORY_TOOL_SPECS,
    RECALL_SPEC,
    REMEMBER_SPEC,
    dispatch_tool_call,
)
from rickshaw.providers.base import Response, TokenUsage, ToolCall


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------

def test_cosine_identical_vectors():
    v = [1.0, 0.0, 0.0]
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_opposite_vectors():
    assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_empty_vectors():
    assert cosine_similarity([], []) == 0.0


def test_cosine_mismatched_lengths():
    assert cosine_similarity([1.0], [1.0, 2.0]) == 0.0


def test_cosine_zero_vector():
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# Local embedder
# ---------------------------------------------------------------------------

def test_local_embedder_deterministic():
    embedder = LocalEmbedder(dim=32)
    v1 = embedder.embed("hello world")
    v2 = embedder.embed("hello world")
    assert v1 == v2


def test_local_embedder_dimension():
    embedder = LocalEmbedder(dim=16)
    assert len(embedder.embed("test")) == 16
    assert embedder.dimension == 16


def test_local_embedder_normalized():
    embedder = LocalEmbedder(dim=32)
    vec = embedder.embed("test")
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_local_embedder_different_texts_differ():
    embedder = LocalEmbedder(dim=32)
    v1 = embedder.embed("apple")
    v2 = embedder.embed("banana")
    assert v1 != v2


# ---------------------------------------------------------------------------
# Provider embedder
# ---------------------------------------------------------------------------

class _FakeEmbeddingProvider:
    def embed(self, text: str) -> list[float]:
        return [float(ord(c)) for c in text[:3]]


def test_provider_embedder():
    fake = _FakeEmbeddingProvider()
    embedder = ProviderEmbedder(fake, dim=3)
    vec = embedder.embed("abc")
    assert vec == [97.0, 98.0, 99.0]
    assert embedder.dimension == 3


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

def test_store_put_and_get():
    store = MemoryStore()
    rec = MemoryRecord(id="r1", text="hello", embedding=[1.0, 0.0])
    store.put(rec)
    fetched = store.get("r1")
    assert fetched is not None
    assert fetched.text == "hello"
    assert fetched.embedding == [1.0, 0.0]


def test_store_get_nonexistent():
    store = MemoryStore()
    assert store.get("nonexistent") is None


def test_store_search_scope_filtering():
    store = MemoryStore()
    r1 = MemoryRecord(id="r1", text="global", embedding=[1.0, 0.0], scope=MemoryScope.GLOBAL)
    r2 = MemoryRecord(id="r2", text="session", embedding=[0.9, 0.1], scope=MemoryScope.SESSION)
    r3 = MemoryRecord(id="r3", text="task", embedding=[0.8, 0.2], scope=MemoryScope.TASK)
    store.put(r1)
    store.put(r2)
    store.put(r3)

    results = store.search([1.0, 0.0], scope_filter=[MemoryScope.GLOBAL])
    assert len(results) == 1
    assert results[0][0].id == "r1"


def test_store_search_excludes_superseded():
    store = MemoryStore()
    r1 = MemoryRecord(id="r1", text="old", embedding=[1.0, 0.0])
    r1.superseded_by = "r2"
    store.put(r1)
    r2 = MemoryRecord(id="r2", text="new", embedding=[1.0, 0.0])
    store.put(r2)

    results = store.search([1.0, 0.0])
    ids = [r.id for r, _ in results]
    assert "r1" not in ids
    assert "r2" in ids


def test_store_delete():
    store = MemoryStore()
    rec = MemoryRecord(id="r1", text="delete me", embedding=[1.0])
    store.put(rec)
    assert store.delete("r1") is True
    assert store.get("r1") is None
    assert store.delete("r1") is False


def test_store_mark_superseded():
    store = MemoryStore()
    rec = MemoryRecord(id="r1", text="original", embedding=[1.0])
    store.put(rec)
    store.mark_superseded("r1", "r2")
    updated = store.get("r1")
    assert updated is not None
    assert updated.superseded_by == "r2"


def test_store_all_records():
    store = MemoryStore()
    store.put(MemoryRecord(id="a", text="a", embedding=[1.0], scope=MemoryScope.GLOBAL))
    store.put(MemoryRecord(id="b", text="b", embedding=[1.0], scope=MemoryScope.SESSION))
    assert len(store.all_records()) == 2
    assert len(store.all_records(scope_filter=[MemoryScope.GLOBAL])) == 1


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------

def test_ranker_basic_ordering():
    ranker = Ranker(w_rel=1.0, w_rec=0.0, w_imp=0.0)
    now = datetime.now(timezone.utc)
    r1 = MemoryRecord(id="r1", text="low", embedding=[1.0], last_used_at=now)
    r2 = MemoryRecord(id="r2", text="high", embedding=[1.0], last_used_at=now)

    candidates = [(r1, 0.3), (r2, 0.9)]
    ranked = ranker.rank(candidates)
    assert ranked[0].id == "r2"


def test_ranker_respects_importance():
    ranker = Ranker(w_rel=0.0, w_rec=0.0, w_imp=1.0)
    now = datetime.now(timezone.utc)
    r1 = MemoryRecord(id="r1", text="low", embedding=[1.0], importance=0.1, last_used_at=now)
    r2 = MemoryRecord(id="r2", text="high", embedding=[1.0], importance=0.9, last_used_at=now)

    ranked = ranker.rank([(r1, 0.5), (r2, 0.5)])
    assert ranked[0].id == "r2"


def test_ranker_empty():
    ranker = Ranker()
    assert ranker.rank([]) == []


def test_ranker_limit():
    ranker = Ranker(w_rel=1.0, w_rec=0.0, w_imp=0.0)
    now = datetime.now(timezone.utc)
    candidates = [
        (MemoryRecord(id=f"r{i}", text=f"r{i}", embedding=[1.0], last_used_at=now), float(i))
        for i in range(10)
    ]
    ranked = ranker.rank(candidates, limit=3)
    assert len(ranked) == 3


def test_ranker_mmr_diversity():
    ranker = Ranker(w_rel=1.0, w_rec=0.0, w_imp=0.0, diversity_penalty=0.5)
    now = datetime.now(timezone.utc)
    # Two similar records and one different
    r1 = MemoryRecord(id="r1", text="a", embedding=[1.0, 0.0], last_used_at=now)
    r2 = MemoryRecord(id="r2", text="b", embedding=[0.99, 0.01], last_used_at=now)
    r3 = MemoryRecord(id="r3", text="c", embedding=[0.0, 1.0], last_used_at=now)

    candidates = [(r1, 0.9), (r2, 0.85), (r3, 0.5)]
    ranked = ranker.rank(candidates, limit=3)
    assert len(ranked) == 3


# ---------------------------------------------------------------------------
# Memory service: dedupe-on-write
# ---------------------------------------------------------------------------

def test_service_dedupe_on_write():
    service = MemoryService(embedder=LocalEmbedder(dim=32))
    rec1 = service.write("the sky is blue")
    assert rec1 is not None
    # Same text should be deduped
    rec2 = service.write("the sky is blue")
    assert rec2 is None


def test_service_write_different_texts():
    service = MemoryService(embedder=LocalEmbedder(dim=32))
    r1 = service.write("apple")
    r2 = service.write("something completely different and unrelated")
    assert r1 is not None
    assert r2 is not None


def test_service_write_observations():
    service = MemoryService(embedder=LocalEmbedder(dim=32))
    resp = Response(text="Hello, I'm the assistant", model="test", usage=TokenUsage())
    records = service.write_observations(resp)
    assert len(records) == 1
    assert records[0].text == "Hello, I'm the assistant"


def test_service_write_observations_empty_text():
    service = MemoryService(embedder=LocalEmbedder(dim=32))
    resp = Response(text="", model="test", usage=TokenUsage())
    records = service.write_observations(resp)
    assert records == []


def test_service_remember_recall_forget():
    service = MemoryService(embedder=LocalEmbedder(dim=32))
    rid = service.remember("user likes dark mode")
    assert isinstance(rid, str)
    assert rid != "duplicate: already stored"

    results = service.recall("dark mode")
    assert len(results) > 0
    assert any("dark mode" in r["text"] for r in results)

    msg = service.forget(rid)
    assert "deleted" in msg

    msg2 = service.forget(rid)
    assert "not found" in msg2


# ---------------------------------------------------------------------------
# Memory tools: schemas and dispatch
# ---------------------------------------------------------------------------

def test_memory_tool_specs_have_required_fields():
    for spec in MEMORY_TOOL_SPECS:
        assert spec.name
        assert spec.description
        assert "type" in spec.parameters
        assert spec.parameters["type"] == "object"


def test_dispatch_remember():
    service = MemoryService(embedder=LocalEmbedder(dim=32))
    tc = ToolCall(id="t1", name="remember", arguments={"fact": "test fact"})
    result = json.loads(dispatch_tool_call(tc, service))
    assert isinstance(result, str)
    assert result != "duplicate: already stored"


def test_dispatch_recall():
    service = MemoryService(embedder=LocalEmbedder(dim=32))
    service.write("apples are fruits")
    tc = ToolCall(id="t2", name="recall", arguments={"query": "fruit"})
    result = json.loads(dispatch_tool_call(tc, service))
    assert isinstance(result, list)


def test_dispatch_forget():
    service = MemoryService(embedder=LocalEmbedder(dim=32))
    rid = service.remember("temporary")
    tc = ToolCall(id="t3", name="forget", arguments={"id": rid})
    result = json.loads(dispatch_tool_call(tc, service))
    assert "deleted" in result


def test_dispatch_unknown_tool():
    service = MemoryService(embedder=LocalEmbedder(dim=32))
    tc = ToolCall(id="t4", name="unknown_tool", arguments={})
    result = json.loads(dispatch_tool_call(tc, service))
    assert "unknown tool" in result
