"""Tests for the memory subsystem — store, ranker, dedupe, embedder, tools."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from rickshaw.memory._chroma_index import chroma_available
from rickshaw.memory._math import cosine_similarity
from rickshaw.memory.embedder import ProviderEmbedder, TFIDFEmbedder
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
# TF-IDF embedder
# ---------------------------------------------------------------------------

def test_tfidf_embedder_deterministic():
    embedder = TFIDFEmbedder(dim=32)
    v1 = embedder.embed("hello world")
    v2 = embedder.embed("hello world")
    assert v1 == v2


def test_tfidf_embedder_dimension():
    embedder = TFIDFEmbedder(dim=16)
    assert len(embedder.embed("test one two")) == 16
    assert embedder.dimension == 16


def test_tfidf_embedder_normalized():
    embedder = TFIDFEmbedder(dim=32)
    vec = embedder.embed("test content here")
    norm = math.sqrt(sum(x * x for x in vec))
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_tfidf_embedder_empty_text():
    embedder = TFIDFEmbedder(dim=32)
    vec = embedder.embed("")
    assert vec == [0.0] * 32


def test_tfidf_embedder_different_texts_differ():
    embedder = TFIDFEmbedder(dim=64)
    v1 = embedder.embed("apple")
    v2 = embedder.embed("banana")
    assert v1 != v2


def test_tfidf_embedder_lexical_overlap_beats_unrelated():
    """Texts sharing words are more similar than texts that don't."""
    embedder = TFIDFEmbedder(dim=256)
    related_a = embedder.embed("the red apple is a sweet ripe fruit")
    related_b = embedder.embed("a sweet apple is my favorite fruit")
    unrelated = embedder.embed("database servers process concurrent queries")

    sim_related = cosine_similarity(related_a, related_b)
    sim_unrelated = cosine_similarity(related_a, unrelated)
    assert sim_related > sim_unrelated


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
    service = MemoryService(embedder=TFIDFEmbedder(dim=32))
    rec1 = service.write("the sky is blue")
    assert rec1 is not None
    # Same text should be deduped
    rec2 = service.write("the sky is blue")
    assert rec2 is None


def test_service_write_different_texts():
    service = MemoryService(embedder=TFIDFEmbedder(dim=32))
    r1 = service.write("apple")
    r2 = service.write("something completely different and unrelated")
    assert r1 is not None
    assert r2 is not None


def test_service_write_observations():
    service = MemoryService(embedder=TFIDFEmbedder(dim=32))
    resp = Response(text="Hello, I'm the assistant", model="test", usage=TokenUsage())
    records = service.write_observations(resp)
    assert len(records) == 1
    assert records[0].text == "Hello, I'm the assistant"


def test_service_write_observations_empty_text():
    service = MemoryService(embedder=TFIDFEmbedder(dim=32))
    resp = Response(text="", model="test", usage=TokenUsage())
    records = service.write_observations(resp)
    assert records == []


def test_service_remember_recall_forget():
    service = MemoryService(embedder=TFIDFEmbedder(dim=32))
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
# Memory service: sensitive filtering in assemble_context (item 5)
# ---------------------------------------------------------------------------

def test_assemble_context_excludes_sensitive():
    """Sensitive records are excluded from assembled context."""
    service = MemoryService(embedder=TFIDFEmbedder(dim=64))
    service.write("public preference dark mode", sensitive=False)
    service.write("secret api token value", sensitive=True)
    ctx = service.assemble_context("dark mode preference")
    texts = [r.text for r in ctx]
    assert "public preference dark mode" in texts
    assert all(not r.sensitive for r in ctx)
    assert "secret api token value" not in texts


def test_assemble_context_budget_filled_by_non_sensitive():
    """Non-sensitive records fill the budget even when sensitive ones exist.

    A sensitive record ranking above a non-sensitive one must not consume a
    context slot: filtering happens before ranking/budgeting.
    """
    service = MemoryService(embedder=TFIDFEmbedder(dim=128), context_budget=2)
    # Two non-sensitive records that should both make it into the budget.
    service.write("alpha topic one about apples", sensitive=False)
    service.write("alpha topic two about apples", sensitive=False)
    # A sensitive record with strong lexical overlap with the query.
    service.write("alpha topic secret about apples", sensitive=True)
    ctx = service.assemble_context("alpha topic apples")
    assert len(ctx) == 2
    assert all(not r.sensitive for r in ctx)


# ---------------------------------------------------------------------------
# Memory store: ChromaDB vector index detection + fallback (item 4)
# ---------------------------------------------------------------------------

def test_store_search_works_regardless_of_backend():
    """Search returns correct results whether ChromaDB or the fallback is used."""
    store = MemoryStore(vector_dim=32)
    # ChromaDB may or may not be installed; either way the store must remain
    # fully functional (indexed KNN or brute-force cosine fallback).
    rec = MemoryRecord(
        text="hello",
        embedding=[1.0] + [0.0] * 31,
        scope=MemoryScope.SESSION,
        type=MemoryType.FACT,
    )
    store.put(rec)
    results = store.search([1.0] + [0.0] * 31, limit=5)
    assert len(results) == 1
    assert results[0][0].id == rec.id
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)


def test_store_use_vector_index_disabled():
    """use_vector_index=False guarantees the brute-force path."""
    store = MemoryStore(vector_dim=32, use_vector_index=False)
    assert store.vector_search_enabled is False


def test_store_search_scope_filter_bruteforce():
    """Scope filter is applied before ranking on the fallback path."""
    store = MemoryStore(vector_dim=8, use_vector_index=False)
    vec = [1.0] + [0.0] * 7
    store.put(MemoryRecord(text="g", embedding=vec, scope=MemoryScope.GLOBAL, type=MemoryType.FACT))
    store.put(MemoryRecord(text="s", embedding=vec, scope=MemoryScope.SESSION, type=MemoryType.FACT))
    results = store.search(vec, scope_filter=[MemoryScope.GLOBAL], limit=5)
    assert len(results) == 1
    assert results[0][0].scope == MemoryScope.GLOBAL


@pytest.mark.skipif(
    not chroma_available(), reason="chromadb not installed"
)
def test_store_chroma_index_path():
    """When ChromaDB is installed, the indexed KNN path is used and correct."""
    store = MemoryStore(vector_dim=8, use_vector_index=True)
    assert store.vector_search_enabled is True
    near = [1.0] + [0.0] * 7
    far = [0.0] * 7 + [1.0]
    r_near = MemoryRecord(text="near", embedding=near, scope=MemoryScope.GLOBAL, type=MemoryType.FACT)
    r_far = MemoryRecord(text="far", embedding=far, scope=MemoryScope.GLOBAL, type=MemoryType.FACT)
    store.put(r_near)
    store.put(r_far)
    results = store.search(near, scope_filter=[MemoryScope.GLOBAL], limit=2)
    assert results[0][0].id == r_near.id
    # Scope filtering happens in the index (metadata filter).
    assert store.search(near, scope_filter=[MemoryScope.SESSION], limit=2) == []
    # Deleted records leave the index.
    store.delete(r_near.id)
    ids = [rec.id for rec, _ in store.search(near, limit=2)]
    assert r_near.id not in ids


def test_resource_tracker_warmup_prevents_fake_stderr_crash():
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root), *[p for p in sys.path if p]]
    )

    failure_script = textwrap.dedent(
        """
        from multiprocessing import shared_memory
        from unittest.mock import patch
        import traceback

        class FakeStderr:
            def write(self, *a, **k): pass
            def flush(self): pass
            def isatty(self): return True
            def fileno(self): return -1

        try:
            with patch('sys.stderr', FakeStderr()):
                shm = shared_memory.SharedMemory(create=True, size=1)
                print('unexpected-success', shm.name)
                shm.close()
                shm.unlink()
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        """
    )
    failure = subprocess.run(
        [sys.executable, "-c", failure_script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert failure.returncode == 1
    assert "bad value(s) in fds_to_keep" in failure.stderr
    assert "resource_tracker.py" in failure.stderr

    success_script = textwrap.dedent(
        """
        import traceback
        from multiprocessing import shared_memory
        from unittest.mock import patch
        from rickshaw.memory._chroma_index import _prestart_resource_tracker

        class FakeStderr:
            def write(self, *a, **k): pass
            def flush(self): pass
            def isatty(self): return True
            def fileno(self): return -1

        _prestart_resource_tracker()
        try:
            with patch('sys.stderr', FakeStderr()):
                shm = shared_memory.SharedMemory(create=True, size=1)
                print('ok', shm.name)
                shm.close()
                shm.unlink()
        except Exception:
            traceback.print_exc()
            raise SystemExit(1)
        """
    )
    success = subprocess.run(
        [sys.executable, "-c", success_script],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert success.returncode == 0, success.stderr
    assert success.stdout.startswith("ok ")


@pytest.mark.skipif(not chroma_available(), reason="chromadb not installed")
def test_chroma_index_initialization_warms_resource_tracker(monkeypatch):
    pytest.importorskip("chromadb")
    calls = []

    def fake_warmup() -> None:
        calls.append(True)

    monkeypatch.setattr("rickshaw.memory._chroma_index._prestart_resource_tracker", fake_warmup)
    store = MemoryStore(vector_dim=8, use_vector_index=True)
    assert store.vector_search_enabled is True
    assert calls == [True]


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
    service = MemoryService(embedder=TFIDFEmbedder(dim=32))
    tc = ToolCall(id="t1", name="remember", arguments={"fact": "test fact"})
    result = json.loads(dispatch_tool_call(tc, service))
    assert isinstance(result, str)
    assert result != "duplicate: already stored"


def test_dispatch_recall():
    service = MemoryService(embedder=TFIDFEmbedder(dim=32))
    service.write("apples are fruits")
    tc = ToolCall(id="t2", name="recall", arguments={"query": "fruit"})
    result = json.loads(dispatch_tool_call(tc, service))
    assert isinstance(result, list)


def test_dispatch_forget():
    service = MemoryService(embedder=TFIDFEmbedder(dim=32))
    rid = service.remember("temporary")
    tc = ToolCall(id="t3", name="forget", arguments={"id": rid})
    result = json.loads(dispatch_tool_call(tc, service))
    assert "deleted" in result


def test_dispatch_unknown_tool():
    service = MemoryService(embedder=TFIDFEmbedder(dim=32))
    tc = ToolCall(id="t4", name="unknown_tool", arguments={})
    result = json.loads(dispatch_tool_call(tc, service))
    # Registry surfaces errors as a structured {"error": ...} payload.
    assert isinstance(result, dict)
    assert "unknown tool" in result["error"]
