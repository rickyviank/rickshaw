"""Tests for the memory subsystem — store, ranker, dedupe, embedder, tools."""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from rickshaw.memory._chroma_index import chroma_available
from rickshaw.memory._math import cosine_similarity
from rickshaw.memory.embedder import (
    Model2VecEmbedder,
    ProviderEmbedder,
    TFIDFEmbedder,
    make_embedder,
)
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


# ---------------------------------------------------------------------------
# Phase 1: unified sqlite-vec store (float32 BLOB + vec0 + brute-force fallback)
# ---------------------------------------------------------------------------

def test_store_embedding_stored_as_float32_blob():
    """Embeddings are persisted as compact float32 BLOBs, not JSON text."""
    store = MemoryStore(vector_dim=8, use_vector_index=False)
    rec = MemoryRecord(id="r1", text="hello", embedding=[1.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
    store.put(rec)
    raw = store._conn.execute("SELECT embedding FROM memories WHERE id='r1'").fetchone()[0]
    assert isinstance(raw, (bytes, bytearray))
    assert len(raw) == 8 * 4  # 8 dims * 4 bytes per float32
    # Round-trips exactly (within float32 precision).
    fetched = store.get("r1")
    assert fetched.embedding[0] == pytest.approx(1.0)
    assert fetched.embedding[2] == pytest.approx(0.5)


def test_store_search_identical_results_indexed_vs_bruteforce():
    """The vec0 indexed path and brute-force path return identical rankings."""
    vec = [1.0] + [0.0] * 15
    recs = [
        MemoryRecord(id="near", text="near", embedding=vec),
        MemoryRecord(id="far", text="far", embedding=[0.0] * 15 + [1.0]),
    ]
    # Force brute-force.
    brute = MemoryStore(vector_dim=16, use_vector_index=False)
    for r in recs:
        brute.put(r)
    brute_hits = brute.search(vec, limit=2)
    # Use the indexed path if available; otherwise this just re-confirms brute.
    indexed = MemoryStore(vector_dim=16, use_vector_index=True)
    for r in recs:
        indexed.put(r)
    indexed_hits = indexed.search(vec, limit=2)
    # Both must agree on the top hit regardless of backend.
    assert brute_hits[0][0].id == "near"
    assert indexed_hits[0][0].id == "near"
    assert brute_hits[0][1] == pytest.approx(1.0, abs=1e-5)
    assert indexed_hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_store_migrates_json_embeddings_to_blob(tmp_path):
    """A legacy DB with JSON-text embeddings is migrated to float32 BLOBs on open."""
    import sqlite3 as _sqlite3
    db = tmp_path / "legacy.db"
    c = _sqlite3.connect(str(db))
    c.execute(
        """CREATE TABLE memories (
            id TEXT PRIMARY KEY, text TEXT NOT NULL, embedding TEXT NOT NULL,
            scope TEXT NOT NULL, type TEXT NOT NULL, importance REAL NOT NULL DEFAULT 0.0,
            created_at TEXT NOT NULL, last_used_at TEXT NOT NULL,
            use_count INTEGER NOT NULL DEFAULT 0, sensitive INTEGER NOT NULL DEFAULT 0,
            superseded_by TEXT)"""
    )
    c.execute(
        "INSERT INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        ("r1", "hello", json.dumps([1.0, 0.5, 0.0, 0.0]), "session", "fact", 0.5,
         "2025-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00", 0, 0, None),
    )
    c.commit()
    c.close()

    store = MemoryStore(str(db), vector_dim=None, use_vector_index=False)
    rec = store.get("r1")
    assert rec is not None
    assert rec.embedding == [1.0, 0.5, 0.0, 0.0]
    raw = store._conn.execute("SELECT embedding FROM memories WHERE id='r1'").fetchone()[0]
    assert isinstance(raw, (bytes, bytearray))  # migrated to BLOB
    assert len(raw) == 16  # 4 floats * 4 bytes


def test_store_vec0_index_backfills_existing_rows(tmp_path):
    """Rows present before the store opens are backfilled into the vec0 index."""
    db = tmp_path / "mem.db"
    vec = [1.0] + [0.0] * 15
    # First store writes a row (no index), then a second store opens with an
    # index and must pick up the pre-existing row.
    base = MemoryStore(str(db), vector_dim=16, use_vector_index=False)
    base.put(MemoryRecord(id="r1", text="hello", embedding=vec))
    base.close()
    # Re-open with index enabled (will use vec0 if available, else fallback).
    store = MemoryStore(str(db), vector_dim=16, use_vector_index=True)
    hits = store.search(vec, limit=5)
    assert any(r.id == "r1" for r, _ in hits)


def test_store_superseded_removed_from_index():
    """mark_superseded drops the record from the index so it can't surface."""
    store = MemoryStore(vector_dim=8, use_vector_index=True)
    vec = [1.0] + [0.0] * 7
    store.put(MemoryRecord(id="old", text="old", embedding=vec))
    store.put(MemoryRecord(id="new", text="new", embedding=vec))
    store.mark_superseded("old", "new")
    hits = store.search(vec, limit=5)
    ids = [r.id for r, _ in hits]
    assert "old" not in ids
    assert "new" in ids


# ---------------------------------------------------------------------------
# Phase 2: tiered embedders (Model2Vec default, FastEmbed opt-in, make_embedder)
# ---------------------------------------------------------------------------

def test_model2vec_embedder_loads_vendored_model():
    """The default Model2VecEmbedder loads the vendored model fully offline."""
    e = Model2VecEmbedder()
    assert e.dimension == 128
    v = e.embed("hello world")
    assert len(v) == 128
    assert any(abs(x) > 0 for x in v)  # not all zeros


def test_model2vec_embedder_semantic_beats_tfidf():
    """Model2Vec captures semantic similarity that TF-IDF cannot."""
    m2v = Model2VecEmbedder()
    # Synonyms with no word overlap should be more similar than unrelated text.
    sim_syn = cosine_similarity(
        m2v.embed("the feline is sleeping on the mat"),
        m2v.embed("a cat is resting on the rug"),
    )
    sim_unrel = cosine_similarity(
        m2v.embed("the feline is sleeping on the mat"),
        m2v.embed("database servers process concurrent queries"),
    )
    assert sim_syn > sim_unrel


def test_model2vec_embedder_deterministic():
    """Same text produces the same vector across calls."""
    e = Model2VecEmbedder()
    v1 = e.embed("deterministic test")
    v2 = e.embed("deterministic test")
    assert v1 == pytest.approx(v2, abs=1e-6)


def test_make_embedder_tiers():
    """make_embedder selects the correct tier by name."""
    assert isinstance(make_embedder("tfidf", dim=16), TFIDFEmbedder)
    assert isinstance(make_embedder("model2vec"), Model2VecEmbedder)
    # Unknown tier raises.
    with pytest.raises(ValueError):
        make_embedder("nonexistent")


def test_make_embedder_default_is_model2vec():
    """The default tier (no arg) is Model2Vec, not TF-IDF."""
    e = make_embedder()
    assert isinstance(e, Model2VecEmbedder)


def test_service_default_embedder_is_model2vec():
    """MemoryService defaults to Model2VecEmbedder when no embedder is passed."""
    service = MemoryService()
    assert isinstance(service.embedder, Model2VecEmbedder)


def test_service_reembeds_on_tier_change(tmp_path):
    """Switching embedder tier re-embeds all existing records."""
    db = str(tmp_path / "mem.db")
    # Write with TF-IDF (32-dim).
    svc_tfidf = MemoryService(
        embedder=TFIDFEmbedder(dim=32), db_path=db,
    )
    svc_tfidf.write("user likes dark mode")
    svc_tfidf.write("python is a great language")
    recs = svc_tfidf.store.all_records()
    assert all(len(r.embedding) == 32 for r in recs)
    svc_tfidf.store.close()

    # Re-open with Model2Vec (128-dim) — records must be re-embedded.
    svc_m2v = MemoryService(
        embedder=Model2VecEmbedder(), db_path=db,
    )
    recs = svc_m2v.store.all_records()
    assert all(len(r.embedding) == 128 for r in recs)
    # Retrieval should still work after re-embedding.
    results = svc_m2v.assemble_context("dark mode preference")
    assert len(results) > 0
