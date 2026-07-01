"""Tests for the deferred worker and job queue."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from rickshaw.memory.embedder import LocalEmbedder
from rickshaw.memory.record import MemoryRecord, MemoryScope
from rickshaw.memory.service import MemoryService
from rickshaw.providers.base import (
    Capabilities,
    Effort,
    LLMProvider,
    Message,
    Response,
    TokenUsage,
    ToolSpec,
)
from rickshaw.queue import Job, JobQueue, JobStatus, JobType
from rickshaw.worker import DeferredWorker


class _StubProvider(LLMProvider):
    """Minimal provider for worker tests."""

    @property
    def name(self) -> str:
        return "stub"

    def complete(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Response:
        return Response(text="0.75", model="stub", effort=effort)

    def stream(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield "0.75"

    def available_models(self) -> list[str]:
        return ["stub"]

    def validate(self) -> None:
        pass

    def capabilities(self) -> Capabilities:
        return Capabilities()


def test_queue_enqueue_dequeue():
    q = JobQueue()
    j = Job(type=JobType.IMPORTANCE_SCORING)
    q.enqueue(j)
    assert q.pending_count == 1
    popped = q.dequeue()
    assert popped is j
    assert q.pending_count == 0


def test_queue_dequeue_empty():
    q = JobQueue()
    assert q.dequeue() is None


def test_worker_importance_scoring_with_provider():
    memory = MemoryService(embedder=LocalEmbedder(dim=32))
    rec = memory.write("important fact")
    assert rec is not None
    queue = JobQueue()
    queue.enqueue(Job(type=JobType.IMPORTANCE_SCORING, payload={"record_id": rec.id}))

    worker = DeferredWorker(queue=queue, memory=memory, provider=_StubProvider())
    processed = worker.process_batch()

    assert processed == 1
    updated = memory.store.get(rec.id)
    assert updated is not None
    assert updated.importance == 0.75


def test_worker_importance_scoring_no_provider():
    memory = MemoryService(embedder=LocalEmbedder(dim=32))
    rec = memory.write("some fact")
    assert rec is not None
    queue = JobQueue()
    queue.enqueue(Job(type=JobType.IMPORTANCE_SCORING, payload={"record_id": rec.id}))

    worker = DeferredWorker(queue=queue, memory=memory, provider=None)
    processed = worker.process_batch()

    assert processed == 1
    updated = memory.store.get(rec.id)
    assert updated is not None
    assert updated.importance > 0


def test_worker_eviction_ttl():
    memory = MemoryService(embedder=LocalEmbedder(dim=32))
    old_time = datetime.now(timezone.utc) - timedelta(hours=48)
    rec = MemoryRecord(
        id="old_rec",
        text="expired session record",
        embedding=LocalEmbedder(dim=32).embed("x"),
        scope=MemoryScope.SESSION,
        created_at=old_time,
        last_used_at=old_time,
    )
    memory.store.put(rec)

    queue = JobQueue()
    queue.enqueue(Job(type=JobType.EVICTION))
    worker = DeferredWorker(queue=queue, memory=memory)
    worker.process_batch()

    assert memory.store.get("old_rec") is None


def test_worker_eviction_superseded():
    memory = MemoryService(embedder=LocalEmbedder(dim=32))
    rec = MemoryRecord(
        id="superseded_rec",
        text="old record",
        embedding=LocalEmbedder(dim=32).embed("x"),
        scope=MemoryScope.SESSION,
        superseded_by="new_rec",
    )
    memory.store.put(rec)

    queue = JobQueue()
    queue.enqueue(Job(type=JobType.EVICTION))
    worker = DeferredWorker(queue=queue, memory=memory)
    worker.process_batch()

    assert memory.store.get("superseded_rec") is None


def test_worker_compaction():
    memory = MemoryService(embedder=LocalEmbedder(dim=32))
    r1 = memory.write("fact A")
    r2 = memory.write("fact B about something else")
    assert r1 is not None and r2 is not None

    queue = JobQueue()
    queue.enqueue(Job(
        type=JobType.COMPACTION,
        payload={"record_ids": [r1.id, r2.id]},
    ))
    worker = DeferredWorker(queue=queue, memory=memory, provider=_StubProvider())
    worker.process_batch()

    # Originals should be superseded
    assert memory.store.get(r1.id) is not None
    assert memory.store.get(r1.id).superseded_by is not None


def test_worker_handles_missing_record():
    memory = MemoryService(embedder=LocalEmbedder(dim=32))
    queue = JobQueue()
    queue.enqueue(Job(
        type=JobType.IMPORTANCE_SCORING,
        payload={"record_id": "nonexistent"},
    ))
    worker = DeferredWorker(queue=queue, memory=memory)
    processed = worker.process_batch()
    assert processed == 1
    assert queue.pending_count == 0
