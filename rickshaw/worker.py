"""Deferred worker — runs off the hot path, batched.

Only component besides the orchestrator allowed to call the provider,
and only in background: importance scoring, compaction/reflection, and
local-only eviction.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from rickshaw.memory.record import MemoryRecord, MemoryScope
from rickshaw.memory.service import MemoryService
from rickshaw.providers.base import Effort, LLMProvider, Message
from rickshaw.queue import Job, JobQueue, JobStatus, JobType

logger = logging.getLogger(__name__)

_SESSION_TTL_HOURS = 24.0


class DeferredWorker:
    """Process deferred jobs from the queue.

    Uses the same injected LLMProvider as the orchestrator (for LLM-gated
    tasks like importance scoring and compaction/reflection). Local-only
    eviction requires no provider.

    FUTURE: run in a background thread/process with its own event loop, support
    a separate (cheaper) provider for background tasks, and a priority queue so
    foreground turns take precedence. See FUTURE.md ("Deferred Worker").
    """

    def __init__(
        self,
        queue: JobQueue,
        memory: MemoryService,
        provider: LLMProvider | None = None,
    ) -> None:
        self.queue = queue
        self.memory = memory
        self.provider = provider

    def process_batch(self, max_jobs: int = 10) -> int:
        """Process up to *max_jobs* from the queue. Returns count processed."""
        processed = 0
        for _ in range(max_jobs):
            job = self.queue.dequeue()
            if job is None:
                break
            job.status = JobStatus.RUNNING
            try:
                self._handle(job)
                self.queue.mark_completed(job)
            except Exception as exc:
                self.queue.mark_failed(job, str(exc))
                logger.warning("Job %s failed: %s", job.id, exc)
            processed += 1
        return processed

    def _handle(self, job: Job) -> None:
        if job.type == JobType.IMPORTANCE_SCORING:
            self._score_importance(job)
        elif job.type == JobType.COMPACTION:
            self._compact(job)
        elif job.type == JobType.EVICTION:
            self._evict(job)

    def _score_importance(self, job: Job) -> None:
        """LLM-gated importance scoring for a record."""
        record_id = job.payload.get("record_id", "")
        record = self.memory.store.get(record_id)
        if record is None:
            return

        if self.provider is None:
            # No provider — assign a heuristic score
            record.importance = min(0.5, record.importance + 0.1)
            self.memory.store.update(record)
            return

        prompt = (
            f"Rate the importance of this memory on a scale of 0.0 to 1.0. "
            f"Reply with ONLY a number.\n\n"
            f"Memory: {record.text}"
        )
        try:
            response = self.provider.complete(
                [Message(role="user", content=prompt)],
                effort=Effort.LOW,
            )
            score = float(response.text.strip())
            record.importance = min(max(score, 0.0), 1.0)
        except (ValueError, TypeError):
            record.importance = 0.5
        except Exception:
            record.importance = 0.5
        self.memory.store.update(record)

    def _compact(self, job: Job) -> None:
        """Summarize many raw records into distilled ones, mark originals superseded."""
        record_ids: list[str] = job.payload.get("record_ids", [])
        records = [
            self.memory.store.get(rid)
            for rid in record_ids
        ]
        records = [r for r in records if r is not None]
        if len(records) < 2:
            return

        combined_text = "\n".join(r.text for r in records)

        if self.provider is not None:
            prompt = (
                f"Summarize these related memories into a single concise statement:\n\n"
                f"{combined_text}"
            )
            try:
                response = self.provider.complete(
                    [Message(role="user", content=prompt)],
                    effort=Effort.LOW,
                )
                summary = response.text.strip()
            except Exception:
                summary = combined_text[:200]
        else:
            summary = combined_text[:200]

        new_record = self.memory.write(
            text=summary,
            scope=records[0].scope,
            type=records[0].type,
            importance=max(r.importance for r in records),
        )

        if new_record is not None:
            for r in records:
                self.memory.store.mark_superseded(r.id, new_record.id)

    def _evict(self, job: Job) -> None:
        """Local-only eviction: TTL/decay for session-scoped records and supersession."""
        now = datetime.now(timezone.utc)
        session_records = self.memory.store.all_records(
            scope_filter=[MemoryScope.SESSION, MemoryScope.TASK]
        )
        for record in session_records:
            # TTL eviction
            age_hours = (now - record.created_at).total_seconds() / 3600
            if age_hours > _SESSION_TTL_HOURS:
                self.memory.store.delete(record.id)
                continue
            # Supersession cleanup
            if record.superseded_by is not None:
                self.memory.store.delete(record.id)
