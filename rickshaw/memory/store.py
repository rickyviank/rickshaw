"""SQLite-backed persistence for MemoryRecords."""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from rickshaw.memory._chroma_index import ChromaVectorIndex
from rickshaw.memory._math import cosine_similarity
from rickshaw.memory.record import MemoryRecord, MemoryScope, MemoryType

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding TEXT NOT NULL,
    scope TEXT NOT NULL,
    type TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 0,
    sensitive INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT
);
"""


def _dt_to_iso(dt: datetime) -> str:
    return dt.isoformat()


def _iso_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class MemoryStore:
    """SQLite-backed store for MemoryRecords."""

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        vector_dim: int | None = None,
        use_vector_index: bool = True,
    ) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()

        # Optional indexed vector search via ChromaDB. SQLite stays the source
        # of truth; the index mirrors embeddings for KNN. Falls back to a
        # brute-force cosine scan when ChromaDB is unavailable (see
        # _chroma_index).
        self._vector_dim = vector_dim
        self._index: ChromaVectorIndex | None = None
        if use_vector_index and vector_dim:
            index = ChromaVectorIndex(db_path, vector_dim)
            if index.enabled:
                self._index = index

    @property
    def vector_search_enabled(self) -> bool:
        """Whether the ChromaDB-backed vector index is active."""
        return self._index is not None

    def close(self) -> None:
        self._conn.close()

    def put(self, record: MemoryRecord) -> None:
        # embedding is stored as JSON text for reconstruction and the
        # brute-force fallback; the ChromaDB index (when active) additionally
        # mirrors the raw vector for indexed KNN search.
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, text, embedding, scope, type, importance,
                created_at, last_used_at, use_count, sensitive, superseded_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.text,
                json.dumps(record.embedding),
                record.scope.value,
                record.type.value,
                record.importance,
                _dt_to_iso(record.created_at),
                _dt_to_iso(record.last_used_at),
                record.use_count,
                int(record.sensitive),
                record.superseded_by,
            ),
        )
        self._conn.commit()
        if self._index is not None:
            self._index.upsert(record)

    def get(self, record_id: str) -> MemoryRecord | None:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def search(
        self,
        query_vec: list[float],
        scope_filter: list[MemoryScope] | None = None,
        limit: int = 20,
    ) -> list[tuple[MemoryRecord, float]]:
        """Return records ranked by similarity, with optional scope filter.

        The metadata scope filter is applied FIRST. When the ChromaDB index is
        active the ranked KNN search runs inside Chroma (scope-filtered via
        metadata); otherwise we fall back to a brute-force cosine scan over the
        scope-filtered candidate rows.
        """
        if self._index is not None:
            try:
                return self._search_vector_index(query_vec, scope_filter, limit)
            except Exception as exc:
                logger.warning(
                    "ChromaDB search failed (%s); "
                    "falling back to brute-force cosine scan.",
                    exc,
                )
        return self._search_bruteforce(query_vec, scope_filter, limit)

    def _search_vector_index(
        self,
        query_vec: list[float],
        scope_filter: list[MemoryScope] | None,
        limit: int,
    ) -> list[tuple[MemoryRecord, float]]:
        """KNN search via ChromaDB; full records are re-hydrated from SQLite."""
        hits = self._index.query(query_vec, scope_filter, limit)
        scored: list[tuple[MemoryRecord, float]] = []
        for record_id, sim in hits:
            record = self.get(record_id)
            if record is None or record.superseded_by is not None:
                continue
            scored.append((record, sim))
        return scored

    def _search_bruteforce(
        self,
        query_vec: list[float],
        scope_filter: list[MemoryScope] | None,
        limit: int,
    ) -> list[tuple[MemoryRecord, float]]:
        """Scope-filter in SQL, then brute-force cosine over candidate rows."""
        if scope_filter:
            placeholders = ",".join("?" for _ in scope_filter)
            query = (
                f"SELECT * FROM memories WHERE scope IN ({placeholders}) "
                "AND superseded_by IS NULL"
            )
            rows = self._conn.execute(
                query, [s.value for s in scope_filter]
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE superseded_by IS NULL"
            ).fetchall()

        scored: list[tuple[MemoryRecord, float]] = []
        for row in rows:
            record = self._row_to_record(row)
            sim = cosine_similarity(query_vec, record.embedding)
            scored.append((record, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def update(self, record: MemoryRecord) -> None:
        self.put(record)

    def mark_superseded(self, record_id: str, superseded_by: str) -> None:
        self._conn.execute(
            "UPDATE memories SET superseded_by = ? WHERE id = ?",
            (superseded_by, record_id),
        )
        self._conn.commit()
        # Superseded records must not surface in search — drop from the index.
        if self._index is not None:
            self._index.delete(record_id)

    def all_records(
        self, scope_filter: list[MemoryScope] | None = None,
    ) -> list[MemoryRecord]:
        if scope_filter:
            placeholders = ",".join("?" for _ in scope_filter)
            rows = self._conn.execute(
                f"SELECT * FROM memories WHERE scope IN ({placeholders})",
                [s.value for s in scope_filter],
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM memories").fetchall()
        return [self._row_to_record(r) for r in rows]

    def delete(self, record_id: str) -> bool:
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE id = ?", (record_id,)
        )
        self._conn.commit()
        if self._index is not None:
            self._index.delete(record_id)
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            text=row["text"],
            embedding=json.loads(row["embedding"]),
            scope=MemoryScope(row["scope"]),
            type=MemoryType(row["type"]),
            importance=row["importance"],
            created_at=_iso_to_dt(row["created_at"]),
            last_used_at=_iso_to_dt(row["last_used_at"]),
            use_count=row["use_count"],
            sensitive=bool(row["sensitive"]),
            superseded_by=row["superseded_by"],
        )
