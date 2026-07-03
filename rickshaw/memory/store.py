"""SQLite-backed persistence for MemoryRecords.

Phase 1 substrate upgrade: embeddings are now stored as compact float32 BLOBs
(used by both the indexed and fallback paths), and KNN search happens inside
SQLite via the ``vec0`` virtual table from ``sqlite-vec`` when the extension can
be loaded. When it cannot (the extension is absent, or this Python's ``sqlite3``
was built without ``enable_load_extension`` support — common on macOS system
Python and some locked-down distros), search falls back to an exact brute-force
cosine scan over the same float32 BLOBs. Both paths return identical results;
sqlite-vec only buys scale.

This replaces the previous SQLite + ChromaDB dual-write design: SQLite is now
the *sole* store, eliminating the cross-store sync problem. Existing databases
with JSON-encoded ``embedding`` columns are migrated to float32 BLOBs on open.

The ``ChromaVectorIndex`` module is still imported lazily for backward
compatibility with callers that explicitly construct it, but :class:`MemoryStore`
no longer uses it by default. The ``use_vector_index`` flag now controls the
``vec0`` index.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import struct
from datetime import datetime
from pathlib import Path

from rickshaw.memory._math import cosine_similarity
from rickshaw.memory._vec_index import (
    blob_to_vec,
    vec_available,
    vec_to_blob,
    Vec0Index,
)
from rickshaw.memory.record import MemoryRecord, MemoryScope, MemoryType

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    scope TEXT NOT NULL,
    type TEXT NOT NULL,
    importance REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL,
    last_used_at TEXT NOT NULL,
    use_count INTEGER NOT NULL DEFAULT 0,
    sensitive INTEGER NOT NULL DEFAULT 0,
    superseded_by TEXT,
    extra TEXT NOT NULL DEFAULT '{}'
);
"""


def _dt_to_iso(dt: datetime) -> str:
    return dt.isoformat()


def _iso_to_dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


class MemoryStore:
    """SQLite-backed store for MemoryRecords.

    Vectors are stored as float32 BLOBs in the ``embedding`` column. KNN search
    uses a ``vec0`` shadow table when sqlite-vec is available, otherwise an
    exact brute-force cosine scan over the same BLOBs. The two paths produce
    identical rankings; sqlite-vec only matters at scale.
    """

    def __init__(
        self,
        db_path: str | Path = ":memory:",
        vector_dim: int | None = None,
        use_vector_index: bool = True,
    ) -> None:
        # check_same_thread=False lets a serialized caller (e.g. the TUI, which
        # runs each turn in an exclusive worker thread) reuse the connection
        # across threads. Access is expected to be serialized by the caller.
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(_SCHEMA)
        self._conn.commit()
        self._migrate()
        self._vector_dim = vector_dim

        # Phase 1 unified vector index: sqlite-vec vec0 inside SQLite. Falls
        # back to brute-force cosine when the extension can't load.
        self._index: Vec0Index | None = None
        if use_vector_index and vector_dim:
            if vec_available(self._conn):
                try:
                    self._index = Vec0Index(self._conn, vector_dim)
                    self._backfill_index()
                except Exception as exc:
                    logger.info(
                        "vec0 index init failed (%s); brute-force fallback.",
                        exc,
                    )
                    self._index = None
            # If sqlite-vec is unavailable we simply proceed without an index;
            # search() transparently uses the brute-force path.

    def _migrate(self) -> None:
        """Add columns introduced after the initial schema and migrate the
        embedding column from JSON text to float32 BLOB."""
        cols = {
            row[1]
            for row in self._conn.execute("PRAGMA table_info(memories)").fetchall()
        }
        if "extra" not in cols:
            self._conn.execute(
                "ALTER TABLE memories ADD COLUMN extra TEXT NOT NULL DEFAULT '{}'"
            )
            self._conn.commit()

        # Migrate legacy JSON-encoded embeddings to float32 BLOBs.
        # Detect JSON text by trying to parse the first row; BLOBs are bytes.
        sample = self._conn.execute("SELECT embedding FROM memories LIMIT 1").fetchone()
        if sample is not None:
            emb = sample["embedding"]
            if isinstance(emb, str) and emb.startswith("["):
                logger.info(
                    "Migrating %d JSON-encoded embeddings to float32 BLOBs.",
                    self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0],
                )
                for row in self._conn.execute(
                    "SELECT id, embedding FROM memories"
                ).fetchall():
                    try:
                        vec = json.loads(row["embedding"])
                        self._conn.execute(
                            "UPDATE memories SET embedding = ? WHERE id = ?",
                            (vec_to_blob(vec), row["id"]),
                        )
                    except (json.JSONDecodeError, struct.error):
                        # Leave unparseable rows as-is; they'll be skipped in
                        # search rather than crashing the migration.
                        continue
                self._conn.commit()

    def _backfill_index(self) -> None:
        """Populate the vec0 shadow table from any rows present at open time."""
        if self._index is None:
            return
        existing = self._index.count()
        base = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        if existing >= base:
            return
        rows = self._conn.execute(
            "SELECT id, embedding, scope, superseded_by FROM memories"
        ).fetchall()
        for row in rows:
            if row["superseded_by"] is not None:
                continue
            try:
                blob = row["embedding"]
                if isinstance(blob, str):
                    blob = vec_to_blob(json.loads(blob))
                self._conn.execute(
                    f"INSERT OR REPLACE INTO {self._index._table} "
                    "(id, embedding, scope, active) VALUES (?, ?, ?, 1)",
                    (row["id"], blob, row["scope"]),
                )
            except Exception:
                continue
        self._conn.commit()

    @property
    def vector_search_enabled(self) -> bool:
        """Whether the vec0-backed vector index is active."""
        return self._index is not None

    def close(self) -> None:
        self._conn.close()

    def put(self, record: MemoryRecord) -> None:
        # embedding is stored as a float32 BLOB (compact, used by both the
        # indexed and brute-force paths); the vec0 shadow table mirrors it for
        # indexed KNN search.
        self._conn.execute(
            """INSERT OR REPLACE INTO memories
               (id, text, embedding, scope, type, importance,
                created_at, last_used_at, use_count, sensitive, superseded_by,
                extra)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.text,
                vec_to_blob(record.embedding),
                record.scope.value,
                record.type.value,
                record.importance,
                _dt_to_iso(record.created_at),
                _dt_to_iso(record.last_used_at),
                record.use_count,
                int(record.sensitive),
                record.superseded_by,
                json.dumps(record.extra),
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

        The metadata scope filter is applied FIRST. When the vec0 index is
        active the ranked KNN search runs inside sqlite-vec (scope-filtered
        via a WHERE clause); otherwise we fall back to an exact brute-force
        cosine scan over the scope-filtered candidate rows. Both paths return
        identical results.
        """
        if self._index is not None:
            try:
                return self._search_vector_index(query_vec, scope_filter, limit)
            except Exception as exc:
                logger.warning(
                    "vec0 search failed (%s); falling back to brute-force cosine.",
                    exc,
                )
        return self._search_bruteforce(query_vec, scope_filter, limit)

    def _search_vector_index(
        self,
        query_vec: list[float],
        scope_filter: list[MemoryScope] | None,
        limit: int,
    ) -> list[tuple[MemoryRecord, float]]:
        """KNN search via vec0; full records are re-hydrated from SQLite."""
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
        blob = row["embedding"]
        if isinstance(blob, (bytes, bytearray)):
            embedding = blob_to_vec(blob)
        else:  # legacy JSON text row that survived migration
            embedding = json.loads(blob) if blob else []
        return MemoryRecord(
            id=row["id"],
            text=row["text"],
            embedding=embedding,
            scope=MemoryScope(row["scope"]),
            type=MemoryType(row["type"]),
            importance=row["importance"],
            created_at=_iso_to_dt(row["created_at"]),
            last_used_at=_iso_to_dt(row["last_used_at"]),
            use_count=row["use_count"],
            sensitive=bool(row["sensitive"]),
            superseded_by=row["superseded_by"],
            extra=json.loads(row["extra"]) if row["extra"] else {},
        )
