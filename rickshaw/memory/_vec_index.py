"""sqlite-vec-backed vector index for indexed KNN search inside SQLite.

``sqlite-vec`` (https://github.com/asg0171/sqlite-vec) provides a ``vec0``
virtual table module that performs KNN search directly inside SQLite, removing
the need for a separate vector DB (e.g. ChromaDB) and the dual-write sync
problem it introduced.

This module is an *optional accelerator*. When the ``sqlite_vec`` extension can
be loaded into the connection, :class:`~rickshaw.memory.store.MemoryStore` uses
a ``vec0`` shadow table for KNN. When it cannot be loaded (the extension is
absent, or the Python ``sqlite3`` module was built without
``enable_load_extension`` support — common on macOS system Python and some
locked-down distros), the store transparently falls back to an exact
brute-force cosine scan over float32 BLOBs. Both paths return identical
results; sqlite-vec only buys scale.
"""

from __future__ import annotations

import logging
import struct
import sqlite3
from pathlib import Path

from rickshaw.memory.record import MemoryRecord, MemoryScope

logger = logging.getLogger(__name__)

# sqlite-vec's vec0 virtual table stores vectors as raw little-endian float32
# blobs. One 4-byte word per dimension.


def _load_vec_extension(conn: sqlite3.Connection) -> bool:
    """Try to load the sqlite-vec extension into *conn*.

    Returns True on success. On any failure (module missing, extension loading
    disabled in this sqlite build, etc.) returns False and leaves the
    connection untouched. Never raises.
    """
    try:
        import sqlite_vec

        if not hasattr(conn, "enable_load_extension"):
            logger.info(
                "sqlite-vec: connection lacks enable_load_extension "
                "(sqlite3 built without loadable-extension support); "
                "using brute-force cosine fallback."
            )
            return False
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        # Sanity check the module is actually registered.
        try:
            conn.execute("SELECT vec_version()").fetchone()
        except sqlite3.OperationalError:
            return False
        return True
    except Exception as exc:  # ImportError, init failure, permissions, etc.
        logger.info(
            "sqlite-vec unavailable (%s); using brute-force cosine fallback "
            "for vector search.",
            exc,
        )
        return False


def vec_available(conn: sqlite3.Connection) -> bool:
    """Return True if sqlite-vec can be loaded on *conn* right now."""
    return _load_vec_extension(conn)


def vec_to_blob(vec: list[float] | bytes) -> bytes:
    """Pack a float vector into a compact little-endian float32 BLOB."""
    if isinstance(vec, (bytes, bytearray)):
        return bytes(vec)
    return struct.pack(f"{len(vec)}f", *[float(x) for x in vec])


def blob_to_vec(blob: bytes) -> list[float]:
    """Unpack a float32 BLOB back into a list of floats."""
    if not blob:
        return []
    n = len(blob) // 4
    return list(struct.unpack(f"{n}f", blob))


class Vec0Index:
    """Shadow ``vec0`` virtual table holding record embeddings for KNN.

    The base ``memories`` table remains the source of truth for full records;
    this index only mirrors (id, embedding, scope) so KNN + scope filtering
    happen inside SQLite. Superseded records are removed from the index so they
    never surface in search results.
    """

    def __init__(self, conn: sqlite3.Connection, dimension: int) -> None:
        self._conn = conn
        self._dimension = dimension
        self._table = "memories_vec"
        conn.execute(
            f"""CREATE VIRTUAL TABLE IF NOT EXISTS {self._table} USING vec0(
                id TEXT PRIMARY KEY,
                embedding FLOAT[{dimension}] NOT NULL,
                scope TEXT NOT NULL,
                +active INTEGER NOT NULL DEFAULT 1
            )"""
        )
        conn.commit()

    @property
    def dimension(self) -> int:
        return self._dimension

    def upsert(self, record: MemoryRecord) -> None:
        """Insert/replace a record's embedding. Superseded records are removed."""
        if record.superseded_by is not None:
            self.delete(record.id)
            return
        active = 1
        self._conn.execute(
            f"""INSERT OR REPLACE INTO {self._table}
               (id, embedding, scope, active)
               VALUES (?, ?, ?, ?)""",
            (record.id, vec_to_blob(record.embedding), record.scope.value, active),
        )
        self._conn.commit()

    def delete(self, record_id: str) -> None:
        self._conn.execute(f"DELETE FROM {self._table} WHERE id = ?", (record_id,))
        self._conn.commit()

    def query(
        self,
        query_vec: list[float],
        scope_filter: list[MemoryScope] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Return ``(record_id, cosine_similarity)`` for the top-*limit* matches.

        Scope filtering is applied via a WHERE clause before KNN. sqlite-vec
        reports cosine *distance*; we convert to similarity (``1 - distance``).
        """
        # Pull a few extra to allow post-filtering of superseded/inactive rows.
        k = max(limit, 1)
        where = "active = 1"
        params: list = [vec_to_blob(query_vec), k]
        if scope_filter:
            scopes = [s.value for s in scope_filter]
            placeholders = ",".join("?" for _ in scopes)
            where += f" AND scope IN ({placeholders})"
            params = [vec_to_blob(query_vec), k, *scopes]
        rows = self._conn.execute(
            f"""SELECT id, distance
                FROM {self._table}
                WHERE embedding MATCH ? AND k = ? AND {where}
                ORDER BY distance""",
            params,
        ).fetchall()
        return [(r[0], 1.0 - float(r[1])) for r in rows]

    def count(self) -> int:
        row = self._conn.execute(f"SELECT COUNT(*) FROM {self._table}").fetchone()
        return int(row[0]) if row else 0
