"""Optional ChromaDB-backed vector index for indexed similarity search.

ChromaDB (https://www.trychroma.com/) is an embedded vector database with ANN
search and metadata filtering. It is an *optional* dependency: when it is not
installed (or fails to initialize), :class:`~rickshaw.memory.store.MemoryStore`
transparently falls back to a brute-force cosine scan.

SQLite remains the source of truth for full records; this index only mirrors
each record's embedding plus the minimal metadata needed to serve scope-filtered
KNN queries (the ``scope``). Superseded records are removed from the index so
they never surface in search results.

Install with::

    pip install chromadb
"""

from __future__ import annotations

import logging
import uuid
from contextlib import suppress
from pathlib import Path
from multiprocessing import resource_tracker, shared_memory

from rickshaw.memory.record import MemoryRecord, MemoryScope

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "memories"


def _prestart_resource_tracker() -> None:
    """Start multiprocessing's resource tracker while stdio still has real fds."""
    try:
        tracker = resource_tracker._resource_tracker
        if getattr(tracker, "_pid", None) is not None:
            return
    except Exception:
        return

    # Textual workers can replace stderr with a fake stream whose fileno() is
    # -1. If the resource tracker is first spawned there, fork_exec rejects the
    # invalid fd list; warming it up here keeps later turn-time multiprocessing
    # use from respawning under redirected stdio.
    with suppress(Exception):
        shm = shared_memory.SharedMemory(create=True, size=1)
        try:
            shm.close()
        finally:
            shm.unlink()


def chroma_available() -> bool:
    """Return True if the ``chromadb`` package can be imported."""
    try:
        import chromadb  # noqa: F401
    except Exception:
        return False
    return True


class ChromaVectorIndex:
    """Thin wrapper over a ChromaDB collection holding record embeddings.

    Uses an in-memory (ephemeral) client for ``:memory:`` stores and a
    persistent client rooted at a sibling directory of the SQLite file
    otherwise, so the vector index shares the store's lifetime/location.
    """

    def __init__(self, db_path: str | Path, dimension: int | None) -> None:
        self._enabled = False
        self._collection = None
        self._dimension = dimension
        try:
            import chromadb
            from chromadb.config import Settings

            settings = Settings(anonymized_telemetry=False, allow_reset=True)
            if str(db_path) == ":memory:":
                # EphemeralClient shares process-global state, so use a unique
                # collection name to keep in-memory stores isolated.
                client = chromadb.EphemeralClient(settings=settings)
                name = f"{_COLLECTION_NAME}_{uuid.uuid4().hex}"
            else:
                p = Path(db_path)
                chroma_dir = str(p.parent / f"{p.name}.chroma")
                client = chromadb.PersistentClient(path=chroma_dir, settings=settings)
                # Stable name so the index is reused across restarts.
                name = _COLLECTION_NAME
            # We supply our own embeddings; cosine space matches our similarity.
            self._collection = client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
            _prestart_resource_tracker()
            self._enabled = True
        except Exception as exc:  # ImportError, init failure, etc.
            logger.warning(
                "ChromaDB unavailable (%s); using brute-force cosine fallback "
                "for vector search.",
                exc,
            )
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def upsert(self, record: MemoryRecord) -> None:
        """Insert/replace a record's embedding. Superseded records are removed."""
        if record.superseded_by is not None:
            self.delete(record.id)
            return
        self._collection.upsert(
            ids=[record.id],
            embeddings=[list(record.embedding)],
            metadatas=[{"scope": record.scope.value}],
        )

    def delete(self, record_id: str) -> None:
        self._collection.delete(ids=[record_id])

    def query(
        self,
        query_vec: list[float],
        scope_filter: list[MemoryScope] | None,
        limit: int,
    ) -> list[tuple[str, float]]:
        """Return ``(record_id, similarity)`` for the top-*limit* matches.

        The metadata ``scope`` filter is applied by Chroma before ranking.
        Cosine *distance* is converted to similarity (``1 - distance``).
        """
        if self._collection.count() == 0:
            return []
        where = None
        if scope_filter:
            where = {"scope": {"$in": [s.value for s in scope_filter]}}
        res = self._collection.query(
            query_embeddings=[list(query_vec)],
            n_results=max(1, limit),
            where=where,
        )
        ids = (res.get("ids") or [[]])[0]
        distances = (res.get("distances") or [[]])[0]
        return [(rid, 1.0 - float(dist)) for rid, dist in zip(ids, distances)]
