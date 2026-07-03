"""Credential storage.

The contract is intentionally tiny. ``modify`` is the ONLY write path — a
serialized read-modify-write — so concurrent requests and processes cannot
double-refresh a rotated OAuth token. OAuth refresh runs *inside* the ``fn``
passed to ``modify`` (see :mod:`rickshaw_ai.auth.resolver`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Awaitable, Callable, Protocol, Union, runtime_checkable

from pydantic import TypeAdapter

logger = logging.getLogger(__name__)

from rickshaw_ai.credentials.types import (
    ApiKeyCredential,
    Credential,
    OAuthCredential,
)

ModifyFn = Callable[
    [Union[Credential, None]],
    Union[Credential, None, Awaitable[Union[Credential, None]]],
]

_ADAPTER: TypeAdapter[Credential] = TypeAdapter(Credential)


@runtime_checkable
class CredentialStore(Protocol):
    """Persistence for one type-tagged credential per provider."""

    async def read(self, provider_id: str) -> Credential | None: ...

    async def modify(self, provider_id: str, fn: ModifyFn) -> Credential | None:
        """Serialized read-modify-write.

        Acquire the provider's lock, read the current value, call ``fn(current)``
        (awaiting if it returns an awaitable), persist the result (``None`` means
        delete), release the lock, and return the new value.
        """
        ...

    async def delete(self, provider_id: str) -> None: ...


async def _call_fn(fn: ModifyFn, current: Credential | None) -> Credential | None:
    result = fn(current)
    if asyncio.iscoroutine(result):
        return await result
    return result  # type: ignore[return-value]


class InMemoryCredentialStore:
    """Default store: a process-local dict with a per-provider lock."""

    def __init__(self, initial: dict[str, Credential] | None = None) -> None:
        self._data: dict[str, Credential] = dict(initial or {})
        self._locks: dict[str, asyncio.Lock] = {}

    def _lock(self, provider_id: str) -> asyncio.Lock:
        lock = self._locks.get(provider_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[provider_id] = lock
        return lock

    async def read(self, provider_id: str) -> Credential | None:
        return self._data.get(provider_id)

    async def modify(self, provider_id: str, fn: ModifyFn) -> Credential | None:
        async with self._lock(provider_id):
            current = self._data.get(provider_id)
            new = await _call_fn(fn, current)
            if new is None:
                self._data.pop(provider_id, None)
            else:
                self._data[provider_id] = new
            return new

    async def delete(self, provider_id: str) -> None:
        async with self._lock(provider_id):
            self._data.pop(provider_id, None)

    # test/introspection helper
    def set(self, provider_id: str, credential: Credential) -> None:
        self._data[provider_id] = credential


class FileCredentialStore:
    """A JSON-file-backed store using a cross-process file lock around writes.

    Shape mirrors pi's ``auth.json``::

        { "<provider_id>": { "type": "api_key" | "oauth", ... }, ... }

    The lock (an adjacent ``<path>.lock`` file, ``fcntl``-based on POSIX)
    serializes ``modify`` across processes so a rotated OAuth token is never
    double-refreshed.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._async_locks: dict[str, asyncio.Lock] = {}

    def _async_lock(self, provider_id: str) -> asyncio.Lock:
        lock = self._async_locks.get(provider_id)
        if lock is None:
            lock = asyncio.Lock()
            self._async_locks[provider_id] = lock
        return lock

    def _load(self) -> dict[str, Credential]:
        if not self.path.is_file():
            return {}
        try:
            raw = json.loads(self.path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Credential file %s is corrupt or unreadable, "
                "starting with empty credentials: %s",
                self.path, exc,
            )
            return {}
        out: dict[str, Credential] = {}
        for pid, data in (raw or {}).items():
            try:
                out[pid] = _ADAPTER.validate_python(data)
            except Exception as exc:
                logger.warning(
                    "Skipping invalid credential for provider %r: %s",
                    pid, exc,
                )
                continue
        return out

    def _dump(self, data: dict[str, Credential]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            pid: json.loads(_ADAPTER.dump_json(cred)) for pid, cred in data.items()
        }
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as fh:
                json.dump(serializable, fh, indent=2)
            os.replace(tmp, str(self.path))
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    async def read(self, provider_id: str) -> Credential | None:
        return self._load().get(provider_id)

    async def modify(self, provider_id: str, fn: ModifyFn) -> Credential | None:
        async with self._async_lock(provider_id):
            loop = asyncio.get_running_loop()
            return await asyncio.to_thread(self._modify_locked, provider_id, fn, loop)

    def _modify_locked(
        self,
        provider_id: str,
        fn: ModifyFn,
        loop: asyncio.AbstractEventLoop,
    ) -> Credential | None:
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with open(lock_path, "w") as lock_file:
            _flock(lock_file)
            try:
                data = self._load()
                current = data.get(provider_id)
                future = asyncio.run_coroutine_threadsafe(
                    _call_fn(fn, current), loop,
                )
                new = future.result()
                if new is None:
                    data.pop(provider_id, None)
                else:
                    data[provider_id] = new
                self._dump(data)
                return new
            finally:
                _funlock(lock_file)

    async def delete(self, provider_id: str) -> None:
        async def _drop(_: Credential | None) -> None:
            return None

        await self.modify(provider_id, _drop)


_FLOCK_WARNED = False


def _flock(fh) -> None:  # pragma: no cover - platform dependent
    global _FLOCK_WARNED
    try:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
    except ImportError:
        if not _FLOCK_WARNED:
            logger.warning(
                "fcntl unavailable; credential file locking disabled. "
                "Concurrent writes may corrupt the credential store."
            )
            _FLOCK_WARNED = True
    except OSError as exc:
        logger.warning("Failed to acquire credential file lock: %s", exc)


def _funlock(fh) -> None:  # pragma: no cover - platform dependent
    try:
        import fcntl

        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except (ImportError, OSError):
        pass


__all__ = [
    "CredentialStore",
    "InMemoryCredentialStore",
    "FileCredentialStore",
    "ApiKeyCredential",
    "OAuthCredential",
]
