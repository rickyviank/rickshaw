"""Persistent user settings stored in ``~/.rickshaw/settings.json``.

The file is created with sensible defaults on first access.  Writes use an
atomic temp-file + :func:`os.replace` pattern so a crash mid-write never
corrupts the file.

**Security invariant**: API keys are NEVER written to disk.  Only the *name*
of the environment variable holding the key (``api_key_env``) is persisted.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

_CURRENT_VERSION = 1

_DEFAULT_SETTINGS: dict = {
    "version": _CURRENT_VERSION,
    "provider": "openai",
    "effort": "medium",
    "embedding_provider": "openai",
    "embedding_model": "text-embedding-3-small",
    "providers": {},
}

# Keys that may never appear inside a persisted provider block.
_FORBIDDEN_KEYS = {"api_key", "key", "secret", "token", "password"}


def default_settings_path() -> Path:
    """Return ``~/.rickshaw/settings.json``."""
    return Path.home() / ".rickshaw" / "settings.json"


def _ensure_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _seed_defaults(path: Path) -> dict:
    """Write the default settings file and return its contents."""
    _ensure_dir(path)
    data = dict(_DEFAULT_SETTINGS)
    _atomic_write(path, data)
    return data


def _atomic_write(path: Path, data: dict) -> None:
    """Write *data* as JSON via a temp file + :func:`os.replace`."""
    _ensure_dir(path)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=".settings-", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
            fh.write("\n")
        os.replace(tmp, str(path))
    except BaseException:
        # Clean up the temp file on any error.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _strip_secrets(data: dict) -> dict:
    """Remove any accidentally-included secret values before writing."""
    providers = data.get("providers", {})
    for _name, pdata in providers.items():
        if isinstance(pdata, dict):
            for forbidden in _FORBIDDEN_KEYS:
                pdata.pop(forbidden, None)
    return data


def _migrate(data: dict) -> dict:
    """Apply version-keyed migrations and return the (possibly updated) dict.

    Currently only version 1 exists; future versions add migration steps here.
    """
    version = data.get("version", 0)
    if version < 1:
        # Pre-v1 files lack the version key; backfill defaults.
        for key, default in _DEFAULT_SETTINGS.items():
            data.setdefault(key, default)
        data["version"] = _CURRENT_VERSION
    return data


def load_settings(path: Path | None = None) -> dict:
    """Load settings from *path* (default ``~/.rickshaw/settings.json``).

    If the file does not exist it is seeded with defaults.
    """
    path = path or default_settings_path()
    if not path.is_file():
        return _seed_defaults(path)
    with open(path) as fh:
        data = json.load(fh)
    data = _migrate(data)
    return data


def save_settings(data: dict, path: Path | None = None) -> None:
    """Persist *data* to *path* atomically.

    Strips any accidentally-included secret values before writing.
    """
    path = path or default_settings_path()
    cleaned = _strip_secrets(dict(data))  # shallow copy top-level
    _atomic_write(path, cleaned)
