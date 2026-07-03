"""Tests for rickshaw/settings.py — persistent user settings."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from rickshaw.settings import (
    _CURRENT_VERSION,
    _DEFAULT_SETTINGS,
    _strip_secrets,
    load_settings,
    save_settings,
)


@pytest.fixture()
def settings_dir(tmp_path: Path) -> Path:
    """Return a fresh settings file path inside *tmp_path*."""
    return tmp_path / ".rickshaw" / "settings.json"


# ---- load / save round-trip ------------------------------------------------


def test_load_creates_default_on_missing(settings_dir: Path):
    data = load_settings(settings_dir)
    assert settings_dir.is_file()
    assert data["version"] == _CURRENT_VERSION
    assert data["provider"] == ""
    assert data["effort"] == "medium"


def test_save_and_load_round_trip(settings_dir: Path):
    original = dict(_DEFAULT_SETTINGS)
    original["provider"] = "anthropic"
    original["effort"] = "high"
    save_settings(original, settings_dir)

    loaded = load_settings(settings_dir)
    assert loaded["provider"] == "anthropic"
    assert loaded["effort"] == "high"


def test_save_overwrites_existing(settings_dir: Path):
    load_settings(settings_dir)  # seed
    data = load_settings(settings_dir)
    data["effort"] = "low"
    save_settings(data, settings_dir)

    reloaded = load_settings(settings_dir)
    assert reloaded["effort"] == "low"


# ---- default seeding -------------------------------------------------------


def test_default_seeding_has_all_keys(settings_dir: Path):
    data = load_settings(settings_dir)
    for key in _DEFAULT_SETTINGS:
        assert key in data


# ---- version migration -----------------------------------------------------


def test_migration_from_v0(settings_dir: Path):
    settings_dir.parent.mkdir(parents=True, exist_ok=True)
    settings_dir.write_text(json.dumps({"provider": "devin"}))

    data = load_settings(settings_dir)
    assert data["version"] == _CURRENT_VERSION
    assert data["provider"] == "devin"  # preserved
    assert data["effort"] == "medium"  # backfilled


# ---- API keys never written to disk ----------------------------------------


def test_api_keys_never_written(settings_dir: Path):
    data = dict(_DEFAULT_SETTINGS)
    data["providers"] = {
        "custom": {
            "base_url": "https://example.com",
            "model": "m",
            "api_key_env": "CUSTOM_KEY",
            "wire_format": "openai",
            "api_key": "sk-SHOULD-NOT-PERSIST",
            "secret": "also-bad",
        }
    }
    save_settings(data, settings_dir)

    raw = json.loads(settings_dir.read_text())
    custom = raw["providers"]["custom"]
    assert "api_key" not in custom
    assert "secret" not in custom
    assert custom["api_key_env"] == "CUSTOM_KEY"


def test_strip_secrets_removes_forbidden_keys():
    data = {
        "providers": {
            "p": {
                "api_key": "bad",
                "key": "bad",
                "token": "bad",
                "password": "bad",
                "api_key_env": "GOOD",
                "base_url": "https://x",
            }
        }
    }
    cleaned = _strip_secrets(data)
    p = cleaned["providers"]["p"]
    assert "api_key" not in p
    assert "key" not in p
    assert "token" not in p
    assert "password" not in p
    assert p["api_key_env"] == "GOOD"


# ---- precedence with env vars (via config) ---------------------------------


def test_env_var_overrides_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    settings_file = tmp_path / ".rickshaw" / "settings.json"
    data = dict(_DEFAULT_SETTINGS)
    data["provider"] = "anthropic"
    data["effort"] = "low"
    settings_file.parent.mkdir(parents=True)
    settings_file.write_text(json.dumps(data))

    monkeypatch.setenv("RICKSHAW_PROVIDER", "devin")
    monkeypatch.setenv("RICKSHAW_EFFORT", "high")

    # Patch default_settings_path so load_config picks up our file
    import rickshaw.settings as settings_mod

    monkeypatch.setattr(settings_mod, "default_settings_path", lambda: settings_file)

    from rickshaw.config import load_config

    cfg = load_config()
    assert cfg.provider == "devin"
    assert cfg.effort.value == "high"
