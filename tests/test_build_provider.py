"""Tests for the build_provider_from_profile helper."""

from __future__ import annotations

import os

import pytest

from rickshaw.config import ProviderProfile
from rickshaw.providers.build import build_provider_from_profile
from rickshaw.providers.openai_provider import OpenAIProvider


def test_openai_wire_format_builds_openai_provider(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MY_CUSTOM_KEY", "sk-test-key-123")
    profile = ProviderProfile(
        base_url="https://custom.example.com/v1",
        model="custom-model",
        api_key_env="MY_CUSTOM_KEY",
        wire_format="openai",
    )
    provider = build_provider_from_profile("deepseek", profile)
    assert isinstance(provider, OpenAIProvider)
    assert provider._api_key == "sk-test-key-123"
    assert provider._base_url == "https://custom.example.com/v1"
    assert provider._model == "custom-model"


def test_api_key_read_from_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEEPSEEK_KEY", "ds-key-456")
    profile = ProviderProfile(
        base_url="https://api.deepseek.com/v1",
        model="deepseek-chat",
        api_key_env="DEEPSEEK_KEY",
        wire_format="openai",
    )
    provider = build_provider_from_profile("deepseek", profile)
    assert provider._api_key == "ds-key-456"


def test_missing_env_var_gives_empty_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NONEXISTENT_KEY", raising=False)
    profile = ProviderProfile(
        base_url="https://example.com/v1",
        model="m",
        api_key_env="NONEXISTENT_KEY",
        wire_format="openai",
    )
    provider = build_provider_from_profile("test", profile)
    assert provider._api_key == ""


def test_anthropic_wire_format(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ANTHRO_KEY", "ak-test")
    profile = ProviderProfile(
        base_url="https://api.anthropic.com",
        model="claude-3-5-sonnet-latest",
        api_key_env="ANTHRO_KEY",
        wire_format="anthropic",
    )
    provider = build_provider_from_profile("anthropic", profile)
    assert provider.name == "anthropic"
    assert provider._api_key == "ak-test"


def test_devin_wire_format(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("DEVIN_KEY", "dk-test")
    profile = ProviderProfile(
        base_url="https://api.devin.ai",
        model="devin",
        api_key_env="DEVIN_KEY",
        wire_format="devin",
    )
    provider = build_provider_from_profile("devin", profile)
    assert provider.name == "devin"
    assert provider._api_key == "dk-test"


def test_embedding_model_passed_through(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OAI_KEY", "sk-x")
    profile = ProviderProfile(
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        api_key_env="OAI_KEY",
        wire_format="openai",
    )
    provider = build_provider_from_profile("openai", profile, embedding_model="ada-002")
    assert provider._embedding_model == "ada-002"
