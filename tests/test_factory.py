"""Tests for the provider factory."""

import pytest

from rickshaw.providers.factory import get_provider, list_providers


def test_get_openai_provider():
    provider = get_provider("openai", api_key="test-key")
    assert provider.name == "openai"


def test_get_devin_provider():
    provider = get_provider("devin", api_key="test-key")
    assert provider.name == "devin"


def test_case_insensitive():
    provider = get_provider("OpenAI", api_key="test-key")
    assert provider.name == "openai"


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        get_provider("nonexistent")


def test_list_providers():
    names = list_providers()
    assert "openai" in names
    assert "devin" in names
