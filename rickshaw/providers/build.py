"""Build a provider instance from a :class:`ProviderProfile`."""

from __future__ import annotations

import os

from rickshaw.config import ProviderProfile
from rickshaw.providers.base import LLMProvider
from rickshaw.providers.factory import get_provider


def build_provider_from_profile(
    name: str,
    profile: ProviderProfile,
    embedding_model: str = "",
) -> LLMProvider:
    """Instantiate a provider from a :class:`ProviderProfile`.

    For ``wire_format == "openai"`` the :class:`OpenAIProvider` is used,
    allowing any OpenAI-compatible endpoint (e.g. DeepSeek, Ollama).

    The API key is always resolved from the environment variable named by
    ``profile.api_key_env`` -- never from disk.
    """
    api_key = os.environ.get(profile.api_key_env, "")

    if profile.wire_format == "openai":
        return get_provider(
            "openai",
            api_key=api_key,
            base_url=profile.base_url,
            model=profile.model,
            embedding_model=embedding_model or None,
        )

    if profile.wire_format == "anthropic":
        return get_provider(
            "anthropic",
            api_key=api_key,
            base_url=profile.base_url,
            model=profile.model,
        )

    if profile.wire_format == "devin":
        return get_provider(
            "devin",
            api_key=api_key,
            base_url=profile.base_url,
        )

    # Fallback: try the factory by name (may work for user-registered providers)
    return get_provider(name, api_key=api_key)
