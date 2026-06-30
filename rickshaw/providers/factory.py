"""Provider registry and factory."""

from __future__ import annotations

from typing import Any

from rickshaw.providers.base import LLMProvider

_REGISTRY: dict[str, type[LLMProvider]] = {}


def _ensure_builtins() -> None:
    """Lazily register the built-in providers on first access."""
    if _REGISTRY:
        return

    from rickshaw.providers.devin_provider import DevinProvider
    from rickshaw.providers.openai_provider import OpenAIProvider

    register("openai", OpenAIProvider)
    register("devin", DevinProvider)


def register(name: str, cls: type[LLMProvider]) -> None:
    """Add a provider class to the registry.

    This is the extension point for third-party providers::

        from rickshaw.providers.factory import register
        register("my_llm", MyLLMProvider)
    """
    _REGISTRY[name.lower()] = cls


def get_provider(name: str, **config: Any) -> LLMProvider:
    """Instantiate a provider by its registered name.

    Raises :class:`ValueError` if *name* is not registered.
    """
    _ensure_builtins()
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown provider {name!r}. Available providers: {available}"
        )
    return _REGISTRY[key](**config)


def list_providers() -> list[str]:
    """Return sorted list of registered provider names."""
    _ensure_builtins()
    return sorted(_REGISTRY)
