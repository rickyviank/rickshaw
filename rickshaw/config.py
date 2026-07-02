"""Configuration loading from environment variables and optional config file.

Resolution order (later wins):
  1. YAML config file (``config.yaml``)
  2. ``~/.rickshaw/settings.json`` (user-editable persistent settings)
  3. Real environment variables

API keys are **always** read from the environment (via each profile's
``api_key_env``), never from the settings file.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

from rickshaw.providers.base import Effort

_EFFORT_NAMES = {e.value: e for e in Effort}


@dataclass
class ProviderProfile:
    """Descriptor for a single LLM provider endpoint.

    ``api_key_env`` is the *name* of the environment variable holding the API
    key — the key itself is never stored on disk.
    """

    base_url: str
    model: str
    api_key_env: str
    wire_format: str = "openai"


def _parse_effort(raw: str) -> Effort:
    key = raw.strip().lower()
    if key not in _EFFORT_NAMES:
        valid = ", ".join(sorted(_EFFORT_NAMES))
        raise ValueError(
            f"Invalid effort level {raw!r}. Valid values: {valid}"
        )
    return _EFFORT_NAMES[key]


@dataclass
class RickshawConfig:
    """Resolved configuration for a Rickshaw session."""

    provider: str = "openai"
    effort: Effort = Effort.MEDIUM

    # OpenAI-specific
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

    # Devin-specific
    devin_api_key: str = ""
    devin_base_url: str = "https://api.devin.ai"

    # Anthropic-specific
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com"
    anthropic_model: str = "claude-3-5-sonnet-latest"

    # Separate embedding provider (may differ from the chat provider)
    embedding_provider: str = ""

    # Provider profiles (name → ProviderProfile)
    providers: dict[str, ProviderProfile] = field(default_factory=dict)

    # Path to the user settings file (populated by load_config)
    settings_path: str | None = None

    # Extra overrides loaded from the config file
    extra: dict[str, str] = field(default_factory=dict)


_BUILTIN_PROFILES: dict[str, ProviderProfile] = {
    "openai": ProviderProfile(
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        wire_format="openai",
    ),
    "devin": ProviderProfile(
        base_url="https://api.devin.ai",
        model="devin",
        api_key_env="DEVIN_API_KEY",
        wire_format="devin",
    ),
    "anthropic": ProviderProfile(
        base_url="https://api.anthropic.com",
        model="claude-3-5-sonnet-latest",
        api_key_env="ANTHROPIC_API_KEY",
        wire_format="anthropic",
    ),
}


def load_config(
    config_path: str | Path | None = None,
    dotenv_path: str | Path | None = None,
) -> RickshawConfig:
    """Build a :class:`RickshawConfig` from env vars and an optional YAML file.

    Resolution order (later wins):
      1. YAML config file (``config.yaml``)
      2. ``~/.rickshaw/settings.json`` (user-editable persistent settings)
      3. Real environment variables

    API keys are always read from the environment (via each profile's
    ``api_key_env``), never from the settings file.
    """
    # 1. Load .env
    if dotenv_path:
        load_dotenv(dotenv_path)
    else:
        load_dotenv()  # searches CWD and parents

    # 2. Load optional YAML config
    file_values: dict[str, str] = {}
    if config_path is None:
        for candidate in ("config.yaml", "config.yml"):
            if Path(candidate).is_file():
                config_path = candidate
                break
    if config_path and Path(config_path).is_file():
        with open(config_path) as fh:
            raw = yaml.safe_load(fh) or {}
        if isinstance(raw, dict):
            file_values = {str(k).upper(): str(v) for k, v in raw.items()}

    # 3. Load settings.json (lazy import to avoid circular dependency)
    from rickshaw.settings import default_settings_path, load_settings

    settings_path = str(default_settings_path())
    settings = load_settings()

    def _get(key: str, default: str = "") -> str:
        """Return value: env (highest) → settings → yaml → default."""
        env_val = os.environ.get(key)
        if env_val is not None:
            return env_val
        return file_values.get(key, default)

    # Merge provider profiles: builtins → settings.json → overrides
    providers: dict[str, ProviderProfile] = dict(_BUILTIN_PROFILES)
    for pname, pdata in settings.get("providers", {}).items():
        providers[pname] = ProviderProfile(
            base_url=pdata.get("base_url", ""),
            model=pdata.get("model", ""),
            api_key_env=pdata.get("api_key_env", ""),
            wire_format=pdata.get("wire_format", "openai"),
        )

    # Settings-level provider/effort/embedding overrides (below env)
    settings_provider = settings.get("provider", "")
    settings_effort = settings.get("effort", "")
    settings_emb_provider = settings.get("embedding_provider", "")
    settings_emb_model = settings.get("embedding_model", "")

    provider_name = _get(
        "RICKSHAW_PROVIDER",
        settings_provider or file_values.get("RICKSHAW_PROVIDER", "openai"),
    )
    effort_raw = _get(
        "RICKSHAW_EFFORT",
        settings_effort or file_values.get("RICKSHAW_EFFORT", "medium"),
    )
    embedding_provider = _get(
        "RICKSHAW_EMBEDDING_PROVIDER",
        settings_emb_provider or file_values.get("RICKSHAW_EMBEDDING_PROVIDER", ""),
    )

    # Resolve per-provider env overrides for the built-in providers
    openai_base = _get("OPENAI_BASE_URL", providers["openai"].base_url)
    openai_model = _get("OPENAI_MODEL", providers["openai"].model)
    openai_emb = _get("OPENAI_EMBEDDING_MODEL", settings_emb_model or "text-embedding-3-small")
    devin_base = _get("DEVIN_BASE_URL", providers["devin"].base_url)
    anthropic_base = _get("ANTHROPIC_BASE_URL", providers["anthropic"].base_url)
    anthropic_model = _get("ANTHROPIC_MODEL", providers["anthropic"].model)

    # Keep profiles in sync with resolved values
    providers["openai"] = ProviderProfile(
        base_url=openai_base, model=openai_model,
        api_key_env="OPENAI_API_KEY", wire_format="openai",
    )
    providers["devin"] = ProviderProfile(
        base_url=devin_base, model="devin",
        api_key_env="DEVIN_API_KEY", wire_format="devin",
    )
    providers["anthropic"] = ProviderProfile(
        base_url=anthropic_base, model=anthropic_model,
        api_key_env="ANTHROPIC_API_KEY", wire_format="anthropic",
    )

    cfg = RickshawConfig(
        provider=provider_name,
        effort=_parse_effort(effort_raw),
        openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
        openai_base_url=openai_base,
        openai_model=openai_model,
        openai_embedding_model=openai_emb,
        devin_api_key=os.environ.get("DEVIN_API_KEY", ""),
        devin_base_url=devin_base,
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        anthropic_base_url=anthropic_base,
        anthropic_model=anthropic_model,
        embedding_provider=embedding_provider,
        providers=providers,
        settings_path=settings_path,
        extra={
            k: v
            for k, v in file_values.items()
            if not k.startswith(("RICKSHAW_", "OPENAI_", "DEVIN_", "ANTHROPIC_"))
        },
    )
    return cfg
