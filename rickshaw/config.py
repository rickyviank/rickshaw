"""Configuration loading from environment variables and optional config file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

from rickshaw.providers.base import Effort

_EFFORT_NAMES = {e.value: e for e in Effort}


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

    # Extra overrides loaded from the config file
    extra: dict[str, str] = field(default_factory=dict)


def load_config(
    config_path: str | Path | None = None,
    dotenv_path: str | Path | None = None,
) -> RickshawConfig:
    """Build a :class:`RickshawConfig` from env vars and an optional YAML file.

    Resolution order (later wins):
      1. ``.env`` file (loaded via *python-dotenv*)
      2. YAML config file (``config.yaml``)
      3. Real environment variables
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

    def _get(key: str, default: str = "") -> str:
        """Return the value from env (highest priority) or file or default."""
        return os.environ.get(key, file_values.get(key, default))

    effort_raw = _get("RICKSHAW_EFFORT", "medium")

    return RickshawConfig(
        provider=_get("RICKSHAW_PROVIDER", "openai"),
        effort=_parse_effort(effort_raw),
        openai_api_key=_get("OPENAI_API_KEY"),
        openai_base_url=_get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=_get("OPENAI_MODEL", "gpt-4o"),
        openai_embedding_model=_get(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        ),
        devin_api_key=_get("DEVIN_API_KEY"),
        devin_base_url=_get("DEVIN_BASE_URL", "https://api.devin.ai"),
        anthropic_api_key=_get("ANTHROPIC_API_KEY"),
        anthropic_base_url=_get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        anthropic_model=_get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
        embedding_provider=_get("RICKSHAW_EMBEDDING_PROVIDER"),
        extra={
            k: v
            for k, v in file_values.items()
            if not k.startswith(("RICKSHAW_", "OPENAI_", "DEVIN_", "ANTHROPIC_"))
        },
    )
