"""Interactive CLI entrypoint for Rickshaw."""

from __future__ import annotations

import argparse

from rickshaw.config import RickshawConfig, load_config
from rickshaw.providers.base import Effort, LLMProvider
from rickshaw.providers.build import build_provider_from_profile
from rickshaw.providers.factory import get_provider

_EFFORT_NAMES = {e.value: e for e in Effort}


def _build_provider(name: str, cfg: RickshawConfig) -> LLMProvider:
    """Instantiate the requested provider from its profile in *cfg*."""
    profile = cfg.providers.get(name)
    if profile is not None:
        return build_provider_from_profile(
            name, profile, embedding_model=cfg.openai_embedding_model,
        )
    # Fallback for providers not in the profile map
    return get_provider(name)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rickshaw",
        description="Multi-LLM provider harness with effort levels",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider name (e.g. openai, devin, anthropic). Overrides config/env.",
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Default reasoning effort level for the session.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate provider connectivity and exit.",
    )
    return parser.parse_args(argv)


# _print_header and _run_repl have been removed; the plain REPL is replaced
# by the Textual TUI (rickshaw.tui).  The symbols below are preserved so
# existing imports from rickshaw.cli keep working.
