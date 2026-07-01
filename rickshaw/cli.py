"""Interactive CLI entrypoint for Rickshaw."""

from __future__ import annotations

import argparse
import sys

from rickshaw.config import RickshawConfig, load_config
from rickshaw.providers.base import Effort, LLMProvider, Message
from rickshaw.providers.factory import get_provider

_EFFORT_NAMES = {e.value: e for e in Effort}


def _build_provider(name: str, cfg: RickshawConfig) -> LLMProvider:
    """Instantiate the requested provider with config values."""
    if name == "openai":
        return get_provider(
            name,
            api_key=cfg.openai_api_key,
            base_url=cfg.openai_base_url,
            model=cfg.openai_model,
            embedding_model=cfg.openai_embedding_model,
        )
    if name == "devin":
        return get_provider(
            name,
            api_key=cfg.devin_api_key,
            base_url=cfg.devin_base_url,
        )
    if name == "anthropic":
        return get_provider(
            name,
            api_key=cfg.anthropic_api_key,
            base_url=cfg.anthropic_base_url,
            model=cfg.anthropic_model,
        )
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


def _print_header(provider: LLMProvider, effort: Effort) -> None:
    # Imported lazily to avoid a circular import (tui imports from cli) and to
    # keep the branding strings in one place.
    from rickshaw.tui import RICKSHAW_LOGO, RICKSHAW_SLOGAN

    caps = provider.capabilities()
    effort_support = caps.effort_levels
    print(f"{RICKSHAW_LOGO} \u00b7 {RICKSHAW_SLOGAN}")
    print(f"Rickshaw  provider={provider.name}  effort={effort.value}")
    if effort_support:
        levels = ", ".join(e.value for e in effort_support)
        print(f"  Supported effort levels: {levels}")
    else:
        print("  (provider does not advertise effort level support)")
    print('Type "/effort <level>" to change effort mid-session.')
    print('Type "/quit" or Ctrl-D to exit.\n')


def _run_repl(provider: LLMProvider, effort: Effort) -> None:
    """Simple read-eval-print loop."""
    messages: list[Message] = []
    current_effort = effort
    caps = provider.capabilities()

    while True:
        try:
            user_input = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            print("Goodbye.")
            break

        # Per-turn effort override
        if user_input.lower().startswith("/effort "):
            level_str = user_input.split(maxsplit=1)[1].strip().lower()
            if level_str in _EFFORT_NAMES:
                current_effort = _EFFORT_NAMES[level_str]
                if caps.effort_levels and current_effort not in caps.effort_levels:
                    print(
                        f"  Warning: {provider.name} does not honor "
                        f"effort={current_effort.value}; it may be ignored."
                    )
                print(f"  Effort set to {current_effort.value} for subsequent turns.")
            else:
                print(f"  Invalid effort level {level_str!r}. Use: low, medium, high.")
            continue

        messages.append(Message(role="user", content=user_input))

        # Warn once if the chosen effort isn't supported
        if caps.effort_levels and current_effort not in caps.effort_levels:
            print(
                f"  Warning: {provider.name} does not honor "
                f"effort={current_effort.value}."
            )

        try:
            response = provider.complete(messages, effort=current_effort)
        except Exception as exc:
            print(f"  Error: {exc}")
            continue

        messages.append(Message(role="assistant", content=response.text))

        print(f"\n[effort: {response.effort.value}]  ({response.model})")
        print(response.text)
        if response.usage.total_tokens:
            print(
                f"  tokens: {response.usage.prompt_tokens} prompt"
                f" + {response.usage.completion_tokens} completion"
                f" = {response.usage.total_tokens} total"
            )
        print()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config()

    provider_name = args.provider or cfg.provider
    effort = _EFFORT_NAMES.get(args.effort, cfg.effort) if args.effort else cfg.effort

    provider = _build_provider(provider_name, cfg)

    # Validate connectivity
    try:
        provider.validate()
    except Exception as exc:
        print(f"Provider validation failed ({provider_name}): {exc}", file=sys.stderr)
        if args.validate_only:
            sys.exit(1)
        print("Continuing anyway; calls may fail.\n", file=sys.stderr)

    if args.validate_only:
        print(f"Provider {provider_name!r} validated successfully.")
        return

    _print_header(provider, effort)
    _run_repl(provider, effort)


if __name__ == "__main__":
    main()
