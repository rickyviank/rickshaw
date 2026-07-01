"""Rich-based terminal UI for Rickshaw.

A lightweight alternative to the plain-text REPL in :mod:`rickshaw.cli`. This
is *not* a full-screen app framework (no Textual, no fixed panes): it is a
normal ``input()``/``print()`` loop with :mod:`rich` layered on top purely for
formatting. Every user turn is routed through :meth:`Orchestrator.run_turn`, so
the semantic memory layer is active and degradation info is surfaced.

Install the optional extra to use it::

    pip install -e ".[tui]"

then launch::

    rickshaw-tui --provider openai --effort high
"""

from __future__ import annotations

import argparse
import sys

# Reuse provider/config wiring from the CLI so behavior stays consistent.
from rickshaw.cli import _EFFORT_NAMES, _build_provider, load_config
from rickshaw.orchestrator import Orchestrator, TurnResult
from rickshaw.memory.service import MemoryService
from rickshaw.providers.base import Effort, LLMProvider

# Branding — kept module-level so cli.py can import and reuse them.
RICKSHAW_LOGO = "o--o  rickshaw"
RICKSHAW_SLOGAN = "your driver, your memory"
RICKSHAW_BANNER = f"{RICKSHAW_LOGO} \u00b7 {RICKSHAW_SLOGAN}"

# Where the memory layer persists across sessions (vs. the default ":memory:").
_DEFAULT_DB_PATH = "rickshaw_memory.db"

_RICH_MISSING_MSG = (
    "The Rich terminal UI requires the 'rich' package, which is not installed.\n"
    "Install the optional extra with:\n\n"
    '    pip install "rickshaw[tui]"\n'
)


def _import_rich():
    """Import Rich lazily, raising a clear, actionable error if it's absent."""
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel
    except ImportError as exc:  # pragma: no cover - exercised via message text
        raise SystemExit(_RICH_MISSING_MSG) from exc
    return Console, Markdown, Panel


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rickshaw-tui",
        description="Rich terminal UI for the Rickshaw provider harness.",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider name (e.g. openai, devin). Overrides config/env.",
    )
    parser.add_argument(
        "--effort",
        choices=["low", "medium", "high"],
        default=None,
        help="Default reasoning effort level for the session.",
    )
    parser.add_argument(
        "--db-path",
        default=_DEFAULT_DB_PATH,
        help=(
            "SQLite path for the persistent memory layer "
            f"(default: {_DEFAULT_DB_PATH})."
        ),
    )
    return parser.parse_args(argv)


def _render_header(console, Panel, provider: LLMProvider, effort: Effort) -> None:
    """Render the branded startup panel with provider/effort info."""
    caps = provider.capabilities()
    lines = [f"[bold]{RICKSHAW_BANNER}[/bold]", ""]
    lines.append(f"provider: [cyan]{provider.name}[/cyan]    effort: [cyan]{effort.value}[/cyan]")
    if caps.effort_levels:
        levels = ", ".join(e.value for e in caps.effort_levels)
        lines.append(f"supported effort levels: {levels}")
    else:
        lines.append("(provider does not advertise effort level support)")
    lines.append("")
    lines.append('[dim]Type "/effort <level>" to change effort, "/quit" or Ctrl-D to exit.[/dim]')
    console.print(Panel("\n".join(lines), border_style="cyan", expand=False))


def _render_status(console, result: TurnResult) -> None:
    """Print a dim status line summarizing the turn's side info."""
    parts = [f"tool calls: {result.tool_calls_made}"]
    if result.degraded:
        parts.append("[yellow]degraded (local memory)[/yellow]")
    for warning in result.warnings:
        parts.append(f"[yellow]{warning}[/yellow]")
    console.print(f"[dim]{'  |  '.join(parts)}[/dim]")


def _run_tui(orchestrator: Orchestrator, provider: LLMProvider, effort: Effort) -> None:
    """Rich-formatted, line-oriented loop routing turns through the Orchestrator."""
    Console, Markdown, Panel = _import_rich()
    console = Console()

    orchestrator.effort = effort
    _render_header(console, Panel, provider, effort)
    caps = provider.capabilities()

    while True:
        try:
            user_input = console.input("[bold green]you>[/bold green] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye.[/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("/quit", "/exit"):
            console.print("[dim]Goodbye.[/dim]")
            break

        if user_input.lower().startswith("/effort "):
            level_str = user_input.split(maxsplit=1)[1].strip().lower()
            if level_str in _EFFORT_NAMES:
                new_effort = _EFFORT_NAMES[level_str]
                orchestrator.effort = new_effort
                if caps.effort_levels and new_effort not in caps.effort_levels:
                    console.print(
                        f"[yellow]Warning: {provider.name} does not honor "
                        f"effort={new_effort.value}; it may be ignored.[/yellow]"
                    )
                console.print(
                    f"[dim]Effort set to {new_effort.value} for subsequent turns.[/dim]"
                )
            else:
                console.print(
                    f"[yellow]Invalid effort level {level_str!r}. "
                    "Use: low, medium, high.[/yellow]"
                )
            continue

        try:
            result = orchestrator.run_turn(user_input)
        except Exception as exc:  # keep the loop alive on unexpected errors
            console.print(f"[red]Error: {exc}[/red]")
            continue

        console.print(Markdown(result.text))
        _render_status(console, result)
        console.print()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    cfg = load_config()

    provider_name = args.provider or cfg.provider
    effort = _EFFORT_NAMES.get(args.effort, cfg.effort) if args.effort else cfg.effort

    provider = _build_provider(provider_name, cfg)

    try:
        provider.validate()
    except Exception as exc:
        print(f"Provider validation failed ({provider_name}): {exc}", file=sys.stderr)
        print("Continuing anyway; calls may fail.\n", file=sys.stderr)

    memory = MemoryService(db_path=args.db_path)
    orchestrator = Orchestrator(provider=provider, memory=memory, effort=effort)

    _run_tui(orchestrator, provider, effort)


if __name__ == "__main__":
    main()
