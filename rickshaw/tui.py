"""Full-screen terminal UI for Rickshaw, built on Textual.

A Claude-Code / Codex-style TUI: a scrollable transcript, a pinned input at the
bottom, a status bar (provider · model · effort · tokens), streaming replies, a
"thinking" indicator, Esc to interrupt an in-flight turn, and slash-command
autocomplete. Every turn is routed through :meth:`Orchestrator.run_turn`, so the
semantic memory layer (remember / recall / forget) and graceful-degradation info
are active and surfaced.

Launch::

    rickshaw --provider openai --effort high

The module itself (and the branding constants below) import fine without Textual
installed -- the framework is imported lazily, only when the app is built.
"""

from __future__ import annotations

import argparse
import sys

from rickshaw.cli import _EFFORT_NAMES, _build_provider, load_config
from rickshaw.config import ProviderProfile, RickshawConfig
from rickshaw.memory.service import MemoryService
from rickshaw.orchestrator import Orchestrator
from rickshaw.providers.base import Effort, LLMProvider
from rickshaw.providers.build import build_provider_from_profile
from rickshaw.providers.factory import get_provider
from rickshaw.settings import load_settings, save_settings

# Branding — module-level so cli.py can import and reuse them.
RICKSHAW_LOGO = "o--o  rickshaw"
RICKSHAW_SLOGAN = "your driver, your memory"
RICKSHAW_BANNER = f"{RICKSHAW_LOGO} \u00b7 {RICKSHAW_SLOGAN}"

# Where the memory layer persists across sessions (vs. the default ":memory:").
_DEFAULT_DB_PATH = "rickshaw_memory.db"

# Slash-commands, used for help text and inline autocomplete.
_COMMANDS = {
    "/help": "Show this help.",
    "/status": "Show provider, model, and effort.",
    "/settings": "Show current settings and usage hints.",
    "/clear": "Clear the transcript.",
    "/provider": "/provider [name|add] -- show, switch, or register a provider.",
    "/effort": "/effort <low|medium|high> -- set reasoning effort.",
    "/model": "/model [name] -- show or switch the chat model.",
    "/memory": "List recently stored memories.",
    "/quit": "Exit.",
    "/exit": "Exit.",
}

_DEFAULT_HINT = "/help  ·  esc interrupt  ·  ^c quit"
_USER_MARK = "[#d98a3d]\u203a[/]"  # amber angle-quote before each user message

_TEXTUAL_MISSING_MSG = (
    "The Rickshaw terminal UI requires Textual, which is not installed.\n"
    "Install it with:\n\n"
    '    pip install "rickshaw"\n'
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="rickshaw",
        description="Multi-LLM provider harness with effort levels.",
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
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate provider connectivity and exit.",
    )
    return parser.parse_args(argv)


def _rebuild_provider(name: str, cfg: RickshawConfig, model: str) -> LLMProvider:
    """Build a provider with a model override (used by /model and /settings).

    Works for any provider whose profile has ``wire_format == 'openai'``,
    ``'anthropic'``, or ``'devin'``.
    """
    profile = cfg.providers.get(name)
    if profile is not None:
        overridden = ProviderProfile(
            base_url=profile.base_url,
            model=model,
            api_key_env=profile.api_key_env,
            wire_format=profile.wire_format,
        )
        return build_provider_from_profile(
            name, overridden, embedding_model=cfg.openai_embedding_model,
        )
    raise ValueError(f"no profile found for provider {name!r}")


def make_app(
    orchestrator: Orchestrator,
    provider: LLMProvider,
    effort: Effort,
    cfg: RickshawConfig | None = None,
):
    """Build the Textual app instance. Imports Textual lazily.

    Kept as a factory (rather than a module-level class) so importing this
    module does not require Textual to be installed.
    """
    try:
        from textual import work
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import VerticalScroll
        from textual.suggester import SuggestFromList
        from textual.widgets import Input, Markdown, Rule, Static
    except ImportError as exc:  # pragma: no cover - exercised via message text
        raise SystemExit(_TEXTUAL_MISSING_MSG) from exc

    cfg = cfg or RickshawConfig()

    # ---- Provider-add wizard steps ------------------------------------

    _PROVIDER_ADD_STEPS = [
        ("name", "name: "),
        ("base_url", "base url: "),
        ("api_key_env", "api key env var: "),
        ("wire_format", "wire format (openai/anthropic/devin) [openai]: "),
    ]

    # ---- Main TUI app --------------------------------------------------

    class RickshawTUI(App):
        """Textual application driving turns through the Orchestrator."""

        TITLE = "rickshaw"
        SUB_TITLE = RICKSHAW_SLOGAN
        # Minimalist: no footer, no filled status bar, no boxed input.
        # Near-monochrome with a single amber accent; hairline rules separate
        # turns. Chrome is intentionally almost invisible.
        CSS = """
        Screen { layout: vertical; background: #0e0f11; }
        #head { height: auto; color: #4b5563; padding: 1 3 0 3; }
        #transcript {
            height: 1fr;
            padding: 1 3;
            scrollbar-size: 1 1;
            scrollbar-color: #2a2e37;
            scrollbar-color-hover: #3a3f47;
            scrollbar-color-active: #3a3f47;
            scrollbar-background: #0e0f11;
        }
        #transcript > Static { margin: 0 0 1 0; }
        #transcript > Markdown {
            margin: 0 0 1 0;
            padding: 0;
            background: transparent;
        }
        #transcript > Rule { color: #22252b; margin: 0 0 1 0; }
        .u { color: #dfe2e7; }
        .a { color: #9aa0a8; }
        .meta { color: #5c6370; }
        .warn { color: #c98a3d; }
        #hint { height: 1; color: #3a3f47; padding: 0 3 1 3; }
        #prompt {
            border: none;
            background: #0e0f11;
            color: #dfe2e7;
            padding: 0 3;
        }
        #prompt:focus { border: none; }
        """

        BINDINGS = [
            Binding("escape", "interrupt", "Interrupt", show=False),
            Binding("ctrl+l", "clear", "Clear", show=False),
            Binding("ctrl+c", "quit", "Quit", show=False),
        ]

        def __init__(self) -> None:
            super().__init__()
            self.orchestrator = orchestrator
            self.provider = provider
            self.effort = effort
            self.cfg = cfg
            self.orchestrator.effort = effort
            self._buffer = ""
            self._current_md: Markdown | None = None
            self._turn_active = False
            self._has_turns = False
            self._provider_add_state: dict | None = None

        # ---- layout -----------------------------------------------------

        def compose(self) -> ComposeResult:
            yield Static(RICKSHAW_BANNER, id="head")
            yield VerticalScroll(id="transcript")
            yield Static(_DEFAULT_HINT, id="hint")
            yield Input(
                placeholder="Message rickshaw…",
                id="prompt",
                suggester=SuggestFromList(sorted(_COMMANDS), case_sensitive=False),
            )

        def on_mount(self) -> None:
            caps = self.provider.capabilities()
            model = getattr(self.provider, "_model", "") or self.provider.name
            self._write(
                f"{self.provider.name} · {model} · effort "
                f"{self.orchestrator.effort.value} · /help",
                cls="meta",
            )
            if not caps.function_calling:
                self._write(
                    "tools off — recall is harness-driven for this provider.",
                    cls="meta",
                )
            self.query_one("#prompt", Input).focus()

        # ---- transcript helpers ----------------------------------------

        def _write(self, text: str, cls: str = "") -> Static:
            """Append a plain (Rich-markup) line to the transcript."""
            widget = Static(text, classes=cls)
            self.query_one("#transcript", VerticalScroll).mount(widget)
            self._scroll_end()
            return widget

        def _begin_assistant(self) -> None:
            self._buffer = ""
            md = Markdown("")
            self.query_one("#transcript", VerticalScroll).mount(md)
            self._current_md = md
            self._scroll_end()

        def _scroll_end(self) -> None:
            self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

        def _set_hint(self, text: str) -> None:
            self.query_one("#hint", Static).update(text)

        # ---- input handling --------------------------------------------

        def on_input_submitted(self, event: Input.Submitted) -> None:
            value = event.value.strip()
            event.input.value = ""
            # Provider-add wizard intercepts all input while active.
            if self._provider_add_state is not None:
                self._provider_add_step(value)
                return
            if not value:
                return
            if value.startswith("/"):
                self._handle_command(value)
                return
            if self._turn_active:
                self._write("A turn is already running; press Esc to interrupt.", "warn")
                return
            self._start_turn(value)

        def _handle_command(self, value: str) -> None:
            parts = value.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ""

            if cmd in ("/quit", "/exit"):
                self.exit()
            elif cmd == "/help":
                self._cmd_help()
            elif cmd == "/status":
                self._cmd_status()
            elif cmd == "/clear":
                self.action_clear()
            elif cmd == "/effort":
                self._cmd_effort(arg)
            elif cmd == "/model":
                self._cmd_model(arg)
            elif cmd == "/settings":
                self._cmd_settings()
            elif cmd in ("/provider", "/engine"):
                self._cmd_provider(arg)
            elif cmd == "/memory":
                self._cmd_memory()
            else:
                self._write(f"Unknown command {cmd!r}. Try /help.", "warn")

        def _cmd_help(self) -> None:
            for name, desc in _COMMANDS.items():
                self._write(f"{name}  {desc}", "meta")
            self._write("esc interrupts a running turn · ^c quits", "meta")

        def _cmd_status(self) -> None:
            model = getattr(self.provider, "_model", "") or self.provider.name
            caps = self.provider.capabilities()
            tools = "tools on" if caps.function_calling else "tools off"
            self._write(
                f"provider · {self.provider.name} · {model} · effort "
                f"{self.orchestrator.effort.value} · {tools}",
                "meta",
            )

        def _cmd_effort(self, arg: str) -> None:
            level = arg.lower()
            if level not in _EFFORT_NAMES:
                self._write(f"Invalid effort {arg!r}. Use: low, medium, high.", "warn")
                return
            new_effort = _EFFORT_NAMES[level]
            caps = self.provider.capabilities()
            if caps.effort_levels and new_effort not in caps.effort_levels:
                supported = ", ".join(e.value for e in caps.effort_levels)
                self._write(
                    f"{self.provider.name} does not support effort "
                    f"{new_effort.value}. Supported: {supported}.",
                    "warn",
                )
                return
            self.orchestrator.effort = new_effort
            self.effort = new_effort
            settings = load_settings()
            settings["effort"] = new_effort.value
            save_settings(settings)
            self._write(f"effort · {new_effort.value}", "meta")

        def _cmd_model(self, arg: str) -> None:
            if not arg:
                # List the current provider's available models.
                model = getattr(self.provider, "_model", "") or "(unknown)"
                self._write(
                    f"current · {self.provider.name} ({model})", "meta",
                )
                self._write("", "meta")
                try:
                    models = self.provider.available_models()
                except Exception as exc:
                    self._write(f"Cannot list models: {exc}", "warn")
                    return
                self._write("  available models:", "meta")
                for m in models:
                    marker = "♦" if m == model else " "
                    self._write(f"    {m:<32} {marker}", "meta")
                return

            # Strict validation: only allow models from the current provider.
            try:
                valid_models = self.provider.available_models()
            except Exception as exc:
                self._write(f"Cannot validate model: {exc}", "warn")
                return

            if arg not in valid_models:
                display = ", ".join(valid_models)
                self._write(
                    f"Unknown model {arg!r} for {self.provider.name}. "
                    f"Available: {display}",
                    "warn",
                )
                return

            try:
                new_provider = _rebuild_provider(self.provider.name, self.cfg, arg)
            except Exception as exc:
                self._write(f"Cannot switch model: {exc}", "warn")
                return
            self.provider = new_provider
            self.orchestrator.provider = new_provider

            # Effort reconciliation: reset to medium if unsupported.
            caps = new_provider.capabilities()
            old_effort = self.orchestrator.effort
            if caps.effort_levels and old_effort not in caps.effort_levels:
                default_effort = Effort.MEDIUM
                self.orchestrator.effort = default_effort
                self.effort = default_effort
                self._write(
                    f"note: {arg} does not support effort "
                    f"{old_effort.value}. Reset to medium.",
                    "warn",
                )

            settings = load_settings()
            settings["model"] = arg
            settings["effort"] = self.orchestrator.effort.value
            save_settings(settings)
            self._write(f"model · {arg}", "meta")

        def _cmd_memory(self) -> None:
            try:
                records = self.orchestrator.memory.store.all_records()
            except Exception as exc:  # pragma: no cover - defensive
                self._write(f"Could not read memory: {exc}", "warn")
                return
            if not records:
                self._write("no memories stored yet.", "meta")
                return
            self._write(f"memories · {len(records)}", "meta")
            for rec in records[-10:]:
                snippet = rec.text if len(rec.text) <= 100 else rec.text[:97] + "…"
                self._write(f"  {snippet}", "meta")

        def _cmd_settings(self) -> None:
            """Read-only display of current settings + usage hints."""
            model = getattr(self.provider, "_model", "") or self.provider.name
            settings = load_settings()
            emb_prov = settings.get(
                "embedding_provider",
                self.cfg.embedding_provider or "openai",
            )
            emb_model = settings.get(
                "embedding_model", self.cfg.openai_embedding_model,
            )
            lines = [
                "Settings",
                "\u2500" * 44,
                f"  provider         {self.provider.name}",
                f"  model            {model}",
                f"  effort           {self.orchestrator.effort.value}",
                f"  embedding        {emb_prov} / {emb_model}",
                "",
                "  Use:",
                "    /provider <name>          switch provider",
                "    /provider                 list available providers",
                "    /model <name>             switch chat model",
                "    /effort <low|medium|high> set reasoning effort",
                "    /provider add             register a custom provider",
                "\u2500" * 44,
            ]
            for line in lines:
                self._write(line, "meta")

        def _cmd_provider(self, arg: str) -> None:
            """Show, switch, or register providers."""
            if not arg:
                self._cmd_provider_list()
            elif arg.lower() == "add":
                self._cmd_provider_add_start()
            else:
                self._cmd_provider_switch(arg)

        def _cmd_provider_list(self) -> None:
            """List available providers with the active one marked."""
            model = getattr(self.provider, "_model", "") or self.provider.name
            self._write(
                f"current \u00b7 {self.provider.name} ({model})", "meta",
            )
            self._write("", "meta")
            self._write("  available providers:", "meta")
            for name in sorted(self.cfg.providers):
                profile = self.cfg.providers[name]
                marker = "\u2666" if name == self.provider.name else " "
                self._write(
                    f"    {name:<16} {marker} {profile.api_key_env}", "meta",
                )
            self._write("", "meta")
            self._write(
                "  /provider <name> to switch \u00b7 /provider add to register",
                "meta",
            )

        def _cmd_provider_switch(self, name: str) -> None:
            """Switch the active provider to *name*."""
            profile = self.cfg.providers.get(name)
            if profile is None:
                available = ", ".join(sorted(self.cfg.providers))
                self._write(
                    f"Unknown provider {name!r}. Available: {available}", "warn",
                )
                return
            try:
                new_provider = build_provider_from_profile(
                    name, profile,
                    embedding_model=self.cfg.openai_embedding_model,
                )
            except Exception as exc:
                self._write(f"Cannot switch provider: {exc}", "warn")
                return
            self.provider = new_provider
            self.orchestrator.provider = new_provider

            # Effort mismatch: reset to medium if unsupported.
            caps = new_provider.capabilities()
            old_effort = self.orchestrator.effort
            if caps.effort_levels and old_effort not in caps.effort_levels:
                default_effort = Effort.MEDIUM
                self.orchestrator.effort = default_effort
                self.effort = default_effort
                self._write(
                    f"note: {name} does not support effort "
                    f"{old_effort.value}. Reset to medium.",
                    "warn",
                )

            settings = load_settings()
            settings["provider"] = name
            settings["effort"] = self.orchestrator.effort.value
            save_settings(settings)

            model = getattr(new_provider, "_model", "") or name
            self._write(
                f"{name} \u00b7 {model} \u00b7 effort "
                f"{self.orchestrator.effort.value}",
                "meta",
            )

        def _cmd_provider_add_start(self) -> None:
            """Begin the interactive provider-registration wizard."""
            self._provider_add_state = {"step": 0, "data": {}}
            _key, prompt = _PROVIDER_ADD_STEPS[0]
            self._set_hint(f"{prompt}(Enter to submit, Esc to cancel)")

        def _provider_add_step(self, value: str) -> None:
            """Process one step of the provider-add wizard."""
            state = self._provider_add_state
            if state is None:
                return
            step_idx = state["step"]
            key, prompt_text = _PROVIDER_ADD_STEPS[step_idx]

            # Apply default for wire_format.
            if key == "wire_format" and not value:
                value = "openai"

            # Validate required fields.
            if not value and key != "wire_format":
                self._write(f"{key} is required.", "warn")
                self._write(_PROVIDER_ADD_STEPS[step_idx][1], "meta")
                return

            # Echo the user's input next to the prompt so the transcript
            # reads like a CLI conversation.
            self._write(f"{prompt_text}{value}", "meta")

            state["data"][key] = value
            step_idx += 1
            state["step"] = step_idx

            if step_idx < len(_PROVIDER_ADD_STEPS):
                _next_key, next_prompt = _PROVIDER_ADD_STEPS[step_idx]
                self._set_hint(f"{next_prompt}(Enter to submit, Esc to cancel)")
            else:
                self._provider_add_finish(state["data"])

        def _provider_add_finish(self, data: dict) -> None:
            """Register the new provider and persist it."""
            self._provider_add_state = None
            self._set_hint(_DEFAULT_HINT)

            name = data["name"]
            profile = ProviderProfile(
                base_url=data["base_url"],
                model="",
                api_key_env=data["api_key_env"],
                wire_format=data.get("wire_format", "openai"),
            )
            self.cfg.providers[name] = profile

            settings = load_settings()
            settings.setdefault("providers", {})[name] = {
                "base_url": profile.base_url,
                "api_key_env": profile.api_key_env,
                "wire_format": profile.wire_format,
                "model": profile.model,
            }
            save_settings(settings)

            self._write(
                f"provider registered \u00b7 {name} "
                f"({profile.wire_format} wire format)",
                "meta",
            )

        # ---- turn execution --------------------------------------------

        def _start_turn(self, text: str) -> None:
            self._turn_active = True
            if self._has_turns:
                self.query_one("#transcript", VerticalScroll).mount(Rule())
            self._has_turns = True
            self._write(f"{_USER_MARK} {text}", "u")
            self._begin_assistant()
            self._set_hint("thinking…  ·  esc to interrupt")
            self.query_one("#prompt", Input).disabled = True
            self._run_turn(text)

        @work(thread=True, exclusive=True, group="turn")
        def _run_turn(self, text: str) -> None:
            def on_delta(chunk: str) -> None:
                if not self._turn_active:
                    return
                self.call_from_thread(self._append_delta, chunk)

            try:
                result = self.orchestrator.run_turn(text, on_delta=on_delta)
            except Exception as exc:  # keep the app alive on unexpected errors
                self.call_from_thread(self._turn_error, exc)
                return
            self.call_from_thread(self._turn_done, result)

        def _append_delta(self, chunk: str) -> None:
            self._buffer += chunk
            if self._current_md is not None:
                self._current_md.update(self._buffer)
            self._scroll_end()

        def _turn_done(self, result) -> None:
            if self._current_md is not None and self._buffer != result.text:
                # Non-streaming providers deliver everything in one delta; make
                # sure the final rendered text matches the result exactly.
                self._current_md.update(result.text)
            # Keep the transcript quiet: only surface a dim meta line when there
            # is something noteworthy (tokens, tool calls, or degradation).
            parts: list[str] = []
            if result.usage is not None and result.usage.total_tokens:
                parts.append(f"{result.usage.total_tokens} tok")
            if result.tool_calls_made:
                parts.append(f"{result.tool_calls_made} tool calls")
            if result.degraded:
                parts.append("degraded · local memory")
            if parts:
                self._write(" · ".join(parts), "meta")
            self._finish_turn()

        def _turn_error(self, exc: Exception) -> None:
            self._write(f"Error: {exc}", "warn")
            self._finish_turn()

        def _finish_turn(self) -> None:
            self._turn_active = False
            self._current_md = None
            self._set_hint(_DEFAULT_HINT)
            prompt = self.query_one("#prompt", Input)
            prompt.disabled = False
            prompt.focus()

        # ---- actions ----------------------------------------------------

        def action_interrupt(self) -> None:
            if self._provider_add_state is not None:
                self._provider_add_state = None
                self._write("(cancelled)", "warn")
                self._set_hint(_DEFAULT_HINT)
                return
            if not self._turn_active:
                return
            self.workers.cancel_group(self, "turn")
            self._write("(interrupted)", "warn")
            self._finish_turn()

        def action_clear(self) -> None:
            self.query_one("#transcript", VerticalScroll).remove_children()
            self._has_turns = False
            self._write("cleared.", "meta")

    return RickshawTUI()


def _run_app(
    orchestrator: Orchestrator,
    provider: LLMProvider,
    effort: Effort,
    cfg: RickshawConfig,
) -> None:
    """Build and run the Textual app (separated out for testability)."""
    make_app(orchestrator, provider, effort, cfg).run()


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
        if args.validate_only:
            sys.exit(1)
        print("Continuing anyway; calls may fail.\n", file=sys.stderr)

    if args.validate_only:
        print(f"Provider {provider_name!r} validated successfully.")
        return

    memory = MemoryService(db_path=args.db_path)
    orchestrator = Orchestrator(provider=provider, memory=memory, effort=effort)

    _run_app(orchestrator, provider, effort, cfg)


if __name__ == "__main__":
    main()
