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
    "/settings": "Open the settings panel.",
    "/clear": "Clear the transcript.",
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
        from textual.containers import Vertical, VerticalScroll
        from textual.screen import ModalScreen
        from textual.suggester import SuggestFromList
        from textual.widgets import (
            Button,
            Input,
            Label,
            Markdown,
            Rule,
            Select,
            Static,
        )
    except ImportError as exc:  # pragma: no cover - exercised via message text
        raise SystemExit(_TEXTUAL_MISSING_MSG) from exc

    cfg = cfg or RickshawConfig()

    # ---- Settings modal screen -----------------------------------------

    class SettingsScreen(ModalScreen):
        """Modal for viewing / editing runtime settings."""

        CSS = """
        SettingsScreen { align: center middle; }
        #settings-container {
            width: 70;
            max-height: 36;
            background: #1a1b1e;
            border: solid #3a3f47;
            padding: 1 2;
        }
        #settings-container Label { margin: 1 0 0 0; color: #9aa0a8; }
        #settings-container Select { margin: 0 0 1 0; }
        #settings-container Input { margin: 0 0 1 0; }
        #settings-container Button { margin: 1 1 0 0; }
        .section-title { color: #d98a3d; text-style: bold; margin: 1 0 0 0; }
        """

        BINDINGS = [
            Binding("escape", "dismiss_settings", "Close", show=False),
        ]

        def __init__(self, app_cfg: RickshawConfig) -> None:
            super().__init__()
            self._cfg = app_cfg
            self._advanced = False

        def compose(self) -> ComposeResult:
            provider_choices = [
                (name, name) for name in sorted(self._cfg.providers)
            ]
            settings = load_settings()
            current_provider = settings.get("provider", self._cfg.provider)
            current_effort = settings.get("effort", self._cfg.effort.value)
            effort_choices = [
                ("low", "low"), ("medium", "medium"), ("high", "high"),
            ]
            current_emb_provider = settings.get(
                "embedding_provider", self._cfg.embedding_provider or "openai",
            )
            current_emb_model = settings.get(
                "embedding_model", self._cfg.openai_embedding_model,
            )

            with Vertical(id="settings-container"):
                yield Label("Settings", classes="section-title")

                yield Label("Provider")
                yield Select(
                    provider_choices,
                    value=current_provider,
                    id="sel-provider",
                )

                yield Label("Chat model")
                profile = self._cfg.providers.get(current_provider)
                default_model = profile.model if profile else ""
                yield Input(
                    value=default_model,
                    placeholder="model name",
                    id="inp-model",
                )

                yield Label("Effort")
                yield Select(
                    effort_choices,
                    value=current_effort,
                    id="sel-effort",
                )

                # Advanced section (always rendered, toggled by button)
                yield Label("Advanced", classes="section-title", id="lbl-advanced")

                yield Label("Add / edit provider", id="lbl-adv-header")
                yield Label("Name", id="lbl-adv-name")
                yield Input(placeholder="e.g. deepseek", id="inp-adv-name")
                yield Label("Base URL", id="lbl-adv-url")
                yield Input(placeholder="https://...", id="inp-adv-url")
                yield Label("API key env var", id="lbl-adv-key")
                yield Input(placeholder="DEEPSEEK_API_KEY", id="inp-adv-key")
                yield Label("Wire format", id="lbl-adv-wire")
                yield Select(
                    [("openai", "openai"), ("anthropic", "anthropic"), ("devin", "devin")],
                    value="openai",
                    id="sel-adv-wire",
                )

                yield Label("Embedding provider", id="lbl-emb-provider")
                yield Input(
                    value=current_emb_provider,
                    placeholder="openai",
                    id="inp-emb-provider",
                )
                yield Label("Embedding model", id="lbl-emb-model")
                yield Input(
                    value=current_emb_model,
                    placeholder="text-embedding-3-small",
                    id="inp-emb-model",
                )

                yield Button("Save", variant="primary", id="btn-save")
                yield Button("Cancel", id="btn-cancel")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "btn-cancel":
                self.dismiss(None)
                return
            if event.button.id == "btn-save":
                result = self._collect()
                self.dismiss(result)

        def _collect(self) -> dict:
            data: dict = {}
            data["provider"] = self.query_one("#sel-provider", Select).value
            data["model"] = self.query_one("#inp-model", Input).value.strip()
            data["effort"] = self.query_one("#sel-effort", Select).value
            data["embedding_provider"] = self.query_one(
                "#inp-emb-provider", Input
            ).value.strip()
            data["embedding_model"] = self.query_one(
                "#inp-emb-model", Input
            ).value.strip()

            adv_name = self.query_one("#inp-adv-name", Input).value.strip()
            adv_url = self.query_one("#inp-adv-url", Input).value.strip()
            adv_key = self.query_one("#inp-adv-key", Input).value.strip()
            adv_wire = self.query_one("#sel-adv-wire", Select).value
            if adv_name and adv_url and adv_key:
                data["new_provider"] = {
                    "name": adv_name,
                    "base_url": adv_url,
                    "api_key_env": adv_key,
                    "wire_format": adv_wire,
                    "model": data.get("model", ""),
                }
            return data

        def on_select_changed(self, event: Select.Changed) -> None:
            if event.select.id == "sel-provider":
                profile = self._cfg.providers.get(str(event.value))
                if profile:
                    self.query_one("#inp-model", Input).value = profile.model

        def action_dismiss_settings(self) -> None:
            self.dismiss(None)

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
                f"{self.provider.name} · {model} · effort "
                f"{self.orchestrator.effort.value} · {tools}",
                "meta",
            )

        def _cmd_effort(self, arg: str) -> None:
            level = arg.lower()
            if level not in _EFFORT_NAMES:
                self._write(f"Invalid effort {arg!r}. Use: low, medium, high.", "warn")
                return
            new_effort = _EFFORT_NAMES[level]
            self.orchestrator.effort = new_effort
            caps = self.provider.capabilities()
            if caps.effort_levels and new_effort not in caps.effort_levels:
                self._write(
                    f"note: {self.provider.name} may ignore "
                    f"effort={new_effort.value}.",
                    "warn",
                )
            self._write(f"effort · {new_effort.value}", "meta")

        def _cmd_model(self, arg: str) -> None:
            if not arg:
                model = getattr(self.provider, "_model", "") or "(unknown)"
                self._write(f"Current model: {model}", "meta")
                return
            try:
                new_provider = _rebuild_provider(self.provider.name, self.cfg, arg)
            except Exception as exc:
                self._write(f"Cannot switch model: {exc}", "warn")
                return
            self.provider = new_provider
            self.orchestrator.provider = new_provider
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
            self.push_screen(SettingsScreen(self.cfg), callback=self._on_settings_dismiss)

        def _on_settings_dismiss(self, result) -> None:
            if result is None:
                return
            settings = load_settings()

            # Handle new custom provider
            new_prov = result.pop("new_provider", None)
            if new_prov:
                name = new_prov.pop("name")
                settings.setdefault("providers", {})[name] = new_prov
                self.cfg.providers[name] = ProviderProfile(
                    base_url=new_prov["base_url"],
                    model=new_prov.get("model", ""),
                    api_key_env=new_prov["api_key_env"],
                    wire_format=new_prov.get("wire_format", "openai"),
                )

            chosen_provider = result.get("provider", self.cfg.provider)
            chosen_model = result.get("model", "")
            chosen_effort = result.get("effort", self.orchestrator.effort.value)
            emb_provider = result.get("embedding_provider", "")
            emb_model = result.get("embedding_model", "")

            settings["provider"] = chosen_provider
            settings["effort"] = chosen_effort
            settings["embedding_provider"] = emb_provider
            settings["embedding_model"] = emb_model
            save_settings(settings)

            # Rebuild provider
            try:
                profile = self.cfg.providers.get(chosen_provider)
                if profile and chosen_model:
                    overridden = ProviderProfile(
                        base_url=profile.base_url,
                        model=chosen_model,
                        api_key_env=profile.api_key_env,
                        wire_format=profile.wire_format,
                    )
                    new_provider = build_provider_from_profile(
                        chosen_provider, overridden,
                        embedding_model=emb_model or self.cfg.openai_embedding_model,
                    )
                else:
                    new_provider = _build_provider(chosen_provider, self.cfg)
                self.provider = new_provider
                self.orchestrator.provider = new_provider
            except Exception as exc:
                self._write(f"Could not switch provider: {exc}", "warn")

            # Update effort
            if chosen_effort in _EFFORT_NAMES:
                self.orchestrator.effort = _EFFORT_NAMES[chosen_effort]
                self.effort = _EFFORT_NAMES[chosen_effort]

            self._cmd_status()
            self._write("settings saved.", "meta")

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
