"""Full-screen terminal UI for Rickshaw, built on Textual.

A Claude-Code / Codex-style TUI: a scrollable transcript, a pinned input at the
bottom, a status bar (provider · model · effort · tokens), streaming replies, a
"thinking" indicator, Esc to interrupt an in-flight turn, and slash-command
autocomplete. Every turn is routed through :meth:`Orchestrator.run_turn`, so the
semantic memory layer (remember / recall / forget) and graceful-degradation info
are active and surfaced.

Launch::

    rickshaw                       # prompts for provider on startup
    rickshaw --provider openai     # optional override
    rickshaw --effort high

When launched without ``--provider`` and with no persisted provider in
``~/.rickshaw/settings.json``, the TUI opens in a *no-provider-selected* state
and immediately shows an interactive provider picker. OAuth-capable providers
trigger a login flow. ``--provider`` remains available as an optional override.

The module itself (and the branding constants below) import fine without Textual
installed -- the framework is imported lazily, only when the app is built.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import webbrowser
from urllib.parse import quote, urlencode

import httpx

from rickshaw.cli import _EFFORT_NAMES, _build_provider, load_config

logger = logging.getLogger(__name__)
from rickshaw.config import ProviderProfile, RickshawConfig
from rickshaw.memory.service import MemoryService
from rickshaw.orchestrator import Orchestrator
from rickshaw.providers.base import Effort, LLMProvider
from rickshaw.providers.build import build_provider_from_profile
from rickshaw.providers.factory import get_provider
from rickshaw.settings import load_settings, save_settings

from rickshaw_ai._builtins import default_providers as _builtin_providers
from rickshaw_ai.credentials.store import FileCredentialStore
from rickshaw_ai.factory import builtin_models as _builtin_models
from rickshaw.providers import _bridge
from rickshaw.providers._bridge import run_sync

# Branding — module-level so cli.py can import and reuse them.
RICKSHAW_LOGO = "o--o  rickshaw"
RICKSHAW_SLOGAN = "your driver, your memory"
RICKSHAW_BANNER = f"{RICKSHAW_LOGO} \u00b7 {RICKSHAW_SLOGAN}"

# Where the memory layer persists across sessions (vs. the default ":memory:").
_DEFAULT_DB_PATH = "rickshaw_memory.db"

_OAUTH_QUIRKS = {
    "anthropic": {
        "authorize_extra": {"code": "true"},
        "token_encoding": "json",
        "token_include_state": True,
    }
}
_DEFAULT_OAUTH_QUIRK = {
    "authorize_extra": {},
    "token_encoding": "form",
    "token_include_state": False,
}

# Slash-commands, used for help text and inline autocomplete.
_COMMANDS = {
    "/help": "Show this help.",
    "/status": "Show provider, model, and effort.",
    "/settings": "Interactive provider/model picker (also shows current settings).",
    "/models": "List the current provider's available models.",
    "/clear": "Clear the transcript.",
    "/provider": "/provider [name|add] -- show, switch, or register a provider.",
    "/effort": "/effort <low|medium|high> -- set reasoning effort.",
    "/model": "/model [name] -- show or switch the chat model.",
    "/login": "Authenticate (or re-authenticate) the active provider via OAuth.",
    "/memory": "List recently stored memories.",
    "/quit": "Exit.",
    "/exit": "Exit.",
}

_DEFAULT_HINT = "/help  ·  esc interrupt  ·  ^c quit"
_USER_MARK = "[#d98a3d]\u203a[/]"  # amber angle-quote before each user message
_PROMPT_GLYPH = "\u203a"  # in-frame prompt glyph

_TEXTUAL_MISSING_MSG = (
    "The Rickshaw terminal UI requires Textual, which is not installed.\n"
    "Install it with:\n\n"
    '    pip install "rickshaw"\n'
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from rickshaw import __version__

    parser = argparse.ArgumentParser(
        prog="rickshaw",
        description="Multi-LLM provider harness with effort levels.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"rickshaw {__version__} ({os.path.abspath(__file__)})",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider name (e.g. openai, devin). Overrides config/env.",
    )
    parser.add_argument(
        "--oauth-url",
        metavar="PROVIDER",
        default=None,
        help="Print the OAuth authorize URL for PROVIDER and exit.",
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
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help=(
            "Continue launching even when provider validation fails. "
            "Without this flag, validation failure exits non-zero."
        ),
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


def _get_builtin_provider_names() -> list[str]:
    """Return sorted ids of the built-in providers from rickshaw_ai."""
    return sorted(p.id for p in _builtin_providers())


def _builtin_provider_info(provider_id: str):
    """Look up a ProviderInfo by id from the builtins."""
    for p in _builtin_providers():
        if p.id == provider_id:
            return p
    return None


def _oauth_quirk(provider_id: str) -> dict[str, object]:
    quirk = dict(_DEFAULT_OAUTH_QUIRK)
    quirk.update(_OAUTH_QUIRKS.get(provider_id, {}))
    return quirk


def _build_authorize_url(oauth_cfg, *, state: str, code_challenge: str | None, extra):
    params = {
        "response_type": "code",
        "client_id": oauth_cfg.client_id,
        "scope": " ".join(oauth_cfg.scopes),
        "state": state,
    }
    if oauth_cfg.redirect_uri:
        params["redirect_uri"] = oauth_cfg.redirect_uri
    if code_challenge:
        params["code_challenge"] = code_challenge
        params["code_challenge_method"] = "S256"
    params.update(extra)
    return f"{oauth_cfg.authorize_url}?{urlencode(params, quote_via=quote)}"


def _build_auth_code_authorize_url(provider_id: str, models) -> tuple[str, str | None, str]:
    from rickshaw_ai.auth.oauth import generate_pkce
    import base64, os as _os

    try:
        info = models.provider_info(provider_id)
    except KeyError as exc:
        raise ValueError(f"unknown provider {provider_id!r}") from exc
    oauth_cfg = info.oauth
    if oauth_cfg is None:
        raise ValueError(f"provider {provider_id!r} does not support OAuth")

    verifier = challenge = None
    if oauth_cfg.use_pkce:
        verifier, challenge = generate_pkce()
    state = base64.urlsafe_b64encode(_os.urandom(16)).rstrip(b"=").decode()
    url = _build_authorize_url(
        oauth_cfg,
        state=state,
        code_challenge=challenge,
        extra=_oauth_quirk(provider_id)["authorize_extra"],
    )
    return url, verifier, state


def make_app(
    orchestrator: Orchestrator,
    provider: LLMProvider | None,
    effort: Effort,
    cfg: RickshawConfig | None = None,
):
    """Build the Textual app instance. Imports Textual lazily.

    *provider* may be ``None`` when launching without a pre-selected provider.
    The TUI then shows an interactive picker on mount.

    Kept as a factory (rather than a module-level class) so importing this
    module does not require Textual to be installed.
    """
    try:
        from textual import work
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.containers import Horizontal, VerticalScroll
        from textual.message import Message
        from textual.widgets import Markdown, Rule, Static, TextArea
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

    class PromptArea(TextArea):
        """Multi-line prompt. Enter submits; Shift+Enter / Ctrl+J insert a newline."""

        class Submitted(Message):
            def __init__(self, value: str) -> None:
                self.value = value
                super().__init__()

        @property
        def value(self) -> str:
            return self.text

        @value.setter
        def value(self, new: str) -> None:
            self.text = new

        def _on_key(self, event) -> None:
            if event.key == "enter":
                event.prevent_default()
                event.stop()
                self.post_message(self.Submitted(self.text))
                return
            if event.key in ("shift+enter", "ctrl+j"):
                event.prevent_default()
                event.stop()
                self.insert("\n")
                return

    class RickshawTUI(App):
        """Textual application driving turns through the Orchestrator."""

        TITLE = "rickshaw"
        SUB_TITLE = RICKSHAW_SLOGAN
        # Minimalist: no footer, no filled status bar, no boxed input.
        # Near-monochrome with a single amber accent; hairline rules separate
        # turns. Chrome is intentionally almost invisible.
        CSS = """
        $rk-bg: #0f1113;
        $rk-surface: #16181b;
        $rk-border: #2a2f36;
        $rk-text: #e6e8ea;
        $rk-meta: #8b929c;
        $rk-accent: #e0a86b;
        $rk-assistant: #7fb0c9;
        $rk-warn: #d98a3d;
        $rk-error: #d16a5a;
        $rk-success: #7fae7f;
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
        .degraded-banner {
            color: #1a1a1a;
            background: #c98a3d;
            text-style: bold;
            padding: 0 1;
        }
        #hint { height: 1; color: #3a3f47; padding: 0 3 1 3; }
        #prompt-box {
            height: auto;
            max-height: 12;
            margin: 0 3 1 3;
            padding: 0 1;
            border: round $rk-border;
            background: transparent;
        }
        #prompt-box:focus-within { border: round $rk-accent; }
        #prompt-glyph {
            width: 2;
            height: auto;
            color: $rk-accent;
            padding: 0;
        }
        #prompt {
            height: auto;
            max-height: 10;
            border: none;
            padding: 0;
            background: transparent;
            color: $rk-text;
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
            self._settings_state: dict | None = None
            self._login_state: dict | None = None

        # ---- layout -----------------------------------------------------

        def compose(self) -> ComposeResult:
            yield Static(RICKSHAW_BANNER, id="head")
            yield VerticalScroll(id="transcript")
            yield Static(_DEFAULT_HINT, id="hint")
            with Horizontal(id="prompt-box"):
                yield Static(_PROMPT_GLYPH, id="prompt-glyph")
                yield PromptArea(id="prompt")

        def on_mount(self) -> None:
            if self.provider is None:
                self._write("no provider selected", cls="meta")
                self._start_provider_picker()
            else:
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
            self.query_one("#prompt", PromptArea).focus()

        # ---- on-launch provider picker ---------------------------------

        def _start_provider_picker(self) -> None:
            """Display the provider picker (builtins + configured)."""
            builtin_names = _get_builtin_provider_names()
            configured_names = sorted(self.cfg.providers)
            all_names = sorted(set(builtin_names) | set(configured_names))
            if not all_names:
                self._write("No providers available.", "warn")
                return
            self._write("", "meta")
            self._write("  Pick a provider (enter name, Esc to cancel):", "meta")
            current = self.provider.name if self.provider else ""
            for name in all_names:
                info = _builtin_provider_info(name)
                oauth_tag = ""
                if info and info.oauth:
                    oauth_tag = " (oauth)"
                marker = "\u2666" if name == current else " "
                self._write(f"    {name:<16} {marker}{oauth_tag}", "meta")
            self._settings_state = {
                "step": "provider",
                "providers": all_names,
                "on_launch": self.provider is None,
            }
            self._set_hint("provider name (Enter to submit, Esc to cancel)")

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

        def on_prompt_area_submitted(self, event: "PromptArea.Submitted") -> None:
            value = event.value.strip()
            self.query_one("#prompt", PromptArea).text = ""
            # Wizard intercepts all input while active.
            if self._login_state is not None:
                self._login_step(value)
                return
            if self._settings_state is not None:
                self._settings_step(value)
                return
            if self._provider_add_state is not None:
                self._provider_add_step(value)
                return
            if not value:
                return
            if value.startswith("/"):
                self._handle_command(value)
                return
            if self.provider is None:
                self._write("No provider selected. Use /settings to pick one.", "warn")
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
            elif cmd == "/models":
                self._cmd_models()
            elif cmd in ("/provider", "/engine"):
                self._cmd_provider(arg)
            elif cmd == "/login":
                self._cmd_login()
            elif cmd == "/memory":
                self._cmd_memory()
            else:
                self._write(f"Unknown command {cmd!r}. Try /help.", "warn")

        def _cmd_help(self) -> None:
            for name, desc in _COMMANDS.items():
                self._write(f"{name}  {desc}", "meta")
            self._write("esc interrupts a running turn · ^c quits", "meta")

        def _cmd_status(self) -> None:
            if self.provider is None:
                self._write("provider · (none) · /settings to pick one", "meta")
                return
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
                    logger.exception("Failed to list models for /model")
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
                logger.exception("Failed to validate model %r", arg)
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
                logger.exception("Failed to switch model to %r", arg)
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

        def _cmd_models(self) -> None:
            """Non-interactive list of the current provider's models."""
            model = getattr(self.provider, "_model", "") or "(unknown)"
            self._write(
                f"current \u00b7 {self.provider.name} ({model})", "meta",
            )
            self._write("", "meta")
            try:
                models = self.provider.available_models()
            except Exception as exc:
                logger.exception("Failed to list models for /models")
                self._write(f"Cannot list models: {exc}", "warn")
                return
            self._write("  available models:", "meta")
            for m in models:
                marker = "\u2666" if m == model else " "
                self._write(f"    {m:<32} {marker}", "meta")

        def _cmd_settings(self) -> None:
            """Interactive provider/model picker with settings header."""
            prov_name = self.provider.name if self.provider else "(none)"
            model = (getattr(self.provider, "_model", "") or prov_name) if self.provider else "(none)"
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
                f"  provider         {prov_name}",
                f"  model            {model}",
                f"  effort           {self.orchestrator.effort.value}",
                f"  embedding        {emb_prov} / {emb_model}",
                "\u2500" * 44,
            ]
            for line in lines:
                self._write(line, "meta")

            self._start_provider_picker()

        def _settings_step(self, value: str) -> None:
            """Process one step of the interactive /settings wizard."""
            state = self._settings_state
            if state is None:
                return

            if state["step"] == "provider":
                if not value:
                    self._write("(cancelled)", "warn")
                    self._settings_state = None
                    self._set_hint(_DEFAULT_HINT)
                    return
                chosen = value.strip()
                all_providers = state["providers"]
                if chosen not in all_providers:
                    available = ", ".join(all_providers)
                    self._write(
                        f"Unknown provider {chosen!r}. Available: {available}",
                        "warn",
                    )
                    return
                self._write(f"  provider: {chosen}", "meta")

                # Check if this is an OAuth-capable builtin that needs login.
                info = _builtin_provider_info(chosen)
                if info and info.oauth:
                    if "oauth" in info.auth_methods:
                        self._settings_state = None
                        self._set_hint(_DEFAULT_HINT)
                        self._start_oauth_login(chosen, info)
                        return

                # Build a temporary provider to list its models.
                profile = self.cfg.providers.get(chosen)
                if profile is None:
                    self._write(f"No profile for {chosen!r}; use /provider add.", "warn")
                    self._settings_state = None
                    self._set_hint(_DEFAULT_HINT)
                    return
                try:
                    temp = build_provider_from_profile(
                        chosen, profile,
                        embedding_model=self.cfg.openai_embedding_model,
                    )
                    models = temp.available_models()
                except Exception as exc:
                    logger.exception("Failed to list models for provider %r", chosen)
                    self._write(f"Cannot list models for {chosen}: {exc}", "warn")
                    self._settings_state = None
                    self._set_hint(_DEFAULT_HINT)
                    return

                if not models:
                    self._write(f"No models available for {chosen}.", "warn")
                    self._settings_state = None
                    self._set_hint(_DEFAULT_HINT)
                    return

                current_model = (
                    getattr(self.provider, "_model", "")
                    if self.provider and chosen == self.provider.name
                    else ""
                )
                self._write("", "meta")
                self._write("  Pick a model (enter name, Esc to cancel):", "meta")
                for m in models:
                    marker = "\u2666" if m == current_model else " "
                    self._write(f"    {m:<32} {marker}", "meta")
                state["step"] = "model"
                state["chosen_provider"] = chosen
                state["valid_models"] = models
                self._set_hint("model name (Enter to submit, Esc to cancel)")

            elif state["step"] == "model":
                if not value:
                    self._write("(cancelled)", "warn")
                    self._settings_state = None
                    self._set_hint(_DEFAULT_HINT)
                    return
                model_name = value.strip()
                valid_models = state["valid_models"]
                chosen_provider = state["chosen_provider"]
                if model_name not in valid_models:
                    display = ", ".join(valid_models)
                    self._write(
                        f"Unknown model {model_name!r}. Available: {display}",
                        "warn",
                    )
                    return
                self._write(f"  model: {model_name}", "meta")

                # Apply the provider + model switch.
                self._settings_apply(chosen_provider, model_name)
                self._settings_state = None
                self._set_hint(_DEFAULT_HINT)

        def _settings_apply(self, provider_name: str, model_name: str) -> None:
            """Apply provider + model selection from /settings wizard."""
            profile = self.cfg.providers[provider_name]
            try:
                new_provider = _rebuild_provider(
                    provider_name, self.cfg, model_name,
                )
            except Exception as exc:
                logger.exception("Failed to switch provider/model via /settings")
                self._write(f"Cannot switch: {exc}", "warn")
                return
            self.provider = new_provider
            self.orchestrator.provider = new_provider

            # Effort reconciliation.
            caps = new_provider.capabilities()
            old_effort = self.orchestrator.effort
            if caps.effort_levels and old_effort not in caps.effort_levels:
                default_effort = Effort.MEDIUM
                self.orchestrator.effort = default_effort
                self.effort = default_effort
                self._write(
                    f"note: {provider_name} does not support effort "
                    f"{old_effort.value}. Reset to medium.",
                    "warn",
                )

            settings = load_settings()
            settings["provider"] = provider_name
            settings["model"] = model_name
            settings["effort"] = self.orchestrator.effort.value
            save_settings(settings)

            self._write(
                f"{provider_name} \u00b7 {model_name} \u00b7 effort "
                f"{self.orchestrator.effort.value}",
                "meta",
            )

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
                logger.exception("Failed to switch provider to %r", name)
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

        # ---- OAuth login ------------------------------------------------

        def _start_oauth_login(self, provider_id: str, info=None) -> None:
            """Begin the OAuth login flow for *provider_id*."""
            if info is None:
                info = _builtin_provider_info(provider_id)
            if info is None or info.oauth is None:
                self._write(f"{provider_id} does not support OAuth.", "warn")
                return

            store = FileCredentialStore(_bridge.credential_store_path())
            models = _builtin_models(credentials=store)

            if info.oauth.mode == "device_code":
                self._write(f"logging in to {provider_id} (device code)…", "meta")
                self._run_device_code_login(provider_id, models)
            else:
                self._write(f"logging in to {provider_id} (browser)…", "meta")
                self._write("a browser window will open — paste the code below.", "meta")
                self._login_state = {
                    "provider_id": provider_id,
                    "models": models,
                }
                self._set_hint("paste authorization code (Enter to submit, Esc to cancel)")

                def _open_browser(url: str) -> None:
                    self.call_from_thread(
                        self._write,
                        f"open this URL if the browser didn't launch:\n{url}",
                        "meta",
                    )
                    webbrowser.open(url)

                # Start the login but wait for user to paste the code
                self._login_state["open_browser"] = _open_browser
                self._run_auth_code_login_start(provider_id, models, _open_browser)

        @work(thread=True, exclusive=True, group="login")
        def _run_auth_code_login_start(self, provider_id, models, open_browser):
            """Start auth-code login: build authorize URL and open browser."""
            url, verifier, state = _build_auth_code_authorize_url(provider_id, models)
            open_browser(url)
            # Save PKCE state for later code exchange
            self.call_from_thread(self._save_login_pkce, provider_id, verifier, state)

        def _save_login_pkce(self, provider_id, verifier, state):
            if self._login_state and self._login_state.get("provider_id") == provider_id:
                self._login_state["verifier"] = verifier
                self._login_state["state"] = state

        def _login_step(self, value: str) -> None:
            """Handle user pasting the authorization code."""
            state = self._login_state
            if state is None:
                return
            if not value:
                self._write("(cancelled)", "warn")
                self._login_state = None
                self._set_hint(_DEFAULT_HINT)
                return
            provider_id = state["provider_id"]
            models = state["models"]
            code_raw = value.strip()
            verifier = state.get("verifier")
            pkce_state = state.get("state")
            self._login_state = None
            self._set_hint(_DEFAULT_HINT)
            self._run_auth_code_exchange(provider_id, models, code_raw, verifier, pkce_state)

        @work(thread=True, exclusive=True, group="login")
        def _run_auth_code_exchange(self, provider_id, models, code_raw, verifier, pkce_state):
            """Exchange authorization code for tokens."""
            from rickshaw_ai.auth.oauth import _parse_callback

            info = models.provider_info(provider_id)
            oauth_cfg = info.oauth
            quirk = _oauth_quirk(provider_id)

            code, returned_state = _parse_callback(code_raw)
            if returned_state is not None and pkce_state and returned_state != pkce_state:
                self.call_from_thread(
                    self._write,
                    "OAuth state mismatch — possible CSRF; please retry /login",
                    "warn",
                )
                return

            form = {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": oauth_cfg.client_id,
            }
            if oauth_cfg.redirect_uri:
                form["redirect_uri"] = oauth_cfg.redirect_uri
            if verifier:
                form["code_verifier"] = verifier
            if quirk["token_include_state"] and pkce_state:
                form["state"] = pkce_state

            import httpx
            try:
                credential = run_sync(
                    self._exchange_token(
                        oauth_cfg.token_url,
                        form,
                        quirk["token_encoding"],
                    )
                )
            except Exception as exc:
                self.call_from_thread(self._write, f"login failed: {exc}", "warn")
                return

            # Store credential
            async def _set_cred(existing):
                return credential

            try:
                run_sync(models.credentials.modify(provider_id, _set_cred))
            except Exception as exc:
                self.call_from_thread(self._write, f"failed to store credential: {exc}", "warn")
                return

            self.call_from_thread(self._oauth_login_done, provider_id, info)

        @staticmethod
        async def _exchange_token(token_url, form, encoding):
            from rickshaw_ai.auth.oauth import _credential_from_token
            async with httpx.AsyncClient(timeout=30) as http:
                if encoding == "json":
                    resp = await http.post(token_url, json=form)
                else:
                    resp = await http.post(token_url, data=form)
                if resp.status_code != 200:
                    raise RuntimeError(f"token request rejected ({resp.status_code}): {resp.text}")
                return _credential_from_token(resp.json())

        @work(thread=True, exclusive=True, group="login")
        def _run_device_code_login(self, provider_id, models):
            """Run device-code login (e.g. GitHub Copilot)."""
            def show_user_code(code: str, uri: str) -> None:
                msg = f"enter code {code}"
                if uri:
                    msg += f" at {uri}"
                self.call_from_thread(self._write, msg, "meta")

            try:
                run_sync(models.login(
                    provider_id,
                    show_user_code=show_user_code,
                ))
            except Exception as exc:
                self.call_from_thread(self._write, f"login failed: {exc}", "warn")
                return

            info = models.provider_info(provider_id)
            self.call_from_thread(self._oauth_login_done, provider_id, info)

        def _oauth_login_done(self, provider_id, info) -> None:
            """Finalize after successful OAuth login: switch to the provider."""
            self._write(f"authenticated · {provider_id}", "meta")

            # Ensure the provider has a profile in cfg
            if provider_id not in self.cfg.providers and info:
                wf = "openai"
                if info.protocol == "anthropic":
                    wf = "anthropic"
                elif info.protocol in ("openai", "openai_compatible"):
                    wf = "openai"
                default_model = info.models[0].model if info.models else ""
                self.cfg.providers[provider_id] = ProviderProfile(
                    base_url=info.base_url,
                    model=default_model,
                    api_key_env=info.env_keys[0] if info.env_keys else "",
                    wire_format=wf,
                )

            # Build the provider and switch
            profile = self.cfg.providers.get(provider_id)
            if profile is None:
                self._write(f"no profile for {provider_id}; cannot build provider", "warn")
                return
            try:
                new_provider = build_provider_from_profile(
                    provider_id, profile,
                    embedding_model=self.cfg.openai_embedding_model,
                )
            except Exception as exc:
                logger.exception("Failed to build provider after OAuth login")
                self._write(f"Cannot switch: {exc}", "warn")
                return

            self.provider = new_provider
            self.orchestrator.provider = new_provider

            settings = load_settings()
            settings["provider"] = provider_id
            save_settings(settings)

            model = getattr(new_provider, "_model", "") or provider_id
            self._write(
                f"{provider_id} · {model} · effort "
                f"{self.orchestrator.effort.value}",
                "meta",
            )

        def _cmd_login(self) -> None:
            """Authenticate (or re-authenticate) the active provider via OAuth."""
            if self.provider is None:
                self._write("No provider selected. Use /settings first.", "warn")
                return
            provider_id = self.provider.name
            info = _builtin_provider_info(provider_id)
            if info is None or info.oauth is None:
                self._write(
                    f"{provider_id} does not support OAuth login. "
                    f"Set the API key via its env var instead.",
                    "warn",
                )
                return
            self._start_oauth_login(provider_id, info)

        # ---- turn execution --------------------------------------------

        def _start_turn(self, text: str) -> None:
            self._turn_active = True
            if self._has_turns:
                self.query_one("#transcript", VerticalScroll).mount(Rule())
            self._has_turns = True
            self._write(f"{_USER_MARK} {text}", "u")
            self._begin_assistant()
            self._set_hint("thinking…  ·  esc to interrupt")
            self.query_one("#prompt", PromptArea).disabled = True
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
                self._write(
                    "\u26a0 DEGRADED: provider unreachable \u2014 showing local memory only",
                    "degraded-banner",
                )
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
            prompt = self.query_one("#prompt", PromptArea)
            prompt.disabled = False
            prompt.focus()

        # ---- actions ----------------------------------------------------

        def action_interrupt(self) -> None:
            if self._login_state is not None:
                self._login_state = None
                self._write("(cancelled)", "warn")
                self._set_hint(_DEFAULT_HINT)
                return
            if self._settings_state is not None:
                self._settings_state = None
                self._write("(cancelled)", "warn")
                self._set_hint(_DEFAULT_HINT)
                return
            if self._provider_add_state is not None:
                self._provider_add_state = None
                self._write("(cancelled)", "warn")
                self._set_hint(_DEFAULT_HINT)
                return
            if self._turn_active:
                self.workers.cancel_group(self, "turn")
                self._write("(interrupted)", "warn")
                self._finish_turn()
                return
            prompt = self.query_one("#prompt", PromptArea)
            if prompt.text:
                prompt.text = ""

        def action_clear(self) -> None:
            self.query_one("#transcript", VerticalScroll).remove_children()
            self._has_turns = False
            self._write("cleared.", "meta")

    return RickshawTUI()


def _run_app(
    orchestrator: Orchestrator,
    provider: LLMProvider | None,
    effort: Effort,
    cfg: RickshawConfig,
) -> None:
    """Build and run the Textual app (separated out for testability)."""
    make_app(orchestrator, provider, effort, cfg).run()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)

    if args.oauth_url:
        models = _builtin_models(credentials=FileCredentialStore(_bridge.credential_store_path()))
        try:
            url, _, _ = _build_auth_code_authorize_url(args.oauth_url, models)
        except Exception as exc:
            print(f"{exc}", file=sys.stderr)
            sys.exit(1)
        print(url)
        sys.exit(0)

    cfg = load_config()

    settings = load_settings()
    effort = _EFFORT_NAMES.get(args.effort, cfg.effort) if args.effort else cfg.effort

    # Determine provider: explicit flag > env var > persisted setting > None.
    # A fresh install seeds settings.json with provider="" so the TUI prompts.
    provider_source: str | None
    if args.provider:
        provider_name: str | None = args.provider
        provider_source = "flag"
    elif os.environ.get("RICKSHAW_PROVIDER"):
        provider_name = os.environ["RICKSHAW_PROVIDER"]
        provider_source = "env"
    elif settings.get("provider"):
        provider_name = settings["provider"]
        provider_source = "settings"
    else:
        provider_name = None
        provider_source = None

    provider: LLMProvider | None = None
    if provider_name is not None:
        try:
            provider = _build_provider(provider_name, cfg)
            provider.validate()
        except Exception as exc:
            if args.validate_only:
                print(
                    f"Provider validation failed ({provider_name}): {exc}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if provider is None:
                if provider_source == "settings":
                    settings["provider"] = ""
                    save_settings(settings)
                else:
                    print(
                        f"Could not use provider {provider_name!r}: {exc}. "
                        "Launching provider picker.",
                        file=sys.stderr,
                    )
            elif args.allow_unvalidated:
                print(
                    f"Provider validation failed ({provider_name}): {exc}",
                    file=sys.stderr,
                )
                print(
                    "--allow-unvalidated set; continuing anyway — calls may fail.\n",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Could not use provider {provider_name!r}: {exc}. "
                    "Launching provider picker.",
                    file=sys.stderr,
                )
                provider = None

        if args.validate_only:
            if provider is None:
                sys.exit(1)
            print(f"Provider {provider_name!r} validated successfully.")
            return
    elif args.validate_only:
        print("No provider specified; nothing to validate.", file=sys.stderr)
        sys.exit(1)

    memory = MemoryService(db_path=args.db_path)
    orchestrator = Orchestrator(provider=provider, memory=memory, effort=effort)  # type: ignore[arg-type]

    _run_app(orchestrator, provider, effort, cfg)


if __name__ == "__main__":
    main()
