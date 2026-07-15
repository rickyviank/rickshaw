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
import time
import webbrowser
from urllib.parse import quote, urlencode

import httpx

from rickshaw.cli import _EFFORT_NAMES, _build_provider, load_config
from rickshaw.config import (
    ProviderProfile,
    RickshawConfig,
    is_local_url,
    local_no_models_hint,
    local_server_down_hint,
)
from rickshaw.memory.service import MemoryService
from rickshaw.history import append_history, load_history
from rickshaw.orchestrator import Orchestrator
from rickshaw.providers.base import Effort, LLMProvider
from rickshaw.providers.build import build_provider_from_profile
from rickshaw.providers.factory import get_provider
from rickshaw.prompt.builder import _estimate_tokens
from rickshaw.settings import load_settings, save_settings
from rickshaw.trace_store import TraceStore

from rickshaw import events
from rickshaw.trace_render import format_trace, TraceLine, TraceView

from rickshaw_ai._builtins import default_providers as _builtin_providers
from rickshaw_ai.credentials.store import FileCredentialStore
from rickshaw_ai.factory import builtin_models as _builtin_models
from rickshaw.providers import _bridge
from rickshaw.providers._bridge import run_sync

logger = logging.getLogger(__name__)

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
_ARG_COMMANDS = {"/effort", "/model"}
_EFFORT_VALUES = ["low", "medium", "high"]

STATUS_BAR_VOCABULARY = ("provider", "model", "effort", "context", "tokens", "price")
STATUS_BAR_DEFAULT_SEGMENTS = [
    "provider",
    "model",
    "effort",
    "context",
    "tokens",
    "price",
]
STATUS_BAR_KEEP_ALWAYS = {"provider", "model", "effort"}
STATUS_BAR_DROP_ORDER = ("price", "tokens", "context")
STATUS_BAR_NARROW_WIDTH = 80

_DEFAULT_HINT = "/help  ·  ^o expand trace  ·  ctrl+up/down navigate  ·  esc interrupt  ·  ^c quit"
_TRACE_HINT = "r raw  ·  tab expand event  ·  esc return to prompt"

# Map formatter color class names to Textual markup styles.
_TRACE_STYLE_MAP = {
    "trace-context": "$rk-assistant",
    "trace-tool": "$rk-accent",
    "trace-llm": "$rk-success",
    "trace-answer": "$rk-text",
    "trace-thinking": "$rk-meta",
    "trace-error": "$rk-error",
    "trace-retry": "$rk-warn",
    "trace-memory": "$rk-success",
    "trace-job": "$rk-meta",
    "trace-prompt": "$rk-assistant",
    "trace-done": "$rk-success",
}


def _style_for(color_class: str) -> str:
    """Return a Textual markup style string for a formatter color class."""
    if not color_class:
        return ""
    if color_class.startswith(("$", "#")) or color_class in (
        "dim", "bold", "italic", "reverse",
    ):
        return color_class
    return _TRACE_STYLE_MAP.get(color_class, "$rk-text")


_USER_MARK = "[#e0a86b]›[/]"  # amber angle-quote before each user message
_PROMPT_GLYPH = "›"
_ASSISTANT_LABEL = "o--o [dim]rickshaw[/]"
_SPINNER_FRAMES = "|/-\\"


def _status_segment_value(
    name: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    model_info: object | None = None,
    context_tokens: int = 0,
    session_tokens: int | None = 0,
    session_cost: float | None = None,
    warnings: list[str] | None = None,
) -> str:
    def _warn(msg: str) -> None:
        if warnings is not None and msg not in warnings:
            warnings.append(msg)

    if name == "provider":
        return provider or "—"
    if name == "model":
        return model or "—"
    if name == "effort":
        return effort or "—"
    if name == "context":
        window = getattr(model_info, "context_window", 0) or 0
        if window <= 0:
            _warn("context window unknown for the active model")
            return "—"
        pct = round(100 * max(context_tokens, 0) / window)
        return f"{pct}%"
    if name == "tokens":
        return "—" if session_tokens is None else f"{session_tokens} tok"
    if name == "price":
        pricing = getattr(model_info, "pricing", None)
        in_rate = getattr(pricing, "input", 0.0) or 0.0
        out_rate = getattr(pricing, "output", 0.0) or 0.0
        if in_rate <= 0 and out_rate <= 0:
            _warn("pricing unknown for the active model")
            return "—"
        return f"${(session_cost or 0.0):.4f}"
    _warn(f"unknown status-bar segment: {name!r}")
    return "—"

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
            timeout=profile.timeout,
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


def _find_model_info(provider_id: str, model: str):
    """Return the ModelInfo for provider_id/model from the builtins, or None."""
    for p in _builtin_providers():
        if p.id == provider_id:
            for m in p.models:
                if m.model == model:
                    return m
    return None


def _is_connection_error(exc: Exception) -> bool:
    """Whether *exc* looks like a failure to reach the endpoint at all."""
    if isinstance(exc, (httpx.TransportError, ConnectionError)):
        return True
    msg = str(exc).lower()
    return "unreachable" in msg or "connect" in msg or "refused" in msg


def _local_hint_message(
    name: str, profile: ProviderProfile | None, exc: Exception
) -> str:
    """Return ``str(exc)`` with an actionable hint appended for local endpoints.

    "no models" failures get the per-server model hint; connection failures get
    the "is <server> running?" hint (PRD local-providers §2.2/§2.5). The base
    URL is added when the underlying message doesn't already include it.
    Non-local profiles pass through unchanged.
    """
    message = str(exc)
    if profile is None or not profile.is_local_endpoint():
        return message
    if "no models" in message.lower():
        return f"{message} — {local_no_models_hint(name)}"
    if _is_connection_error(exc):
        hint = local_server_down_hint(name)
        if profile.base_url and profile.base_url not in message:
            return f"{message} — {name} unreachable at {profile.base_url} — {hint}"
        return f"{message} — {hint}"
    return message


def _local_turn_hint(name: str, exc: Exception) -> str:
    """Actionable suffix for a failed turn against a local endpoint (J6/J10)."""
    msg = str(exc).lower()
    if not isinstance(exc, httpx.ConnectTimeout) and (
        isinstance(exc, httpx.TimeoutException)
        or "timed out" in msg
        or "timeout" in msg
    ):
        return f"increase providers.{name}.timeout in ~/.rickshaw/settings.json"
    if _is_connection_error(exc):
        return local_server_down_hint(name)
    return ""


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
    trace_store: TraceStore | None = None,
):
    """Build the Textual app instance. Imports Textual lazily.

    *provider* may be ``None`` when launching without a pre-selected provider.
    The TUI then shows an interactive picker on mount.

    *trace_store* may be provided by the caller (``main()`` uses the persistent
    database path); otherwise a transient ``:memory:`` store is created.

    Kept as a factory (rather than a module-level class) so importing this
    module does not require Textual to be installed.
    """
    try:
        from textual import work
        from textual.app import App, ComposeResult
        from textual.binding import Binding
        from textual.css.query import NoMatches
        from textual.containers import Horizontal, Vertical, VerticalScroll
        from textual.message import Message
        from textual.suggester import SuggestFromList
        from textual.widgets import Markdown, Rule, Static, TextArea
    except ImportError as exc:  # pragma: no cover - exercised via message text
        raise SystemExit(_TEXTUAL_MISSING_MSG) from exc

    cfg = cfg or RickshawConfig()

    # Resolve the trace store: explicit arg > orchestrator's store > in-memory.
    if trace_store is None:
        trace_store = getattr(orchestrator, "trace_store", None) or getattr(
            orchestrator, "_trace_store", None
        )
    if trace_store is None:
        trace_store = TraceStore(":memory:")
    orchestrator.trace_store = trace_store
    orchestrator._trace_store = trace_store

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
            app = self.app
            if event.key == "tab":
                if app._menu_open and app._menu_items:
                    if app._menu_accept(via_enter=False):
                        event.prevent_default()
                        event.stop()
                        return
                if (
                    app._login_state is None
                    and app._settings_state is None
                    and app._provider_add_state is None
                    and app._turns
                ):
                    app.focus_trace()
                    event.prevent_default()
                    event.stop()
                    return
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

    class TraceLineWidget(Vertical):
        """A single focusable line inside an expanded trace block."""

        can_focus = True

        def __init__(
            self,
            trace_block: "TraceBlock",
            line: TraceLine,
            index: int,
            max_content_height: int,
        ) -> None:
            super().__init__(classes="trace-line")
            self.trace_block = trace_block
            self.line = line
            self.index = index
            self.max_content_height = max_content_height
            self._summary_widget: Static | None = None
            self._content_widget: Static | None = None
            self._expanded = line.content is not None and not line.is_capped

        def compose(self) -> ComposeResult:
            from rich.markup import escape

            yield Static(self._line_text(), classes="trace-line-summary", markup=True)

        def on_mount(self) -> None:
            self._summary_widget = self.query_one(".trace-line-summary", Static)
            self._refresh()

        def _line_text(self) -> str:
            from rich.markup import escape

            ts = self.line.timestamp or ""
            style = _style_for(self.line.color_class)
            if self.line.label:
                label = escape(self.line.label)
                label_markup = f"[{style}]{label}[/]" if style else label
            else:
                label_markup = ""
            text = f"{ts} {label_markup}".rstrip()

            marker = ""
            if self.line.expandable and not self.trace_block._raw_mode:
                marker = "[-]" if self._expanded else "[+]"
                text += f" [dim]{marker}[/]"

            if self.trace_block._raw_mode:
                body = self.line.raw_json or self.line.summary or ""
            else:
                body = self.line.summary
            if body:
                text += f" {escape(body)}"
            return text

        def _ensure_content_widget(self) -> None:
            if self._content_widget is not None:
                return
            if self.trace_block._raw_mode:
                return
            if self.line.content is not None:
                text = self.line.content
                classes = "trace-content"
            elif self.line.raw_json:
                text = self.line.raw_json
                classes = "trace-content trace-raw"
            else:
                return
            self._content_widget = Static(
                text, markup=False, classes=classes
            )
            if self.line.is_capped and self.line.content is not None:
                self._content_widget.styles.max_height = self.max_content_height
                self._content_widget.styles.overflow_y = "scroll"
            self.mount(self._content_widget)
            self._content_widget.display = False

        def _refresh(self) -> None:
            if self._summary_widget is not None:
                self._summary_widget.update(self._line_text())
            if self._expanded and not self.trace_block._raw_mode:
                self._ensure_content_widget()
                if self._content_widget is not None:
                    self._content_widget.display = True
            elif self._content_widget is not None:
                self._content_widget.display = False

        def toggle_expand(self) -> None:
            if self.trace_block._raw_mode or not self.line.expandable:
                return
            self._expanded = not self._expanded
            self._refresh()

        def set_raw_mode(self, raw_mode: bool) -> None:
            self._expanded = (
                not raw_mode
                and self.line.content is not None
                and not self.line.is_capped
            )
            if self._content_widget is not None:
                self._content_widget.remove()
                self._content_widget = None
            self._refresh()

        def on_key(self, event) -> None:
            if event.key == "up":
                self.trace_block.focus_line(self.index - 1)
                event.stop()
                event.prevent_default()
            elif event.key == "down":
                self.trace_block.focus_line(self.index + 1)
                event.stop()
                event.prevent_default()
            elif event.key == "enter":
                self.toggle_expand()
                event.stop()
                event.prevent_default()
            elif event.key in ("r", "R"):
                self.trace_block.toggle_raw_mode()
                event.stop()
                event.prevent_default()
            elif event.key == "escape":
                self.trace_block.focus_prompt()
                event.stop()
                event.prevent_default()

        def on_focus(self, event) -> None:
            if hasattr(self.trace_block, "_on_line_focus_changed"):
                self.trace_block._on_line_focus_changed()

        def on_blur(self, event) -> None:
            if hasattr(self.trace_block, "_on_line_focus_changed"):
                self.trace_block._on_line_focus_changed()

    class TraceBlock(Vertical):
        """Collapsible per-turn trace summary/details block."""

        def __init__(
            self,
            event_records: list[tuple[events.TurnEvent, float]],
            turn_id: str,
            duration: float,
            status: str,
            task_input: str = "",
            provider: str = "",
            model: str = "",
            width: int = 80,
            height: int = 24,
        ) -> None:
            super().__init__(classes="trace-block")
            self.event_records = event_records
            self.turn_id = turn_id
            self.duration = duration
            self.status = status
            self.task_input = task_input
            self.provider = provider
            self.model = model
            self._width = width
            self._height = height
            self._expanded = False
            self._raw_mode = False
            self.view: TraceView | None = None
            self._line_widgets: list[TraceLineWidget] = []

        def _build_view(self) -> None:
            if self.view is not None:
                return
            self.view = format_trace(
                self.event_records,
                task_input=self.task_input,
                provider=self.provider,
                model=self.model,
                status=self.status,
                duration=self.duration,
                width=self._width,
                height=self._height,
            )

        def compose(self) -> ComposeResult:
            from rich.markup import escape

            self._build_view()
            yield Static(escape(self._summary_text()), classes="summary")
            yield Vertical(classes="details")

        def on_mount(self) -> None:
            self.summary = self.query_one(".summary", Static)
            self.details = self.query_one(".details", Vertical)
            self._render_details()

        def _summary_text(self) -> str:
            if self.view is not None and self.view.summary:
                return self.view.summary
            n = len(self.event_records)
            return f"{n} events · {self.status} · {self.duration:.1f}s"

        def _render_details(self) -> None:
            if self.details is None or self.view is None:
                return
            self.details.remove_children()
            self._line_widgets.clear()
            for header in self.view.header_lines:
                self.details.mount(Static(header, classes="trace-header"))
            app_height = getattr(self.app, "size", None)
            height = app_height.height if app_height is not None else self._height
            max_h = max(5, int(height * 0.3))
            for index, line in enumerate(self.view.lines):
                widget = TraceLineWidget(self, line, index, max_h)
                self.details.mount(widget)
                self._line_widgets.append(widget)

        def toggle(self) -> None:
            from rich.markup import escape

            self._expanded = not self._expanded
            if self._expanded:
                self.details.display = True
                self.add_class("expanded")
            else:
                focused = self.app.focused
                if isinstance(focused, TraceLineWidget) and focused.trace_block is self:
                    self.focus_prompt()
                self.details.display = False
                self.remove_class("expanded")
            self.summary.update(escape(self._summary_text()))
            if hasattr(self.app, "_update_trace_hint"):
                self.app._update_trace_hint()

        def expand(self) -> None:
            if not self._expanded:
                self.toggle()

        def collapse(self) -> None:
            if self._expanded:
                self.toggle()

        def focus_first_line(self) -> None:
            if self._line_widgets:
                line = self._line_widgets[0]
                line.focus(scroll_visible=False)
                line.scroll_visible(animate=False, immediate=True)

        def focus_line(self, index: int) -> None:
            if not self._line_widgets:
                return
            index = max(0, min(index, len(self._line_widgets) - 1))
            line = self._line_widgets[index]
            line.focus(scroll_visible=False)
            line.scroll_visible(animate=False, immediate=True)

        def focus_prompt(self) -> None:
            try:
                self.app.query_one("#prompt", PromptArea).focus()
            except Exception:
                pass

        def toggle_raw_mode(self) -> None:
            self._raw_mode = not self._raw_mode
            for widget in self._line_widgets:
                widget.set_raw_mode(self._raw_mode)

        def _on_line_focus_changed(self) -> None:
            if hasattr(self.app, "_update_trace_hint"):
                self.app._update_trace_hint()

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

          Screen { layout: vertical; background: $rk-bg; }
          #welcome {
              height: auto;
              width: 1fr;
              background: $rk-surface;
              color: $rk-text;
              border: round $rk-border;
              padding: 0 2;
              margin: 0 0 1 0;
          }
          #welcome.compact { padding: 0 1; }
          #transcript {
              height: 1fr;
              padding: 1 3;
              scrollbar-size: 1 1;
              scrollbar-color: #2a2e37;
              scrollbar-color-hover: #3a3f47;
              scrollbar-color-active: #3a3f47;
              scrollbar-background: $rk-bg;
          }
          #transcript > Static { margin: 0 0 1 0; }
          #transcript > Markdown {
              margin: 0 0 1 0;
              padding: 0;
              background: transparent;
          }
          #transcript > Rule { color: $rk-border; margin: 0 0 1 0; }
          #transcript > Markdown.assistant { padding: 0 0 0 2; }
          .u { color: $rk-text; }
          .a { color: $rk-meta; }
          .a-label { color: $rk-assistant; }
          .meta { color: $rk-meta; }
          .warn { color: $rk-warn; }
          #turn-indicator { color: $rk-meta; height: auto; }
          .degraded-banner {
              color: #1a1a1a;
              background: $rk-error;
              text-style: bold;
              padding: 0 1;
          }
          #hint { height: 1; color: #3a3f47; padding: 0 3 1 3; }
          #slashmenu {
              display: none;
              background: $rk-surface;
              color: $rk-meta;
              border: round $rk-border;
              padding: 0 1;
              margin: 0 3;
              height: auto;
          }
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
          #statusbar {
              dock: bottom;
              height: 1;
              background: $rk-surface;
              color: $rk-meta;
              padding: 0 3;
          }
          .trace-block {
              background: $rk-surface;
              border: round $rk-border;
              padding: 0 1;
              margin: 0 0 1 0;
              height: auto;
          }
          .trace-block .summary {
              color: $rk-meta;
              height: auto;
          }
          .trace-block .details {
              color: $rk-text;
              height: auto;
              display: none;
          }
          .trace-block.expanded .details {
              display: block;
          }
          .u.selected {
              background: #2a2f36;
          }
          .trace-block.selected {
              border: tall $rk-accent;
          }
          .trace-line { height: auto; color: $rk-text; }
          .trace-line:focus { background: $rk-border; }
          .trace-line-summary { height: auto; width: 1fr; }
          .trace-header { color: $rk-meta; height: auto; }
          .trace-content { height: auto; width: 1fr; color: $rk-text; }
          .trace-raw { color: $rk-meta; }
          .trace-context { color: $rk-assistant; }
          .trace-tool { color: $rk-accent; }
          .trace-llm { color: $rk-success; }
          .trace-answer { color: $rk-text; }
          .trace-thinking { color: $rk-meta; }
          .trace-error { color: $rk-error; }
          .trace-retry { color: $rk-warn; }
          .trace-memory { color: $rk-success; }
          .trace-job { color: $rk-meta; }
          .trace-prompt { color: $rk-assistant; }
          .trace-done { color: $rk-success; }
        """

        BINDINGS = [
            Binding("escape", "interrupt", "Interrupt", show=False),
            Binding("ctrl+l", "clear", "Clear", show=False),
            Binding("ctrl+c", "ctrl_c", "Quit", show=False, priority=True),
            Binding("ctrl+up", "prev_turn", "Prev turn", show=False),
            Binding("ctrl+down", "next_turn", "Next turn", show=False),
            Binding("ctrl+o", "toggle_trace", "Toggle trace", show=False),
            Binding("r", "toggle_trace_raw", "Toggle raw trace", show=False),
        ]

        def __init__(self, trace_store: TraceStore | None = None) -> None:
            super().__init__()
            self.orchestrator = orchestrator
            self.provider = provider
            self.effort = effort
            self.cfg = cfg
            self.trace_store = trace_store
            self.orchestrator.effort = effort
            self._history: list[str] = load_history()
            self._history_pos: int = len(self._history)
            self._buffer = ""
            self._current_md: Markdown | None = None
            self._turn_active = False
            self._has_turns = False
            self._provider_add_state: dict | None = None
            self._settings_state: dict | None = None
            self._login_state: dict | None = None
            self._active_profile_name: str | None = None
            self._effort_note_shown: set[str] = set()
            self._turn_seq = 0
            self._indicator: Static | None = None
            self._indicator_timer = None
            self._turn_started = 0.0
            self._turn_input: str = ""
            self._spinner_idx = 0
            self._live_tokens = 0
            self._first_token = False
            self._phase_label = "Thinking…"
            self._current_events: list[tuple[events.TurnEvent, float]] = []
            self._current_turn_id: str | None = None
            self._current_user: Static | None = None
            self._turns: list[dict] = []
            self._selected_turn_index = -1
            self._ctrl_c_pending = False
            self._ctrl_c_timer = None
            self._session_tokens = 0
            self._session_tool_calls = 0
            self._session_cost = 0.0
            self._last_ctx_tokens = 0
            self._ctx_window_warned = False
            self._last_usage = None
            self._menu_open = False
            self._menu_mode = "command"
            self._menu_items: list[tuple[str, str]] = []
            self._menu_index = 0
            self._menu_arg_cmd = ""

        # ---- layout -----------------------------------------------------

        def compose(self) -> ComposeResult:
            yield VerticalScroll(id="transcript")
            yield Static("", id="slashmenu")
            yield Static(_DEFAULT_HINT, id="hint")
            with Horizontal(id="prompt-box"):
                yield Static(_PROMPT_GLYPH, id="prompt-glyph")
                yield PromptArea(id="prompt")
            yield Static("", id="statusbar")

        def on_mount(self) -> None:
            self._render_welcome()
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
            self._update_status_bar()

        # ---- welcome panel ----------------------------------------------

        def _welcome_text(self, compact: bool) -> str:
            """Rich-markup body for the welcome panel (D2)."""
            logo = f"[$rk-assistant]{RICKSHAW_LOGO}[/]"
            if self.provider is None:
                prov = "provider: (none)"
                prov_markup = f"[$rk-meta]{prov}[/]"
            else:
                model = getattr(self.provider, "_model", "") or self.provider.name
                prov = (
                    f"{self.provider.name} \u00b7 {model} \u00b7 effort "
                    f"{self.orchestrator.effort.value}"
                )
                prov_markup = f"[$rk-accent]{prov}[/]"
            if compact:
                return (
                    f"{logo}  [$rk-meta]\u00b7 {RICKSHAW_SLOGAN}[/]\n"
                    f"{prov_markup}  [$rk-meta]\u00b7  /help[/]"
                )
            cwd = os.getcwd()
            return (
                f"{logo}\n"
                f"[$rk-meta]{RICKSHAW_SLOGAN}[/]\n"
                f"\n"
                f"{prov_markup}\n"
                f"[$rk-meta]cwd:[/] {cwd}\n"
                f"\n"
                f"[$rk-meta]/help  \u00b7  esc interrupt  \u00b7  ^c quit[/]"
            )

        def _render_welcome(self) -> None:
            """Mount the welcome panel at the top of the transcript (launch/clear)."""
            width = self.size.width
            compact = bool(width) and width < 80
            panel = Static(self._welcome_text(compact=compact), id="welcome")
            self.query_one("#transcript", VerticalScroll).mount(panel)
            self._scroll_end()

        def _apply_responsive_welcome(self, width: int | None = None) -> None:
            try:
                welcome = self.query_one("#welcome", Static)
            except NoMatches:
                return
            width = width if width is not None else self.size.width
            welcome.update(self._welcome_text(compact=bool(width) and width < 80))

        def on_resize(self, event) -> None:
            self._update_status_bar(event.size.width)
            self._apply_responsive_welcome(event.size.width)

        # ---- on-launch provider picker ---------------------------------

        def _start_provider_picker(self) -> None:
            """Display the provider picker (builtins + configured)."""
            self._close_menu()
            builtin_names = _get_builtin_provider_names()
            configured_names = sorted(self.cfg.providers)
            all_names = sorted(set(builtin_names) | set(configured_names))
            if not all_names:
                self._write("No providers available.", "warn")
                return
            self._write("", "meta")
            self._write("  Pick a provider (enter name, Esc to cancel):", "meta")
            current = (
                self._active_profile_name or self.provider.name
            ) if self.provider else ""
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
            transcript = self.query_one("#transcript", VerticalScroll)
            label = Static(_ASSISTANT_LABEL, classes="a-label")
            md = Markdown("", classes="assistant")
            if self._indicator is not None and self._indicator.parent is not None:
                transcript.mount(label, before=self._indicator)
                transcript.mount(md, before=self._indicator)
            else:
                transcript.mount(label)
                transcript.mount(md)
            self._current_md = md
            self._scroll_end()

        def _scroll_end(self) -> None:
            self.query_one("#transcript", VerticalScroll).scroll_end(animate=False)

        def _set_hint(self, text: str) -> None:
            self.query_one("#hint", Static).update(text)

        # ---- status bar ----------------------------------------------

        def _active_model_info(self):
            """Return the active model info (or provider wrapper) if available."""
            if self.provider is None:
                return None
            model = getattr(self.provider, "_model", "") or ""
            info = _find_model_info(self.provider.name, model)
            if info is None:
                return None
            if hasattr(info, "context_window") or hasattr(info, "pricing"):
                return info
            models = getattr(info, "models", None) or []
            for model_info in models:
                if getattr(model_info, "model", None) == model or getattr(
                    model_info, "id", None
                ) == model:
                    return model_info
            return models[0] if models else info

        def _context_segment(self) -> str:
            info = self._active_model_info()
            window = getattr(info, "context_window", 0) if info is not None else 0
            if not window:
                if not self._ctx_window_warned:
                    logger.warning(
                        "context_window unavailable for active model; rendering '—'"
                    )
                    self._ctx_window_warned = True
                return "—"
            pct = (self._last_ctx_tokens / window) * 100
            return f"{pct:.0f}%"

        def _price_segment(self) -> str:
            info = self._active_model_info()
            if info is None or not getattr(info, "pricing", None):
                return "—"
            return f"~${self._session_cost:.4f}"

        def _status_segment(self, name: str) -> str:
            info = self._active_model_info()
            return _status_segment_value(
                name,
                provider=(
                    self._active_profile_name or self.provider.name
                ) if self.provider else None,
                model=getattr(self.provider, "_model", "") or (
                    self.provider.name if self.provider else None
                ),
                effort=getattr(self.orchestrator.effort, "value", None),
                model_info=info,
                context_tokens=self._last_ctx_tokens,
                session_tokens=self._session_tokens,
                session_cost=self._session_cost,
            )

        def _update_status_bar(self, width: int | None = None) -> None:
            try:
                bar = self.query_one("#statusbar", Static)
            except Exception:
                return
            segments = [s for s in (self.cfg.status_bar or []) if s in STATUS_BAR_VOCABULARY]
            if not segments:
                bar.update("")
                return

            width = width if width is not None else getattr(self.size, "width", 0) or 0
            visible = list(segments)
            if width > 0:
                content_width = max(0, width - 2)
                while True:
                    text = " | ".join(self._status_segment(name) for name in visible)
                    if len(text) <= content_width:
                        break
                    dropped = False
                    for name in STATUS_BAR_DROP_ORDER:
                        if name in visible and name not in STATUS_BAR_KEEP_ALWAYS:
                            visible.remove(name)
                            dropped = True
                            break
                    if not dropped:
                        break
            bar.update(" | ".join(self._status_segment(name) for name in visible))

        def _warn_missing_metadata(self, model_info: object | None) -> list[str]:
            warnings: list[str] = []
            for name in ("context", "price"):
                _status_segment_value(name, model_info=model_info, warnings=warnings)
            for msg in warnings:
                self._write(f"⚠ {msg}", "warn")
            return warnings

        def _render_menu(self) -> None:
            menu = self.query_one("#slashmenu", Static)
            if not self._menu_open or not self._menu_items:
                menu.update("")
                return

            from rich.markup import escape

            lines: list[str] = []
            items = self._menu_items
            for idx, (value, desc) in enumerate(items):
                value = escape(value)
                desc = escape(desc)
                if self._menu_mode == "command":
                    body = f"{value:<10} {desc}"
                else:
                    body = value
                if idx == self._menu_index:
                    lines.append(f"[#e0a86b]›[/] [reverse #e0a86b]{body}[/]")
                else:
                    lines.append(f"[#8b929c]  {body}[/]")
            menu.update("\n".join(lines))

        def _open_menu(
            self,
            mode: str,
            items: list[tuple[str, str]],
            arg_cmd: str = "",
        ) -> None:
            self._menu_open = True
            self._menu_mode = mode
            self._menu_items = items
            self._menu_arg_cmd = arg_cmd
            if self._menu_items:
                self._menu_index = min(self._menu_index, len(self._menu_items) - 1)
            else:
                self._menu_index = 0
            self.query_one("#slashmenu", Static).display = True
            self._render_menu()

        def _close_menu(self) -> None:
            self._menu_open = False
            self._menu_mode = "command"
            self._menu_items = []
            self._menu_index = 0
            self._menu_arg_cmd = ""
            menu = self.query_one("#slashmenu", Static)
            menu.update("")
            menu.display = False

        def _menu_accept(self, *, via_enter: bool) -> bool:
            if not self._menu_open or not self._menu_items:
                return False

            if self._menu_mode == "value":
                arg_cmd = self._menu_arg_cmd
                sel = self._menu_items[self._menu_index][0]
                self.query_one("#prompt", PromptArea).text = ""
                self._close_menu()
                if arg_cmd == "/effort":
                    self._cmd_effort(sel)
                elif arg_cmd == "/model":
                    self._cmd_model(sel)
                return True

            typed = self.query_one("#prompt", PromptArea).text.strip().lower()
            exact = typed in _COMMANDS
            if via_enter and exact:
                self._close_menu()
                return False

            cmd = self._menu_items[self._menu_index][0]
            if cmd in _ARG_COMMANDS:
                self.query_one("#prompt", PromptArea).text = f"{cmd} "
                self._close_menu()
                return True

            self.query_one("#prompt", PromptArea).text = ""
            self._close_menu()
            self._handle_command(cmd)
            return True
        # ---- input handling --------------------------------------------

        def _update_menu_from_prompt(self, text: str) -> None:
            if (
                self._login_state is not None
                or self._settings_state is not None
                or self._provider_add_state is not None
                or self._turn_active
            ):
                self._close_menu()
                return

            if not text.startswith("/"):
                self._close_menu()
                return

            if " " in text:
                cmd, _, rest = text.partition(" ")
                cmd = cmd.lower()
                if cmd in _ARG_COMMANDS:
                    if cmd == "/effort":
                        filter_text = rest.strip().lower()
                        items = [
                            (v, "") for v in _EFFORT_VALUES if v.startswith(filter_text)
                        ]
                    else:
                        if self.provider is None:
                            self._close_menu()
                            return
                        try:
                            models = self.provider.available_models()
                        except Exception:
                            self._close_menu()
                            return
                        filter_text = rest.strip()
                        items = [(m, "") for m in models if m.startswith(filter_text)]
                    if not items:
                        self._close_menu()
                        return
                    self._menu_index = 0
                    self._open_menu("value", items, arg_cmd=cmd)
                    return
                self._close_menu()
                return

            items = [
                (name, desc)
                for name, desc in _COMMANDS.items()
                if name.startswith(text.lower())
            ]
            if not items:
                self._close_menu()
                return
            self._menu_index = 0
            self._open_menu("command", items)

        def on_text_area_changed(self, event: TextArea.Changed) -> None:
            if getattr(event.text_area, "id", None) == "prompt":
                self._update_menu_from_prompt(event.text_area.text)

        def on_prompt_area_changed(self, event: TextArea.Changed) -> None:
            self.on_text_area_changed(event)

        def on_prompt_area_submitted(self, event: "PromptArea.Submitted") -> None:
            value = event.value.strip()
            prompt = self.query_one("#prompt", PromptArea)
            if self._menu_open and self._menu_accept_via_enter(value):
                return
            prompt.text = ""
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
            self._record_history(value)
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

        def _menu_accept_via_enter(self, typed: str) -> bool:
            if not self._menu_open:
                return False
            if self._menu_mode == "value":
                return self._menu_accept(via_enter=True)

            if typed.strip().lower() in _COMMANDS:
                self._close_menu()
                return False
            return self._menu_accept(via_enter=True)

        def on_key(self, event) -> None:
            if self._menu_open and self._menu_items:
                if event.key == "up":
                    self._menu_index = (self._menu_index - 1) % len(self._menu_items)
                    self._render_menu()
                    event.stop()
                    event.prevent_default()
                elif event.key == "down":
                    self._menu_index = (self._menu_index + 1) % len(self._menu_items)
                    self._render_menu()
                    event.stop()
                    event.prevent_default()
                elif event.key == "tab":
                    if self._menu_accept(via_enter=False):
                        event.stop()
                        event.prevent_default()
                elif event.key == "escape":
                    self._close_menu()
                    event.stop()
                    event.prevent_default()
                return

            if event.key not in ("up", "down"):
                return
            if not self._history_nav_allowed(event.key):
                return
            moved = self._history_prev() if event.key == "up" else self._history_next()
            if moved:
                event.prevent_default()
                event.stop()

        def _record_history(self, value: str) -> None:
            append_history(value)
            self._history.append(value)
            if len(self._history) > 1000:
                self._history = self._history[-1000:]
            self._history_pos = len(self._history)

        def _history_nav_allowed(self, direction: str) -> bool:
            if self._login_state is not None:
                return False
            if self._settings_state is not None:
                return False
            if self._provider_add_state is not None:
                return False
            if self._menu_open:
                return False
            prompt = self.query_one("#prompt", PromptArea)
            if not prompt.has_focus:
                return False
            return self._prompt_on_boundary_line(direction)

        def _prompt_on_boundary_line(self, direction: str) -> bool:
            prompt = self.query_one("#prompt", PromptArea)
            if hasattr(prompt, "document"):
                cursor_row = prompt.cursor_location[0]
                last_row = len(prompt.document.lines) - 1
                if direction == "up":
                    return cursor_row == 0
                return cursor_row == last_row
            return True

        def _set_prompt_text(self, text: str) -> None:
            prompt = self.query_one("#prompt", PromptArea)
            prompt.value = text
            if hasattr(prompt, "document"):
                try:
                    prompt.move_cursor(prompt.document.end)
                except AttributeError:
                    pass

        def _history_prev(self) -> bool:
            if self._history_pos <= 0:
                return False
            self._history_pos -= 1
            self._set_prompt_text(self._history[self._history_pos])
            return True

        def _history_next(self) -> bool:
            if self._history_pos >= len(self._history):
                return False
            self._history_pos += 1
            if self._history_pos == len(self._history):
                self._set_prompt_text("")
            else:
                self._set_prompt_text(self._history[self._history_pos])
            return True
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

        def _active_local_profile(self) -> tuple[str, ProviderProfile] | None:
            """Return ``(name, profile)`` when the active provider is local."""
            if self.provider is None:
                return None
            base_url = (getattr(self.provider, "_base_url", "") or "").rstrip("/")
            if not is_local_url(base_url):
                return None
            names = [self._active_profile_name] if self._active_profile_name else []
            names += sorted(self.cfg.providers)
            for name in names:
                profile = self.cfg.providers.get(name)
                if profile is not None and profile.base_url.rstrip("/") == base_url:
                    return name, profile
            return None

        def _exc_with_local_hint(self, exc: Exception) -> str:
            """Failure text for the active provider, hint-enriched when local."""
            local = self._active_local_profile()
            if local is None:
                return str(exc)
            return _local_hint_message(local[0], local[1], exc)

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
            self._write(f"effort · {new_effort.value}", "meta")
            self._update_status_bar()

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
                    self._write(
                        f"Cannot list models: {self._exc_with_local_hint(exc)}",
                        "warn",
                    )
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
                self._write(
                    f"Cannot validate model: {self._exc_with_local_hint(exc)}",
                    "warn",
                )
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
            self._write(f"model · {arg}", "meta")
            self._update_status_bar()

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
                self._write(
                    f"Cannot list models: {self._exc_with_local_hint(exc)}",
                    "warn",
                )
                return
            self._write("  available models:", "meta")
            for m in models:
                marker = "\u2666" if m == model else " "
                self._write(f"    {m:<32} {marker}", "meta")

        def _cmd_settings(self) -> None:
            """Interactive provider/model picker with settings header."""
            self._close_menu()
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
                    if profile.is_local_endpoint():
                        temp.validate()
                    models = temp.available_models()
                except Exception as exc:
                    logger.exception("Failed to list models for provider %r", chosen)
                    self._write(
                        f"Cannot list models for {chosen}: "
                        f"{_local_hint_message(chosen, profile, exc)}",
                        "warn",
                    )
                    self._settings_state = None
                    self._set_hint(_DEFAULT_HINT)
                    return

                if not models:
                    msg = f"No models available for {chosen}."
                    if profile.is_local_endpoint():
                        msg = f"{msg} — {local_no_models_hint(chosen)}"
                    self._write(msg, "warn")
                    self._settings_state = None
                    self._set_hint(_DEFAULT_HINT)
                    return

                # Local model resolution (D4): re-verify the persisted model
                # and adopt silently when the server offers exactly one.
                if profile.is_local_endpoint():
                    if profile.model and profile.model not in models:
                        self._write(
                            f"model {profile.model!r} is no longer available "
                            f"on {chosen}",
                            "meta",
                        )
                    if len(models) == 1:
                        self._write(f"  model: {models[0]}", "meta")
                        self._settings_apply(chosen, models[0])
                        self._settings_state = None
                        self._set_hint(_DEFAULT_HINT)
                        return

                current_model = (
                    getattr(self.provider, "_model", "")
                    if self.provider
                    and chosen == (self._active_profile_name or self.provider.name)
                    else ""
                )
                self._start_model_picker(chosen, models, current_model)

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

        def _start_model_picker(
            self, chosen: str, models: list[str], current_model: str,
        ) -> None:
            """Show the model-picker step (shared by /settings and /provider)."""
            self._write("", "meta")
            self._write("  Pick a model (enter name, Esc to cancel):", "meta")
            for m in models:
                marker = "\u2666" if m == current_model else " "
                self._write(f"    {m:<32} {marker}", "meta")
            self._settings_state = {
                "step": "model",
                "chosen_provider": chosen,
                "valid_models": models,
            }
            self._set_hint("model name (Enter to submit, Esc to cancel)")

        def _settings_apply(self, provider_name: str, model_name: str) -> None:
            """Apply provider + model selection from /settings wizard."""
            profile = self.cfg.providers[provider_name]
            try:
                new_provider = _rebuild_provider(
                    provider_name, self.cfg, model_name,
                )
            except Exception as exc:
                logger.exception("Failed to switch provider/model via /settings")
                self._write(
                    f"Cannot switch: {_local_hint_message(provider_name, profile, exc)}",
                    "warn",
                )
                return
            self.provider = new_provider
            self.orchestrator.provider = new_provider
            self._active_profile_name = provider_name

            # Effort reconciliation.
            caps = new_provider.capabilities()
            old_effort = self.orchestrator.effort
            if profile.is_local_endpoint():
                self._note_local_effort(provider_name)
            elif caps.effort_levels and old_effort not in caps.effort_levels:
                default_effort = Effort.MEDIUM
                self.orchestrator.effort = default_effort
                self.effort = default_effort
                self._write(
                    f"note: {provider_name} does not support effort "
                    f"{old_effort.value}. Reset to medium.",
                    "warn",
                )

            if profile.is_local_endpoint():
                self._persist_local_model(provider_name, model_name)
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
            self._write(
                f"{provider_name} · {model_name} · effort "
                f"{self.orchestrator.effort.value}",
                "meta",
            )
            self.query_one("#prompt", PromptArea).focus()
            self._update_status_bar()

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
            active = self._active_profile_name or self.provider.name
            model = getattr(self.provider, "_model", "") or self.provider.name
            self._write(
                f"current \u00b7 {active} ({model})", "meta",
            )
            self._write("", "meta")
            self._write("  available providers:", "meta")
            for name in sorted(self.cfg.providers):
                profile = self.cfg.providers[name]
                marker = "\u2666" if name == active else " "
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
                self._write(
                    f"Cannot switch provider: "
                    f"{_local_hint_message(name, profile, exc)}",
                    "warn",
                )
                return
            if profile.is_local_endpoint() and not self._resolve_local_model(
                name, profile, new_provider,
            ):
                return
            self.provider = new_provider
            self.orchestrator.provider = new_provider
            self._active_profile_name = name

            # Effort mismatch: reset to medium if unsupported.
            caps = new_provider.capabilities()
            old_effort = self.orchestrator.effort
            if profile.is_local_endpoint():
                self._note_local_effort(name)
            elif caps.effort_levels and old_effort not in caps.effort_levels:
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
            self._write(
                f"{name} · {model} · effort "
                f"{self.orchestrator.effort.value}",
                "meta",
            )
            self._update_status_bar()

        def _note_local_effort(self, name: str) -> None:
            """One-time quiet note: effort is a no-op for local providers (D5)."""
            if name in self._effort_note_shown:
                return
            self._effort_note_shown.add(name)
            self._write(
                f"effort is not applicable to local provider {name} "
                f"— using provider defaults",
                "meta",
            )

        def _resolve_local_model(
            self, name: str, profile: ProviderProfile, provider: LLMProvider,
        ) -> bool:
            """Resolve the model for a local endpoint before committing (D4).

            Returns ``True`` when *provider* is ready to activate. Returns
            ``False`` when activation failed (the previous provider stays
            active) or when the choice was handed to the model picker, which
            applies the switch itself.
            """
            try:
                provider.validate()
                models = provider.available_models()
            except Exception as exc:
                logger.exception("Failed to activate local provider %r", name)
                self._write(
                    f"Cannot switch provider: "
                    f"{_local_hint_message(name, profile, exc)}",
                    "warn",
                )
                return False
            if profile.model and profile.model in models:
                return True
            if profile.model:
                self._write(
                    f"model {profile.model!r} is no longer available on {name}",
                    "meta",
                )
            if not models:
                self._write(
                    f"Cannot switch provider: {name} lists no models — "
                    f"{local_no_models_hint(name)}",
                    "warn",
                )
                return False
            if len(models) == 1:
                provider._model = models[0]
                self._persist_local_model(name, models[0])
                self._write(f"model · {models[0]}", "meta")
                return True
            self._start_model_picker(name, models, "")
            return False

        def _persist_local_model(self, name: str, model: str) -> None:
            """Persist the resolved model for a local provider profile."""
            profile = self.cfg.providers[name]
            self.cfg.providers[name] = ProviderProfile(
                base_url=profile.base_url,
                model=model,
                api_key_env=profile.api_key_env,
                wire_format=profile.wire_format,
                timeout=profile.timeout,
            )
            settings = load_settings()
            entry = settings.setdefault("providers", {}).setdefault(name, {
                "base_url": profile.base_url,
                "api_key_env": profile.api_key_env,
                "wire_format": profile.wire_format,
            })
            entry["model"] = model
            save_settings(settings)

        def _cmd_provider_add_start(self) -> None:
            """Begin the interactive provider-registration wizard."""
            self._close_menu()
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

            # api_key_env is optional when the base URL is local (J7).
            key_optional = key == "api_key_env" and is_local_url(
                state["data"].get("base_url", "")
            )

            # Validate required fields.
            if not value and key != "wire_format" and not key_optional:
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
                next_key, next_prompt = _PROVIDER_ADD_STEPS[step_idx]
                if next_key == "api_key_env" and is_local_url(
                    state["data"].get("base_url", "")
                ):
                    self._set_hint(
                        f"{next_prompt}(optional for local — Enter to skip)"
                    )
                else:
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
            self._close_menu()
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
            self._active_profile_name = provider_id

            settings = load_settings()
            settings["provider"] = provider_id
            save_settings(settings)

            model = getattr(new_provider, "_model", "") or provider_id
            self._write(
                f"{provider_id} · {model} · effort "
                f"{self.orchestrator.effort.value}",
                "meta",
            )
            self._update_status_bar()

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
            self._close_menu()
            self._turn_active = True
            self._turn_seq += 1
            seq = self._turn_seq
            if self._has_turns:
                self.query_one("#transcript", VerticalScroll).mount(Rule())
            self._has_turns = True
            self._current_user = self._write(f"{_USER_MARK} {text}", "u")
            self._buffer = ""
            self._current_md = None
            self._first_token = False
            self._phase_label = "Thinking…"
            self._turn_input = text
            self._current_events = []
            self._current_turn_id = None
            self._live_tokens = 0
            self._spinner_idx = 0
            self._turn_started = time.monotonic()
            indicator = Static("", id="turn-indicator")
            self.query_one("#transcript", VerticalScroll).mount(indicator)
            self._indicator = indicator
            self._tick_indicator()
            self._indicator_timer = self.set_interval(1 / 8, self._tick_indicator)
            self._set_hint("esc to interrupt")
            self.query_one("#prompt", PromptArea).disabled = True
            self._run_turn(text, seq)

        def _indicator_text(self) -> str:
            frame = _SPINNER_FRAMES[self._spinner_idx % len(_SPINNER_FRAMES)]
            secs = int(time.monotonic() - self._turn_started)
            if self._first_token:
                label = "Streaming…"
            else:
                label = self._phase_label or "Thinking…"
            return (
                f"[#e0a86b]{frame}[/] {label} "
                f"({secs}s · ~{self._live_tokens} tok · esc to interrupt)"
            )

        def _tick_indicator(self) -> None:
            if self._indicator is None:
                return
            try:
                self._indicator.update(self._indicator_text())
            except Exception:  # pragma: no cover - widget may be gone mid-tick
                pass
            self._spinner_idx += 1

        def _stop_indicator(self) -> None:
            if self._indicator_timer is not None:
                self._indicator_timer.stop()
                self._indicator_timer = None
            if self._indicator is not None:
                try:
                    self._indicator.remove()
                except Exception:  # pragma: no cover
                    pass
                self._indicator = None

        @work(thread=True, exclusive=True, group="turn")
        def _run_turn(self, text: str, seq: int) -> None:
            def on_delta(chunk: str) -> None:
                if not self._turn_active or seq != self._turn_seq:
                    return
                self.call_from_thread(self._append_delta, chunk, seq)

            def on_event(event: events.TurnEvent) -> None:
                if not self._turn_active or seq != self._turn_seq:
                    return
                self.call_from_thread(self._on_turn_event, event, seq)

            try:
                result = self.orchestrator.run_turn(
                    text,
                    on_delta=on_delta,
                    on_event=on_event,
                    trace_store=self.trace_store,
                )
            except Exception as exc:  # keep the app alive on unexpected errors
                self.call_from_thread(self._turn_error, exc, seq)
                return
            self.call_from_thread(self._turn_done, result, seq)

        def _append_delta(self, chunk: str, seq: int) -> None:
            if seq != self._turn_seq or not self._turn_active:
                return
            if not self._first_token:
                self._first_token = True
                self._begin_assistant()
            self._buffer += chunk
            if self._current_md is not None:
                self._current_md.update(self._buffer)
            self._live_tokens = _estimate_tokens(self._buffer)
            self._scroll_end()

        def _on_turn_event(self, event: events.TurnEvent, seq: int) -> None:
            if seq != self._turn_seq or not self._turn_active:
                return
            self._current_events.append((event, time.monotonic() - self._turn_started))
            if isinstance(event, events.TurnStart):
                self._current_turn_id = event.turn_id

            # Map lifecycle events to live spinner labels.
            if isinstance(event, (events.ContextStart, events.ContextDone)):
                self._phase_label = "Assembling context…"
            elif isinstance(event, events.PromptBuilt):
                self._phase_label = "Building prompt…"
            elif isinstance(event, events.LLMCallStart):
                self._phase_label = "Calling LLM…"
            elif isinstance(event, events.TurnToolCallStart):
                self._phase_label = f"Calling tool {event.tool_name}…"
            elif isinstance(event, events.Retry):
                self._phase_label = "Retrying LLM call…"
            elif isinstance(event, events.Degraded):
                self._phase_label = "Degraded — showing local memory…"
            elif isinstance(event, events.TurnTextDelta):
                self._first_token = True
                if self._current_md is None:
                    self._begin_assistant()

        def _turn_done(self, result, seq: int) -> None:
            if seq != self._turn_seq or not self._turn_active:
                return
            self._stop_indicator()
            if not self._first_token:
                # Nothing streamed (e.g. empty stream or non-delta path).
                self._first_token = True
                self._begin_assistant()
            if self._current_md is not None and self._buffer != result.text:
                # Non-streaming providers deliver everything at once; make sure
                # the final rendered text matches the result exactly.
                self._current_md.update(result.text)
                self._buffer = result.text
            # Accumulate authoritative session totals for a future status bar.
            if result.usage is not None:
                self._session_tokens += result.usage.total_tokens
            self._session_tool_calls += result.tool_calls_made
            self._last_usage = result.usage
            # End-of-turn meta line uses the AUTHORITATIVE real usage totals.
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
            usage = result.usage
            if usage is not None:
                self._session_tokens += usage.total_tokens or 0
                self._last_ctx_tokens = usage.prompt_tokens or 0
                info = self._active_model_info()
                if info is not None:
                    p = info.pricing
                    self._session_cost += (
                        (usage.prompt_tokens or 0) / 1_000_000 * (p.input or 0)
                        + (usage.completion_tokens or 0) / 1_000_000 * (p.output or 0)
                    )
            self._update_status_bar()
            self._finalize_trace("completed")
            self._finish_turn()

        def _turn_error(self, exc: Exception, seq: int) -> None:
            if seq != self._turn_seq or not self._turn_active:
                return
            self._stop_indicator()
            hint = ""
            local = self._active_local_profile()
            if local is not None:
                hint = _local_turn_hint(local[0], exc)
            self._write(
                f"Error: {exc} — {hint}" if hint else f"Error: {exc}", "warn",
            )
            self._finalize_trace("failed")
            self._finish_turn()

        def _finalize_trace(self, status: str) -> None:
            """Mount the per-turn trace block and register the turn for navigation."""
            if self._current_user is None:
                return
            turn_id = self._current_turn_id or ""
            duration = time.monotonic() - self._turn_started
            provider_name = self.provider.name if self.provider is not None else ""
            model_name = (
                getattr(self.provider, "_model", "") or provider_name
                if self.provider is not None
                else ""
            )
            width = getattr(self.size, "width", 0) or 80
            height = getattr(self.size, "height", 0) or 24
            trace = TraceBlock(
                event_records=list(self._current_events),
                turn_id=turn_id,
                duration=duration,
                status=status,
                task_input=self._turn_input,
                provider=provider_name,
                model=model_name,
                width=width,
                height=height,
            )
            self.query_one("#transcript", VerticalScroll).mount(trace)
            self._turns.append(
                {
                    "user": self._current_user,
                    "trace": trace,
                    "turn_id": turn_id,
                }
            )
            self._scroll_end()

        def _finish_turn(self) -> None:
            self._turn_active = False
            self._stop_indicator()
            self._current_md = None
            self._current_user = None
            self._current_events = []
            self._current_turn_id = None
            self._phase_label = "Thinking…"
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
                self._stop_indicator()
                self._write("(interrupted)", "warn")
                self._finalize_trace("interrupted")
                self._turn_active = False
                self._finish_turn()
                return
            if self._menu_open:
                self._close_menu()
                return
            prompt = self.query_one("#prompt", PromptArea)
            if prompt.text:
                prompt.text = ""
        def action_ctrl_c(self) -> None:
            # While a turn runs, a single Ctrl+C cancels it (like Esc) rather
            # than quitting. Otherwise it's the first tap of double-tap-to-quit.
            if self._turn_active:
                self.action_interrupt()
                return
            if self._ctrl_c_pending:
                self.exit()
                return
            self._ctrl_c_pending = True
            self._set_hint("press ctrl+c again to quit")
            if self._ctrl_c_timer is not None:
                self._ctrl_c_timer.stop()
            self._ctrl_c_timer = self.set_timer(1.5, self._reset_ctrl_c)

        def _reset_ctrl_c(self) -> None:
            self._ctrl_c_pending = False
            self._ctrl_c_timer = None
            if not self._turn_active:
                self._set_hint(_DEFAULT_HINT)

        def action_clear(self) -> None:
            self.query_one("#transcript", VerticalScroll).remove_children()
            self._has_turns = False
            self._turns = []
            self._selected_turn_index = -1
            self.call_after_refresh(self._finish_clear)

        def _finish_clear(self) -> None:
            self._render_welcome()
            self._write("cleared.", "meta")

        def _set_selected_turn(self, index: int) -> None:
            old = self._selected_turn_index
            if old >= 0 and old < len(self._turns):
                self._turns[old]["user"].remove_class("selected")
                self._turns[old]["trace"].remove_class("selected")
            self._selected_turn_index = index
            if index >= 0 and index < len(self._turns):
                self._turns[index]["user"].add_class("selected")
                self._turns[index]["trace"].add_class("selected")
                self._turns[index]["trace"].scroll_visible()

        def action_prev_turn(self) -> None:
            if not self._turns:
                return
            if self._selected_turn_index == -1:
                new = len(self._turns) - 1
            else:
                new = max(0, self._selected_turn_index - 1)
            self._set_selected_turn(new)

        def action_next_turn(self) -> None:
            if not self._turns:
                return
            if self._selected_turn_index == -1:
                return
            new = self._selected_turn_index + 1
            if new >= len(self._turns):
                self._set_selected_turn(-1)
                self.query_one("#prompt", PromptArea).focus()
            else:
                self._set_selected_turn(new)

        def action_toggle_trace(self) -> None:
            if not self._turns:
                return
            if self._selected_turn_index == -1:
                self._set_selected_turn(len(self._turns) - 1)
            self._turns[self._selected_turn_index]["trace"].toggle()

        def action_toggle_trace_raw(self) -> None:
            focused = self.focused
            if isinstance(focused, TraceLineWidget):
                focused.trace_block.toggle_raw_mode()

        def focus_trace(self) -> None:
            if not self._turns:
                return
            if self._selected_turn_index == -1:
                self._set_selected_turn(len(self._turns) - 1)
            trace = self._turns[self._selected_turn_index]["trace"]
            if not trace._expanded:
                trace.expand()
            self.call_after_refresh(self._finish_focus_trace, trace)

        def _finish_focus_trace(self, trace: "TraceBlock") -> None:
            trace.focus_first_line()
            self._update_trace_hint()

        def _update_trace_hint(self) -> None:
            focused = self.focused
            if (
                isinstance(focused, TraceLineWidget)
                and focused.trace_block._expanded
            ):
                self._set_hint(_TRACE_HINT)
            else:
                self._set_hint(_DEFAULT_HINT)

    return RickshawTUI(trace_store=trace_store)


def _run_app(
    orchestrator: Orchestrator,
    provider: LLMProvider | None,
    effort: Effort,
    cfg: RickshawConfig,
    trace_store: TraceStore | None = None,
) -> None:
    """Build and run the Textual app (separated out for testability)."""
    make_app(orchestrator, provider, effort, cfg, trace_store=trace_store).run()


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
            exc_msg = _local_hint_message(
                provider_name, cfg.providers.get(provider_name), exc,
            )
            if args.validate_only:
                print(
                    f"Provider validation failed ({provider_name}): {exc_msg}",
                    file=sys.stderr,
                )
                sys.exit(1)
            if provider is None:
                if provider_source == "settings":
                    settings["provider"] = ""
                    save_settings(settings)
                else:
                    print(
                        f"Could not use provider {provider_name!r}: {exc_msg}. "
                        "Launching provider picker.",
                        file=sys.stderr,
                    )
            elif args.allow_unvalidated:
                print(
                    f"Provider validation failed ({provider_name}): {exc_msg}",
                    file=sys.stderr,
                )
                print(
                    "--allow-unvalidated set; continuing anyway — calls may fail.\n",
                    file=sys.stderr,
                )
            else:
                print(
                    f"Could not use provider {provider_name!r}: {exc_msg}. "
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
    trace_store = TraceStore(db_path=args.db_path)
    orchestrator = Orchestrator(
        provider=provider,
        memory=memory,
        effort=effort,
        trace_store=trace_store,
    )  # type: ignore[arg-type]

    _run_app(orchestrator, provider, effort, cfg, trace_store=trace_store)


if __name__ == "__main__":
    main()
