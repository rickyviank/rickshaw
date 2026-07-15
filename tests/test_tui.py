"""Tests for the Textual-based terminal UI (rickshaw.tui).

The TUI is an optional extra (``pip install rickshaw[tui]``). Tests that need
Textual are skipped when it isn't installed; the import/wiring tests below only
need the module to import, which does not require Textual at import time.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs

import httpx
import pytest
import respx

from rickshaw.memory.embedder import TFIDFEmbedder
from rickshaw.history import append_history, default_history_path, load_history
from rickshaw.memory.service import MemoryService
from rickshaw.orchestrator import Orchestrator
from rickshaw import events
from rickshaw.config import (
    LOCAL_PRESET_NAMES,
    ProviderProfile,
    RickshawConfig,
    _parse_status_bar,
    load_config,
    local_no_models_hint,
    local_server_down_hint,
)
from rickshaw.providers.base import (
    Capabilities,
    Effort,
    LLMProvider,
    Message,
    Response,
    ToolSpec,
)

import rickshaw.tui as tui
from rickshaw.settings import load_settings, save_settings
from rickshaw.trace_render import TraceLine, TraceView


class _FakeProvider(LLMProvider):
    """Minimal fake provider — never hits the network.

    ``function_calling=False`` + ``streaming=True`` exercises the real
    token-streaming path in ``Orchestrator.run_turn(on_delta=...)``.
    """

    def __init__(self, function_calling: bool = False) -> None:
        self._function_calling = function_calling
        self._model = "fake-model"
        self.validated = False

    @property
    def name(self) -> str:
        return "fake"

    def complete(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Response:
        return Response(text="Hello from fake", model="fake-model", effort=effort)

    def stream(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        for chunk in ("Hello ", "from ", "fake"):
            yield chunk

    def available_models(self) -> list[str]:
        return ["fake-model"]

    def validate(self) -> None:
        self.validated = True

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            function_calling=self._function_calling,
            effort_levels=list(Effort),
            max_context_tokens=4096,
        )


class _GatedProvider(_FakeProvider):
    """Fake provider whose stream blocks until released — lets tests observe
    the in-flight 'thinking' state deterministically."""

    def __init__(self) -> None:
        super().__init__(function_calling=False)
        import threading

        self.started = threading.Event()
        self.release = threading.Event()

    def stream(self, messages, effort=Effort.MEDIUM, tools=None, **kwargs):
        self.started.set()
        self.release.wait(5)
        for chunk in ("Hello ", "from ", "fake"):
            yield chunk


def _make_orchestrator(function_calling: bool = False):
    provider = _FakeProvider(function_calling=function_calling)
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory)
    return orch, provider, memory


def _statusbar_text(app) -> str:
    return str(app.query_one("#statusbar").render())


def _fake_model_info():
    return SimpleNamespace(
        id="fake",
        models=[
            SimpleNamespace(
                model="fake-model",
                context_window=1000,
                pricing=SimpleNamespace(input=1.0, output=2.0),
            )
        ],
    )


# --- module / wiring tests (no Textual needed) -------------------------------


def test_module_imports_without_textual():
    """The module and branding constants import even without Textual."""
    assert tui.RICKSHAW_LOGO == "o--o  rickshaw"
    assert tui.RICKSHAW_SLOGAN == "your driver, your memory"
    assert "your driver, your memory" in tui.RICKSHAW_BANNER


def test_cli_exports_preserved():
    """cli still exports the symbols tui.py imports."""
    from rickshaw import cli

    assert hasattr(cli, "_EFFORT_NAMES")
    assert hasattr(cli, "_build_provider")
    assert hasattr(cli, "load_config")


def test_parse_args_defaults_and_overrides():
    args = tui._parse_args([])
    assert args.provider is None
    assert args.effort is None
    assert args.db_path == tui._DEFAULT_DB_PATH

    args = tui._parse_args(
        ["--provider", "openai", "--effort", "high", "--db-path", "x.db"]
    )
    assert args.provider == "openai"
    assert args.effort == "high"
    assert args.db_path == "x.db"


def test_status_segment_value_context_missing_and_warns():
    warnings: list[str] = []

    assert tui._status_segment_value("context", model_info=None) == "—"
    assert (
        tui._status_segment_value("context", model_info=None, warnings=warnings)
        == "—"
    )
    assert warnings == ["context window unknown for the active model"]


def test_status_segment_value_context_uses_window_and_percent():
    warnings: list[str] = []
    model_info = SimpleNamespace(context_window=0)

    assert (
        tui._status_segment_value("context", model_info=model_info, warnings=warnings)
        == "—"
    )
    assert warnings == ["context window unknown for the active model"]

    model_info = SimpleNamespace(context_window=200000)
    assert (
        tui._status_segment_value(
            "context",
            model_info=model_info,
            context_tokens=100000,
        )
        == "50%"
    )


def test_status_segment_value_price_missing_and_formats():
    warnings: list[str] = []
    model_info = SimpleNamespace(pricing=SimpleNamespace(input=0.0, output=0.0))

    assert (
        tui._status_segment_value("price", model_info=model_info, warnings=warnings)
        == "—"
    )
    assert warnings == ["pricing unknown for the active model"]

    model_info = SimpleNamespace(pricing=SimpleNamespace(input=3.0, output=0.0))
    assert (
        tui._status_segment_value(
            "price",
            model_info=model_info,
            session_cost=0.1234,
        )
        == "$0.1234"
    )


def test_status_segment_value_passthrough_and_unknown_segment():
    assert tui._status_segment_value("provider", provider="anthropic") == "anthropic"
    assert tui._status_segment_value("provider", provider=None) == "—"
    assert tui._status_segment_value("model", model="claude") == "claude"
    assert tui._status_segment_value("model", model=None) == "—"
    assert tui._status_segment_value("effort", effort="medium") == "medium"
    assert tui._status_segment_value("effort", effort=None) == "—"
    assert tui._status_segment_value("tokens", session_tokens=0) == "0 tok"
    assert tui._status_segment_value("tokens", session_tokens=None) == "—"

    warnings: list[str] = []
    assert tui._status_segment_value("nope", warnings=warnings) == "—"
    assert warnings == ["unknown status-bar segment: 'nope'"]


def test_status_segment_value_warning_deduplicates():
    warnings: list[str] = []

    tui._status_segment_value("context", model_info=None, warnings=warnings)
    tui._status_segment_value("context", model_info=None, warnings=warnings)

    assert warnings == ["context window unknown for the active model"]


def test_oauth_authorize_url_encoding_and_quirks():
    anthropic = tui._builtin_provider_info("anthropic")
    openai = tui._builtin_provider_info("openai")
    assert anthropic is not None
    assert openai is not None

    anthro_url = tui._build_authorize_url(
        anthropic.oauth,
        state="state123",
        code_challenge="challenge",
        extra=tui._oauth_quirk("anthropic")["authorize_extra"],
    )
    assert "scope=user%3Aprofile%20user%3Ainference" in anthro_url
    assert "code=true" in anthro_url
    assert " " not in anthro_url
    assert ":create_api_key" not in anthro_url

    openai_url = tui._build_authorize_url(
        openai.oauth,
        state="state123",
        code_challenge="challenge",
        extra=tui._oauth_quirk("openai")["authorize_extra"],
    )
    assert "scope=openid%20profile%20email%20offline_access" in openai_url
    assert "code=true" not in openai_url
    assert " " not in openai_url
    assert ":openid" not in openai_url


@respx.mock
@pytest.mark.asyncio
async def test_oauth_token_exchange_honors_provider_encoding():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    anthropic_route = respx.post("https://console.anthropic.com/v1/oauth/token").mock(
        return_value=httpx.Response(
            200,
            json={"access_token": "a", "refresh_token": "r", "expires_in": 3600},
        )
    )
    credential = await type(app)._exchange_token(
        "https://console.anthropic.com/v1/oauth/token",
        {
            "grant_type": "authorization_code",
            "code": "code",
            "client_id": "client",
            "redirect_uri": "https://callback",
            "code_verifier": "verifier",
            "state": "state",
        },
        "json",
    )
    assert credential.access == "a"
    anthro_req = anthropic_route.calls[0].request
    assert anthro_req.headers["content-type"].startswith("application/json")
    assert json.loads(anthro_req.content)["state"] == "state"

    openai_route = respx.post("https://auth.openai.com/oauth/token").mock(
        return_value=httpx.Response(
            200,
            json={"access_token": "b", "refresh_token": "r2", "expires_in": 3600},
        )
    )
    credential = await type(app)._exchange_token(
        "https://auth.openai.com/oauth/token",
        {
            "grant_type": "authorization_code",
            "code": "code",
            "client_id": "client",
            "redirect_uri": "https://callback",
            "code_verifier": "verifier",
        },
        "form",
    )
    assert credential.access == "b"
    openai_req = openai_route.calls[0].request
    assert openai_req.headers["content-type"].startswith(
        "application/x-www-form-urlencoded"
    )
    assert parse_qs(openai_req.content.decode())["code"][0] == "code"
    assert "state" not in parse_qs(openai_req.content.decode())


@patch("rickshaw.tui._build_provider")
@patch("rickshaw.tui.load_config")
def test_main_wiring_constructs_orchestrator(mock_config, mock_build):
    """main() builds the provider, MemoryService + Orchestrator, then runs app."""
    from rickshaw.config import RickshawConfig

    mock_config.return_value = RickshawConfig()
    provider = _FakeProvider()
    mock_build.return_value = provider

    captured: dict[str, Any] = {}

    def _fake_run(orchestrator, prov, effort, cfg, trace_store=None):
        captured.update(
            orchestrator=orchestrator,
            provider=prov,
            effort=effort,
            trace_store=trace_store,
        )

    with patch("rickshaw.tui._run_app", side_effect=_fake_run):
        tui.main(["--provider", "fake", "--effort", "high", "--db-path", ":memory:"])

    assert provider.validated is True
    orch = captured["orchestrator"]
    assert isinstance(orch, Orchestrator)
    assert orch.provider is provider
    assert isinstance(orch.memory, MemoryService)
    assert captured["effort"] == Effort.HIGH
    assert isinstance(captured["trace_store"], tui.TraceStore)


@patch("rickshaw.tui._run_app")
@patch("rickshaw.tui._build_provider")
@patch("rickshaw.tui.load_config")
def test_main_validation_failure_falls_back(mock_config, mock_build, mock_run):
    """Validation failure on a normal launch falls back to the picker."""
    from rickshaw.config import RickshawConfig

    mock_config.return_value = RickshawConfig()
    provider = _FakeProvider()
    provider.validate = MagicMock(side_effect=ValueError("bad key"))
    mock_build.return_value = provider

    tui.main(["--provider", "fake", "--db-path", ":memory:"])

    mock_run.assert_called_once()
    args = mock_run.call_args[0]
    kwargs = mock_run.call_args[1]
    assert args[1] is None  # provider
    assert isinstance(kwargs.get("trace_store"), tui.TraceStore)


@patch("rickshaw.tui._run_app")
@patch("rickshaw.tui._build_provider")
@patch("rickshaw.tui.load_config")
def test_main_allow_unvalidated_continues(mock_config, mock_build, mock_run):
    """--allow-unvalidated lets the app launch despite validation failure."""
    from rickshaw.config import RickshawConfig

    mock_config.return_value = RickshawConfig()
    provider = _FakeProvider()
    provider.validate = MagicMock(side_effect=ValueError("bad key"))
    mock_build.return_value = provider

    tui.main(["--provider", "fake", "--db-path", ":memory:", "--allow-unvalidated"])

    mock_run.assert_called_once()


# --- streaming orchestrator behavior ----------------------------------------


def test_orchestrator_streams_deltas_without_tools():
    """on_delta receives real token chunks when tools aren't advertised."""
    orch, _provider, memory = _make_orchestrator(function_calling=False)
    deltas: list[str] = []

    result = orch.run_turn("hi", on_delta=deltas.append)

    assert deltas == ["Hello ", "from ", "fake"]
    assert result.text == "Hello from fake"
    assert result.degraded is False
    # The streamed answer is written to memory.
    assert any("Hello from fake" in r.text for r in memory.store.all_records())


def test_orchestrator_single_delta_with_tools():
    """With function-calling providers, the final text arrives as one delta."""
    orch, _provider, _memory = _make_orchestrator(function_calling=True)
    deltas: list[str] = []

    result = orch.run_turn("hi", on_delta=deltas.append)

    assert result.text == "Hello from fake"
    assert deltas == ["Hello from fake"]


def test_run_turn_without_on_delta_unchanged():
    """Omitting on_delta preserves the original non-streaming behavior."""
    orch, _provider, _memory = _make_orchestrator(function_calling=True)
    result = orch.run_turn("hi")
    assert result.text == "Hello from fake"


# --- Textual app tests (require Textual) -------------------------------------


def test_make_app_builds_instance():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.HIGH)
    # Effort is applied to the orchestrator at construction.
    assert orch.effort == Effort.HIGH
    assert app is not None


@pytest.mark.asyncio
async def test_status_bar_default_shows_all_six_segments_in_order():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=RickshawConfig())

    async with app.run_test() as pilot:
        await pilot.pause()
        assert str(app.query_one("#statusbar").render()) == (
            "fake | fake-model | medium | — | 0 tok | —"
        )


@pytest.mark.asyncio
async def test_status_bar_custom_order_and_removal():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig(status_bar=["effort", "provider"])
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        await pilot.pause()
        assert str(app.query_one("#statusbar").render()) == "medium | fake"


@pytest.mark.asyncio
async def test_status_bar_context_dash_when_window_missing():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=RickshawConfig())

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app._active_model_info() is None
        assert "—" in str(app.query_one("#statusbar").render())


def test_load_config_status_bar_custom_and_unknown_ignored(caplog):
    s = load_settings()
    s["status_bar"] = ["price", "provider", "bogus"]
    save_settings(s)
    cfg = load_config()
    assert cfg.status_bar == ["price", "provider"]

    with caplog.at_level("WARNING", logger="rickshaw.config"):
        assert _parse_status_bar(["provider", "bogus"]) == ["provider"]
    assert any("bogus" in msg for msg in caplog.messages)


@pytest.mark.asyncio
async def test_app_statusbar_drops_lower_priority_segments_on_narrow_width():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    with patch("rickshaw.tui._find_model_info", return_value=_fake_model_info()):
        async with app.run_test(size=(40, 24)) as pilot:
            await pilot.pause()
            rendered = _statusbar_text(app)
            assert "fake" in rendered
            assert "fake-model" in rendered
            assert "medium" in rendered
            assert "$" not in rendered
            assert "0%" in rendered
            assert rendered.startswith("fake")


@pytest.mark.asyncio
async def test_app_statusbar_drops_context_and_tokens_at_extreme_narrow_width():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    with patch("rickshaw.tui._find_model_info", return_value=_fake_model_info()):
        async with app.run_test(size=(30, 24)) as pilot:
            await pilot.pause()
            rendered = _statusbar_text(app)
            assert "fake" in rendered
            assert "fake-model" in rendered
            assert "medium" in rendered
            assert "$" not in rendered
            assert "0%" not in rendered


@pytest.mark.asyncio
async def test_app_statusbar_restores_segments_on_resize():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    with patch("rickshaw.tui._find_model_info", return_value=_fake_model_info()):
        async with app.run_test(size=(200, 24)) as pilot:
            await pilot.pause()
            rendered = _statusbar_text(app)
            assert "fake" in rendered
            assert "fake-model" in rendered
            assert "medium" in rendered
            assert "0%" in rendered
            assert "$0.0000" in rendered

            await pilot.resize_terminal(30, 24)
            rendered = _statusbar_text(app)
            assert "fake" in rendered
            assert "fake-model" in rendered
            assert "medium" in rendered
            assert "0%" not in rendered
            assert "$" not in rendered

            await pilot.resize_terminal(200, 24)
            rendered = _statusbar_text(app)
            assert "0%" in rendered
            assert "$0.0000" in rendered

async def test_app_slash_opens_command_menu():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/"
        await pilot.pause()
        menu = app.query_one("#slashmenu")
        rendered = str(menu.render())
        assert menu.display is True
        assert "/help" in rendered
        assert "/model" in rendered
        assert "/effort" in rendered


@pytest.mark.asyncio
async def test_app_slash_filters_command_menu():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/mo"
        await pilot.pause()
        menu = app.query_one("#slashmenu")
        rendered = str(menu.render())
        assert menu.display is True
        assert "/model" in rendered
        assert "/models" in rendered
        assert "/help" not in rendered


@pytest.mark.asyncio
async def test_app_slash_effort_value_picker_applies_selection():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    orch.effort = Effort.LOW
    app = tui.make_app(orch, provider, Effort.LOW)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/effort "
        await pilot.pause()
        menu = app.query_one("#slashmenu")
        rendered = str(menu.render())
        assert menu.display is True
        assert "low" in rendered
        assert "medium" in rendered
        assert "high" in rendered

        await pilot.press("down")
        await pilot.press("enter")
        await pilot.pause()
        assert orch.effort == Effort.MEDIUM
        assert app.query_one("#slashmenu").display is False


@pytest.mark.asyncio
async def test_app_slash_escape_dismisses_menu():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/"
        await pilot.pause()
        assert app.query_one("#slashmenu").display is True
        await pilot.press("escape")
        await pilot.pause()
        assert app.query_one("#slashmenu").display is False


@pytest.mark.asyncio
async def test_app_slash_tab_completes_arg_command_to_value_picker():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/ef"
        await pilot.pause()
        await pilot.press("tab")
        await pilot.pause()
        prompt = app.query_one("#prompt")
        menu = app.query_one("#slashmenu")
        rendered = str(menu.render())
        assert prompt.value == "/effort "
        assert menu.display is True
        assert "low" in rendered
        assert "medium" in rendered
        assert "high" in rendered


@pytest.mark.asyncio
async def test_app_records_plain_messages_and_slash_commands_in_history():
    pytest.importorskip("textual")
    orch, provider, memory = _make_orchestrator(function_calling=False)
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "remember milk"
        await pilot.press("enter")

        stored = False
        for _ in range(60):
            if any("Hello from fake" in r.text for r in memory.store.all_records()):
                stored = True
                break
            await pilot.pause(0.05)
        assert stored
        for _ in range(60):
            if not app._turn_active:
                break
            await pilot.pause(0.05)

        prompt = app.query_one("#prompt")
        prompt.value = "/help"
        await pilot.press("enter")
        await pilot.pause()

    assert load_history(default_history_path()) == ["remember milk", "/help"]


def test_app_loads_persisted_history_on_construction():
    pytest.importorskip("textual")
    append_history("first message")
    append_history("/help")
    orch, provider, _memory = _make_orchestrator()

    app = tui.make_app(orch, provider, Effort.MEDIUM)

    assert app._history == ["first message", "/help"]
    assert app._history_pos == 2


def test_history_cap_rolls_over_at_1000_entries():
    entries = [f"entry {i}" for i in range(1005)]
    for entry in entries:
        append_history(entry)

    history = load_history()
    assert len(history) == 1000
    assert history[0] == "entry 5"
    assert history[-1] == "entry 1004"


@pytest.mark.asyncio
async def test_app_runs_a_turn_through_orchestrator():
    pytest.importorskip("textual")
    orch, provider, memory = _make_orchestrator(function_calling=False)
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "remember milk"
        await pilot.press("enter")

        # The turn runs in a worker thread; poll until it writes to memory.
        stored = False
        for _ in range(60):
            if any("Hello from fake" in r.text for r in memory.store.all_records()):
                stored = True
                break
            await pilot.pause(0.05)

        transcript = app.query_one("#transcript").query("Static")
        rendered = " ".join(str(w.render()) for w in transcript)
        assert "remember milk" in rendered
        # The streamed assistant reply was routed through run_turn into memory.
        assert stored


@pytest.mark.asyncio
async def test_prompt_newline_via_ctrl_j():
    pytest.importorskip("textual")
    orch, provider, memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.focus()
        await pilot.press("f", "o", "o")
        await pilot.press("ctrl+j")
        await pilot.press("b", "a", "r")
        await pilot.pause()

        assert app.query_one("#prompt").text == "foo\nbar"
        assert not any("Hello from fake" in r.text for r in memory.store.all_records())


@pytest.mark.asyncio
async def test_prompt_newline_via_shift_enter():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.focus()
        await pilot.press("f", "o", "o")
        await pilot.press("shift+enter")
        await pilot.press("b", "a", "r")
        await pilot.pause()

        assert app.query_one("#prompt").text == "foo\nbar"


@pytest.mark.asyncio
async def test_prompt_enter_submits_and_clears():
    pytest.importorskip("textual")
    orch, provider, memory = _make_orchestrator(function_calling=False)
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.focus()
        await pilot.press("h", "i")
        await pilot.press("enter")

        stored = False
        for _ in range(60):
            if any("Hello from fake" in r.text for r in memory.store.all_records()):
                stored = True
                break
            await pilot.pause(0.05)

        transcript = app.query_one("#transcript").query("Static")
        rendered = " ".join(str(w.render()) for w in transcript)
        assert "hi" in rendered
        assert stored
        assert app.query_one("#prompt").text == ""


@pytest.mark.asyncio
async def test_prompt_esc_clears_when_idle():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.focus()
        await pilot.press("x", "y", "z")
        assert app.query_one("#prompt").text == "xyz"
        await pilot.press("escape")
        await pilot.pause()
        assert app.query_one("#prompt").text == ""


@pytest.mark.asyncio
async def test_wizard_advances_on_textarea():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui.build_provider_from_profile", return_value=provider):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/settings"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "Pick a provider" in rendered

            app.query_one("#prompt").value = "fake"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "Pick a model" in rendered


@pytest.mark.asyncio
async def test_app_mounts_welcome_panel():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test():
        welcome = app.query_one("#welcome")
        rendered = str(welcome.render())
        assert "o--o  rickshaw" in rendered
        assert "your driver, your memory" in rendered


@pytest.mark.asyncio
async def test_app_no_provider_welcome_shows_none():
    pytest.importorskip("textual")
    orch, _provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    app = tui.make_app(orch, None, Effort.MEDIUM, cfg=cfg)

    async with app.run_test():
        welcome = app.query_one("#welcome")
        rendered = str(welcome.render())
        assert "provider: (none)" in rendered


@pytest.mark.asyncio
async def test_app_clear_re_renders_welcome_panel():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        original = app.query_one("#welcome")
        app.query_one("#prompt").value = "/clear"
        await pilot.press("enter")
        await pilot.pause()

        refreshed = app.query_one("#welcome")
        assert refreshed is not original
        rendered = " ".join(
            str(w.render()) for w in app.query_one("#transcript").query("Static")
        )
        assert "cleared." in rendered


@pytest.mark.asyncio
async def test_app_boot_smoke_mounts_core_widgets():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#welcome")
        app.query_one("#prompt")
        app.query_one("#statusbar")

        app.query_one("#prompt").value = "/help"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render()) for w in app.query_one("#transcript").query("Static")
        )
        assert "/help" in rendered


@pytest.mark.asyncio
async def test_app_degraded_turn_shows_themed_banner():
    pytest.importorskip("textual")
    from rickshaw.orchestrator import TurnResult

    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)
    orch.run_turn = lambda text, on_delta=None, **kwargs: TurnResult(
        text="local answer", degraded=True
    )

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")

        banner_found = False
        for _ in range(60):
            transcript = app.query_one("#transcript").query("Static")
            for widget in transcript:
                if "degraded-banner" in widget.classes and (
                    "provider unreachable — showing local memory only"
                    in str(widget.render())
                ):
                    banner_found = True
                    break
            if banner_found:
                break
            await pilot.pause(0.05)

        assert banner_found


@pytest.mark.asyncio
async def test_j4_indicator_appears_on_turn_start():
    pytest.importorskip("textual")
    provider = _GatedProvider()
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory)
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")
        for _ in range(100):
            if provider.started.is_set():
                break
            await pilot.pause(0.02)

        indicator = app.query("#turn-indicator")
        assert len(indicator) > 0
        indicator_text = str(app.query_one("#turn-indicator").render())
        assert any(
            label in indicator_text
            for label in (
                "Thinking",
                "Assembling context",
                "Building prompt",
                "Calling LLM",
            )
        )

        provider.release.set()
        for _ in range(200):
            if len(app.query("#turn-indicator")) == 0:
                break
            await pilot.pause(0.02)

        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static, Markdown")
        )
        assert "o--o" in rendered
        assert "rickshaw" in rendered


@pytest.mark.asyncio
async def test_j4_role_glyphs_present():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator(function_calling=False)
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hello"
        await pilot.press("enter")
        for _ in range(100):
            if any("Hello from fake" in r.text for r in _memory.store.all_records()):
                break
            await pilot.pause(0.02)

        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static, Markdown")
        )
        assert "o--o" in rendered
        assert "rickshaw" in rendered
        assert "Hello from fake" in rendered


@pytest.mark.asyncio
async def test_j4_esc_interrupts_running_turn():
    pytest.importorskip("textual")
    provider = _GatedProvider()
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory)
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")
        for _ in range(100):
            if provider.started.is_set():
                break
            await pilot.pause(0.02)

        await pilot.press("escape")
        await pilot.pause(0.05)

        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static, Markdown")
        )
        assert "(interrupted)" in rendered
        assert len(app.query("#turn-indicator")) == 0
        assert app.query_one("#prompt").disabled is False

        provider.release.set()
        for _ in range(100):
            await pilot.pause(0.02)
            if app.query_one("#prompt").disabled is False:
                break


@pytest.mark.asyncio
async def test_j4_ctrl_c_single_press_cancels_running_turn():
    pytest.importorskip("textual")
    provider = _GatedProvider()
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory)
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")
        for _ in range(100):
            if provider.started.is_set():
                break
            await pilot.pause(0.02)

        with patch.object(app, "exit") as mock_exit:
            await pilot.press("ctrl+c")
            await pilot.pause(0.05)
            assert mock_exit.called is False

        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static, Markdown")
        )
        assert "(interrupted)" in rendered

        provider.release.set()
        for _ in range(100):
            await pilot.pause(0.02)
            if app.query_one("#prompt").disabled is False:
                break


@pytest.mark.asyncio
async def test_j4_ctrl_c_double_tap_quits():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        with patch.object(app, "exit") as mock_exit:
            await pilot.press("ctrl+c")
            await pilot.pause(0.05)
            assert mock_exit.called is False
            assert "again to quit" in str(app.query_one("#hint").render())

            await pilot.press("ctrl+c")
            await pilot.pause(0.05)
            assert mock_exit.called is True

async def test_app_history_recall_and_return_to_draft():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator(function_calling=False)
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "recall this"
        await pilot.press("enter")

        for _ in range(60):
            if not app._turn_active:
                break
            await pilot.pause(0.05)

        await pilot.press("up")
        assert app.query_one("#prompt").value == "recall this"

        await pilot.press("down")
        assert app.query_one("#prompt").value == ""


@pytest.mark.asyncio
async def test_app_history_navigation_is_blocked_during_wizard():
    pytest.importorskip("textual")
    append_history("saved history entry")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app._settings_state = {"step": "provider", "providers": ["fake"], "on_launch": False}
        prompt = app.query_one("#prompt")
        prompt.value = "draft"
        prompt.focus()

        await pilot.press("up")

        assert app.query_one("#prompt").value == "draft"


@pytest.mark.asyncio
async def test_app_effort_command_updates_orchestrator():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/effort low"
        await pilot.press("enter")
        await pilot.pause()
        assert orch.effort == Effort.LOW


@pytest.mark.asyncio
async def test_app_settings_command_shows_header():
    """``/settings`` prints settings header and starts the interactive picker."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/settings"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "Settings" in rendered
        assert "provider" in rendered
        assert "Pick a provider" in rendered


@pytest.mark.asyncio
async def test_app_provider_list_shows_providers():
    """``/provider`` (no arg) lists available providers."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/provider"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "available providers" in rendered
        assert "fake" in rendered
        assert "FAKE_KEY" in rendered


@pytest.mark.asyncio
async def test_app_provider_add_registers_provider():
    """``/provider add`` wizard registers a new provider in cfg.providers."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/provider add"
        await pilot.press("enter")
        await pilot.pause()

        # Step through the wizard: name, base_url, api_key_env, wire_format
        for val in ["testeng", "https://test.example.com/v1", "TEST_KEY", ""]:
            prompt.value = val
            await pilot.press("enter")
            await pilot.pause()

        assert "testeng" in cfg.providers
        p = cfg.providers["testeng"]
        assert p.base_url == "https://test.example.com/v1"
        assert p.api_key_env == "TEST_KEY"
        assert p.wire_format == "openai"

        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "provider registered" in rendered


@pytest.mark.asyncio
async def test_app_warn_missing_metadata_writes_warn_statics():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test():
        warnings = app._warn_missing_metadata(None)
        assert warnings == [
            "context window unknown for the active model",
            "pricing unknown for the active model",
        ]

        statics = [
            widget
            for widget in app.query_one("#transcript").query("Static")
            if "warn" in widget.classes and "⚠" in str(widget.render())
        ]
        assert len(statics) == 2


@pytest.mark.asyncio
async def test_app_effort_rejected_when_unsupported():
    """Effort change is rejected when the provider doesn't support it."""
    pytest.importorskip("textual")

    class _LimitedProvider(_FakeProvider):
        def capabilities(self):
            return Capabilities(
                streaming=True,
                function_calling=False,
                effort_levels=[Effort.LOW, Effort.MEDIUM],
                max_context_tokens=4096,
            )

    provider = _LimitedProvider()
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory)
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/effort high"
        await pilot.press("enter")
        await pilot.pause()
        # Effort should NOT have changed since HIGH is unsupported.
        assert orch.effort == Effort.MEDIUM
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "does not support effort high" in rendered


@pytest.mark.asyncio
async def test_app_model_list_shows_available_models():
    """Bare ``/model`` lists the current provider's available_models()."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/model"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "available models" in rendered
        assert "fake-model" in rendered
        assert "♦" in rendered


@pytest.mark.asyncio
async def test_app_model_rejects_unknown_model():
    """``/model bad-name`` is rejected with valid options listed."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/model gpt-4o"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "Unknown model" in rendered
        assert "fake-model" in rendered


@pytest.mark.asyncio
async def test_app_model_scoped_to_active_provider():
    """Model selection is scoped: can't pick models from another provider."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        # Try to select a model that doesn't belong to the fake provider.
        app.query_one("#prompt").value = "/model claude-3-5-sonnet-latest"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "Unknown model" in rendered
        assert "fake-model" in rendered


@pytest.mark.asyncio
async def test_app_model_switch_effort_reset():
    """Effort is reset to medium if unsupported by the new model's provider."""
    pytest.importorskip("textual")

    class _LimitedModelProvider(_FakeProvider):
        """Provider that supports two models but no HIGH effort."""

        def available_models(self):
            return ["fake-model", "fake-model-2"]

        def capabilities(self):
            return Capabilities(
                streaming=True,
                function_calling=False,
                effort_levels=[Effort.LOW, Effort.MEDIUM],
                max_context_tokens=4096,
            )

    provider = _LimitedModelProvider()
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory)
    orch.effort = Effort.HIGH  # Set an unsupported effort level.
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.HIGH, cfg=cfg)

    # Patch _rebuild_provider to return a provider with limited effort.
    rebuilt = _LimitedModelProvider()
    rebuilt._model = "fake-model-2"
    with patch("rickshaw.tui._rebuild_provider", return_value=rebuilt):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/model fake-model-2"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "Reset to medium" in rendered
            assert orch.effort == Effort.MEDIUM


@pytest.mark.asyncio
async def test_app_model_offline_error_surfaces_warning():
    """If available_models() raises, a warning is shown and no switch happens."""
    pytest.importorskip("textual")

    class _OfflineProvider(_FakeProvider):
        def available_models(self):
            raise RuntimeError("offline — no cached model list")

    provider = _OfflineProvider()
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory)
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        # Bare /model should warn on error.
        app.query_one("#prompt").value = "/model"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "Cannot list models" in rendered

        # /model <name> should also warn on error and not switch.
        app.query_one("#prompt").value = "/model some-model"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "Cannot validate model" in rendered


@pytest.mark.asyncio
async def test_app_clear_command():
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/clear"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render()) for w in app.query_one("#transcript").query("Static")
        )
        assert "cleared." in rendered


@pytest.mark.asyncio
async def test_app_help_lists_provider_command():
    """``/help`` includes the /provider command."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/help"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "/provider" in rendered
        assert "/settings" in rendered


@pytest.mark.asyncio
async def test_app_engine_alias_still_works():
    """``/engine`` still works as a deprecated alias for ``/provider``."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/engine"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "available providers" in rendered


@pytest.mark.asyncio
async def test_app_settings_interactive_picker():
    """``/settings`` starts an interactive wizard: pick provider, then model."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui._rebuild_provider", return_value=provider), \
         patch("rickshaw.tui.build_provider_from_profile", return_value=provider):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/settings"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "Settings" in rendered
            assert "Pick a provider" in rendered

            # Step 1: pick the "fake" provider.
            app.query_one("#prompt").value = "fake"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "Pick a model" in rendered
            assert "fake-model" in rendered

            # Step 2: pick the "fake-model" model.
            app.query_one("#prompt").value = "fake-model"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "fake · fake-model" in rendered


@pytest.mark.asyncio
async def test_app_settings_rejects_unknown_provider():
    """``/settings`` rejects an unknown provider name."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/settings"
        await pilot.press("enter")
        await pilot.pause()

        app.query_one("#prompt").value = "nonexistent"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "Unknown provider" in rendered


@pytest.mark.asyncio
async def test_app_settings_rejects_unknown_model():
    """``/settings`` rejects an unknown model in step 2."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui.build_provider_from_profile", return_value=provider):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/settings"
            await pilot.press("enter")
            await pilot.pause()

            # Pick the fake provider.
            app.query_one("#prompt").value = "fake"
            await pilot.press("enter")
            await pilot.pause()

            # Try an invalid model.
            app.query_one("#prompt").value = "bad-model"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "Unknown model" in rendered
            assert "fake-model" in rendered


@pytest.mark.asyncio
async def test_app_settings_cancel_at_provider_step():
    """Pressing Esc cancels /settings at the provider step."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/settings"
        await pilot.press("enter")
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "(cancelled)" in rendered


@pytest.mark.asyncio
async def test_app_models_command():
    """``/models`` lists the current provider's models."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/models"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "available models" in rendered
        assert "fake-model" in rendered
        assert "♦" in rendered


@pytest.mark.asyncio
async def test_app_models_offline_error():
    """``/models`` surfaces an error if available_models() raises."""
    pytest.importorskip("textual")

    class _OfflineProvider(_FakeProvider):
        def available_models(self):
            raise RuntimeError("offline — no cached model list")

    provider = _OfflineProvider()
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory)
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/models"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "Cannot list models" in rendered


@pytest.mark.asyncio
async def test_app_settings_offline_error_aborts():
    """``/settings`` aborts gracefully if available_models() raises."""
    pytest.importorskip("textual")

    class _OfflineProvider(_FakeProvider):
        def available_models(self):
            raise RuntimeError("offline — no cached model list")

    provider = _OfflineProvider()
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory)
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    # Patch build_provider_from_profile so the /settings wizard also gets an
    # offline provider (by default it constructs a real OpenAI provider which
    # may succeed via disk-cached models).
    with patch("rickshaw.tui.build_provider_from_profile", return_value=provider):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/settings"
            await pilot.press("enter")
            await pilot.pause()

            # Pick the fake provider — should fail since models can't be fetched.
            app.query_one("#prompt").value = "fake"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "Cannot list models" in rendered


@pytest.mark.asyncio
async def test_app_model_error_is_logged(caplog):
    """Exception details from /model are logged, not just shown in the TUI."""
    pytest.importorskip("textual")

    class _OfflineProvider(_FakeProvider):
        def available_models(self):
            raise RuntimeError("offline — no cached model list")

    provider = _OfflineProvider()
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory)
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    with caplog.at_level("ERROR"):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/model"
            await pilot.press("enter")
            await pilot.pause()

    assert any("Failed to list models" in m for m in caplog.messages)


@pytest.mark.asyncio
async def test_app_provider_switch_error_is_logged(caplog):
    """Exception details from /provider switch are logged."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    cfg.providers["broken"] = ProviderProfile(
        base_url="", model="",
        api_key_env="BROKEN_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    def _raise(*a, **kw):
        raise ValueError("provider construction failed")

    with caplog.at_level("ERROR"), \
         patch("rickshaw.tui.build_provider_from_profile", side_effect=_raise):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/provider broken"
            await pilot.press("enter")
            await pilot.pause()

    assert any("Failed to switch provider" in m for m in caplog.messages)


# --- No-provider / provider picker tests ------------------------------------


@pytest.mark.asyncio
async def test_app_no_provider_shows_picker():
    """Launching with provider=None shows the provider picker."""
    pytest.importorskip("textual")
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=None, memory=memory)
    cfg = RickshawConfig()
    app = tui.make_app(orch, None, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "no provider selected" in rendered
        assert "Pick a provider" in rendered


@pytest.mark.asyncio
async def test_app_picker_selects_key_based_provider():
    """Selecting a key-based (non-OAuth) provider in the picker works."""
    pytest.importorskip("textual")
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=None, memory=memory)
    cfg = RickshawConfig()
    fake_provider = _FakeProvider()
    cfg.providers["fake"] = ProviderProfile(
        base_url="", model="fake-model",
        api_key_env="FAKE_KEY", wire_format="openai",
    )
    app = tui.make_app(orch, None, Effort.MEDIUM, cfg=cfg)

    # Patch _builtin_provider_info to return None for "fake" (no OAuth)
    # and build_provider_from_profile to return our fake provider.
    with patch("rickshaw.tui._builtin_provider_info", return_value=None), \
         patch("rickshaw.tui.build_provider_from_profile", return_value=fake_provider), \
         patch("rickshaw.tui._get_builtin_provider_names", return_value=[]):
        async with app.run_test() as pilot:
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "Pick a provider" in rendered

            # Select the "fake" provider.
            app.query_one("#prompt").value = "fake"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "Pick a model" in rendered


@pytest.mark.asyncio
async def test_app_picker_oauth_provider_triggers_login():
    """Selecting an OAuth provider triggers the login flow (mocked)."""
    pytest.importorskip("textual")
    from rickshaw_ai.registry import OAuthConfig, ProviderInfo

    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=None, memory=memory)
    cfg = RickshawConfig()
    app = tui.make_app(orch, None, Effort.MEDIUM, cfg=cfg)

    # Create a mock OAuth provider info
    mock_info = MagicMock()
    mock_info.oauth = MagicMock()
    mock_info.oauth.mode = "auth_code"
    mock_info.auth_methods = ["oauth", "api_key"]
    mock_info.id = "testprov"

    with patch("rickshaw.tui._builtin_provider_info", return_value=mock_info), \
         patch("rickshaw.tui._get_builtin_provider_names", return_value=["testprov"]), \
         patch.object(type(app), "_start_oauth_login") as mock_login:
        async with app.run_test() as pilot:
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "Pick a provider" in rendered

            # Select the OAuth provider.
            app.query_one("#prompt").value = "testprov"
            await pilot.press("enter")
            await pilot.pause()
            mock_login.assert_called_once()


@pytest.mark.asyncio
async def test_app_no_provider_rejects_messages():
    """Sending a message with no provider shows a warning."""
    pytest.importorskip("textual")
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=None, memory=memory)
    cfg = RickshawConfig()
    app = tui.make_app(orch, None, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui._get_builtin_provider_names", return_value=[]):
        async with app.run_test() as pilot:
            await pilot.pause()
            # Cancel the picker
            await pilot.press("escape")
            await pilot.pause()

            app.query_one("#prompt").value = "hello world"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "No provider selected" in rendered


@pytest.mark.asyncio
async def test_app_login_command_no_provider():
    """/login with no provider active shows a warning."""
    pytest.importorskip("textual")
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=None, memory=memory)
    cfg = RickshawConfig()
    app = tui.make_app(orch, None, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui._get_builtin_provider_names", return_value=[]):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            app.query_one("#prompt").value = "/login"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "No provider selected" in rendered


@pytest.mark.asyncio
async def test_app_login_command_no_oauth():
    """/login on a non-OAuth provider shows a warning."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui._builtin_provider_info", return_value=None):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/login"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "does not support OAuth" in rendered


@pytest.mark.asyncio
async def test_app_help_includes_login():
    """/help lists the /login command."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/help"
        await pilot.press("enter")
        await pilot.pause()
        rendered = " ".join(
            str(w.render())
            for w in app.query_one("#transcript").query("Static")
        )
        assert "/login" in rendered


@pytest.mark.asyncio
async def test_app_status_no_provider():
    """/status with no provider active shows (none)."""
    pytest.importorskip("textual")
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=None, memory=memory)
    cfg = RickshawConfig()
    app = tui.make_app(orch, None, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui._get_builtin_provider_names", return_value=[]):
        async with app.run_test() as pilot:
            await pilot.pause()
            await pilot.press("escape")
            await pilot.pause()

            app.query_one("#prompt").value = "/status"
            await pilot.press("enter")
            await pilot.pause()
            rendered = " ".join(
                str(w.render())
                for w in app.query_one("#transcript").query("Static")
            )
            assert "(none)" in rendered


# --- Local provider support (PRD: local-providers) ---------------------------


class _LocalProvider(_FakeProvider):
    """Fake local OpenAI-compatible provider — never hits the network."""

    def __init__(
        self,
        models: list[str] | None = None,
        base_url: str = "http://localhost:8080/v1",
    ) -> None:
        super().__init__(function_calling=False)
        self._base_url = base_url
        self._model = ""
        self._models = list(models or [])

    def available_models(self) -> list[str]:
        return list(self._models)


class _DownLocalProvider(_LocalProvider):
    """Fake local provider whose server is unreachable."""

    def validate(self) -> None:
        raise ValueError(
            f"llamacpp unreachable at {self._base_url} — connection refused"
        )

    def available_models(self) -> list[str]:
        raise ConnectionError(
            f"Could not reach local inference server at {self._base_url}: refused"
        )


def _transcript_text(app) -> str:
    return " ".join(
        str(w.render()) for w in app.query_one("#transcript").query("Static")
    )


def test_local_hint_message_appends_no_models_hint():
    profile = ProviderProfile(
        base_url="http://localhost:8080/v1", model="",
        api_key_env="LLAMACPP_API_KEY", wire_format="openai",
    )
    exc = ValueError("llamacpp at http://localhost:8080/v1 lists no models")
    msg = tui._local_hint_message("llamacpp", profile, exc)
    assert str(exc) in msg
    assert msg.endswith(local_no_models_hint("llamacpp"))


def test_local_hint_message_server_down_without_duplicating_url():
    profile = ProviderProfile(
        base_url="http://localhost:8080/v1", model="",
        api_key_env="LLAMACPP_API_KEY", wire_format="openai",
    )
    exc = ValueError("llamacpp unreachable at http://localhost:8080/v1")
    msg = tui._local_hint_message("llamacpp", profile, exc)
    assert msg.endswith(local_server_down_hint("llamacpp"))
    assert msg.count("http://localhost:8080/v1") == 1


def test_local_hint_message_adds_url_when_missing():
    profile = ProviderProfile(
        base_url="http://localhost:11434/v1", model="",
        api_key_env="OLLAMA_API_KEY", wire_format="openai",
    )
    exc = ConnectionError("connection refused")
    msg = tui._local_hint_message("ollama", profile, exc)
    assert "ollama unreachable at http://localhost:11434/v1" in msg
    assert msg.endswith(local_server_down_hint("ollama"))


def test_local_hint_message_hosted_unchanged():
    profile = ProviderProfile(
        base_url="https://api.openai.com/v1", model="gpt-4o",
        api_key_env="OPENAI_API_KEY", wire_format="openai",
    )
    exc = ConnectionError("connection refused")
    assert tui._local_hint_message("openai", profile, exc) == str(exc)
    assert tui._local_hint_message("openai", None, exc) == str(exc)


def test_local_turn_hint_timeout_suggests_profile_timeout():
    hint = tui._local_turn_hint("llamacpp", httpx.ReadTimeout("timed out"))
    assert hint == (
        "increase providers.llamacpp.timeout in ~/.rickshaw/settings.json"
    )


def test_local_turn_hint_connection_suggests_server_check():
    assert tui._local_turn_hint(
        "llamacpp", httpx.ConnectError("connection refused")
    ) == local_server_down_hint("llamacpp")
    assert tui._local_turn_hint(
        "llamacpp", httpx.ConnectTimeout("timed out")
    ) == local_server_down_hint("llamacpp")
    assert tui._local_turn_hint("llamacpp", ValueError("weird")) == ""


def test_local_presets_appear_in_config_providers():
    cfg = load_config()
    for name in LOCAL_PRESET_NAMES:
        assert name in cfg.providers
        assert cfg.providers[name].is_local_endpoint()


@pytest.mark.asyncio
async def test_app_launch_picker_lists_local_presets():
    """J1: the on-launch picker offers the local presets."""
    pytest.importorskip("textual")
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=None, memory=memory)
    cfg = load_config()
    app = tui.make_app(orch, None, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        await pilot.pause()
        rendered = _transcript_text(app)
        assert "Pick a provider" in rendered
        for name in LOCAL_PRESET_NAMES:
            assert name in rendered


@pytest.mark.asyncio
async def test_app_provider_list_includes_local_presets():
    """``/provider`` lists the local presets alongside hosted ones."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = load_config()
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "/provider"
        await pilot.press("enter")
        await pilot.pause()
        rendered = _transcript_text(app)
        assert "available providers" in rendered
        for name in LOCAL_PRESET_NAMES:
            assert name in rendered


@pytest.mark.asyncio
async def test_app_provider_switch_local_auto_selects_single_model():
    """J1: /provider llamacpp with one served model adopts it silently."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = load_config()
    local = _LocalProvider(models=["qwen2.5-7b"])
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui.build_provider_from_profile", return_value=local):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/provider llamacpp"
            await pilot.press("enter")
            await pilot.pause()
            rendered = _transcript_text(app)
            assert "model · qwen2.5-7b" in rendered
            assert "llamacpp · qwen2.5-7b" in rendered
            assert app.provider is local
            assert local._model == "qwen2.5-7b"
            assert "llamacpp" in _statusbar_text(app)

    s = load_settings()
    assert s["provider"] == "llamacpp"
    entry = s["providers"]["llamacpp"]
    assert entry["model"] == "qwen2.5-7b"
    assert entry["base_url"] == "http://localhost:8080/v1"
    assert entry["api_key_env"] == "LLAMACPP_API_KEY"
    assert entry["wire_format"] == "openai"


@pytest.mark.asyncio
async def test_app_provider_switch_local_multi_model_routes_to_picker():
    """J3: several installed models route through the /settings model picker."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = load_config()
    local = _LocalProvider(models=["m-a", "m-b"], base_url="http://localhost:11434/v1")
    applied = _LocalProvider(models=["m-a", "m-b"], base_url="http://localhost:11434/v1")
    applied._model = "m-b"
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui.build_provider_from_profile", return_value=local), \
         patch("rickshaw.tui._rebuild_provider", return_value=applied):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/provider ollama"
            await pilot.press("enter")
            await pilot.pause()
            rendered = _transcript_text(app)
            assert "Pick a model" in rendered
            assert "m-a" in rendered
            assert "m-b" in rendered
            # Not switched yet: the picker applies the choice.
            assert app.provider is provider

            app.query_one("#prompt").value = "m-b"
            await pilot.press("enter")
            await pilot.pause()
            rendered = _transcript_text(app)
            assert "ollama · m-b" in rendered
            assert app.provider is applied

    s = load_settings()
    assert s["provider"] == "ollama"
    assert s["providers"]["ollama"]["model"] == "m-b"


@pytest.mark.asyncio
async def test_app_provider_switch_local_missing_model_falls_back():
    """J9: a persisted model that vanished is re-resolved (note + auto-select)."""
    pytest.importorskip("textual")
    s = load_settings()
    s.setdefault("providers", {})["llamacpp"] = {
        "base_url": "http://localhost:8080/v1",
        "model": "old.gguf",
        "timeout": 300,
    }
    save_settings(s)
    orch, provider, _memory = _make_orchestrator()
    cfg = load_config()
    assert cfg.providers["llamacpp"].model == "old.gguf"
    local = _LocalProvider(models=["new.gguf"])
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui.build_provider_from_profile", return_value=local):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/provider llamacpp"
            await pilot.press("enter")
            await pilot.pause()
            rendered = _transcript_text(app)
            assert "'old.gguf' is no longer available" in rendered
            assert "model · new.gguf" in rendered
            assert app.provider is local

    entry = load_settings()["providers"]["llamacpp"]
    assert entry["model"] == "new.gguf"
    # Pre-existing keys of the entry are preserved.
    assert entry["timeout"] == 300
    assert entry["base_url"] == "http://localhost:8080/v1"


@pytest.mark.asyncio
async def test_app_provider_switch_local_server_down_keeps_previous():
    """J5: a down server fails the switch with a hint; previous provider stays."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = load_config()
    down = _DownLocalProvider()
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui.build_provider_from_profile", return_value=down):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/provider llamacpp"
            await pilot.press("enter")
            await pilot.pause()
            rendered = _transcript_text(app)
            assert "Cannot switch provider" in rendered
            assert local_server_down_hint("llamacpp") in rendered
            assert app.provider is provider
            assert orch.provider is provider

    assert load_settings()["provider"] != "llamacpp"


@pytest.mark.asyncio
async def test_app_local_effort_note_shown_once():
    """J4: entering a local provider notes effort once; effort is untouched."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    orch.effort = Effort.HIGH
    cfg = load_config()
    local = _LocalProvider(models=["m1"], base_url="http://localhost:1234/v1")
    app = tui.make_app(orch, provider, Effort.HIGH, cfg=cfg)

    with patch("rickshaw.tui.build_provider_from_profile", return_value=local):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/provider lmstudio"
            await pilot.press("enter")
            await pilot.pause()
            app.query_one("#prompt").value = "/provider lmstudio"
            await pilot.press("enter")
            await pilot.pause()
            rendered = _transcript_text(app)
            note = (
                "effort is not applicable to local provider lmstudio "
                "— using provider defaults"
            )
            assert rendered.count(note) == 1
            assert orch.effort == Effort.HIGH
            assert "Reset to medium" not in rendered


@pytest.mark.asyncio
async def test_app_hosted_effort_reset_warning_unchanged():
    """Hosted providers keep the effort-reset warning on every switch."""
    pytest.importorskip("textual")

    class _NoHighProvider(_FakeProvider):
        def capabilities(self):
            return Capabilities(
                streaming=True,
                function_calling=False,
                effort_levels=[Effort.LOW, Effort.MEDIUM],
                max_context_tokens=4096,
            )

    orch, provider, _memory = _make_orchestrator()
    orch.effort = Effort.HIGH
    cfg = RickshawConfig()
    cfg.providers["hosted"] = ProviderProfile(
        base_url="https://api.example.com/v1", model="m",
        api_key_env="HOSTED_KEY", wire_format="openai",
    )
    hosted = _NoHighProvider()
    app = tui.make_app(orch, provider, Effort.HIGH, cfg=cfg)

    with patch("rickshaw.tui.build_provider_from_profile", return_value=hosted):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/provider hosted"
            await pilot.press("enter")
            await pilot.pause()
            assert orch.effort == Effort.MEDIUM
            orch.effort = Effort.HIGH
            app.query_one("#prompt").value = "/provider hosted"
            await pilot.press("enter")
            await pilot.pause()
            rendered = _transcript_text(app)
            assert rendered.count(
                "note: hosted does not support effort high. Reset to medium."
            ) == 2
            assert "effort is not applicable" not in rendered


@pytest.mark.asyncio
async def test_app_settings_flow_local_auto_selects_single_model():
    """J1: the on-launch picker auto-selects a lone local model and persists."""
    pytest.importorskip("textual")
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=None, memory=memory)
    cfg = load_config()
    local = _LocalProvider(models=["solo-model"])
    applied = _LocalProvider(models=["solo-model"])
    applied._model = "solo-model"
    app = tui.make_app(orch, None, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui.build_provider_from_profile", return_value=local), \
         patch("rickshaw.tui._rebuild_provider", return_value=applied):
        async with app.run_test() as pilot:
            await pilot.pause()
            assert "Pick a provider" in _transcript_text(app)

            app.query_one("#prompt").value = "llamacpp"
            await pilot.press("enter")
            await pilot.pause()
            rendered = _transcript_text(app)
            assert "Pick a model" not in rendered
            assert "model: solo-model" in rendered
            assert "llamacpp · solo-model" in rendered
            assert app.provider is applied
            assert app._settings_state is None
            assert "llamacpp" in _statusbar_text(app)

    s = load_settings()
    assert s["provider"] == "llamacpp"
    assert s["providers"]["llamacpp"]["model"] == "solo-model"


@pytest.mark.asyncio
async def test_app_settings_flow_local_no_models_shows_hint():
    """An empty local model list aborts the picker with the per-server hint."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = load_config()
    local = _LocalProvider(models=[])
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    with patch("rickshaw.tui.build_provider_from_profile", return_value=local):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "/settings"
            await pilot.press("enter")
            await pilot.pause()

            app.query_one("#prompt").value = "llamacpp"
            await pilot.press("enter")
            await pilot.pause()
            rendered = _transcript_text(app)
            assert "No models available for llamacpp" in rendered
            assert local_no_models_hint("llamacpp") in rendered
            assert app._settings_state is None
            assert app.provider is provider


@pytest.mark.asyncio
async def test_app_provider_add_wizard_optional_key_for_local_url():
    """J7: the api_key_env step is skippable when the base URL is local."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/provider add"
        await pilot.press("enter")
        await pilot.pause()

        for val in ["myllama", "http://localhost:9090/v1"]:
            prompt.value = val
            await pilot.press("enter")
            await pilot.pause()

        hint = str(app.query_one("#hint").render())
        assert "optional for local — Enter to skip" in hint

        for val in ["", ""]:
            prompt.value = val
            await pilot.press("enter")
            await pilot.pause()

        assert "myllama" in cfg.providers
        p = cfg.providers["myllama"]
        assert p.base_url == "http://localhost:9090/v1"
        assert p.api_key_env == ""
        assert p.wire_format == "openai"
        assert "provider registered" in _transcript_text(app)

    assert load_settings()["providers"]["myllama"]["api_key_env"] == ""


@pytest.mark.asyncio
async def test_app_provider_add_wizard_requires_key_for_hosted_url():
    """Hosted base URLs still require an api_key_env in the wizard."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    cfg = RickshawConfig()
    app = tui.make_app(orch, provider, Effort.MEDIUM, cfg=cfg)

    async with app.run_test() as pilot:
        prompt = app.query_one("#prompt")
        prompt.value = "/provider add"
        await pilot.press("enter")
        await pilot.pause()

        for val in ["hostedeng", "https://api.example.com/v1"]:
            prompt.value = val
            await pilot.press("enter")
            await pilot.pause()

        hint = str(app.query_one("#hint").render())
        assert "optional for local" not in hint

        prompt.value = ""
        await pilot.press("enter")
        await pilot.pause()
        assert "api_key_env is required." in _transcript_text(app)
        assert "hostedeng" not in cfg.providers


@pytest.mark.asyncio
async def test_app_turn_error_connection_shows_server_hint():
    """J6: a connection error on a turn carries the local server hint."""
    pytest.importorskip("textual")
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    local = _LocalProvider(models=["m1"])
    local._model = "m1"
    orch = Orchestrator(provider=local, memory=memory)
    cfg = load_config()
    app = tui.make_app(orch, local, Effort.MEDIUM, cfg=cfg)

    def _boom(text, on_delta=None, **kwargs):
        raise httpx.ConnectError("connection refused")

    orch.run_turn = _boom
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")

        rendered = ""
        for _ in range(100):
            rendered = _transcript_text(app)
            if "Error:" in rendered:
                break
            await pilot.pause(0.05)
        assert "connection refused" in rendered
        assert local_server_down_hint("llamacpp") in rendered


@pytest.mark.asyncio
async def test_app_turn_error_timeout_suggests_timeout_setting():
    """J10: a generation timeout suggests raising the per-profile timeout."""
    pytest.importorskip("textual")
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    local = _LocalProvider(models=["m1"])
    local._model = "m1"
    orch = Orchestrator(provider=local, memory=memory)
    cfg = load_config()
    app = tui.make_app(orch, local, Effort.MEDIUM, cfg=cfg)

    def _slow(text, on_delta=None, **kwargs):
        raise httpx.ReadTimeout("timed out")

    orch.run_turn = _slow
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")

        rendered = ""
        for _ in range(100):
            rendered = _transcript_text(app)
            if "Error:" in rendered:
                break
            await pilot.pause(0.05)
        assert (
            "increase providers.llamacpp.timeout in ~/.rickshaw/settings.json"
            in rendered
        )


@pytest.mark.asyncio
async def test_app_turn_error_hosted_has_no_local_hint():
    """Hosted turn failures render exactly as before."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    def _boom(text, on_delta=None, **kwargs):
        raise httpx.ConnectError("connection refused")

    orch.run_turn = _boom
    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hi"
        await pilot.press("enter")

        rendered = ""
        for _ in range(100):
            rendered = _transcript_text(app)
            if "Error:" in rendered:
                break
            await pilot.pause(0.05)
        assert "Error: connection refused" in rendered
        assert "running?" not in rendered


@patch("rickshaw.tui._build_provider")
def test_main_validate_only_local_failure_prints_hint(mock_build, capsys):
    """--validate-only against a down local server exits 1 with the hint."""
    provider = _FakeProvider()
    provider.validate = MagicMock(side_effect=ValueError(
        "llamacpp unreachable at http://localhost:8080/v1 — connection refused"
    ))
    mock_build.return_value = provider

    with pytest.raises(SystemExit) as excinfo:
        tui.main(
            ["--provider", "llamacpp", "--validate-only", "--db-path", ":memory:"]
        )

    assert excinfo.value.code == 1
    err = capsys.readouterr().err
    assert "Provider validation failed (llamacpp)" in err
    assert local_server_down_hint("llamacpp") in err


# --- LLM visibility / trace block tests -------------------------------------


@pytest.mark.asyncio
async def test_trace_block_appears_after_turn():
    """A collapsed trace block is appended after the assistant turn completes."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator(function_calling=False)
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hello"
        await pilot.press("enter")

        for _ in range(100):
            if not app._turn_active:
                break
            await pilot.pause(0.02)

        trace_blocks = app.query(".trace-block")
        assert len(trace_blocks) == 1
        summary = str(trace_blocks[0].query_one(".summary").render())
        assert "steps" in summary
        assert "s" in summary  # duration suffix

        # Expanded details contain the grouped answer block.
        app.action_toggle_trace()
        await pilot.pause()
        line_texts = "\n".join(
            str(line._summary_widget.render())
            for line in trace_blocks[0]._line_widgets
            if line._summary_widget is not None
        )
        assert "answer" in line_texts.lower()
        answer_content = "\n".join(
            str(line._content_widget.render())
            for line in trace_blocks[0]._line_widgets
            if line._content_widget is not None
        )
        assert "Hello from fake" in answer_content


@pytest.mark.asyncio
async def test_spinner_text_updates_for_events():
    """The turn indicator label reflects the latest lifecycle event type."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator()
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        await pilot.pause()
        app._turn_active = True
        app._turn_seq = 1
        app._first_token = False
        app._phase_label = "Thinking…"
        app._live_tokens = 0
        app._turn_started = 0.0

        cases = [
            (events.ContextStart(), "Assembling context…"),
            (events.ContextDone(record_count=2, token_estimate=10), "Assembling context…"),
            (events.PromptBuilt(message_count=3, token_estimate=20), "Building prompt…"),
            (events.LLMCallStart(attempt=1, model="fake"), "Calling LLM…"),
            (events.TurnToolCallStart(call_id="1", tool_name="recall", arguments={}), "Calling tool recall…"),
            (events.Retry(attempt=1, max_retries=2, delay=1.0, error="boom"), "Retrying LLM call…"),
            (events.Degraded(reason="Falling back to local memory"), "Degraded — showing local memory…"),
        ]
        for event, expected in cases:
            app._on_turn_event(event, 1)
            text = app._indicator_text()
            assert expected in text, f"expected {expected!r} in {text!r}"

        # After the first text delta the indicator should switch to Streaming.
        app._on_turn_event(events.TurnTextDelta(text="hi"), 1)
        assert "Streaming…" in app._indicator_text()


@pytest.mark.asyncio
async def test_ctrl_o_expands_and_collapses_trace_block():
    """Ctrl+O toggles the selected turn's trace block details."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator(function_calling=False)
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hello"
        await pilot.press("enter")

        for _ in range(100):
            if not app._turn_active:
                break
            await pilot.pause(0.02)

        trace = app.query_one(".trace-block")
        assert not trace._expanded

        app.action_toggle_trace()
        await pilot.pause()
        assert trace._expanded
        assert trace.details.display is True
        # Details are rendered as a list of per-line widgets.
        assert len(trace._line_widgets) > 0
        line_texts = "\n".join(
            str(line._summary_widget.render())
            for line in trace._line_widgets
            if line._summary_widget is not None
        )
        assert "answer" in line_texts.lower()
        answer_content = "\n".join(
            str(line._content_widget.render())
            for line in trace._line_widgets
            if line._content_widget is not None
        )
        assert "Hello from fake" in answer_content

        app.action_toggle_trace()
        await pilot.pause()
        assert not trace._expanded
        assert trace.details.display is False


@pytest.mark.asyncio
async def test_ctrl_up_down_navigates_turns():
    """Ctrl+Up/Down moves the selection highlight between turns."""
    pytest.importorskip("textual")
    orch, provider, _memory = _make_orchestrator(function_calling=False)
    app = tui.make_app(orch, provider, Effort.MEDIUM)

    async with app.run_test() as pilot:
        for value in ("first", "second"):
            app.query_one("#prompt").value = value
            await pilot.press("enter")
            for _ in range(100):
                if not app._turn_active:
                    break
                await pilot.pause(0.02)

        assert len(app._turns) == 2
        assert app._selected_turn_index == -1

        # Ctrl+Up from prompt selects the most recent turn.
        app.action_prev_turn()
        assert app._selected_turn_index == 1
        assert "selected" in app._turns[1]["user"].classes
        assert "selected" in app._turns[1]["trace"].classes

        # Ctrl+Up again moves to the older turn.
        app.action_prev_turn()
        assert app._selected_turn_index == 0
        assert "selected" not in app._turns[1]["trace"].classes
        assert "selected" in app._turns[0]["trace"].classes

        # Ctrl+Down moves back to the newer turn and then to the prompt.
        app.action_next_turn()
        assert app._selected_turn_index == 1
        app.action_next_turn()
        assert app._selected_turn_index == -1


@pytest.mark.asyncio
async def test_trace_events_persisted_to_store(tmp_path):
    """Turn lifecycle events are written to the provided TraceStore."""
    pytest.importorskip("textual")
    db = tmp_path / "trace.db"
    trace_store = tui.TraceStore(db_path=str(db))
    provider = _FakeProvider(function_calling=False)
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory, trace_store=trace_store)
    app = tui.make_app(orch, provider, Effort.MEDIUM, trace_store=trace_store)

    async with app.run_test() as pilot:
        app.query_one("#prompt").value = "hello"
        await pilot.press("enter")

        for _ in range(100):
            if not app._turn_active:
                break
            await pilot.pause(0.02)

        trace_store.flush(timeout=2.0)
        turn_id = app._turns[-1]["turn_id"]
        trace = trace_store.get_trace(turn_id)
        assert trace is not None
        event_types = [e["type"] for e in trace["events"]]
        assert "turn_start" in event_types
        assert "turn_done" in event_types

    trace_store.close()


@pytest.mark.asyncio
async def test_trace_r_raw_toggle():
    """Pressing ``r`` on a focused trace line toggles raw JSON for the block."""
    pytest.importorskip("textual")

    def _mock_format(event_records, **kwargs):
        return TraceView(
            summary="1 step · 0 tool calls · 0 retries · completed · 0.0s",
            header_lines=["hello · completed · 0.00s", "fake/fake-model"],
            lines=[
                TraceLine(
                    timestamp="+0.10s",
                    label="tool",
                    summary="recall(query)",
                    raw_json='{"type": "tool_call_done"}',
                    expandable=True,
                    color_class="trace-tool",
                )
            ],
            step_count=1,
        )

    orch, provider, _memory = _make_orchestrator(function_calling=False)
    app = tui.make_app(orch, provider, Effort.MEDIUM)
    with patch("rickshaw.tui.format_trace", side_effect=_mock_format):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "hello"
            await pilot.press("enter")
            for _ in range(100):
                if not app._turn_active:
                    break
                await pilot.pause(0.02)

            trace = app.query_one(".trace-block")
            app.action_toggle_trace()
            await pilot.pause()
            line = trace._line_widgets[0]
            line.focus()
            await pilot.pause()

            rendered = str(line._summary_widget.render())
            assert "recall(query)" in rendered

            await pilot.press("r")
            await pilot.pause()
            rendered = str(line._summary_widget.render())
            assert '"type": "tool_call_done"' in rendered

            await pilot.press("r")
            await pilot.pause()
            rendered = str(line._summary_widget.render())
            assert "recall(query)" in rendered


@pytest.mark.asyncio
async def test_trace_per_event_expand():
    """Enter on a focused trace line expands/collapses its raw payload."""
    pytest.importorskip("textual")

    def _mock_format(event_records, **kwargs):
        return TraceView(
            summary="1 step · 0 tool calls · 0 retries · completed · 0.0s",
            header_lines=["hello · completed · 0.00s", "fake/fake-model"],
            lines=[
                TraceLine(
                    timestamp="+0.10s",
                    label="tool",
                    summary="recall(query)",
                    raw_json='{"type": "tool_call_done"}',
                    expandable=True,
                    color_class="trace-tool",
                )
            ],
            step_count=1,
        )

    orch, provider, _memory = _make_orchestrator(function_calling=False)
    app = tui.make_app(orch, provider, Effort.MEDIUM)
    with patch("rickshaw.tui.format_trace", side_effect=_mock_format):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "hello"
            await pilot.press("enter")
            for _ in range(100):
                if not app._turn_active:
                    break
                await pilot.pause(0.02)

            trace = app.query_one(".trace-block")
            app.action_toggle_trace()
            await pilot.pause()
            line = trace._line_widgets[0]
            line.focus()
            await pilot.pause()

            assert not line._expanded
            await pilot.press("enter")
            await pilot.pause()
            assert line._expanded
            assert line._content_widget is not None
            assert line._content_widget.display is True
            assert '"type": "tool_call_done"' in str(line._content_widget.render())

            await pilot.press("enter")
            await pilot.pause()
            assert not line._expanded
            assert line._content_widget.display is False


@pytest.mark.asyncio
async def test_trace_grouped_answer_and_thinking():
    """Grouped answer and thinking blocks render with their merged content."""
    pytest.importorskip("textual")

    def _mock_format(event_records, **kwargs):
        return TraceView(
            summary="2 steps · 0 tool calls · 0 retries · completed · 0.0s",
            header_lines=["hello · completed · 0.00s", "fake/fake-model"],
            lines=[
                TraceLine(
                    timestamp="+0.10s",
                    label="thinking",
                    summary="(2 Δ)",
                    content="step one\nstep two",
                    expandable=True,
                    color_class="trace-thinking",
                ),
                TraceLine(
                    timestamp="+0.20s",
                    label="answer",
                    summary="(1 Δ)",
                    content="final answer",
                    expandable=True,
                    color_class="trace-answer",
                ),
            ],
            step_count=2,
        )

    orch, provider, _memory = _make_orchestrator(function_calling=False)
    app = tui.make_app(orch, provider, Effort.MEDIUM)
    with patch("rickshaw.tui.format_trace", side_effect=_mock_format):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "hello"
            await pilot.press("enter")
            for _ in range(100):
                if not app._turn_active:
                    break
                await pilot.pause(0.02)

            trace = app.query_one(".trace-block")
            app.action_toggle_trace()
            await pilot.pause()
            line_texts = "\n".join(
                str(line._summary_widget.render())
                for line in trace._line_widgets
                if line._summary_widget is not None
            )
            assert "thinking" in line_texts.lower()
            assert "answer" in line_texts.lower()
            content_text = "\n".join(
                str(line._content_widget.render())
                for line in trace._line_widgets
                if line._content_widget is not None
            )
            assert "final answer" in content_text
            assert "step one" in content_text


@pytest.mark.asyncio
async def test_trace_contextual_hint():
    """The bottom hint updates when a trace line has focus."""
    pytest.importorskip("textual")

    def _mock_format(event_records, **kwargs):
        return TraceView(
            summary="1 step · 0 tool calls · 0 retries · completed · 0.0s",
            header_lines=["hello · completed · 0.00s", "fake/fake-model"],
            lines=[
                TraceLine(
                    timestamp="+0.10s",
                    label="answer",
                    summary="(1 Δ)",
                    content="hi",
                    expandable=True,
                    color_class="trace-answer",
                )
            ],
            step_count=1,
        )

    orch, provider, _memory = _make_orchestrator(function_calling=False)
    app = tui.make_app(orch, provider, Effort.MEDIUM)
    with patch("rickshaw.tui.format_trace", side_effect=_mock_format):
        async with app.run_test() as pilot:
            app.query_one("#prompt").value = "hello"
            await pilot.press("enter")
            for _ in range(100):
                if not app._turn_active:
                    break
                await pilot.pause(0.02)

            trace = app.query_one(".trace-block")
            app.action_toggle_trace()
            await pilot.pause()
            trace._line_widgets[0].focus()
            await pilot.pause()

            hint_text = str(app.query_one("#hint").render())
            assert tui._TRACE_HINT in hint_text

            await pilot.press("escape")
            await pilot.pause()
            hint_text = str(app.query_one("#hint").render())
            assert tui._DEFAULT_HINT in hint_text
