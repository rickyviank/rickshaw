"""Tests for the Textual-based terminal UI (rickshaw.tui).

The TUI is an optional extra (``pip install rickshaw[tui]``). Tests that need
Textual are skipped when it isn't installed; the import/wiring tests below only
need the module to import, which does not require Textual at import time.
"""

from __future__ import annotations

import json
from typing import Any, Iterator
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs

import httpx
import pytest
import respx

from rickshaw.memory.embedder import TFIDFEmbedder
from rickshaw.memory.service import MemoryService
from rickshaw.orchestrator import Orchestrator
from rickshaw.config import ProviderProfile, RickshawConfig
from rickshaw.providers.base import (
    Capabilities,
    Effort,
    LLMProvider,
    Message,
    Response,
    ToolSpec,
)

import rickshaw.tui as tui


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


def _make_orchestrator(function_calling: bool = False):
    provider = _FakeProvider(function_calling=function_calling)
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory)
    return orch, provider, memory


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

    def _fake_run(orchestrator, prov, effort, cfg):
        captured.update(orchestrator=orchestrator, provider=prov, effort=effort)

    with patch("rickshaw.tui._run_app", side_effect=_fake_run):
        tui.main(["--provider", "fake", "--effort", "high", "--db-path", ":memory:"])

    assert provider.validated is True
    orch = captured["orchestrator"]
    assert isinstance(orch, Orchestrator)
    assert orch.provider is provider
    assert isinstance(orch.memory, MemoryService)
    assert captured["effort"] == Effort.HIGH


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
    _, call_provider, _, _ = mock_run.call_args[0]
    assert call_provider is None


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
