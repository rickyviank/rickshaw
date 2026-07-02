"""Tests for the Textual-based terminal UI (rickshaw.tui).

The TUI is an optional extra (``pip install rickshaw[tui]``). Tests that need
Textual are skipped when it isn't installed; the import/wiring tests below only
need the module to import, which does not require Textual at import time.
"""

from __future__ import annotations

from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from rickshaw.memory.embedder import TFIDFEmbedder
from rickshaw.memory.service import MemoryService
from rickshaw.orchestrator import Orchestrator
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


@patch("rickshaw.tui._build_provider")
@patch("rickshaw.tui.load_config")
def test_main_survives_validation_failure(mock_config, mock_build):
    from rickshaw.config import RickshawConfig

    mock_config.return_value = RickshawConfig()
    provider = _FakeProvider()
    provider.validate = MagicMock(side_effect=ValueError("bad key"))
    mock_build.return_value = provider

    with patch("rickshaw.tui._run_app") as mock_run:
        tui.main(["--provider", "fake", "--db-path", ":memory:"])

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
