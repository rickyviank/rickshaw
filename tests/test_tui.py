"""Tests for the Rich-based terminal UI (rickshaw.tui).

The TUI is an optional extra (``pip install rickshaw[tui]``). Rendering tests
that need Rich are skipped when it isn't installed; the wiring tests below only
need the module to import, which does not require Rich at import time.
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
    """Minimal fake provider — never hits the network."""

    def __init__(self, function_calling: bool = False) -> None:
        self._function_calling = function_calling
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
        return Response(text="Hello from fake", model="fake", effort=effort)

    def stream(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        yield self.complete(messages, effort=effort, tools=tools).text

    def available_models(self) -> list[str]:
        return ["fake"]

    def validate(self) -> None:
        self.validated = True

    def capabilities(self) -> Capabilities:
        return Capabilities(
            function_calling=self._function_calling,
            effort_levels=list(Effort),
            max_context_tokens=4096,
        )


def test_module_imports_without_rich():
    """The module (and branding constants) import even without Rich installed."""
    assert tui.RICKSHAW_LOGO == "o--o  rickshaw"
    assert tui.RICKSHAW_SLOGAN == "your driver, your memory"
    assert "rickshaw" in tui.RICKSHAW_BANNER
    assert "your driver, your memory" in tui.RICKSHAW_BANNER


def test_cli_reuses_branding_constants():
    """cli._print_header pulls the same constants from tui (item 5)."""
    from rickshaw import cli

    provider = _FakeProvider()
    cli._print_header(provider, Effort.MEDIUM)  # exercises the lazy import path


def test_parse_args_defaults_and_overrides():
    args = tui._parse_args([])
    assert args.provider is None
    assert args.effort is None
    assert args.db_path == tui._DEFAULT_DB_PATH

    args = tui._parse_args(["--provider", "openai", "--effort", "high", "--db-path", "x.db"])
    assert args.provider == "openai"
    assert args.effort == "high"
    assert args.db_path == "x.db"


@patch("rickshaw.tui._build_provider")
@patch("rickshaw.tui.load_config")
def test_main_wiring_constructs_and_runs_loop(mock_config, mock_build):
    """main() builds the provider, MemoryService + Orchestrator, and runs the loop.

    The loop itself is stubbed so no Rich or live input is needed; we assert the
    orchestrator handed to it is wired to the fake provider and in-memory store.
    """
    from rickshaw.config import RickshawConfig

    mock_config.return_value = RickshawConfig()
    provider = _FakeProvider()
    mock_build.return_value = provider

    captured: dict[str, Any] = {}

    def _fake_loop(orchestrator, prov, effort):
        captured["orchestrator"] = orchestrator
        captured["provider"] = prov
        captured["effort"] = effort

    with patch("rickshaw.tui._run_tui", side_effect=_fake_loop):
        tui.main(["--provider", "fake", "--effort", "high", "--db-path", ":memory:"])

    assert provider.validated is True
    orch = captured["orchestrator"]
    assert isinstance(orch, Orchestrator)
    assert orch.provider is provider
    assert isinstance(orch.memory, MemoryService)
    assert captured["effort"] == Effort.HIGH
    assert orch.effort == Effort.HIGH


@patch("rickshaw.tui._build_provider")
@patch("rickshaw.tui.load_config")
def test_main_survives_validation_failure(mock_config, mock_build):
    """A failing provider.validate() is caught; main still wires and runs."""
    from rickshaw.config import RickshawConfig

    mock_config.return_value = RickshawConfig()
    provider = _FakeProvider()
    provider.validate = MagicMock(side_effect=ValueError("bad key"))
    mock_build.return_value = provider

    with patch("rickshaw.tui._run_tui") as mock_loop:
        tui.main(["--provider", "fake", "--db-path", ":memory:"])

    mock_loop.assert_called_once()


def test_run_tui_routes_turn_through_orchestrator():
    """A single turn is rendered via the Orchestrator (requires Rich)."""
    pytest.importorskip("rich")

    provider = _FakeProvider()
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory)

    run_turn_spy = MagicMock(wraps=orch.run_turn)
    orch.run_turn = run_turn_spy  # type: ignore[method-assign]

    # Feed one user turn then EOF (Ctrl-D) to exit the loop.
    inputs = iter(["hello there", EOFError()])

    def _fake_input(_prompt):
        value = next(inputs)
        if isinstance(value, BaseException):
            raise value
        return value

    with patch("rich.console.Console.input", side_effect=_fake_input):
        tui._run_tui(orch, provider, Effort.MEDIUM)

    run_turn_spy.assert_called_once_with("hello there")


def test_run_tui_effort_command_updates_orchestrator():
    """/effort <level> mid-session updates orchestrator.effort (requires Rich)."""
    pytest.importorskip("rich")

    provider = _FakeProvider()
    memory = MemoryService(embedder=TFIDFEmbedder(dim=32))
    orch = Orchestrator(provider=provider, memory=memory, effort=Effort.MEDIUM)

    inputs = iter(["/effort low", "/quit"])

    with patch("rich.console.Console.input", side_effect=lambda _p: next(inputs)):
        tui._run_tui(orch, provider, Effort.MEDIUM)

    assert orch.effort == Effort.LOW
