"""Tests for CLI argument parsing, effort handling, and the unified entry point.

After unification, the ``rickshaw`` console script points at
:func:`rickshaw.tui.main`.  ``rickshaw.cli`` still exports ``_parse_args``,
``_build_provider``, ``_EFFORT_NAMES``, and ``load_config`` for backward
compatibility.
"""

from unittest.mock import MagicMock, patch

import pytest

from rickshaw.cli import _parse_args, _build_provider, _EFFORT_NAMES
from rickshaw.providers.base import (
    Capabilities,
    Effort,
)


def test_parse_effort_flag():
    args = _parse_args(["--effort", "high"])
    assert args.effort == "high"


def test_parse_provider_flag():
    args = _parse_args(["--provider", "devin"])
    assert args.provider == "devin"


def test_parse_validate_only():
    args = _parse_args(["--validate-only"])
    assert args.validate_only is True


def test_parse_defaults():
    args = _parse_args([])
    assert args.effort is None
    assert args.provider is None
    assert args.validate_only is False


def test_effort_names_mapping():
    assert _EFFORT_NAMES["low"] == Effort.LOW
    assert _EFFORT_NAMES["medium"] == Effort.MEDIUM
    assert _EFFORT_NAMES["high"] == Effort.HIGH


@patch("rickshaw.tui._run_app")
@patch("rickshaw.tui._build_provider")
@patch("rickshaw.tui.load_config")
def test_main_passes_effort_to_tui(mock_config, mock_build, mock_run):
    """--effort flag is resolved and passed to _run_app."""
    from rickshaw.config import RickshawConfig
    from rickshaw.tui import main

    mock_config.return_value = RickshawConfig()
    provider = MagicMock()
    provider.name = "openai"
    provider.validate = MagicMock()
    provider.capabilities.return_value = Capabilities(effort_levels=list(Effort))
    mock_build.return_value = provider

    main(["--effort", "high", "--db-path", ":memory:"])

    mock_run.assert_called_once()
    _, _, call_effort, _ = mock_run.call_args[0]
    assert call_effort == Effort.HIGH


@patch("rickshaw.tui._build_provider")
@patch("rickshaw.tui.load_config")
def test_main_validate_only_success(mock_config, mock_build, capsys):
    from rickshaw.config import RickshawConfig
    from rickshaw.tui import main

    mock_config.return_value = RickshawConfig()
    provider = MagicMock()
    provider.name = "openai"
    provider.validate = MagicMock()
    mock_build.return_value = provider

    main(["--provider", "openai", "--validate-only"])

    captured = capsys.readouterr()
    assert "validated successfully" in captured.out


@patch("rickshaw.tui._build_provider")
@patch("rickshaw.tui.load_config")
def test_main_validate_only_failure(mock_config, mock_build):
    from rickshaw.config import RickshawConfig
    from rickshaw.tui import main

    mock_config.return_value = RickshawConfig()
    provider = MagicMock()
    provider.name = "openai"
    provider.validate.side_effect = ValueError("bad key")
    mock_build.return_value = provider

    with pytest.raises(SystemExit):
        main(["--validate-only"])


@patch("rickshaw.tui._build_provider")
@patch("rickshaw.tui.load_config")
def test_main_validation_failure_exits_by_default(mock_config, mock_build):
    """Without --allow-unvalidated, validation failure exits non-zero."""
    from rickshaw.config import RickshawConfig
    from rickshaw.tui import main

    mock_config.return_value = RickshawConfig()
    provider = MagicMock()
    provider.name = "openai"
    provider.validate.side_effect = ValueError("bad key")
    mock_build.return_value = provider

    with pytest.raises(SystemExit):
        main(["--provider", "openai"])


@patch("rickshaw.tui._run_app")
@patch("rickshaw.tui._build_provider")
@patch("rickshaw.tui.load_config")
def test_main_allow_unvalidated_continues(mock_config, mock_build, mock_run):
    """--allow-unvalidated lets the app launch despite validation failure."""
    from rickshaw.config import RickshawConfig
    from rickshaw.tui import main

    mock_config.return_value = RickshawConfig()
    provider = MagicMock()
    provider.name = "openai"
    provider.validate.side_effect = ValueError("bad key")
    mock_build.return_value = provider

    main(["--provider", "openai", "--allow-unvalidated", "--db-path", ":memory:"])

    mock_run.assert_called_once()


def test_parse_allow_unvalidated_flag():
    from rickshaw.tui import _parse_args

    args = _parse_args(["--allow-unvalidated"])
    assert args.allow_unvalidated is True

    args = _parse_args([])
    assert args.allow_unvalidated is False


@patch("rickshaw.tui._run_app")
@patch("rickshaw.tui.load_config")
def test_main_bare_launch_no_error(mock_config, mock_run):
    """A bare ``rickshaw`` launch (no --provider) no longer errors."""
    from rickshaw.config import RickshawConfig
    from rickshaw.tui import main

    mock_config.return_value = RickshawConfig()

    main(["--db-path", ":memory:"])

    mock_run.assert_called_once()
    _, call_provider, _, _ = mock_run.call_args[0]
    # Provider should be None when nothing is configured.
    assert call_provider is None


@patch("rickshaw.tui._run_app")
@patch("rickshaw.tui._build_provider")
@patch("rickshaw.tui.load_config")
def test_main_provider_flag_still_works(mock_config, mock_build, mock_run):
    """--provider flag is respected and builds the provider normally."""
    from rickshaw.config import RickshawConfig
    from rickshaw.tui import main

    mock_config.return_value = RickshawConfig()
    provider = MagicMock()
    provider.name = "openai"
    provider.validate = MagicMock()
    provider.capabilities.return_value = Capabilities(effort_levels=list(Effort))
    mock_build.return_value = provider

    main(["--provider", "openai", "--db-path", ":memory:"])

    mock_build.assert_called_once()
    mock_run.assert_called_once()
    _, call_provider, _, _ = mock_run.call_args[0]
    assert call_provider is provider
