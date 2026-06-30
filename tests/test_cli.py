"""Tests for CLI argument parsing and effort handling."""

from unittest.mock import MagicMock, patch

import pytest

from rickshaw.cli import _parse_args, main
from rickshaw.providers.base import (
    Capabilities,
    Effort,
    Message,
    Response,
    TokenUsage,
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


@patch("rickshaw.cli._run_repl")
@patch("rickshaw.cli._build_provider")
@patch("rickshaw.cli.load_config")
def test_main_passes_effort_to_repl(mock_config, mock_build, mock_repl):
    """--effort flag is resolved and passed to _run_repl."""
    from rickshaw.config import RickshawConfig

    mock_config.return_value = RickshawConfig()
    provider = MagicMock()
    provider.name = "openai"
    provider.validate = MagicMock()
    provider.capabilities.return_value = Capabilities(effort_levels=list(Effort))
    mock_build.return_value = provider

    main(["--effort", "high"])

    mock_repl.assert_called_once()
    _, call_effort = mock_repl.call_args[0]
    assert call_effort == Effort.HIGH


@patch("rickshaw.cli._build_provider")
@patch("rickshaw.cli.load_config")
def test_main_validate_only_success(mock_config, mock_build, capsys):
    from rickshaw.config import RickshawConfig

    mock_config.return_value = RickshawConfig()
    provider = MagicMock()
    provider.name = "openai"
    provider.validate = MagicMock()
    mock_build.return_value = provider

    main(["--validate-only"])

    captured = capsys.readouterr()
    assert "validated successfully" in captured.out


@patch("rickshaw.cli._build_provider")
@patch("rickshaw.cli.load_config")
def test_main_validate_only_failure(mock_config, mock_build):
    from rickshaw.config import RickshawConfig

    mock_config.return_value = RickshawConfig()
    provider = MagicMock()
    provider.name = "openai"
    provider.validate.side_effect = ValueError("bad key")
    mock_build.return_value = provider

    with pytest.raises(SystemExit):
        main(["--validate-only"])
