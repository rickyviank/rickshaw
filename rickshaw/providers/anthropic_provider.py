"""Anthropic (Claude) provider — a sync facade over :mod:`rickshaw_ai`.

Message hoisting, extended-thinking translation, tool payloads, HTTP, and
retries are delegated to ``rickshaw_ai``'s Anthropic adapter. This class keeps
the harness's synchronous :class:`~rickshaw.providers.base.LLMProvider`
contract.
"""

from __future__ import annotations

import os
from typing import Any

from rickshaw.providers import _bridge
from rickshaw.providers.base import (
    Capabilities,
    Effort,
    LLMProvider,
    Message,
    Response,
    ToolCall,
    ToolSpec,
)
from rickshaw_ai.registry import ModelInfo, ProviderInfo

# Models that support Anthropic's extended-thinking parameter. Effort is only
# translated into thinking parameters for these; otherwise it is reflected back
# in the Response but not sent.
_MODELS_SUPPORTING_THINKING = ("claude-3-7", "claude-opus-4", "claude-sonnet-4")

_EFFORT_MAP: dict[Effort, str] = {
    Effort.LOW: "low",
    Effort.MEDIUM: "medium",
    Effort.HIGH: "high",
}


def _model_supports_thinking(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _MODELS_SUPPORTING_THINKING)


class AnthropicProvider(LLMProvider):
    """Provider for the Anthropic Messages API (no embeddings)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base_url = (
            base_url or os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        ).rstrip("/")
        self._model = model or os.environ.get(
            "ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"
        )

    @property
    def name(self) -> str:
        return "anthropic"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }

    # -- rickshaw_ai wiring ------------------------------------------------

    def _provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            id="anthropic",
            base_url=self._base_url,
            protocol="anthropic",
            api_key_header="x-api-key",
            api_key_prefix="",
        )

    def _model_info(self) -> ModelInfo:
        return ModelInfo(
            id=f"anthropic/{self._model}",
            provider_id="anthropic",
            model=self._model,
            supports_tools=True,
            supports_reasoning=_model_supports_thinking(self._model),
            supports_vision_input=True,
        )

    @staticmethod
    def _parse_tool_calls(content_blocks: list[dict[str, Any]]) -> list[ToolCall]:
        """Parse Anthropic ``tool_use`` blocks into normalized :class:`ToolCall`s."""
        parsed: list[ToolCall] = []
        for block in content_blocks:
            if block.get("type") != "tool_use":
                continue
            args = block.get("input")
            if not isinstance(args, dict):
                args = {}
            parsed.append(
                ToolCall(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=args,
                    raw=block,
                )
            )
        return parsed

    # -- completion (delegated) --------------------------------------------

    def complete(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Response:
        req = _bridge.GenerateRequest(
            messages=_bridge.to_ai_messages(messages),
            tools=_bridge.to_ai_tools(tools),
            tool_choice=tool_choice if tools else None,
            reasoning=_bridge.Reasoning(effort=_EFFORT_MAP[effort]),
            provider_options=dict(kwargs),
        )
        result = _bridge.generate(
            self._provider_info(), self._model_info(), self._api_key, req
        )
        raw = result.metadata.get("raw", {})
        tool_calls = self._parse_tool_calls(raw.get("content", []))
        return _bridge.to_response(
            result, effort=effort, tool_calls=tool_calls, fallback_model=self._model
        )

    # stream() is inherited from LLMProvider — falls back to complete().

    @staticmethod
    def _static_models() -> list[str]:
        return [
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

    def available_models(self) -> list[str]:
        from rickshaw.config import is_local_url

        return self._cached_available_models(
            self._static_models,
            cache_key=f"anthropic:{self._base_url}",
            is_local=is_local_url(self._base_url),
        )

    def validate(self) -> None:
        if _bridge.has_stored_credential(self.name):
            return
        if not self._api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY is not set. "
                "Set the environment variable or pass api_key to the provider."
            )

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            function_calling=True,
            vision=True,
            embeddings=False,
            max_context_tokens=200_000,
            effort_levels=[],
        )
