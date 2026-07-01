"""Anthropic (Claude) provider implementation."""

from __future__ import annotations

import os
from typing import Any

import httpx

from rickshaw.providers.base import (
    Capabilities,
    Effort,
    LLMProvider,
    Message,
    Response,
    ToolCall,
    ToolSpec,
    TokenUsage,
)

_ANTHROPIC_VERSION = "2023-06-01"

# Default number of tokens Claude may generate; the Messages API requires
# ``max_tokens`` to be set explicitly, unlike OpenAI's chat completions.
_DEFAULT_MAX_TOKENS = 4096

# Models that support Anthropic's extended-thinking parameter. Effort is only
# translated into thinking parameters for these; otherwise it is reflected back
# in the Response but not sent (mirroring DevinProvider's ignore-if-unsupported
# behavior).
_MODELS_SUPPORTING_THINKING = ("claude-3-7", "claude-opus-4", "claude-sonnet-4")

# Map the normalized effort onto an extended-thinking token budget. Only used
# for models that advertise thinking support.
_EFFORT_THINKING_BUDGET: dict[Effort, int] = {
    Effort.LOW: 1024,
    Effort.MEDIUM: 4096,
    Effort.HIGH: 16384,
}

# Map OpenAI-style tool_choice values onto Anthropic's shape.
_TOOL_CHOICE_MAP: dict[str, dict[str, str]] = {
    "auto": {"type": "auto"},
    "any": {"type": "any"},
    "required": {"type": "any"},  # OpenAI's "required" == Anthropic's "any"
    "none": {"type": "none"},
}


def _model_supports_thinking(model: str) -> bool:
    return any(model.startswith(prefix) for prefix in _MODELS_SUPPORTING_THINKING)


class AnthropicProvider(LLMProvider):
    """Provider for the Anthropic Messages API.

    Anthropic has no embeddings API, so this provider does not use
    :class:`EmbeddingMixin` and reports ``embeddings=False``.
    """

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
        # Anthropic uses ``x-api-key`` plus an API-version header, NOT the
        # ``Authorization: Bearer`` scheme used by OpenAI/Devin.
        return {
            "x-api-key": self._api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_tool_calls(content_blocks: list[dict[str, Any]]) -> list[ToolCall]:
        """Parse Anthropic ``tool_use`` blocks into normalized :class:`ToolCall`s.

        Unlike OpenAI, whose ``arguments`` are a JSON-encoded string, Anthropic's
        ``input`` is already a decoded dict, so no JSON parsing is required.
        """
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

    @staticmethod
    def _tools_payload(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """Convert normalized ToolSpec list into Anthropic's tools format."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            }
            for t in tools
        ]

    @staticmethod
    def _split_messages(
        messages: list[Message],
    ) -> tuple[str | None, list[dict[str, str]]]:
        """Hoist system prompts out into a single top-level ``system`` string.

        Anthropic's Messages API takes the system prompt as a top-level
        ``system`` field rather than a ``role="system"`` entry in the messages
        list. Multiple system messages are joined with blank lines.
        """
        system_parts: list[str] = []
        chat: list[dict[str, str]] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                chat.append({"role": m.role, "content": m.content})
        system = "\n\n".join(system_parts) if system_parts else None
        return system, chat

    def complete(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Response:
        system, chat = self._split_messages(messages)

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": chat,
            "max_tokens": _DEFAULT_MAX_TOKENS,
        }
        if system is not None:
            payload["system"] = system

        # Effort is always reflected back in the Response. It is only translated
        # into extended-thinking parameters when the target model supports them;
        # otherwise it is ignored (like DevinProvider).
        if _model_supports_thinking(self._model):
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": _EFFORT_THINKING_BUDGET[effort],
            }

        if tools:
            payload["tools"] = self._tools_payload(tools)
            if tool_choice is not None:
                mapped = _TOOL_CHOICE_MAP.get(tool_choice)
                if mapped is not None:
                    payload["tool_choice"] = mapped

        payload.update(kwargs)

        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{self._base_url}/v1/messages",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        content_blocks = data.get("content", [])
        text = "".join(
            block.get("text", "")
            for block in content_blocks
            if block.get("type") == "text"
        )
        usage_data = data.get("usage", {})
        parsed_tool_calls = self._parse_tool_calls(content_blocks)

        return Response(
            text=text,
            model=data.get("model", self._model),
            usage=TokenUsage(
                prompt_tokens=usage_data.get("input_tokens", 0),
                completion_tokens=usage_data.get("output_tokens", 0),
                total_tokens=(
                    usage_data.get("input_tokens", 0)
                    + usage_data.get("output_tokens", 0)
                ),
            ),
            effort=effort,
            raw=data,
            tool_calls=parsed_tool_calls,
        )

    # stream() is inherited from LLMProvider — falls back to complete().
    # TODO: Override with native SSE streaming of Anthropic message deltas.

    def available_models(self) -> list[str]:
        # Anthropic exposes no public list-models endpoint, so return a static
        # list of well-known Claude model identifiers.
        return [
            "claude-3-5-sonnet-latest",
            "claude-3-5-haiku-latest",
            "claude-3-opus-latest",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

    def validate(self) -> None:
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
