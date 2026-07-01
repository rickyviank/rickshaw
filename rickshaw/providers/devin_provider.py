"""Devin provider implementation.

This provider connects to the Devin API for agentic code-generation tasks.
Many endpoint details are left as TODOs because the exact Devin API contract
may evolve; fill them in from the Devin API documentation.
"""

from __future__ import annotations

import os
from typing import Any, Iterator

import httpx

from rickshaw.providers.base import (
    Capabilities,
    Effort,
    LLMProvider,
    Message,
    Response,
    TokenUsage,
    ToolCall,
    ToolSpec,
)


class DevinProvider(LLMProvider):
    """Provider for the Devin API.

    Devin is a coding agent and may not expose a traditional chat-completions
    interface.  This implementation provides the skeleton; fill in TODOs once
    the API specifics are available.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("DEVIN_API_KEY", "")
        self._base_url = (
            base_url or os.environ.get("DEVIN_BASE_URL", "https://api.devin.ai")
        ).rstrip("/")

    @property
    def name(self) -> str:
        return "devin"

    def _headers(self) -> dict[str, str]:
        # TODO: Confirm the exact auth header expected by the Devin API.
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """Parse Devin-format tool calls into normalized :class:`ToolCall`s.

        Devin does not yet support function-calling, so this returns an empty
        list. Implement once the Devin API documents its tool-call format.
        """
        # TODO: Parse tool calls once the Devin API supports function-calling.
        return []

    def complete(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Response:
        # TODO: Forward and parse tool calls once Devin API supports function-calling.
        # Currently reports function_calling=False; tools/tool_choice are accepted but ignored.
        # TODO: Replace with the actual Devin API endpoint and request shape.
        # TODO: Map the normalized ``Effort`` to Devin's reasoning/effort/iteration parameter.
        payload: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            # TODO: Add effort/iteration mapping here once the Devin API
            #       documents the parameter name (e.g. "effort", "iterations").
        }
        payload.update(kwargs)

        with httpx.Client(timeout=300) as client:
            # TODO: Confirm the endpoint path (e.g. /v1/chat/completions).
            resp = client.post(
                f"{self._base_url}/v1/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        # TODO: Adjust response parsing to match the actual Devin API response shape.
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        usage_data = data.get("usage", {})

        return Response(
            text=text,
            model=data.get("model", "devin"),
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            effort=effort,
            raw=data,
        )

    # stream() is inherited from LLMProvider — falls back to complete().
    # TODO: Override with native streaming once the Devin API supports it.

    def available_models(self) -> list[str]:
        # TODO: Query the Devin API for available models/agents.
        return ["devin"]

    def validate(self) -> None:
        if not self._api_key:
            raise ValueError(
                "DEVIN_API_KEY is not set. "
                "Set the environment variable or pass api_key to the provider."
            )
        # TODO: Hit a lightweight Devin API endpoint to verify connectivity.

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=False,
            function_calling=False,
            vision=False,
            embeddings=False,
            max_context_tokens=128_000,
            # TODO: Populate effort_levels once Devin's effort parameter is confirmed.
            effort_levels=[],
        )
