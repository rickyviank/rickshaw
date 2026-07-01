"""OpenAI provider implementation."""

from __future__ import annotations

import json as _json
import os
from typing import Any, Iterator

import httpx

from rickshaw.providers.base import (
    Capabilities,
    EmbeddingMixin,
    Effort,
    LLMProvider,
    Message,
    Response,
    ToolCall,
    ToolSpec,
    TokenUsage,
)

_EFFORT_MAP: dict[Effort, str] = {
    Effort.LOW: "low",
    Effort.MEDIUM: "medium",
    Effort.HIGH: "high",
}

_MODELS_SUPPORTING_EFFORT = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"}


def _model_supports_effort(model: str) -> bool:
    base = model.split("-")[0] if "-" in model else model
    return any(model.startswith(prefix) for prefix in _MODELS_SUPPORTING_EFFORT)


class OpenAIProvider(EmbeddingMixin, LLMProvider):
    """Provider for the OpenAI chat completions and embeddings APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        embedding_model: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = (
            base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
        self._embedding_model = embedding_model or os.environ.get(
            "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
        )

    @property
    def name(self) -> str:
        return "openai"

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """Parse OpenAI-format tool calls into normalized :class:`ToolCall`s.

        Handles the ``{"function": {"name": ..., "arguments": "..."}}`` shape,
        where ``arguments`` is a JSON-encoded string.
        """
        parsed: list[ToolCall] = []
        for raw_call in raw_calls:
            func = raw_call.get("function", {})
            args_str = func.get("arguments", "{}")
            try:
                args = _json.loads(args_str)
            except (_json.JSONDecodeError, TypeError):
                args = {}
            parsed.append(
                ToolCall(
                    id=raw_call.get("id", ""),
                    name=func.get("name", ""),
                    arguments=args,
                    raw=raw_call,
                )
            )
        return parsed

    @staticmethod
    def _tools_payload(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """Convert normalized ToolSpec list into OpenAI tools format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for t in tools
        ]

    def complete(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Response:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
        }

        if _model_supports_effort(self._model):
            payload["reasoning_effort"] = _EFFORT_MAP[effort]

        if tools:
            payload["tools"] = self._tools_payload(tools)
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        payload.update(kwargs)

        with httpx.Client(timeout=120) as client:
            resp = client.post(
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        message = choice["message"]
        usage_data = data.get("usage", {})

        parsed_tool_calls = self._parse_tool_calls(message.get("tool_calls", []))

        return Response(
            text=message.get("content") or "",
            model=data.get("model", self._model),
            usage=TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
            ),
            effort=effort,
            raw=data,
            tool_calls=parsed_tool_calls,
        )

    def stream(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        # TODO: streaming tool-call parsing can be deferred
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "stream": True,
        }

        if _model_supports_effort(self._model):
            payload["reasoning_effort"] = _EFFORT_MAP[effort]

        payload.update(kwargs)

        with httpx.Client(timeout=120) as client:
            with client.stream(
                "POST",
                f"{self._base_url}/chat/completions",
                headers=self._headers(),
                json=payload,
            ) as resp:
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data_str = line[len("data: "):]
                    if data_str.strip() == "[DONE]":
                        break
                    import json

                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    content = delta.get("content")
                    if content:
                        yield content

    def embed(self, text: str) -> list[float]:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                f"{self._base_url}/embeddings",
                headers=self._headers(),
                json={"model": self._embedding_model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
        return data["data"][0]["embedding"]

    def available_models(self) -> list[str]:
        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
            )
            resp.raise_for_status()
            data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    def validate(self) -> None:
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Set the environment variable or pass api_key to the provider."
            )
        with httpx.Client(timeout=15) as client:
            resp = client.get(
                f"{self._base_url}/models",
                headers=self._headers(),
            )
            resp.raise_for_status()

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=True,
            function_calling=True,
            vision=True,
            embeddings=True,
            max_context_tokens=128_000,
            effort_levels=(
                list(Effort) if _model_supports_effort(self._model) else []
            ),
        )
