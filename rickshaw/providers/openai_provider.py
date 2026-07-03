"""OpenAI provider — a sync facade over :mod:`rickshaw_ai`.

Request translation, HTTP, retries, and streaming are delegated to
``rickshaw_ai``'s OpenAI-compatible adapter. This class keeps the harness's
synchronous :class:`~rickshaw.providers.base.LLMProvider` contract (and its
embeddings/model-listing helpers, which live outside ``rickshaw_ai``'s
tool-calling scope).
"""

from __future__ import annotations

import json as _json
import os
from typing import Any, Iterator

import httpx

from rickshaw.config import is_local_url
from rickshaw.providers import _bridge
from rickshaw.providers.base import (
    Capabilities,
    EmbeddingMixin,
    Effort,
    LLMProvider,
    Message,
    Response,
    ToolCall,
    ToolSpec,
)
from rickshaw_ai.registry import ModelInfo, ProviderInfo

_EFFORT_MAP: dict[Effort, str] = {
    Effort.LOW: "low",
    Effort.MEDIUM: "medium",
    Effort.HIGH: "high",
}

_MODELS_SUPPORTING_EFFORT = {"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"}


def _model_supports_effort(model: str) -> bool:
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

    # -- rickshaw_ai wiring ------------------------------------------------

    def _provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            id="openai",
            base_url=self._base_url,
            protocol="openai",
            api_key_header="Authorization",
            api_key_prefix="Bearer ",
        )

    def _model_info(self) -> ModelInfo:
        return ModelInfo(
            id=f"openai/{self._model}",
            provider_id="openai",
            model=self._model,
            supports_tools=True,
            supports_reasoning=_model_supports_effort(self._model),
            supports_vision_input=True,
        )

    # -- tool-call parsing (canonical helpers, reused by the response map) --

    @staticmethod
    def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolCall]:
        """Parse OpenAI-format tool calls into normalized :class:`ToolCall`s."""
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

    # -- completion / streaming (delegated) --------------------------------

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
        message = (raw.get("choices") or [{}])[0].get("message", {})
        tool_calls = self._parse_tool_calls(message.get("tool_calls", []))
        return _bridge.to_response(
            result, effort=effort, tool_calls=tool_calls, fallback_model=self._model
        )

    def stream(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        req = _bridge.GenerateRequest(
            messages=_bridge.to_ai_messages(messages),
            reasoning=_bridge.Reasoning(effort=_EFFORT_MAP[effort]),
            provider_options=dict(kwargs),
        )
        yield from _bridge.stream_text(
            self._provider_info(), self._model_info(), self._api_key, req
        )

    # -- embeddings / model listing (outside rickshaw_ai scope) ------------

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

    def _fetch_models(self) -> list[str]:
        with httpx.Client(timeout=30) as client:
            resp = client.get(f"{self._base_url}/models", headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        return [m["id"] for m in data.get("data", [])]

    def available_models(self) -> list[str]:
        return self._cached_available_models(
            self._fetch_models,
            cache_key=f"openai:{self._base_url}",
            is_local=is_local_url(self._base_url),
        )

    def validate(self) -> None:
        if _bridge.has_stored_credential(self.name):
            return
        if not self._api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Set the environment variable or pass api_key to the provider."
            )
        with httpx.Client(timeout=15) as client:
            resp = client.get(f"{self._base_url}/models", headers=self._headers())
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
