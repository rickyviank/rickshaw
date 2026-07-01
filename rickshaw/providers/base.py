"""Normalized types and abstract provider interface."""

from __future__ import annotations

import enum
import json as _json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator


class Effort(enum.Enum):
    """Reasoning effort level requested for a completion."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class Message:
    """A single message in a conversation."""

    role: str
    content: str


@dataclass
class TokenUsage:
    """Token counts for a completion."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ToolSpec:
    """Description of a tool the model may call."""

    name: str
    description: str
    parameters: dict[str, Any]


@dataclass
class ToolCall:
    """A normalized tool/function call returned by the model."""

    id: str
    name: str
    arguments: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_openai(cls, raw_call: dict[str, Any]) -> ToolCall:
        """Parse an OpenAI-format tool call dict into a normalized ToolCall."""
        func = raw_call.get("function", {})
        args_str = func.get("arguments", "{}")
        try:
            args = _json.loads(args_str)
        except (_json.JSONDecodeError, TypeError):
            args = {}
        return cls(
            id=raw_call.get("id", ""),
            name=func.get("name", ""),
            arguments=args,
            raw=raw_call,
        )


@dataclass
class Response:
    """Normalized response from any LLM provider."""

    text: str
    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    effort: Effort = Effort.MEDIUM
    raw: dict[str, Any] = field(default_factory=dict)
    tool_calls: list[ToolCall] = field(default_factory=list)


@dataclass
class Capabilities:
    """Structured description of what a provider supports."""

    streaming: bool = False
    function_calling: bool = False
    vision: bool = False
    embeddings: bool = False
    max_context_tokens: int = 0
    effort_levels: list[Effort] = field(default_factory=lambda: list(Effort))


class EmbeddingMixin:
    """Optional mixin for providers that support embeddings."""

    def embed(self, text: str) -> list[float]:
        """Return an embedding vector for *text*.

        Providers that support embeddings should override this method and
        report ``embeddings=True`` in :meth:`capabilities`.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support embeddings"
        )


class LLMProvider(ABC):
    """Abstract base class every provider must implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used by the factory / CLI (e.g. ``'openai'``)."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Response:
        """Send *messages* and return a normalized :class:`Response`.

        *tools* is an optional list of tool specifications the model may call.
        Providers that do not support function-calling should ignore it.
        """

    def stream(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield incremental text chunks.

        The default implementation falls back to :meth:`complete` and yields
        the full text as a single chunk, so providers without native streaming
        still satisfy the interface.
        """
        response = self.complete(messages, effort=effort, tools=tools, **kwargs)
        yield response.text

    @abstractmethod
    def available_models(self) -> list[str]:
        """Return a list of model identifiers this provider can serve."""

    @abstractmethod
    def validate(self) -> None:
        """Verify credentials and connectivity.

        Should raise a descriptive exception on failure so the CLI can
        surface a clear error early.
        """

    @abstractmethod
    def capabilities(self) -> Capabilities:
        """Return a :class:`Capabilities` object describing this provider."""
