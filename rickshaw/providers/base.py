"""Normalized types and abstract provider interface."""

from __future__ import annotations

import enum
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
    """Description of a tool the model may call.

    ``category`` classifies the tool (e.g. ``"memory"`` vs ``"general"``) so the
    orchestrator can apply category-specific handling. ``side_effect`` marks
    whether invoking the tool mutates state: read-only tools (``side_effect=
    False``) do not count against the orchestrator's bounded tool-round budget.
    """

    name: str
    description: str
    parameters: dict[str, Any]
    category: str = "general"
    side_effect: bool = True


@dataclass
class ToolCall:
    """A normalized tool/function call returned by the model.

    This is a pure, vendor-neutral data container. Provider-specific parsing
    (e.g. from OpenAI's wire format) lives on each provider via a
    ``_parse_tool_calls`` method, not on this dataclass.
    """

    id: str
    name: str
    arguments: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)


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
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Response:
        """Send *messages* and return a normalized :class:`Response`.

        *tools* advertises the tool specifications available to the model.
        Providers that do not support function-calling should ignore it.

        *tool_choice* controls whether the model is encouraged, required, or
        forbidden from selecting a tool. Accepts ``"auto"`` (model decides),
        ``"none"`` (never call a tool), ``"required"`` (must call a tool), or
        ``None`` (provider default). It only has effect when *tools* is set.
        """

    def stream(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield incremental text chunks.

        The default implementation falls back to :meth:`complete` and yields
        the full text as a single chunk, so providers without native streaming
        still satisfy the interface.
        """
        response = self.complete(
            messages, effort=effort, tools=tools, tool_choice=tool_choice, **kwargs
        )
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
