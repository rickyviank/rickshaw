"""Orchestrator — owns the turn loop.

The only hot-path caller of the provider. Depends on LLMProvider via
dependency injection, forwards Effort, advertises tool specs from an injected
:class:`ToolRegistry`, and dispatches returned tool calls.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable

import httpx

from rickshaw.memory.service import MemoryService
from rickshaw.memory.tools import build_memory_registry
from rickshaw.prompt.builder import PromptBuilder
from rickshaw.providers.base import (
    Effort,
    LLMProvider,
    Message,
    Response,
    TokenUsage,
)
from rickshaw.queue import Job, JobQueue, JobType
from rickshaw.tool_registry import ToolRegistry

# Callback invoked with incremental text as a turn's final answer is produced.
StreamCallback = Callable[[str], None]

logger = logging.getLogger(__name__)

_MAX_TOOL_ROUNDS = 3
_MAX_RETRIES = 2
_RETRY_BACKOFF = 1.0  # seconds; delay = backoff * 2**attempt (1s, 2s)
# Absolute safety cap on loop iterations, so a stream of read-only tool calls
# (which don't count against max_tool_rounds) can't spin forever.
_HARD_ITERATION_CAP = 20

_PROVIDER_UNREACHABLE_MSG = "Provider unreachable — showing cached results"

_DEFAULT_SYSTEM = (
    "You are a helpful assistant with access to a semantic memory layer. "
    "Use the provided tools to remember, recall, or forget information."
)


@dataclass
class TurnResult:
    """Structured result of a single turn.

    ``text`` is the assistant's final text. ``warnings`` surfaces degradation
    (provider unreachable, function-calling unsupported) so callers/CLIs can
    display it without parsing ``text``. ``tool_calls_made`` counts dispatched
    tool calls. ``degraded`` is True when the turn fell back to local memory.
    """

    text: str
    warnings: list[str] = field(default_factory=list)
    tool_calls_made: int = 0
    degraded: bool = False
    model: str = ""
    usage: TokenUsage | None = None

    def __str__(self) -> str:  # convenience for print()/logging
        return self.text


def _is_transient_error(exc: Exception) -> bool:
    """Whether *exc* is a transient provider error worth retrying."""
    if isinstance(exc, (httpx.TransportError, ConnectionError, TimeoutError)):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        code = exc.response.status_code
        return code == 429 or code >= 500
    return False


class Orchestrator:
    """Turn loop with memory-augmented retrieval and tool dispatch.

    Degrades gracefully if:
    * The provider is unreachable (retries with backoff, then falls back to
      local remember/recall/ranking).
    * The provider reports ``function_calling=False`` (skips tool advertising
      and surfaces a warning).
    """

    def __init__(
        self,
        provider: LLMProvider,
        memory: MemoryService,
        prompt_builder: PromptBuilder | None = None,
        queue: JobQueue | None = None,
        registry: ToolRegistry | None = None,
        system: str = _DEFAULT_SYSTEM,
        effort: Effort = Effort.MEDIUM,
        max_tool_rounds: int = _MAX_TOOL_ROUNDS,
        max_retries: int = _MAX_RETRIES,
        retry_backoff: float = _RETRY_BACKOFF,
    ) -> None:
        self.provider = provider
        self.memory = memory
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.queue = queue or JobQueue()
        # Tool dispatch is decoupled from MemoryService via the registry. Memory
        # tools are registered here at construction; callers may inject a
        # pre-populated registry (e.g. with additional non-memory tools).
        self.registry = registry or build_memory_registry(memory)
        self.system = system
        self.effort = effort
        self.max_tool_rounds = max_tool_rounds
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # Session-start capability notice (item 7).
        if self.provider is not None and not self.provider.capabilities().function_calling:
            logger.info(
                "Provider '%s' does not support function-calling; memory tools "
                "will not be advertised to the model. Context retrieval is "
                "harness-driven.",
                self.provider.name,
            )

    def _complete_with_retry(self, messages: list[Message], tool_specs) -> Response:
        """Call the provider, retrying transient errors with exponential backoff."""
        attempt = 0
        while True:
            try:
                return self.provider.complete(
                    messages, effort=self.effort, tools=tool_specs,
                )
            except Exception as exc:
                if _is_transient_error(exc) and attempt < self.max_retries:
                    delay = self.retry_backoff * (2 ** attempt)
                    logger.warning(
                        "Transient provider error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.max_retries, delay, exc,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    attempt += 1
                    continue
                raise

    def run_turn(
        self,
        task_input: str,
        on_delta: StreamCallback | None = None,
    ) -> TurnResult:
        """Execute a single conversational turn.

        1. Assemble context from memory.
        2. Build the prompt.
        3. Call the provider (with tool specs if supported), retrying transient
           errors with backoff.
        4. Dispatch any tool calls via the registry; loop up to
           *max_tool_rounds* (read-only calls are exempt from the count).
        5. Write observations to memory.
        6. Enqueue deferred jobs (importance scoring).
        7. Return a :class:`TurnResult`.

        If *on_delta* is provided it is called with incremental text as the
        final answer is produced. When the provider supports streaming *and*
        tools are not advertised (no function-calling), text is streamed token
        by token via :meth:`LLMProvider.stream`. Otherwise the final text is
        delivered as a single delta after generation (the tool-call loop can't
        be streamed because tool-call parsing over the stream is provider work
        that is deferred). Passing ``on_delta=None`` preserves the original
        non-streaming behavior exactly.
        """
        warnings: list[str] = []
        ctx = self.memory.assemble_context(task_input)

        caps = self.provider.capabilities()
        use_tools = caps.function_calling
        tool_specs = self.registry.specs() if use_tools else None
        if not use_tools:
            warnings.append(
                f"Provider '{self.provider.name}' does not support function-calling; "
                "memory tools not advertised. Context retrieval is harness-driven."
            )

        messages = self.prompt_builder.build(
            system=self.system,
            tools=tool_specs,
            context=ctx,
            task_input=task_input,
        )

        # Real token streaming is only possible when we won't dispatch tools.
        if on_delta is not None and caps.streaming and not use_tools:
            return self._run_streaming_turn(messages, task_input, warnings, on_delta)

        try:
            response = self._complete_with_retry(messages, tool_specs)
        except Exception as exc:
            logger.warning("Provider unreachable after retries: %s", exc)
            warnings.append(_PROVIDER_UNREACHABLE_MSG)
            results = self.memory.recall(task_input)
            if results:
                text = (
                    f"{_PROVIDER_UNREACHABLE_MSG}:\n"
                    + "; ".join(r["text"] for r in results)
                )
            else:
                text = f"{_PROVIDER_UNREACHABLE_MSG} (no cached results found)."
            return TurnResult(
                text=text, warnings=warnings, tool_calls_made=0, degraded=True,
            )

        # Tool-call dispatch loop.
        tool_calls_made = 0
        rounds_used = 0
        iterations = 0
        while rounds_used < self.max_tool_rounds and iterations < _HARD_ITERATION_CAP:
            iterations += 1
            if not response.tool_calls:
                break

            round_has_side_effect = False
            for tc in response.tool_calls:
                # Errors are surfaced inside the JSON tool result so the model
                # can react (registry returns {"error": ...} on failure).
                result = self.registry.dispatch(tc)
                tool_calls_made += 1
                spec = self.registry.get_spec(tc.name)
                # Unknown tools default to side-effecting (conservative).
                if spec is None or spec.side_effect:
                    round_has_side_effect = True
                messages.append(Message(
                    role="assistant",
                    content=f"[tool_call: {tc.name}({tc.arguments})]",
                ))
                messages.append(Message(role="tool", content=result))

            # Read-only rounds (e.g. only recall) don't consume the budget.
            if round_has_side_effect:
                rounds_used += 1

            try:
                response = self._complete_with_retry(messages, tool_specs)
            except Exception as exc:
                logger.warning("Follow-up provider call failed after retries: %s", exc)
                warnings.append(_PROVIDER_UNREACHABLE_MSG)
                break

        # Write observations
        records = self.memory.write_observations(response)

        # Enqueue deferred jobs
        for rec in records:
            self.queue.enqueue(Job(
                type=JobType.IMPORTANCE_SCORING,
                payload={"record_id": rec.id},
            ))

        # Uniform streaming interface: deliver the final answer as one delta so
        # callers that passed on_delta render through the same path.
        if on_delta is not None and response.text:
            on_delta(response.text)

        return TurnResult(
            text=response.text,
            warnings=warnings,
            tool_calls_made=tool_calls_made,
            degraded=False,
            model=response.model,
            usage=response.usage,
        )

    def _run_streaming_turn(
        self,
        messages: list[Message],
        task_input: str,
        warnings: list[str],
        on_delta: StreamCallback,
    ) -> TurnResult:
        """Stream the final answer token by token (no tool dispatch path).

        Transient errors are retried only before any text has been emitted;
        once streaming has started we can't safely restart. On failure we
        degrade to local recall, mirroring :meth:`run_turn`'s non-streaming
        fallback.
        """
        parts: list[str] = []
        attempt = 0
        while True:
            try:
                for chunk in self.provider.stream(messages, effort=self.effort):
                    parts.append(chunk)
                    on_delta(chunk)
                break
            except Exception as exc:
                if not parts and _is_transient_error(exc) and attempt < self.max_retries:
                    delay = self.retry_backoff * (2 ** attempt)
                    logger.warning(
                        "Transient streaming error (attempt %d/%d), retrying in %.1fs: %s",
                        attempt + 1, self.max_retries, delay, exc,
                    )
                    if delay > 0:
                        time.sleep(delay)
                    attempt += 1
                    continue
                logger.warning("Streaming provider error: %s", exc)
                warnings.append(_PROVIDER_UNREACHABLE_MSG)
                if not parts:
                    results = self.memory.recall(task_input)
                    if results:
                        text = (
                            f"{_PROVIDER_UNREACHABLE_MSG}:\n"
                            + "; ".join(r["text"] for r in results)
                        )
                    else:
                        text = f"{_PROVIDER_UNREACHABLE_MSG} (no cached results found)."
                    on_delta(text)
                    return TurnResult(
                        text=text, warnings=warnings, tool_calls_made=0, degraded=True,
                    )
                break

        response = Response(
            text="".join(parts), model=self.provider.name, effort=self.effort,
        )
        records = self.memory.write_observations(response)
        for rec in records:
            self.queue.enqueue(Job(
                type=JobType.IMPORTANCE_SCORING,
                payload={"record_id": rec.id},
            ))
        return TurnResult(
            text=response.text,
            warnings=warnings,
            tool_calls_made=0,
            degraded=False,
            model=response.model,
            usage=response.usage,
        )
