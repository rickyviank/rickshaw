#!/usr/bin/env python3
"""Offline end-to-end demo using the local embedder and a fake LLM provider.

Demonstrates the full run_turn cycle including tool-call dispatch, memory
write-back, and deferred job enqueueing — all without any network calls.
"""

from __future__ import annotations

import json
from typing import Any, Iterator

from rickshaw.memory.embedder import TFIDFEmbedder
from rickshaw.memory.service import MemoryService
from rickshaw.orchestrator import Orchestrator
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
from rickshaw.queue import JobQueue
from rickshaw.worker import DeferredWorker


class FakeLLMProvider(LLMProvider):
    """Fake provider that returns canned responses with tool calls."""

    def __init__(self) -> None:
        self._call_count = 0

    @property
    def name(self) -> str:
        return "fake"

    def complete(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Response:
        self._call_count += 1

        # First call: return a tool call to remember a fact
        if self._call_count == 1 and tools:
            return Response(
                text="",
                model="fake-model",
                usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                effort=effort,
                tool_calls=[
                    ToolCall(
                        id="call_demo_1",
                        name="remember",
                        arguments={"fact": "The user prefers dark mode."},
                    )
                ],
            )

        # Subsequent calls: return a plain text response
        return Response(
            text="I've noted your preference. How can I help you further?",
            model="fake-model",
            usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
            effort=effort,
        )

    def stream(
        self,
        messages: list[Message],
        effort: Effort = Effort.MEDIUM,
        tools: list[ToolSpec] | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        response = self.complete(messages, effort=effort, tools=tools, **kwargs)
        yield response.text

    def available_models(self) -> list[str]:
        return ["fake-model"]

    def validate(self) -> None:
        pass

    def capabilities(self) -> Capabilities:
        return Capabilities(
            streaming=False,
            function_calling=True,
            vision=False,
            embeddings=False,
            max_context_tokens=4096,
        )


def main() -> None:
    print("=== Rickshaw Offline Demo ===\n")

    # Set up components
    embedder = TFIDFEmbedder(dim=32)
    memory = MemoryService(embedder=embedder)
    provider = FakeLLMProvider()
    queue = JobQueue()
    orchestrator = Orchestrator(
        provider=provider, memory=memory, queue=queue, effort=Effort.MEDIUM,
    )
    worker = DeferredWorker(queue=queue, memory=memory, provider=provider)

    # Turn 1: triggers a tool call (remember)
    print("Turn 1: 'Remember that I prefer dark mode'")
    result = orchestrator.run_turn("Remember that I prefer dark mode")
    print(f"  Response: {result}")
    print(f"  Pending deferred jobs: {queue.pending_count}")

    # Process deferred jobs
    processed = worker.process_batch()
    print(f"  Processed {processed} deferred job(s)\n")

    # Turn 2: recall from memory
    print("Turn 2: 'What are my preferences?'")
    result = orchestrator.run_turn("What are my preferences?")
    print(f"  Response: {result}")

    # Show stored memories
    print("\n--- Stored memories ---")
    records = memory.store.all_records()
    for r in records:
        print(f"  [{r.scope.value}/{r.type.value}] {r.text[:80]}...")

    print(f"\nTotal memories: {len(records)}")
    print("Demo complete.")


if __name__ == "__main__":
    main()
