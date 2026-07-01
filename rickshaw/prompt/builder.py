"""PromptBuilder — assembles the outgoing prompt within a token budget.

Sensitive/local-only records are excluded upstream in
:meth:`MemoryService.assemble_context` (the privacy/egress boundary), so by the
time records reach the builder they are already safe to serialize. The builder's
remaining job is to cap the total prompt size to *max_tokens*.
"""

from __future__ import annotations

from rickshaw.memory.record import MemoryRecord
from rickshaw.providers.base import Message, ToolSpec


def _estimate_tokens(text: str) -> int:
    """Simple token estimate: ~4 chars per token, fallback heuristic."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)


class PromptBuilder:
    """Build a ``list[Message]`` ready for ``LLMProvider.complete()``.

    Sensitive records are filtered out upstream (see
    :meth:`MemoryService.assemble_context`); the builder only enforces the
    *max_tokens* budget.
    """

    def __init__(self, max_tokens: int = 8000) -> None:
        self.max_tokens = max_tokens

    def build(
        self,
        system: str,
        tools: list[ToolSpec] | None,
        context: list[MemoryRecord],
        task_input: str,
    ) -> list[Message]:
        """Assemble a prompt from system instructions, context, and user input.

        * The context section is truncated to fit the token budget.
        * Tool specs are passed through to the provider via the ``tools``
          argument (not embedded in message text).

        Sensitive records are expected to have already been excluded by
        :meth:`MemoryService.assemble_context`.
        """
        messages: list[Message] = []

        # System message
        system_tokens = _estimate_tokens(system)
        messages.append(Message(role="system", content=system))

        # Budget for context = max_tokens - system - user_input - headroom
        user_tokens = _estimate_tokens(task_input)
        headroom = 200  # reserve for tool overhead / response
        remaining = self.max_tokens - system_tokens - user_tokens - headroom

        # Serialize context records within the remaining token budget.
        context_parts: list[str] = []
        used = 0
        for record in context:
            part = f"[{record.scope.value}/{record.type.value}] {record.text}"
            part_tokens = _estimate_tokens(part)
            if used + part_tokens > remaining:
                break
            context_parts.append(part)
            used += part_tokens

        if context_parts:
            context_block = "Relevant context:\n" + "\n".join(context_parts)
            messages.append(Message(role="system", content=context_block))

        # User message
        messages.append(Message(role="user", content=task_input))

        return messages
