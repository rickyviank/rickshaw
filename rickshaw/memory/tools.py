"""Memory tools — remember/recall/forget as normalized tool specs + registry wiring."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from rickshaw.providers.base import ToolCall, ToolSpec
from rickshaw.tool_registry import ToolRegistry

if TYPE_CHECKING:
    from rickshaw.memory.service import MemoryService


REMEMBER_SPEC = ToolSpec(
    name="remember",
    description="Store a fact or observation in long-term memory.",
    parameters={
        "type": "object",
        "properties": {
            "fact": {
                "type": "string",
                "description": "The fact or observation to remember.",
            },
        },
        "required": ["fact"],
    },
    category="memory",
    side_effect=True,
)

RECALL_SPEC = ToolSpec(
    name="recall",
    description="Retrieve relevant memories matching a query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "A natural-language query to search memories.",
            },
        },
        "required": ["query"],
    },
    category="memory",
    side_effect=False,  # read-only: does not count against the tool-round budget
)

FORGET_SPEC = ToolSpec(
    name="forget",
    description="Delete a memory record by its id.",
    parameters={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The id of the memory record to delete.",
            },
        },
        "required": ["id"],
    },
    category="memory",
    side_effect=True,
)

MEMORY_TOOL_SPECS: list[ToolSpec] = [REMEMBER_SPEC, RECALL_SPEC, FORGET_SPEC]


def build_memory_registry(
    memory_service: MemoryService,
    registry: ToolRegistry | None = None,
) -> ToolRegistry:
    """Register the memory tools (remember/recall/forget) on a ToolRegistry.

    Non-memory tools (web search, file ops, ...) can be registered separately on
    the same registry. Returns the registry for convenience.
    """
    registry = registry or ToolRegistry()
    registry.register(
        "remember", lambda args: memory_service.remember(args.get("fact", "")), REMEMBER_SPEC
    )
    registry.register(
        "recall", lambda args: memory_service.recall(args.get("query", "")), RECALL_SPEC
    )
    registry.register(
        "forget", lambda args: memory_service.forget(args.get("id", "")), FORGET_SPEC
    )
    return registry


def dispatch_tool_call(
    tool_call: ToolCall,
    memory_service: MemoryService,
) -> str:
    """Backward-compatible convenience wrapper that dispatches via a registry.

    Prefer constructing a :class:`ToolRegistry` (see :func:`build_memory_registry`)
    and calling :meth:`ToolRegistry.dispatch` directly.
    """
    return build_memory_registry(memory_service).dispatch(tool_call)
