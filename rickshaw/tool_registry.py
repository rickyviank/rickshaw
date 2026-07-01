"""Generalized, provider-agnostic tool dispatch registry.

`ToolRegistry` decouples tool dispatch from any particular backend (e.g.
`MemoryService`). Handlers are registered by name alongside their `ToolSpec`;
the registry validates arguments against the spec's JSON schema, then invokes
the handler. It supports both synchronous and asynchronous handlers so future
tools (provider-backed recall, web search, file ops) can be async without
changing callers.

Results are always returned JSON-serialized, ready to be used as the content of
a ``role="tool"`` message. Errors (unknown tool, validation failure, handler
exception) are surfaced as ``{"error": ...}`` payloads so the model can react.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Union

from rickshaw.providers.base import ToolCall, ToolSpec

# A handler receives the tool-call arguments dict and returns a JSON-serializable
# result (sync) or an awaitable of one (async).
Handler = Callable[[dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass
class _Entry:
    handler: Handler
    spec: ToolSpec


def _validate_arguments(spec: ToolSpec, arguments: dict[str, Any]) -> str | None:
    """Validate *arguments* against *spec.parameters*.

    Uses ``jsonschema`` when available; otherwise falls back to a simple
    required-fields check. Returns an error string, or ``None`` if valid.
    """
    params = spec.parameters or {}
    try:
        import jsonschema

        try:
            jsonschema.validate(instance=arguments, schema=params)
        except jsonschema.ValidationError as exc:
            return f"invalid arguments for '{spec.name}': {exc.message}"
        return None
    except ImportError:
        # Minimal fallback: enforce presence of declared required fields.
        for field in params.get("required", []):
            if field not in arguments:
                return f"invalid arguments for '{spec.name}': missing required field '{field}'"
        return None


class ToolRegistry:
    """Maps tool names to handler callables and their specs."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def register(self, name: str, handler: Handler, spec: ToolSpec) -> None:
        """Register *handler* (and its *spec*) under *name*."""
        if spec.name != name:
            raise ValueError(
                f"spec.name ({spec.name!r}) does not match registered name ({name!r})"
            )
        self._entries[name] = _Entry(handler=handler, spec=spec)

    def unregister(self, name: str) -> None:
        self._entries.pop(name, None)

    def __contains__(self, name: object) -> bool:
        return name in self._entries

    def specs(self) -> list[ToolSpec]:
        """Return the specs of all registered tools (for advertising to a model)."""
        return [entry.spec for entry in self._entries.values()]

    def get_spec(self, name: str) -> ToolSpec | None:
        entry = self._entries.get(name)
        return entry.spec if entry else None

    def _prepare(self, tool_call: ToolCall) -> tuple[_Entry | None, str | None]:
        """Look up + validate a tool call. Returns (entry, error_json_or_None)."""
        entry = self._entries.get(tool_call.name)
        if entry is None:
            return None, json.dumps({"error": f"unknown tool: {tool_call.name}"})
        err = _validate_arguments(entry.spec, tool_call.arguments)
        if err is not None:
            return None, json.dumps({"error": err, "type": "validation_error"})
        return entry, None

    def dispatch(self, tool_call: ToolCall) -> str:
        """Synchronously dispatch *tool_call*, returning a JSON string.

        Async handlers are run to completion via ``asyncio.run``.
        """
        entry, err = self._prepare(tool_call)
        if err is not None:
            return err
        assert entry is not None
        try:
            result = entry.handler(tool_call.arguments)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
        except Exception as exc:
            return json.dumps({"error": str(exc), "type": "handler_error"})
        return json.dumps(result)

    async def async_dispatch(self, tool_call: ToolCall) -> str:
        """Asynchronously dispatch *tool_call*, returning a JSON string.

        Async handlers are awaited directly; sync handlers run in a worker
        thread so they do not block the event loop.
        """
        entry, err = self._prepare(tool_call)
        if err is not None:
            return err
        assert entry is not None
        handler = entry.handler
        try:
            if inspect.iscoroutinefunction(handler):
                result = await handler(tool_call.arguments)
            else:
                # Run the (potentially blocking) sync handler off the event loop.
                result = await asyncio.to_thread(handler, tool_call.arguments)
                if inspect.isawaitable(result):
                    result = await result
        except Exception as exc:
            return json.dumps({"error": str(exc), "type": "handler_error"})
        return json.dumps(result)
