"""Tool definition, validation, dispatch, and streaming assembly.

A :class:`Tool` is the provider-neutral description of a callable the model may
invoke. Each provider adapter translates it to the appropriate wire shape
(Anthropic ``tools``, OpenAI ``tools[type=function]``, Google
``functionDeclarations``). Tool *arguments* returned by a model are validated
against the tool's JSON Schema before use.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Awaitable, Callable, Union, get_type_hints

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field, TypeAdapter

from rickshaw_ai.errors import ToolInputError

Handler = Callable[..., Union[Any, Awaitable[Any]]]


class ToolCall(BaseModel):
    """A tool call requested by a model, in canonical form."""

    id: str
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Tool(BaseModel):
    """A callable advertised to the model.

    ``parameters`` is a JSON Schema object describing the arguments. ``handler``
    is optional — callers may execute tools themselves. ``category`` and
    ``side_effect`` carry through orchestration hints used by consumers.
    """

    model_config = {"arbitrary_types_allowed": True}

    name: str
    description: str = ""
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {"type": "object", "properties": {}}
    )
    handler: Handler | None = None
    category: str = "general"
    side_effect: bool = True

    async def invoke(self, arguments: dict[str, Any]) -> Any:
        """Run the tool's handler, awaiting async handlers."""
        if self.handler is None:
            raise ToolInputError(f"tool {self.name!r} has no handler")
        result = self.handler(arguments)
        if inspect.isawaitable(result):
            result = await result
        return result


def _schema_from_signature(func: Callable[..., Any]) -> dict[str, Any]:
    """Derive a JSON Schema object from a function's typed signature."""
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception as exc:  # pragma: no cover - exotic annotations
        logger.warning(
            "Cannot resolve type hints for %s, falling back to "
            "untyped parameters: %s",
            func.__qualname__, exc,
        )
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("self", "cls"):
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue
        annotation = hints.get(name, str)
        try:
            schema = TypeAdapter(annotation).json_schema()
        except Exception as exc:  # pragma: no cover - unresolved annotation
            logger.warning(
                "Cannot derive schema for parameter %r of %s, "
                "defaulting to string: %s",
                name, func.__qualname__, exc,
            )
            schema = {"type": "string"}
        properties[name] = schema
        if param.default is inspect.Parameter.empty:
            required.append(name)

    out: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        out["required"] = required
    return out


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    category: str = "general",
    side_effect: bool = True,
) -> Any:
    """Decorator that turns a typed function into a :class:`Tool`.

    The parameter schema is derived from the signature/type hints and the
    description defaults to the function's docstring. The wrapped function
    becomes the tool's handler, called with a single ``arguments`` dict.
    """

    def _wrap(fn: Callable[..., Any]) -> Tool:
        params = _schema_from_signature(fn)

        def _handler(arguments: dict[str, Any]) -> Any:
            return fn(**arguments)

        if inspect.iscoroutinefunction(fn):
            async def _ahandler(arguments: dict[str, Any]) -> Any:
                return await fn(**arguments)

            handler: Handler = _ahandler
        else:
            handler = _handler

        return Tool(
            name=name or fn.__name__,
            description=description or (inspect.getdoc(fn) or ""),
            parameters=params,
            handler=handler,
            category=category,
            side_effect=side_effect,
        )

    if func is not None:
        return _wrap(func)
    return _wrap


def validate_arguments(spec_parameters: dict[str, Any], arguments: dict[str, Any]) -> None:
    """Validate *arguments* against a JSON-Schema *spec_parameters*.

    Uses ``jsonschema`` when installed; otherwise falls back to enforcing the
    presence of declared ``required`` fields. Raises :class:`ToolInputError`.
    """
    params = spec_parameters or {}
    try:
        import jsonschema  # type: ignore

        try:
            jsonschema.validate(instance=arguments, schema=params)
        except jsonschema.ValidationError as exc:  # type: ignore[attr-defined]
            raise ToolInputError(f"invalid tool arguments: {exc.message}") from exc
        return
    except ImportError:
        for field in params.get("required", []):
            if field not in arguments:
                raise ToolInputError(
                    f"invalid tool arguments: missing required field {field!r}"
                )


def validate_call(tools: list[Tool], call: ToolCall) -> dict[str, Any]:
    """Validate *call* against the matching tool in *tools*.

    Returns the validated arguments. Raises :class:`ToolInputError` for an
    unknown tool name or invalid arguments.
    """
    by_name = {t.name: t for t in tools}
    spec = by_name.get(call.name)
    if spec is None:
        raise ToolInputError(f"unknown tool: {call.name}")
    validate_arguments(spec.parameters, call.arguments)
    return call.arguments


class ToolCallAssembler:
    """Accumulates streamed tool-call fragments into complete :class:`ToolCall`s.

    Providers stream tool arguments as partial JSON deltas keyed by an index or
    id. Feed fragments via :meth:`start`/:meth:`delta`, then :meth:`finish` to
    parse and return the assembled calls. Malformed JSON raises
    :class:`ToolInputError`.
    """

    def __init__(self) -> None:
        self._order: list[str] = []
        self._names: dict[str, str] = {}
        self._ids: dict[str, str] = {}
        self._buffers: dict[str, str] = {}

    def start(self, key: str, *, call_id: str, name: str) -> None:
        if key not in self._buffers:
            self._order.append(key)
            self._buffers[key] = ""
        self._ids[key] = call_id
        self._names[key] = name

    def delta(self, key: str, fragment: str) -> None:
        self._buffers.setdefault(key, "")
        if key not in self._order:
            self._order.append(key)
        self._buffers[key] += fragment

    def finish(self) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for key in self._order:
            raw = self._buffers.get(key, "") or "{}"
            try:
                args = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as exc:
                raise ToolInputError(
                    f"tool call {self._names.get(key, key)!r} produced invalid "
                    f"JSON arguments: {exc}"
                ) from exc
            if not isinstance(args, dict):
                raise ToolInputError(
                    f"tool call {self._names.get(key, key)!r} arguments must be an object"
                )
            calls.append(
                ToolCall(
                    id=self._ids.get(key, key),
                    name=self._names.get(key, ""),
                    arguments=args,
                )
            )
        return calls
