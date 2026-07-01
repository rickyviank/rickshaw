"""Tests for the generalized ToolRegistry (item 10)."""

from __future__ import annotations

import asyncio
import json

import pytest

from rickshaw.memory.embedder import TFIDFEmbedder
from rickshaw.memory.service import MemoryService
from rickshaw.memory.tools import RECALL_SPEC, REMEMBER_SPEC, build_memory_registry
from rickshaw.providers.base import ToolCall, ToolSpec
from rickshaw.tool_registry import ToolRegistry

_ECHO_SPEC = ToolSpec(
    name="echo",
    description="Echo a message back.",
    parameters={
        "type": "object",
        "properties": {"msg": {"type": "string"}},
        "required": ["msg"],
    },
)


def test_register_and_dispatch_sync():
    registry = ToolRegistry()
    registry.register("echo", lambda args: args["msg"].upper(), _ECHO_SPEC)
    result = json.loads(registry.dispatch(ToolCall(id="1", name="echo", arguments={"msg": "hi"})))
    assert result == "HI"


def test_register_name_mismatch_raises():
    registry = ToolRegistry()
    with pytest.raises(ValueError):
        registry.register("wrong", lambda args: None, _ECHO_SPEC)


def test_dispatch_unknown_tool_returns_error():
    registry = ToolRegistry()
    result = json.loads(registry.dispatch(ToolCall(id="1", name="nope", arguments={})))
    assert "unknown tool" in result["error"]


def test_dispatch_validation_failure():
    registry = ToolRegistry()
    registry.register("echo", lambda args: args["msg"], _ECHO_SPEC)
    # Missing the required "msg" field.
    result = json.loads(registry.dispatch(ToolCall(id="1", name="echo", arguments={})))
    assert result["type"] == "validation_error"


def test_dispatch_handler_error_surfaced():
    registry = ToolRegistry()

    def _boom(args):
        raise RuntimeError("kaboom")

    registry.register("echo", _boom, _ECHO_SPEC)
    result = json.loads(registry.dispatch(ToolCall(id="1", name="echo", arguments={"msg": "x"})))
    assert result["type"] == "handler_error"
    assert "kaboom" in result["error"]


def test_specs_and_get_spec():
    registry = ToolRegistry()
    registry.register("echo", lambda args: None, _ECHO_SPEC)
    assert registry.get_spec("echo") is _ECHO_SPEC
    assert registry.get_spec("missing") is None
    assert [s.name for s in registry.specs()] == ["echo"]


def test_async_dispatch_with_async_handler():
    registry = ToolRegistry()

    async def _ahandler(args):
        await asyncio.sleep(0)
        return args["msg"] + "!"

    registry.register("echo", _ahandler, _ECHO_SPEC)
    result = json.loads(
        asyncio.run(registry.async_dispatch(ToolCall(id="1", name="echo", arguments={"msg": "hey"})))
    )
    assert result == "hey!"


def test_async_dispatch_wraps_sync_handler():
    registry = ToolRegistry()
    registry.register("echo", lambda args: args["msg"], _ECHO_SPEC)
    result = json.loads(
        asyncio.run(registry.async_dispatch(ToolCall(id="1", name="echo", arguments={"msg": "sync"})))
    )
    assert result == "sync"


def test_build_memory_registry_registers_all_tools():
    service = MemoryService(embedder=TFIDFEmbedder(dim=32))
    registry = build_memory_registry(service)
    names = {s.name for s in registry.specs()}
    assert names == {"remember", "recall", "forget"}
    # recall is read-only; remember/forget are side-effecting.
    assert registry.get_spec("recall").side_effect is False
    assert registry.get_spec("remember").side_effect is True


def test_build_memory_registry_dispatch_roundtrip():
    service = MemoryService(embedder=TFIDFEmbedder(dim=32))
    registry = build_memory_registry(service)
    rid = json.loads(registry.dispatch(ToolCall(id="1", name="remember", arguments={"fact": "cats purr"})))
    assert isinstance(rid, str)
    results = json.loads(registry.dispatch(ToolCall(id="2", name="recall", arguments={"query": "cats"})))
    assert any("cats purr" in r["text"] for r in results)
