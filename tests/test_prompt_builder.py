"""Tests for the prompt builder — token budget enforcement.

Sensitive-record filtering now lives in MemoryService.assemble_context (see
test_memory.py); the builder only enforces the token budget.
"""

from __future__ import annotations

from rickshaw.memory.record import MemoryRecord, MemoryScope, MemoryType
from rickshaw.prompt.builder import PromptBuilder, _estimate_tokens
from rickshaw.providers.base import Message, ToolSpec


def test_build_basic_structure():
    builder = PromptBuilder(max_tokens=8000)
    messages = builder.build(
        system="You are helpful.",
        tools=None,
        context=[],
        task_input="Hi",
    )
    assert len(messages) == 2  # system + user
    assert messages[0].role == "system"
    assert messages[-1].role == "user"
    assert messages[-1].content == "Hi"


def test_build_includes_context():
    builder = PromptBuilder(max_tokens=8000)
    ctx = [
        MemoryRecord(text="fact one", scope=MemoryScope.SESSION, type=MemoryType.FACT),
    ]
    messages = builder.build(
        system="System",
        tools=None,
        context=ctx,
        task_input="Hello",
    )
    assert len(messages) == 3  # system + context + user
    assert "fact one" in messages[1].content


def test_build_serializes_all_provided_records():
    """The builder serializes whatever records it receives (filtering is upstream).

    Records reaching the builder are already non-sensitive because
    MemoryService.assemble_context excludes sensitive ones before ranking.
    """
    builder = PromptBuilder(max_tokens=8000)
    ctx = [
        MemoryRecord(text="public info one", sensitive=False),
        MemoryRecord(text="public info two", sensitive=False),
    ]
    messages = builder.build(
        system="System",
        tools=None,
        context=ctx,
        task_input="Query",
    )
    all_content = " ".join(m.content for m in messages)
    assert "public info one" in all_content
    assert "public info two" in all_content


def test_build_respects_token_budget():
    builder = PromptBuilder(max_tokens=50)
    long_text = "word " * 500
    ctx = [
        MemoryRecord(text=long_text, sensitive=False),
    ]
    messages = builder.build(
        system="S",
        tools=None,
        context=ctx,
        task_input="Q",
    )
    # Context should be truncated (skipped) since it blows the budget
    # Either 2 messages (system+user) or 3 with truncated context
    total_text = " ".join(m.content for m in messages)
    total_tokens = _estimate_tokens(total_text)
    # The output shouldn't include the full 500-word context
    assert long_text not in total_text


def test_build_returns_list_of_messages():
    builder = PromptBuilder()
    messages = builder.build("sys", None, [], "input")
    assert all(isinstance(m, Message) for m in messages)


def test_estimate_tokens_positive():
    assert _estimate_tokens("hello world") > 0
    assert _estimate_tokens("") >= 1
