"""Tests for rickshaw.trace_render."""

from __future__ import annotations

import json

import pytest

from rickshaw import events, trace_render
from rickshaw.providers.base import TokenUsage


def _fmt(*records: tuple[events.TurnEvent, float]) -> list[tuple[events.TurnEvent, float]]:
    """Convenience wrapper to build event records."""
    return list(records)


def _done(
    text: str = "",
    tool_calls_made: int = 0,
    total_tokens: int = 0,
    completion_tokens: int = 0,
    degraded: bool = False,
    model: str = "gpt-4o",
    ts: float = 1.0,
) -> tuple[events.TurnDone, float]:
    usage = TokenUsage(
        prompt_tokens=total_tokens - completion_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )
    return (
        events.TurnDone(
            text=text,
            tool_calls_made=tool_calls_made,
            degraded=degraded,
            model=model,
            usage=usage,
        ),
        ts,
    )


def test_format_timestamp():
    assert trace_render._format_timestamp(0.0) == "+0.00s"
    assert trace_render._format_timestamp(0.423) == "+0.42s"
    assert trace_render._format_timestamp(1.0) == "+1.00s"


def test_truncate_payload_short_text_unchanged():
    text = "short"
    assert trace_render._truncate_payload(text, width=80) == text


def test_truncate_payload_long_text_appends_hint():
    text = "x" * 100
    truncated = trace_render._truncate_payload(text, width=80)
    assert truncated.endswith(" chars)")
    assert "… (+" in truncated
    assert len(truncated) <= 40


def test_truncate_payload_respects_prefix_len():
    text = "y" * 100
    truncated = trace_render._truncate_payload(text, width=80, prefix_len=20)
    assert "… (+" in truncated
    assert len(truncated) <= 20


def test_header_and_collapsed_summary():
    records = [
        (events.ContextStart(), 0.0),
        (events.ContextDone(record_count=2, token_estimate=50), 0.01),
        _done(text="hi", total_tokens=10, completion_tokens=10),
    ]
    view = trace_render.format_trace(
        records,
        task_input="hello",
        provider="openai",
        model="gpt-4o",
        status="completed",
        duration=1.234,
        width=80,
        height=24,
    )
    assert view.header_lines == ['"hello" · completed · 1.23s', "openai/gpt-4o"]
    assert view.summary == "3 steps · 0 tool calls · 0 retries · 1.2s"
    assert view.step_count == 3
    assert not view.degraded


def test_bracket_labels():
    records = [
        (events.ContextStart(), 0.0),
        (events.ContextDone(record_count=3, token_estimate=120), 0.01),
        (events.PromptBuilt(message_count=4, token_estimate=340), 0.02),
        (events.LLMCallStart(attempt=1, model="openai"), 0.05),
        (
            events.LLMCallDone(
                model="gpt-4o",
                usage=TokenUsage(
                    prompt_tokens=200,
                    completion_tokens=10,
                    total_tokens=210,
                ),
            ),
            0.05,
        ),
        (
            events.TurnToolCallStart(
                call_id="tc1",
                tool_name="recall",
                arguments={"query": "what do I like?"},
            ),
            0.42,
        ),
        (
            events.TurnToolCallDone(
                call_id="tc1",
                tool_name="recall",
                result="3 records",
                duration_ms=45,
            ),
            0.42,
        ),
        (events.Retry(attempt=1, max_retries=2, delay=1.0, error="429"), 0.50),
        (events.Degraded(reason="falling back to local memory"), 0.55),
        (events.MemoryWrite(record_ids=["r1", "r2"]), 0.60),
        (events.JobEnqueue(job_type="score", payload={"record_id": "r1"}), 0.65),
        (events.Error(message="something went wrong"), 0.70),
        _done(text="done", total_tokens=210, completion_tokens=10),
    ]
    view = trace_render.format_trace(
        records,
        task_input="what do I like?",
        provider="openai",
        model="gpt-4o",
        status="completed",
        duration=1.2,
        width=120,
        height=24,
    )
    labels = [line.label for line in view.lines]
    assert labels == [
        "[context]",
        "[prompt]",
        "[llm]",
        "[tool]",
        "[retry]",
        "[degraded]",
        "[memory]",
        "[job]",
        "[error]",
        "[thinking]",
        "[answer]",
        "[done]",
    ]

    tool_line = view.lines[labels.index("[tool]")]
    assert 'recall({"query": "what do I like?"}) -> 3 records (45ms)' == tool_line.summary

    retry_line = view.lines[labels.index("[retry]")]
    assert retry_line.summary == "attempt 1/2 in 1.0s"

    memory_line = view.lines[labels.index("[memory]")]
    assert memory_line.summary == "wrote 2 records"

    error_line = view.lines[labels.index("[error]")]
    assert error_line.summary == "something went wrong"


def test_delta_grouping():
    records = [
        (events.TurnThinkingDelta(text="C"), 0.1),
        (events.TurnThinkingDelta(text="D"), 0.2),
        (events.TurnTextDelta(text="A"), 0.3),
        (events.TurnTextDelta(text="B"), 0.4),
        _done(text="CDAB", total_tokens=4, completion_tokens=4),
    ]
    view = trace_render.format_trace(
        records,
        task_input="hi",
        provider="openai",
        model="gpt-4o",
        status="completed",
        duration=1.0,
        width=80,
        height=24,
    )
    labels = [line.label for line in view.lines]
    assert labels == ["[thinking]", "[answer]", "[done]"]

    thinking = view.lines[labels.index("[thinking]")]
    assert not thinking.is_placeholder
    assert thinking.content == "CD"

    answer = view.lines[labels.index("[answer]")]
    assert answer.content == "AB"


def test_summary_and_step_count():
    records = [
        (events.ContextStart(), 0.0),
        (events.ContextDone(record_count=3, token_estimate=120), 0.01),
        (events.PromptBuilt(message_count=4, token_estimate=340), 0.02),
        (events.LLMCallStart(attempt=1, model="openai"), 0.05),
        (
            events.LLMCallDone(
                model="gpt-4o",
                usage=TokenUsage(total_tokens=210, completion_tokens=10),
            ),
            0.05,
        ),
        (
            events.TurnToolCallStart(
                call_id="tc1",
                tool_name="recall",
                arguments={"query": "x"},
            ),
            0.42,
        ),
        (
            events.TurnToolCallDone(
                call_id="tc1",
                tool_name="recall",
                result="y",
                duration_ms=45,
            ),
            0.42,
        ),
        (events.TurnTextDelta(text="answer"), 1.0),
        (events.MemoryWrite(record_ids=["r1"]), 1.1),
        _done(text="answer", tool_calls_made=1, total_tokens=300, completion_tokens=10),
    ]
    view = trace_render.format_trace(
        records,
        task_input="hi",
        provider="openai",
        model="gpt-4o",
        status="completed",
        duration=1.2,
        width=120,
        height=24,
    )
    assert view.tool_calls == 1
    assert view.retries == 0
    assert view.step_count == 7
    assert view.summary == "7 steps · 1 tool call · 0 retries · 1.2s"


def test_truncation():
    long_result = "record " * 50
    records = [
        (
            events.TurnToolCallStart(
                call_id="tc1",
                tool_name="recall",
                arguments={"query": "x"},
            ),
            0.1,
        ),
        (
            events.TurnToolCallDone(
                call_id="tc1",
                tool_name="recall",
                result=long_result,
                duration_ms=45,
            ),
            0.2,
        ),
        _done(text="ok", total_tokens=10, completion_tokens=10),
    ]
    view = trace_render.format_trace(
        records,
        task_input="hi",
        provider="openai",
        model="gpt-4o",
        status="completed",
        duration=0.5,
        width=80,
        height=24,
    )
    tool_line = [line for line in view.lines if line.label == "[tool]"][0]
    assert "… (+" in tool_line.summary
    assert "chars)" in tool_line.summary


def test_terminal_height_cap():
    long_text = "\n".join(f"line {i}" for i in range(20))
    records = [
        (events.TurnTextDelta(text=long_text), 0.1),
        _done(text=long_text, total_tokens=20, completion_tokens=20),
    ]
    view = trace_render.format_trace(
        records,
        task_input="hi",
        provider="openai",
        model="gpt-4o",
        status="completed",
        duration=1.0,
        width=80,
        height=24,
    )
    answer = [line for line in view.lines if line.label == "[answer]"][0]
    assert answer.is_capped
    assert "… (+13 lines)" in answer.summary
    assert answer.content == long_text
    assert answer.summary.count("\nline ") == 7


def test_terminal_height_no_cap():
    short_text = "\n".join(f"line {i}" for i in range(3))
    records = [
        (events.TurnTextDelta(text=short_text), 0.1),
        _done(text=short_text, total_tokens=3, completion_tokens=3),
    ]
    view = trace_render.format_trace(
        records,
        task_input="hi",
        provider="openai",
        model="gpt-4o",
        status="completed",
        duration=1.0,
        width=80,
        height=24,
    )
    answer = [line for line in view.lines if line.label == "[answer]"][0]
    assert not answer.is_capped
    assert "lines)" not in answer.summary
    assert short_text in answer.content
    assert "Δ" in answer.summary


def test_empty_and_partial_handling():
    records = [
        _done(text="", tool_calls_made=0, total_tokens=0, completion_tokens=0),
    ]
    view = trace_render.format_trace(
        records,
        task_input="hi",
        provider="openai",
        model="gpt-4o",
        status="completed",
        duration=1.0,
        width=80,
        height=24,
    )
    labels = [line.label for line in view.lines]
    assert "[thinking]" in labels
    assert "[answer]" in labels
    thinking = [line for line in view.lines if line.label == "[thinking]"][0]
    answer = [line for line in view.lines if line.label == "[answer]"][0]
    assert thinking.is_placeholder
    assert thinking.summary == "(none)"
    assert answer.is_placeholder
    assert answer.summary == "(empty)"
    assert view.step_count == 1  # only [done]


def test_partial_answer_on_interrupt():
    records = [
        (events.TurnTextDelta(text="partial "), 0.5),
        (events.TurnTextDelta(text="answer"), 0.6),
        _done(text="partial answer", total_tokens=2, completion_tokens=2),
    ]
    view = trace_render.format_trace(
        records,
        task_input="hi",
        provider="openai",
        model="gpt-4o",
        status="interrupted",
        duration=1.0,
        width=80,
        height=24,
    )
    answer = [line for line in view.lines if line.label.startswith("[partial")][0]
    assert answer.label == "[partial answer]"
    assert answer.content == "partial answer"
    assert "interrupted" in view.summary


def test_turn_done_produces_answer_when_no_deltas():
    records = [_done(text="final", tool_calls_made=1, total_tokens=15, completion_tokens=15)]
    view = trace_render.format_trace(
        records,
        task_input="hi",
        provider="openai",
        model="gpt-4o",
        status="completed",
        duration=0.5,
        width=80,
        height=24,
    )
    labels = [line.label for line in view.lines]
    assert "[answer]" in labels
    assert "[done]" in labels
    answer = [line for line in view.lines if line.label == "[answer]"][0]
    assert answer.content == "final"
    done = [line for line in view.lines if line.label == "[done]"][0]
    assert "1 tool call" in done.summary
    assert "15 total tokens" in done.summary


def test_raw_json_for_combined_and_delta_blocks():
    start = events.TurnToolCallStart(
        call_id="tc1", tool_name="recall", arguments={"query": "x"}
    )
    done = events.TurnToolCallDone(
        call_id="tc1",
        tool_name="recall",
        result="y",
        duration_ms=10,
    )
    delta1 = events.TurnTextDelta(text="a")
    delta2 = events.TurnTextDelta(text="b")
    records = [
        (start, 0.1),
        (done, 0.2),
        (delta1, 0.3),
        (delta2, 0.4),
        _done(text="ab", total_tokens=2, completion_tokens=2),
    ]
    view = trace_render.format_trace(
        records,
        task_input="hi",
        provider="openai",
        model="gpt-4o",
        status="completed",
        duration=0.5,
        width=80,
        height=24,
    )
    tool_line = [line for line in view.lines if line.label == "[tool]"][0]
    parts = tool_line.raw_json.split("\n\n")
    assert len(parts) == 2
    assert json.loads(parts[0])["type"] == "tool_call_start"
    assert json.loads(parts[1])["type"] == "tool_call_done"

    answer = [line for line in view.lines if line.label == "[answer]"][0]
    deltas = json.loads(answer.raw_json)
    assert isinstance(deltas, list)
    assert len(deltas) == 2
    assert deltas[0]["text"] == "a"
    assert deltas[1]["text"] == "b"


def test_retry_flushes_failed_llm_call():
    records = [
        (events.LLMCallStart(attempt=1, model="openai"), 0.05),
        (events.Error(message="429"), 0.05),
        (events.Retry(attempt=1, max_retries=2, delay=1.0, error="429"), 0.05),
        (events.LLMCallStart(attempt=2, model="openai"), 1.05),
        (
            events.LLMCallDone(
                model="gpt-4o",
                usage=TokenUsage(total_tokens=210, completion_tokens=10),
            ),
            1.06,
        ),
        _done(text="ok", total_tokens=210, completion_tokens=10),
    ]
    view = trace_render.format_trace(
        records,
        task_input="hi",
        provider="openai",
        model="gpt-4o",
        status="completed",
        duration=1.1,
        width=80,
        height=24,
    )
    labels = [line.label for line in view.lines]
    assert labels.count("[llm]") == 2
    assert "[retry]" in labels
    failed = [line for line in view.lines if line.label == "[llm]" and "failed" in line.summary][0]
    assert failed.summary == "openai (attempt 1) -> failed"
    assert view.retries == 1
    assert "1 retry" in view.summary


def test_color_classes():
    records = [
        (events.ContextStart(), 0.0),
        (events.ContextDone(record_count=1, token_estimate=10), 0.01),
        (events.PromptBuilt(message_count=1, token_estimate=10), 0.02),
        (events.LLMCallStart(attempt=1, model="openai"), 0.03),
        (events.LLMCallDone(model="gpt-4o", usage=TokenUsage(total_tokens=1)), 0.04),
        (
            events.TurnToolCallStart(call_id="tc1", tool_name="recall", arguments={}),
            0.05,
        ),
        (
            events.TurnToolCallDone(call_id="tc1", tool_name="recall", result="r", duration_ms=1),
            0.06,
        ),
        (events.TurnTextDelta(text="a"), 0.07),
        (events.Retry(attempt=1, max_retries=2, delay=1.0, error="e"), 0.08),
        (events.Degraded(reason="d"), 0.09),
        (events.MemoryWrite(record_ids=["r1"]), 0.10),
        (events.JobEnqueue(job_type="j", payload={}), 0.11),
        (events.Error(message="e"), 0.12),
        _done(text="a", total_tokens=1, completion_tokens=1),
    ]
    view = trace_render.format_trace(
        records,
        task_input="hi",
        provider="openai",
        model="gpt-4o",
        status="completed",
        duration=1.0,
        width=120,
        height=24,
    )
    expected = {
        "[context]": "trace-context",
        "[prompt]": "trace-prompt",
        "[llm]": "trace-llm",
        "[tool]": "trace-tool",
        "[answer]": "trace-answer",
        "[retry]": "trace-retry",
        "[degraded]": "trace-degraded",
        "[memory]": "trace-memory",
        "[job]": "trace-job",
        "[error]": "trace-error",
        "[done]": "trace-done",
    }
    for line in view.lines:
        if line.label in expected:
            assert line.color_class == expected[line.label], f"{line.label} color_class mismatch"
