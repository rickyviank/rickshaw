"""Human-readable rendering for turn traces.

This module is a presentation-layer helper for the TUI. It takes the raw
`TurnEvent` stream produced by the orchestrator and turns it into a structured,
color-coded, human-readable trace view. All raw event data is preserved and can
be exposed on demand for debugging.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from rickshaw import events


@dataclass
class TraceLine:
    """One display line inside an expanded trace.

    The TUI renders ``summary`` by default. When the user expands the line the
    TUI shows either ``content`` (for answer/thinking blocks) or ``raw_json``
    (for all other events). Global raw mode always renders ``raw_json``.
    """

    timestamp: str
    label: str
    summary: str
    raw_json: str = ""
    content: str | None = None
    expandable: bool = False
    is_capped: bool = False
    is_placeholder: bool = False
    color_class: str = ""


@dataclass
class TraceView:
    """The complete rendered view of one turn trace."""

    header_lines: list[str] = field(default_factory=list)
    summary: str = ""
    lines: list[TraceLine] = field(default_factory=list)
    step_count: int = 0
    tool_calls: int = 0
    retries: int = 0
    degraded: bool = False


def format_trace(
    event_records: list[tuple[events.TurnEvent, float]],
    *,
    task_input: str,
    provider: str,
    model: str,
    status: str,
    duration: float,
    width: int = 80,
    height: int = 24,
) -> TraceView:
    """Format a turn's events into a human-readable ``TraceView``."""
    records = list(event_records)

    # Pre-scan for the TurnDone event so delta blocks can show token counts and
    # the final summary line is available while we stream through records.
    turn_done: events.TurnDone | None = None
    turn_done_ts = duration
    for event, ts in records:
        if isinstance(event, events.TurnDone):
            turn_done = event
            turn_done_ts = ts
            break

    view = TraceView()
    view.header_lines = _header_lines(task_input, status, duration, provider, model)
    view.degraded = _is_degraded(records, turn_done)
    view.tool_calls = _count_tool_calls(records)
    view.retries = _count_retries(records)

    pending: dict[str, tuple[events.TurnEvent, float] | None] = {
        "context": None,
        "llm": None,
        "tool": None,
    }
    pending_delta: tuple[str, list[tuple[events.TurnEvent, float]]] | None = None

    text_block_index: int | None = None
    thinking_block_index: int | None = None
    text_block_ts: float | None = None
    thinking_block_ts: float | None = None
    seen_text_block = False
    seen_thinking_block = False

    def flush_delta() -> None:
        nonlocal pending_delta, text_block_index, thinking_block_index
        nonlocal text_block_ts, thinking_block_ts
        nonlocal seen_text_block, seen_thinking_block
        if pending_delta is None:
            return
        kind, items = pending_delta
        line = _flush_delta_block(kind, items, status, turn_done, width, height)
        first_ts = items[0][1]
        if kind == "text":
            text_block_index = len(view.lines)
            text_block_ts = first_ts
            seen_text_block = True
        else:
            thinking_block_index = len(view.lines)
            thinking_block_ts = first_ts
            seen_thinking_block = True
        view.lines.append(line)
        pending_delta = None

    def flush_unmatched(event: events.TurnEvent | None) -> None:
        """Emit failed/incomplete start lines when the matching done never arrives."""
        if pending["llm"] is not None and (
            event is None or not isinstance(event, events.LLMCallDone)
        ):
            ev, ts = pending["llm"]
            pending["llm"] = None
            summary = f"{ev.model} (attempt {ev.attempt}) -> failed"
            view.lines.append(
                _line(
                    _format_timestamp(ts),
                    "[llm]",
                    summary,
                    raw_json=ev.model_dump_json(indent=2),
                    color_class="trace-llm",
                    expandable=True,
                )
            )
        if pending["tool"] is not None and (
            event is None or not isinstance(event, events.TurnToolCallDone)
        ):
            ev, ts = pending["tool"]
            pending["tool"] = None
            args_str = json.dumps(ev.arguments, ensure_ascii=False, separators=(", ", ": "))
            summary = f"{ev.tool_name}({args_str}) -> (incomplete)"
            view.lines.append(
                _line(
                    _format_timestamp(ts),
                    "[tool]",
                    summary,
                    raw_json=ev.model_dump_json(indent=2),
                    color_class="trace-tool",
                    expandable=True,
                )
            )

    for event, ts in records:
        if isinstance(event, events.TurnTextDelta):
            if pending_delta is None or pending_delta[0] != "text":
                flush_delta()
                pending_delta = ("text", [])
            pending_delta[1].append((event, ts))
        elif isinstance(event, events.TurnThinkingDelta):
            if pending_delta is None or pending_delta[0] != "thinking":
                flush_delta()
                pending_delta = ("thinking", [])
            pending_delta[1].append((event, ts))
        else:
            flush_delta()
            flush_unmatched(event)
            if isinstance(event, events.TurnDone):
                # Handled after the loop so placeholders and final counts line up.
                continue
            line = _process_non_delta(event, ts, pending, width)
            if line is not None:
                view.lines.append(line)

    flush_delta()
    flush_unmatched(None)

    # Final answer block from TurnDone when the turn produced no text deltas.
    if turn_done is not None and not seen_text_block:
        if turn_done.text:
            line = _delta_block_from_content(
                "text",
                turn_done.text,
                turn_done_ts,
                status,
                turn_done,
                width,
                height,
            )
            # The canonical raw JSON for this synthetic answer block is the
            # TurnDone event that carried the final text.
            line.raw_json = turn_done.model_dump_json(indent=2)
            line.label = "[partial answer]" if status == "interrupted" else "[answer]"
            text_block_index = len(view.lines)
            text_block_ts = turn_done_ts
            view.lines.append(line)
            seen_text_block = True

    # Placeholders for transparency; these are not counted as steps.
    if not seen_thinking_block:
        placeholder_ts = text_block_ts if text_block_ts is not None else turn_done_ts
        placeholder = _line(
            _format_timestamp(placeholder_ts),
            "[thinking]",
            "(none)",
            is_placeholder=True,
            color_class="trace-thinking",
        )
        if text_block_index is not None:
            view.lines.insert(text_block_index, placeholder)
            text_block_index += 1
        else:
            view.lines.append(placeholder)

    if not seen_text_block:
        view.lines.append(
            _line(
                _format_timestamp(turn_done_ts),
                "[answer]",
                "(empty)",
                is_placeholder=True,
                color_class="trace-answer",
            )
        )

    if turn_done is not None:
        view.lines.append(_done_line(turn_done, turn_done_ts))

    view.step_count = sum(1 for line in view.lines if not line.is_placeholder)
    view.summary = _collapsed_summary(
        view.step_count,
        view.tool_calls,
        view.retries,
        view.degraded,
        status,
        duration,
    )
    return view


def _format_timestamp(seconds: float) -> str:
    """Return a human relative timestamp like ``+0.42s``."""
    return f"+{seconds:.2f}s"


def _truncate_payload(text: str, width: int, prefix_len: int = 0) -> str:
    """Truncate ``text`` so it fits in the terminal width budget.

    The budget is roughly half of ``width`` minus ``prefix_len``. If the text
    exceeds the budget it is truncated and a ``… (+N chars)`` hint is appended.
    """
    budget = max(8, width // 2 - prefix_len)
    if len(text) <= budget:
        return text
    for cut in range(budget - 1, 0, -1):
        remainder = len(text) - cut
        suffix = f" … (+{remainder} chars)"
        if cut + len(suffix) <= budget:
            return text[:cut] + suffix
    return text[: budget - 1] + "…"


def _render_event(
    event: events.TurnEvent,
    context: dict[str, Any],
    width: int,
) -> TraceLine | None:
    """Render a single non-delta event into a ``TraceLine``.

    ``context`` is a mutable dict used to carry pending start events (e.g. an
    ``LLMCallStart`` waiting for its matching ``LLMCallDone``).
    """
    ts = context.get("__ts", 0.0)
    pending: dict[str, tuple[events.TurnEvent, float] | None] = context.setdefault(
        "_pending",
        {"context": None, "llm": None, "tool": None},
    )
    return _process_non_delta(event, ts, pending, width)


def _process_non_delta(
    event: events.TurnEvent,
    ts: float,
    pending: dict[str, tuple[events.TurnEvent, float] | None],
    width: int,
) -> TraceLine | None:
    """Convert a single non-delta event (or start/done pair) into a TraceLine."""
    if isinstance(event, events.ContextStart):
        pending["context"] = (event, ts)
        return None

    if isinstance(event, events.ContextDone):
        start = pending.get("context")
        if start is not None:
            start_ev, start_ts = start
            pending["context"] = None
            ts = start_ts
            raw = _combine_raw(start_ev, event)
        else:
            raw = event.model_dump_json(indent=2)
        summary = f"{event.record_count} memories, ~{event.token_estimate} tokens"
        return _line(
            _format_timestamp(ts),
            "[context]",
            summary,
            raw_json=raw,
            color_class="trace-context",
            expandable=True,
        )

    if isinstance(event, events.PromptBuilt):
        summary = f"{event.message_count} messages, ~{event.token_estimate} tokens"
        return _line(
            _format_timestamp(ts),
            "[prompt]",
            summary,
            raw_json=event.model_dump_json(indent=2),
            color_class="trace-prompt",
            expandable=True,
        )

    if isinstance(event, events.LLMCallStart):
        pending["llm"] = (event, ts)
        return None

    if isinstance(event, events.LLMCallDone):
        start = pending.get("llm")
        if start is not None:
            start_ev, start_ts = start
            pending["llm"] = None
            ts = start_ts
            raw = _combine_raw(start_ev, event)
            attempt = getattr(start_ev, "attempt", 0)
            start_model = start_ev.model
        else:
            raw = event.model_dump_json(indent=2)
            attempt = 0
            start_model = ""
        done_model = event.model or "unknown"
        total = _total_tokens(event.usage)
        if start_model:
            summary = f"{start_model} (attempt {attempt}) -> {done_model}, {total} tokens"
        else:
            summary = f"-> {done_model}, {total} tokens"
        return _line(
            _format_timestamp(ts),
            "[llm]",
            summary,
            raw_json=raw,
            color_class="trace-llm",
            expandable=True,
        )

    if isinstance(event, events.TurnToolCallStart):
        pending["tool"] = (event, ts)
        return None

    if isinstance(event, events.TurnToolCallDone):
        start = pending.get("tool")
        if start is not None:
            start_ev, start_ts = start
            pending["tool"] = None
            ts = start_ts
            raw = _combine_raw(start_ev, event)
            args = start_ev.arguments
        else:
            raw = event.model_dump_json(indent=2)
            args = {}
        args_str = json.dumps(args, ensure_ascii=False, separators=(", ", ": "))
        result_preview = _truncate_payload(event.result.replace("\n", " "), width)
        summary = f"{event.tool_name}({args_str}) -> {result_preview} ({event.duration_ms}ms)"
        summary = _truncate_payload(summary, width)
        return _line(
            _format_timestamp(ts),
            "[tool]",
            summary,
            raw_json=raw,
            color_class="trace-tool",
            expandable=True,
        )

    if isinstance(event, events.Retry):
        summary = f"attempt {event.attempt}/{event.max_retries} in {event.delay:.1f}s"
        return _line(
            _format_timestamp(ts),
            "[retry]",
            summary,
            raw_json=event.model_dump_json(indent=2),
            color_class="trace-retry",
            expandable=True,
        )

    if isinstance(event, events.Degraded):
        return _line(
            _format_timestamp(ts),
            "[degraded]",
            _truncate_payload(event.reason, width),
            raw_json=event.model_dump_json(indent=2),
            color_class="trace-degraded",
            expandable=True,
        )

    if isinstance(event, events.MemoryWrite):
        summary = f"wrote {len(event.record_ids)} records"
        return _line(
            _format_timestamp(ts),
            "[memory]",
            summary,
            raw_json=event.model_dump_json(indent=2),
            color_class="trace-memory",
            expandable=True,
        )

    if isinstance(event, events.JobEnqueue):
        payload_str = json.dumps(
            event.payload, ensure_ascii=False, separators=(", ", ": ")
        )
        summary = f"enqueued {event.job_type}({payload_str})"
        summary = _truncate_payload(summary, width)
        return _line(
            _format_timestamp(ts),
            "[job]",
            summary,
            raw_json=event.model_dump_json(indent=2),
            color_class="trace-job",
            expandable=True,
        )

    if isinstance(event, events.Error):
        return _line(
            _format_timestamp(ts),
            "[error]",
            _truncate_payload(event.message, width),
            raw_json=event.model_dump_json(indent=2),
            color_class="trace-error",
            expandable=True,
        )

    if isinstance(event, events.TurnDone):
        return _done_line(event, ts)

    # TurnStart is represented by the header and does not produce a body line.
    return None


def _flush_delta_block(
    kind: str,
    items: list[tuple[events.TurnEvent, float]],
    status: str,
    turn_done: events.TurnDone | None,
    width: int,
    height: int,
) -> TraceLine:
    """Render a grouped answer/thinking delta block."""
    events_in_block = [item[0] for item in items]
    first_ts = items[0][1]
    content = "".join(getattr(ev, "text", "") for ev in events_in_block)
    return _delta_block_from_content(
        kind, content, first_ts, status, turn_done, width, height, events_in_block
    )


def _delta_block_from_content(
    kind: str,
    content: str,
    ts: float,
    status: str,
    turn_done: events.TurnDone | None,
    width: int,
    height: int,
    delta_events: list[events.TurnEvent] | None = None,
) -> TraceLine:
    """Build a TraceLine for an answer or thinking block."""
    if kind == "text":
        label = "[partial answer]" if status == "interrupted" else "[answer]"
        token_count = _answer_tokens(turn_done)
        color_class = "trace-answer"
    else:
        label = "[partial thinking]" if status == "interrupted" else "[thinking]"
        token_count = 0
        color_class = "trace-thinking"

    if not content:
        return _line(
            _format_timestamp(ts),
            label,
            "(empty)" if kind == "text" else "(none)",
            is_placeholder=True,
            color_class=color_class,
        )

    delta_count = len(delta_events) if delta_events else 1
    if delta_events:
        raw = json.dumps(
            [ev.model_dump(mode="json") for ev in delta_events],
            ensure_ascii=False,
            indent=2,
        )
    else:
        raw = ""

    cap_lines = max(1, int(height * 0.3))
    lines = content.splitlines()
    header = f"({delta_count} Δ, {token_count} tokens)"
    if len(lines) > cap_lines:
        preview = "\n".join(lines[:cap_lines])
        summary = f"{header}\n{preview}\n… (+{len(lines) - cap_lines} lines)"
        is_capped = True
    else:
        summary = header
        is_capped = False

    return _line(
        _format_timestamp(ts),
        label,
        summary,
        raw_json=raw,
        content=content,
        expandable=True,
        is_capped=is_capped,
        color_class=color_class,
    )


def _done_line(turn_done: events.TurnDone, ts: float) -> TraceLine:
    """Render the final TurnDone summary line."""
    tool_calls = turn_done.tool_calls_made
    total = _total_tokens(turn_done.usage)
    tc_str = f"{tool_calls} tool call{'s' if tool_calls != 1 else ''}"
    summary = f"{tc_str}, {total} total tokens"
    return _line(
        _format_timestamp(ts),
        "[done]",
        summary,
        raw_json=turn_done.model_dump_json(indent=2),
        color_class="trace-done",
        expandable=True,
    )


def _line(
    timestamp: str,
    label: str,
    summary: str,
    *,
    raw_json: str = "",
    content: str | None = None,
    expandable: bool = False,
    is_capped: bool = False,
    is_placeholder: bool = False,
    color_class: str = "",
) -> TraceLine:
    """Convenience constructor for ``TraceLine``."""
    return TraceLine(
        timestamp=timestamp,
        label=label,
        summary=summary,
        raw_json=raw_json,
        content=content,
        expandable=expandable,
        is_capped=is_capped,
        is_placeholder=is_placeholder,
        color_class=color_class,
    )


def _header_lines(
    task_input: str,
    status: str,
    duration: float,
    provider: str,
    model: str,
) -> list[str]:
    """Build the two-line trace header."""
    line1 = f'"{task_input}" · {status} · {duration:.2f}s'
    if provider and model:
        line2 = f"{provider}/{model}"
    elif provider:
        line2 = provider
    elif model:
        line2 = model
    else:
        line2 = ""
    return [line1, line2]


def _collapsed_summary(
    step_count: int,
    tool_calls: int,
    retries: int,
    degraded: bool,
    status: str,
    duration: float,
) -> str:
    """Build the collapsed summary string for a TraceBlock."""
    parts = [f"{step_count} steps"]
    parts.append(f"{tool_calls} tool call{'s' if tool_calls != 1 else ''}")
    if retries == 1:
        parts.append("1 retry")
    else:
        parts.append(f"{retries} retries")
    if degraded:
        parts.append("degraded")
    if status not in ("completed",):
        parts.append(status)
    parts.append(f"{duration:.1f}s")
    return " · ".join(parts)


def _is_degraded(
    records: list[tuple[events.TurnEvent, float]],
    turn_done: events.TurnDone | None,
) -> bool:
    """Return True if the turn contains a Degraded event or ended degraded."""
    if turn_done is not None and turn_done.degraded:
        return True
    return any(isinstance(ev, events.Degraded) for ev, _ in records)


def _count_tool_calls(records: list[tuple[events.TurnEvent, float]]) -> int:
    return sum(1 for ev, _ in records if isinstance(ev, events.TurnToolCallStart))


def _count_retries(records: list[tuple[events.TurnEvent, float]]) -> int:
    return sum(1 for ev, _ in records if isinstance(ev, events.Retry))


def _total_tokens(usage: Any) -> int:
    """Safely extract the total token count from a TokenUsage dataclass/dict."""
    if usage is None:
        return 0
    if isinstance(usage, dict):
        return usage.get("total_tokens", 0)
    return getattr(usage, "total_tokens", 0)


def _answer_tokens(turn_done: events.TurnDone | None) -> int:
    """Return the best available token count for the final answer block."""
    if turn_done is None or turn_done.usage is None:
        return 0
    usage = turn_done.usage
    completion = getattr(usage, "completion_tokens", 0)
    if completion:
        return completion
    return getattr(usage, "total_tokens", 0)


def _combine_raw(*events: events.TurnEvent) -> str:
    """Join the canonical JSON for multiple events with a blank line separator."""
    return "\n\n".join(ev.model_dump_json(indent=2) for ev in events)
