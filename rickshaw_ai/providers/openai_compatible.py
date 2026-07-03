"""Adapter for the OpenAI Chat Completions protocol.

Serves native OpenAI plus every OpenAI-compatible endpoint (Groq, xAI, Mistral,
DeepSeek, Together, Fireworks) and gateways (OpenRouter, Cloudflare AI Gateway).
The wire body it emits for plain-text messages is byte-compatible with the
classic ``{"role", "content"}`` shape.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator

import httpx

logger = logging.getLogger(__name__)

from rickshaw_ai.generate import GenerateRequest, GenerateResult, StopReason, Usage
from rickshaw_ai.messages import (
    ImageBlock,
    Message,
    TextBlock,
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from rickshaw_ai.providers.base import ProviderAdapter, aiter_sse
from rickshaw_ai.registry import ModelInfo, ProviderInfo
from rickshaw_ai.streaming import (
    StreamDone,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from rickshaw_ai.tools import Tool, ToolCall, ToolCallAssembler

_EFFORT_FROM_BUDGET = [(2048, "low"), (8192, "medium")]

_FINISH_MAP = {
    "stop": StopReason.end_turn,
    "length": StopReason.max_output_tokens,
    "tool_calls": StopReason.tool_use,
    "function_call": StopReason.tool_use,
    "content_filter": StopReason.content_filter,
}


def _effort(req: GenerateRequest) -> str | None:
    if req.reasoning is None:
        return None
    if req.reasoning.effort is not None:
        return req.reasoning.effort
    if req.reasoning.budget_tokens is not None:
        for threshold, label in _EFFORT_FROM_BUDGET:
            if req.reasoning.budget_tokens <= threshold:
                return label
        return "high"
    return None


def _tools_payload(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            },
        }
        for t in tools
    ]


def _content_for(message: Message) -> Any:
    """Serialize a message's content: a plain string when text-only, else parts."""
    has_image = any(isinstance(b, ImageBlock) for b in message.content)
    if not has_image:
        return "".join(b.text for b in message.content if isinstance(b, TextBlock))
    parts: list[dict[str, Any]] = []
    for block in message.content:
        if isinstance(block, TextBlock):
            parts.append({"type": "text", "text": block.text})
        elif isinstance(block, ImageBlock):
            if block.source == "url":
                url = block.data
            else:
                url = f"data:{block.media_type};base64,{block.data}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
    return parts


def _wire_messages(req: GenerateRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if req.system:
        out.append({"role": "system", "content": req.system})
    for msg in req.messages:
        tool_uses = [b for b in msg.content if isinstance(b, ToolUseBlock)]
        tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]

        if tool_results:
            for tr in tool_results:
                text = "".join(
                    b.text for b in tr.content if isinstance(b, TextBlock)
                )
                out.append(
                    {"role": "tool", "tool_call_id": tr.tool_use_id, "content": text}
                )
            continue

        if msg.role == "assistant" and tool_uses:
            entry: dict[str, Any] = {
                "role": "assistant",
                "content": "".join(
                    b.text for b in msg.content if isinstance(b, TextBlock)
                )
                or None,
                "tool_calls": [
                    {
                        "id": tu.id,
                        "type": "function",
                        "function": {
                            "name": tu.name,
                            "arguments": json.dumps(tu.arguments),
                        },
                    }
                    for tu in tool_uses
                ],
            }
            out.append(entry)
            continue

        out.append({"role": msg.role, "content": _content_for(msg)})
    return out


def _parse_usage(data: dict[str, Any], model: ModelInfo) -> Usage:
    u = data.get("usage") or {}
    details = u.get("completion_tokens_details") or {}
    prompt_details = u.get("prompt_tokens_details") or {}
    usage = Usage(
        input_tokens=u.get("prompt_tokens", 0),
        output_tokens=u.get("completion_tokens", 0),
        reasoning_tokens=details.get("reasoning_tokens", 0),
        cache_read_tokens=prompt_details.get("cached_tokens", 0),
    )
    usage.cost_usd = usage.compute_cost(model.pricing)
    return usage


def _parse_tool_calls(raw_calls: list[dict[str, Any]]) -> list[ToolUseBlock]:
    blocks: list[ToolUseBlock] = []
    for rc in raw_calls or []:
        func = rc.get("function", {})
        args_str = func.get("arguments", "{}")
        try:
            args = json.loads(args_str) if args_str else {}
        except (json.JSONDecodeError, TypeError) as exc:
            logger.warning(
                "Malformed tool-call arguments for %r, "
                "defaulting to empty: %s",
                func.get("name", "?"), exc,
            )
            args = {}
        blocks.append(
            ToolUseBlock(id=rc.get("id", ""), name=func.get("name", ""), arguments=args)
        )
    return blocks


class OpenAICompatibleAdapter(ProviderAdapter):
    protocol = "openai"

    def endpoint(self, provider: ProviderInfo, model: ModelInfo, *, stream: bool) -> str:
        return f"{provider.base_url.rstrip('/')}/chat/completions"

    def build_body(
        self, req: GenerateRequest, model: ModelInfo, *, stream: bool
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": model.model,
            "messages": _wire_messages(req),
        }
        if stream:
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        if req.max_output_tokens is not None:
            body["max_tokens"] = req.max_output_tokens
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.stop_sequences:
            body["stop"] = req.stop_sequences
        if model.supports_reasoning:
            effort = _effort(req)
            if effort is not None:
                body["reasoning_effort"] = effort
        if req.tools:
            body["tools"] = _tools_payload(req.tools)
            if req.tool_choice is not None:
                body["tool_choice"] = req.tool_choice
        body.update(req.provider_options)
        return body

    def parse_response(
        self, data: dict[str, Any], model: ModelInfo, provider: ProviderInfo
    ) -> GenerateResult:
        choice = (data.get("choices") or [{}])[0]
        wire_msg = choice.get("message", {})
        content: list = []
        text = wire_msg.get("content") or ""
        if text:
            content.append(TextBlock(text=text))
        tool_blocks = _parse_tool_calls(wire_msg.get("tool_calls", []))
        content.extend(tool_blocks)

        finish = choice.get("finish_reason")
        stop = _FINISH_MAP.get(finish, StopReason.end_turn)
        if tool_blocks:
            stop = StopReason.tool_use

        return GenerateResult(
            message=Message(role="assistant", content=content),
            stop_reason=stop,
            usage=_parse_usage(data, model),
            model_id=model.id,
            provider_id=provider.id,
            metadata={
                "raw_stop_reason": finish,
                "response_model": data.get("model", model.model),
                "raw": data,
            },
        )

    async def parse_stream(
        self, response: httpx.Response, model: ModelInfo, provider: ProviderInfo
    ) -> AsyncIterator[StreamEvent]:
        assembler = ToolCallAssembler()
        text_parts: list[str] = []
        started: set[str] = set()
        finish: str | None = None
        usage_data: dict[str, Any] = {}

        async for payload in aiter_sse(response):
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage_data = {"usage": chunk["usage"]}
            choices = chunk.get("choices") or []
            if not choices:
                continue
            choice = choices[0]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
            delta = choice.get("delta", {})
            if delta.get("content"):
                text_parts.append(delta["content"])
                yield TextDelta(text=delta["content"])
            for tc in delta.get("tool_calls", []) or []:
                idx = str(tc.get("index", 0))
                func = tc.get("function", {})
                if idx not in started:
                    started.add(idx)
                    assembler.start(
                        idx, call_id=tc.get("id", idx), name=func.get("name", "")
                    )
                    yield ToolCallStart(id=tc.get("id", idx), name=func.get("name", ""))
                if func.get("arguments"):
                    assembler.delta(idx, func["arguments"])
                    yield ToolCallDelta(
                        id=tc.get("id", idx), arguments_fragment=func["arguments"]
                    )

        calls: list[ToolCall] = assembler.finish()
        content: list = []
        if text_parts:
            content.append(TextBlock(text="".join(text_parts)))
        for call in calls:
            yield ToolCallEnd(call=call)
            content.append(
                ToolUseBlock(id=call.id, name=call.name, arguments=call.arguments)
            )

        stop = _FINISH_MAP.get(finish, StopReason.end_turn)
        if calls:
            stop = StopReason.tool_use
        result = GenerateResult(
            message=Message(role="assistant", content=content),
            stop_reason=stop,
            usage=_parse_usage(usage_data, model),
            model_id=model.id,
            provider_id=provider.id,
            metadata={"raw_stop_reason": finish},
        )
        yield StreamDone(result=result)
