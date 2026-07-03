"""Adapter for the Anthropic Messages API.

Supports API-key auth (``x-api-key``) and Claude Pro/Max OAuth (``Authorization:
Bearer`` + the OAuth beta header). Unifies extended thinking into
:class:`ThinkingBlock`, preserving signatures for same-provider replay.
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
    ThinkingDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
)
from rickshaw_ai.tools import Tool, ToolCall

_ANTHROPIC_VERSION = "2023-06-01"
_OAUTH_BETA = "oauth-2025-04-20"
_DEFAULT_MAX_TOKENS = 4096

_EFFORT_BUDGET = {"low": 1024, "medium": 4096, "high": 16384}

_TOOL_CHOICE_MAP = {
    "auto": {"type": "auto"},
    "required": {"type": "any"},
    "none": {"type": "none"},
}

_STOP_MAP = {
    "end_turn": StopReason.end_turn,
    "max_tokens": StopReason.max_output_tokens,
    "tool_use": StopReason.tool_use,
    "stop_sequence": StopReason.stop_sequence,
    "refusal": StopReason.refusal,
    "pause_turn": StopReason.pause,
}


def _thinking_budget(req: GenerateRequest) -> int | None:
    if req.reasoning is None:
        return None
    if req.reasoning.budget_tokens is not None:
        return req.reasoning.budget_tokens
    if req.reasoning.effort is not None:
        return _EFFORT_BUDGET.get(req.reasoning.effort)
    return None


def _tools_payload(tools: list[Tool]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in tools
    ]


def _block_content(blocks: list) -> Any:
    """Serialize content blocks; a plain string when text-only."""
    if all(isinstance(b, TextBlock) for b in blocks):
        return "".join(b.text for b in blocks)
    out: list[dict[str, Any]] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            out.append({"type": "text", "text": b.text})
        elif isinstance(b, ImageBlock):
            if b.source == "url":
                out.append(
                    {"type": "image", "source": {"type": "url", "url": b.data}}
                )
            else:
                out.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": b.media_type,
                            "data": b.data,
                        },
                    }
                )
    return out


def _wire_messages(req: GenerateRequest) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    if req.system:
        system_parts.append(req.system)
    chat: list[dict[str, Any]] = []

    for msg in req.messages:
        if msg.role == "system":
            system_parts.append(msg.text)
            continue

        tool_uses = [b for b in msg.content if isinstance(b, ToolUseBlock)]
        tool_results = [b for b in msg.content if isinstance(b, ToolResultBlock)]
        thinking = [b for b in msg.content if isinstance(b, ThinkingBlock)]
        plain = [b for b in msg.content if isinstance(b, (TextBlock, ImageBlock))]

        if tool_results:
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_use_id,
                    "content": _block_content(tr.content) or "",
                    **({"is_error": True} if tr.is_error else {}),
                }
                for tr in tool_results
            ]
            chat.append({"role": "user", "content": content})
            continue

        if msg.role == "assistant" and (tool_uses or thinking):
            content: list[dict[str, Any]] = []
            # Replay signed thinking blocks (same-provider only).
            for th in thinking:
                if th.provider == "anthropic" and th.signature and not th.redacted:
                    content.append(
                        {"type": "thinking", "thinking": th.text, "signature": th.signature}
                    )
            for b in plain:
                if isinstance(b, TextBlock) and b.text:
                    content.append({"type": "text", "text": b.text})
            for tu in tool_uses:
                content.append(
                    {"type": "tool_use", "id": tu.id, "name": tu.name, "input": tu.arguments}
                )
            chat.append({"role": "assistant", "content": content})
            continue

        chat.append({"role": msg.role, "content": _block_content(msg.content)})

    system = "\n\n".join(p for p in system_parts if p) if system_parts else None
    return system, chat


def _parse_usage(u: dict[str, Any], model: ModelInfo) -> Usage:
    usage = Usage(
        input_tokens=u.get("input_tokens", 0),
        output_tokens=u.get("output_tokens", 0),
        cache_read_tokens=u.get("cache_read_input_tokens", 0),
        cache_write_tokens=u.get("cache_creation_input_tokens", 0),
    )
    usage.cost_usd = usage.compute_cost(model.pricing)
    return usage


class AnthropicAdapter(ProviderAdapter):
    protocol = "anthropic"

    def endpoint(self, provider: ProviderInfo, model: ModelInfo, *, stream: bool) -> str:
        return f"{provider.base_url.rstrip('/')}/v1/messages"

    def extra_headers(self, provider: ProviderInfo, auth) -> dict[str, str]:
        headers = {"anthropic-version": _ANTHROPIC_VERSION}
        if "Authorization" in auth.headers:  # OAuth (Claude Pro/Max)
            headers["anthropic-beta"] = _OAUTH_BETA
        return headers

    def build_body(
        self, req: GenerateRequest, model: ModelInfo, *, stream: bool
    ) -> dict[str, Any]:
        system, chat = _wire_messages(req)
        body: dict[str, Any] = {
            "model": model.model,
            "messages": chat,
            "max_tokens": req.max_output_tokens or _DEFAULT_MAX_TOKENS,
        }
        if system is not None:
            body["system"] = system
        if stream:
            body["stream"] = True
        if req.temperature is not None:
            body["temperature"] = req.temperature
        if req.stop_sequences:
            body["stop_sequences"] = req.stop_sequences
        if model.supports_reasoning:
            budget = _thinking_budget(req)
            if budget is not None:
                body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if req.tools:
            body["tools"] = _tools_payload(req.tools)
            if req.tool_choice is not None:
                mapped = _TOOL_CHOICE_MAP.get(req.tool_choice)
                if mapped is not None:
                    body["tool_choice"] = mapped
        body.update(req.provider_options)
        return body

    def parse_response(
        self, data: dict[str, Any], model: ModelInfo, provider: ProviderInfo
    ) -> GenerateResult:
        content: list = []
        for block in data.get("content", []):
            btype = block.get("type")
            if btype == "text":
                content.append(TextBlock(text=block.get("text", "")))
            elif btype == "thinking":
                content.append(
                    ThinkingBlock(
                        text=block.get("thinking", ""),
                        signature=block.get("signature"),
                        provider="anthropic",
                    )
                )
            elif btype == "redacted_thinking":
                content.append(ThinkingBlock(redacted=True, provider="anthropic"))
            elif btype == "tool_use":
                content.append(
                    ToolUseBlock(
                        id=block.get("id", ""),
                        name=block.get("name", ""),
                        arguments=block.get("input") or {},
                    )
                )

        raw_stop = data.get("stop_reason")
        stop = _STOP_MAP.get(raw_stop, StopReason.end_turn)
        return GenerateResult(
            message=Message(role="assistant", content=content),
            stop_reason=stop,
            usage=_parse_usage(data.get("usage", {}), model),
            model_id=model.id,
            provider_id=provider.id,
            metadata={
                "raw_stop_reason": raw_stop,
                "response_model": data.get("model", model.model),
                "raw": data,
            },
        )

    async def parse_stream(
        self, response: httpx.Response, model: ModelInfo, provider: ProviderInfo
    ) -> AsyncIterator[StreamEvent]:
        blocks: dict[int, dict[str, Any]] = {}
        input_tokens = 0
        output_tokens = 0
        raw_stop: str | None = None

        async for payload in aiter_sse(response):
            event = json.loads(payload)
            etype = event.get("type")

            if etype == "message_start":
                usage = event.get("message", {}).get("usage", {})
                input_tokens = usage.get("input_tokens", 0)
            elif etype == "content_block_start":
                idx = event["index"]
                block = event["content_block"]
                blocks[idx] = {"block": block, "text": "", "json": "", "thinking": "", "sig": None}
                if block.get("type") == "tool_use":
                    yield ToolCallStart(id=block.get("id", ""), name=block.get("name", ""))
            elif etype == "content_block_delta":
                idx = event["index"]
                delta = event["delta"]
                dtype = delta.get("type")
                if dtype == "text_delta":
                    blocks[idx]["text"] += delta.get("text", "")
                    yield TextDelta(text=delta.get("text", ""))
                elif dtype == "thinking_delta":
                    blocks[idx]["thinking"] += delta.get("thinking", "")
                    yield ThinkingDelta(text=delta.get("thinking", ""))
                elif dtype == "signature_delta":
                    blocks[idx]["sig"] = delta.get("signature")
                elif dtype == "input_json_delta":
                    frag = delta.get("partial_json", "")
                    blocks[idx]["json"] += frag
                    yield ToolCallDelta(
                        id=blocks[idx]["block"].get("id", ""), arguments_fragment=frag
                    )
            elif etype == "message_delta":
                raw_stop = event.get("delta", {}).get("stop_reason", raw_stop)
                output_tokens = event.get("usage", {}).get("output_tokens", output_tokens)
            elif etype == "message_stop":
                break

        content: list = []
        for idx in sorted(blocks):
            entry = blocks[idx]
            block = entry["block"]
            btype = block.get("type")
            if btype == "text":
                content.append(TextBlock(text=entry["text"]))
            elif btype == "thinking":
                content.append(
                    ThinkingBlock(
                        text=entry["thinking"], signature=entry["sig"], provider="anthropic"
                    )
                )
            elif btype == "tool_use":
                try:
                    args = json.loads(entry["json"]) if entry["json"].strip() else {}
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Malformed tool-call arguments for %r in stream, "
                        "defaulting to empty: %s",
                        block.get("name", "?"), exc,
                    )
                    args = {}
                call = ToolCall(id=block.get("id", ""), name=block.get("name", ""), arguments=args)
                yield ToolCallEnd(call=call)
                content.append(
                    ToolUseBlock(id=call.id, name=call.name, arguments=call.arguments)
                )

        usage = Usage(input_tokens=input_tokens, output_tokens=output_tokens)
        usage.cost_usd = usage.compute_cost(model.pricing)
        yield StreamDone(
            result=GenerateResult(
                message=Message(role="assistant", content=content),
                stop_reason=_STOP_MAP.get(raw_stop, StopReason.end_turn),
                usage=usage,
                model_id=model.id,
                provider_id=provider.id,
                metadata={"raw_stop_reason": raw_stop},
            )
        )
