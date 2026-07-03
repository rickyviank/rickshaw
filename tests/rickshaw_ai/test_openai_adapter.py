"""OpenAI-compatible adapter: translation, tools, streaming, cost, images."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from rickshaw_ai import GenerateRequest, Message, Pricing, Reasoning, StopReason, Tool
from rickshaw_ai.messages import ImageBlock, TextBlock
from tests.rickshaw_ai.conftest import make_models

URL = "https://oai.test/chat/completions"

CHAT = {
    "model": "test-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi there"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
}

TOOL_CALL = {
    "model": "test-model",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "call_1", "type": "function",
                     "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 8},
}


def _models(**kw):
    return make_models(protocol="openai", provider_id="oai", base_url="https://oai.test", **kw)


@respx.mock
async def test_generate_normalizes_response():
    respx.post(URL).mock(return_value=httpx.Response(200, json=CHAT))
    models = _models(pricing=Pricing(input=1.0, output=2.0))
    result = await models.get("oai/test-model").generate(
        GenerateRequest(messages=[Message.user("hello")])
    )
    assert result.text == "hi there"
    assert result.stop_reason == StopReason.end_turn
    assert result.usage.input_tokens == 10
    # cost = 10/1e6*1 + 5/1e6*2
    assert result.usage.cost_usd == pytest.approx(10 / 1e6 * 1 + 5 / 1e6 * 2)


@respx.mock
async def test_plain_text_uses_string_content():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=CHAT))
    models = _models()
    await models.get("oai/test-model").generate(
        GenerateRequest(system="be nice", messages=[Message.user("hi")])
    )
    sent = json.loads(route.calls[0].request.content)
    assert sent["messages"] == [
        {"role": "system", "content": "be nice"},
        {"role": "user", "content": "hi"},
    ]


@respx.mock
async def test_tool_calls_parsed_and_forwarded():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=TOOL_CALL))
    models = _models()
    tool = Tool(name="get_weather", description="w",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}})
    result = await models.get("oai/test-model").generate(
        GenerateRequest(messages=[Message.user("weather?")], tools=[tool], tool_choice="required")
    )
    sent = json.loads(route.calls[0].request.content)
    assert sent["tools"][0]["function"]["name"] == "get_weather"
    assert sent["tool_choice"] == "required"

    assert result.stop_reason == StopReason.tool_use
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "get_weather"
    assert result.tool_calls[0].arguments == {"city": "Paris"}


@respx.mock
async def test_reasoning_effort_forwarded_only_when_supported():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=CHAT))
    models = _models(reasoning=True)
    await models.get("oai/test-model").generate(
        GenerateRequest(messages=[Message.user("hi")], reasoning=Reasoning(effort="high"))
    )
    sent = json.loads(route.calls[0].request.content)
    assert sent["reasoning_effort"] == "high"


@respx.mock
async def test_reasoning_budget_maps_to_effort():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=CHAT))
    models = _models(reasoning=True)
    await models.get("oai/test-model").generate(
        GenerateRequest(messages=[Message.user("hi")], reasoning=Reasoning(budget_tokens=1000))
    )
    sent = json.loads(route.calls[0].request.content)
    assert sent["reasoning_effort"] == "low"


@respx.mock
async def test_image_input_translated():
    route = respx.post(URL).mock(return_value=httpx.Response(200, json=CHAT))
    models = _models(vision=True)
    msg = Message(role="user", content=[
        TextBlock(text="what is this"),
        ImageBlock(media_type="image/png", source="base64", data="AAAA"),
    ])
    await models.get("oai/test-model").generate(GenerateRequest(messages=[msg]))
    sent = json.loads(route.calls[0].request.content)
    parts = sent["messages"][0]["content"]
    assert parts[0] == {"type": "text", "text": "what is this"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"].startswith("data:image/png;base64,")


def _sse(chunks):
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


@respx.mock
async def test_streaming_tool_calls_assemble_and_match_nonstream():
    chunks = [
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "id": "call_1", "function": {"name": "get_weather", "arguments": ""}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '{"city": '}}]}}]},
        {"choices": [{"index": 0, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": '"Paris"}'}}]}}]},
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
         "usage": {"prompt_tokens": 20, "completion_tokens": 8}},
    ]
    respx.post(URL).mock(return_value=_sse(chunks))
    models = _models()
    tool = Tool(name="get_weather", parameters={"type": "object", "properties": {"city": {"type": "string"}}})

    events = []
    async for ev in models.get("oai/test-model").stream(
        GenerateRequest(messages=[Message.user("weather?")], tools=[tool])
    ):
        events.append(ev)

    done = events[-1]
    assert done.type == "done"
    assert done.result.stop_reason == StopReason.tool_use
    assert done.result.tool_calls[0].arguments == {"city": "Paris"}
    assert any(e.type == "tool_call_start" for e in events)
    assert any(e.type == "tool_call_end" for e in events)


@respx.mock
async def test_malformed_tool_call_args_logs_warning(caplog):
    """Malformed tool-call JSON arguments should log a warning."""
    import logging

    bad_response = {
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": "{{not json}}",
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }
    models = make_models(
        protocol="openai", provider_id="oai", base_url="https://oai.test/v1"
    )
    respx.post("https://oai.test/v1/chat/completions").respond(json=bad_response)
    with caplog.at_level(
        logging.WARNING, logger="rickshaw_ai.providers.openai_compatible"
    ):
        result = await models.get("oai/test-model").generate(
            GenerateRequest(messages=[Message.user("hi")])
        )
    assert result.tool_calls[0].arguments == {}
    assert "Malformed tool-call arguments" in caplog.text
