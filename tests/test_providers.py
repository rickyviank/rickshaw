"""Tests for provider implementations (mocked HTTP)."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
import respx

from rickshaw.config import ProviderProfile, is_local_url
from rickshaw.providers.base import Effort, Message, Response, ToolCall, ToolSpec
from rickshaw.providers.openai_provider import OpenAIProvider
from rickshaw.providers.devin_provider import DevinProvider
from rickshaw.providers.anthropic_provider import AnthropicProvider
from rickshaw_ai.credentials.types import OAuthCredential


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

OPENAI_CHAT_RESPONSE = {
    "id": "chatcmpl-test",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from OpenAI!"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
}


@respx.mock
def test_openai_complete_returns_normalized_response():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_CHAT_RESPONSE)
    )

    provider = OpenAIProvider(api_key="sk-test")
    messages = [Message(role="user", content="Hi")]
    response = provider.complete(messages, effort=Effort.MEDIUM)

    assert isinstance(response, Response)
    assert response.text == "Hello from OpenAI!"
    assert response.model == "gpt-4o"
    assert response.usage.total_tokens == 15
    assert response.effort == Effort.MEDIUM


@respx.mock
def test_openai_complete_with_effort_high():
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_CHAT_RESPONSE)
    )

    provider = OpenAIProvider(api_key="sk-test", model="o3-mini")
    messages = [Message(role="user", content="Hi")]
    response = provider.complete(messages, effort=Effort.HIGH)

    assert response.effort == Effort.HIGH


@respx.mock
def test_openai_validate_success():
    respx.get("https://api.openai.com/v1/models").mock(
        return_value=httpx.Response(200, json={"data": [{"id": "gpt-4o"}]})
    )
    provider = OpenAIProvider(api_key="sk-test")
    provider.validate()


def test_openai_validate_no_key():
    provider = OpenAIProvider(api_key="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        provider.validate()


def _write_credential_file(path: Path, provider_id: str, credential) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({provider_id: credential.model_dump(mode="json")}))


@respx.mock
def test_openai_complete_uses_stored_oauth_credential(tmp_path, monkeypatch):
    cred_path = tmp_path / "credentials.json"
    monkeypatch.setenv("RICKSHAW_CREDENTIALS_PATH", str(cred_path))
    credential = OAuthCredential(
        access="oauth-access-token",
        refresh="oauth-refresh-token",
        expires=int((time.time() + 3600) * 1000),
    )
    _write_credential_file(cred_path, "openai", credential)

    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_CHAT_RESPONSE)
    )

    provider = OpenAIProvider(api_key="")
    response = provider.complete([Message(role="user", content="Hi")])

    assert response.text == "Hello from OpenAI!"
    assert route.calls
    assert route.calls[0].request.headers.get("authorization") == "Bearer oauth-access-token"


@respx.mock
def test_openai_validate_passes_with_stored_oauth_credential(
    tmp_path, monkeypatch
):
    cred_path = tmp_path / "credentials.json"
    monkeypatch.setenv("RICKSHAW_CREDENTIALS_PATH", str(cred_path))
    credential = OAuthCredential(
        access="oauth-access-token",
        refresh="oauth-refresh-token",
        expires=int((time.time() + 3600) * 1000),
    )
    _write_credential_file(cred_path, "openai", credential)

    provider = OpenAIProvider(api_key="")
    provider.validate()


@respx.mock
def test_openai_seed_api_key_still_works_without_stored_credential(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("RICKSHAW_CREDENTIALS_PATH", str(tmp_path / "missing.json"))
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_CHAT_RESPONSE)
    )

    provider = OpenAIProvider(api_key="sk-test")
    response = provider.complete([Message(role="user", content="Hi")])
    assert response.text == "Hello from OpenAI!"

    provider = OpenAIProvider(api_key="")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        provider.validate()


def test_openai_capabilities():
    provider = OpenAIProvider(api_key="sk-test")
    caps = provider.capabilities()
    assert caps.streaming is True
    assert caps.embeddings is True
    assert caps.max_context_tokens > 0


OPENAI_TOOL_CALL_RESPONSE = {
    "id": "chatcmpl-tool",
    "object": "chat.completion",
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_abc123",
                        "type": "function",
                        "function": {
                            "name": "remember",
                            "arguments": '{"fact": "user prefers dark mode"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ],
    "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
}


@respx.mock
def test_openai_complete_with_tool_calls():
    """Tool calls in the response are parsed into normalized ToolCall objects."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_TOOL_CALL_RESPONSE)
    )

    provider = OpenAIProvider(api_key="sk-test")
    tools = [
        ToolSpec(
            name="remember",
            description="Store a fact",
            parameters={"type": "object", "properties": {"fact": {"type": "string"}}},
        )
    ]
    messages = [Message(role="user", content="Remember that I like dark mode")]
    response = provider.complete(messages, tools=tools)

    assert isinstance(response, Response)
    assert response.text == ""
    assert len(response.tool_calls) == 1

    tc = response.tool_calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "call_abc123"
    assert tc.name == "remember"
    assert tc.arguments == {"fact": "user prefers dark mode"}
    assert tc.raw["type"] == "function"


@respx.mock
def test_openai_complete_without_tool_calls_defaults_empty():
    """Responses without tool calls default to an empty list."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_CHAT_RESPONSE)
    )

    provider = OpenAIProvider(api_key="sk-test")
    messages = [Message(role="user", content="Hi")]
    response = provider.complete(messages)

    assert response.tool_calls == []


def test_openai_parse_tool_calls_directly():
    """OpenAIProvider._parse_tool_calls parses raw OpenAI tool calls."""
    raw = [
        {
            "id": "call_xyz",
            "type": "function",
            "function": {
                "name": "recall",
                "arguments": '{"query": "dark mode"}',
            },
        }
    ]
    parsed = OpenAIProvider._parse_tool_calls(raw)
    assert len(parsed) == 1
    assert isinstance(parsed[0], ToolCall)
    assert parsed[0].id == "call_xyz"
    assert parsed[0].name == "recall"
    assert parsed[0].arguments == {"query": "dark mode"}


def test_openai_parse_tool_calls_malformed_arguments():
    """Malformed JSON arguments fall back to an empty dict."""
    raw = [{"id": "c1", "function": {"name": "remember", "arguments": "not-json"}}]
    parsed = OpenAIProvider._parse_tool_calls(raw)
    assert parsed[0].arguments == {}


def test_devin_parse_tool_calls_returns_empty():
    """DevinProvider does not support tool calls yet — returns []."""
    assert DevinProvider._parse_tool_calls([{"id": "x"}]) == []


@respx.mock
def test_openai_forwards_tool_choice():
    """tool_choice is forwarded in the payload when tools are provided."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_CHAT_RESPONSE)
    )
    provider = OpenAIProvider(api_key="sk-test")
    tools = [
        ToolSpec(
            name="recall",
            description="Recall memories",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
    ]
    provider.complete([Message(role="user", content="hi")], tools=tools, tool_choice="required")
    sent = json.loads(route.calls[0].request.content)
    assert sent["tool_choice"] == "required"


@respx.mock
def test_openai_omits_tool_choice_without_tools():
    """tool_choice is not sent when no tools are advertised."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_CHAT_RESPONSE)
    )
    provider = OpenAIProvider(api_key="sk-test")
    provider.complete([Message(role="user", content="hi")], tool_choice="required")
    sent = json.loads(route.calls[0].request.content)
    assert "tool_choice" not in sent


@respx.mock
def test_openai_complete_forwards_tools_in_payload():
    """When tools are provided, they are forwarded in the OpenAI tools format."""
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=OPENAI_CHAT_RESPONSE)
    )

    provider = OpenAIProvider(api_key="sk-test")
    tools = [
        ToolSpec(
            name="recall",
            description="Recall memories",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )
    ]
    messages = [Message(role="user", content="What do you remember?")]
    provider.complete(messages, tools=tools)

    sent = json.loads(route.calls[0].request.content)
    assert "tools" in sent
    assert sent["tools"][0]["type"] == "function"
    assert sent["tools"][0]["function"]["name"] == "recall"


@respx.mock
def test_openai_embed():
    respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"embedding": [0.1, 0.2, 0.3]}]},
        )
    )
    provider = OpenAIProvider(api_key="sk-test")
    vec = provider.embed("hello")
    assert vec == [0.1, 0.2, 0.3]


@respx.mock
def test_openai_available_models():
    respx.get("https://api.openai.com/v1/models").mock(
        return_value=httpx.Response(
            200, json={"data": [{"id": "gpt-4o"}, {"id": "gpt-3.5-turbo"}]}
        )
    )
    provider = OpenAIProvider(api_key="sk-test")
    models = provider.available_models()
    assert "gpt-4o" in models
    assert "gpt-3.5-turbo" in models


# ---------------------------------------------------------------------------
# Devin provider
# ---------------------------------------------------------------------------

DEVIN_CHAT_RESPONSE = {
    "model": "devin",
    "choices": [
        {
            "message": {"role": "assistant", "content": "Hello from Devin!"},
        }
    ],
    "usage": {"prompt_tokens": 8, "completion_tokens": 4, "total_tokens": 12},
}


@respx.mock
def test_devin_complete_returns_normalized_response():
    respx.post("https://api.devin.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=DEVIN_CHAT_RESPONSE)
    )
    provider = DevinProvider(api_key="test-key")
    messages = [Message(role="user", content="Hi")]
    response = provider.complete(messages, effort=Effort.MEDIUM)

    assert isinstance(response, Response)
    assert response.text == "Hello from Devin!"
    assert response.model == "devin"
    assert response.usage.total_tokens == 12


@respx.mock
def test_devin_complete_preserves_effort_in_response():
    """The effort passed to complete() should be reflected in the Response."""
    respx.post("https://api.devin.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=DEVIN_CHAT_RESPONSE)
    )
    provider = DevinProvider(api_key="test-key")
    messages = [Message(role="user", content="Hi")]

    for effort in Effort:
        response = provider.complete(messages, effort=effort)
        assert response.effort == effort


@respx.mock
def test_devin_complete_with_extra_kwargs():
    """Extra kwargs should be forwarded in the request payload."""
    route = respx.post("https://api.devin.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=DEVIN_CHAT_RESPONSE)
    )
    provider = DevinProvider(api_key="test-key")
    messages = [Message(role="user", content="Hi")]
    provider.complete(messages, temperature=0.5)

    sent_payload = json.loads(route.calls[0].request.content)
    assert sent_payload["temperature"] == 0.5


@respx.mock
def test_devin_complete_sends_correct_message_format():
    """Messages should be serialized as [{role, content}, ...]."""
    route = respx.post("https://api.devin.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=DEVIN_CHAT_RESPONSE)
    )
    provider = DevinProvider(api_key="test-key")
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
    ]
    provider.complete(messages)

    sent_payload = json.loads(route.calls[0].request.content)
    assert sent_payload["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hello"},
    ]


@respx.mock
def test_devin_complete_sends_auth_header():
    """The Authorization header should carry the API key."""
    route = respx.post("https://api.devin.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=DEVIN_CHAT_RESPONSE)
    )
    provider = DevinProvider(api_key="my-secret-key")
    provider.complete([Message(role="user", content="Hi")])

    auth = route.calls[0].request.headers["authorization"]
    assert auth == "Bearer my-secret-key"


@respx.mock
def test_devin_complete_handles_http_error():
    """HTTP errors should propagate as exceptions."""
    respx.post("https://api.devin.ai/v1/chat/completions").mock(
        return_value=httpx.Response(500, json={"error": "internal"})
    )
    provider = DevinProvider(api_key="test-key")
    with pytest.raises(httpx.HTTPStatusError):
        provider.complete([Message(role="user", content="Hi")])


@respx.mock
def test_devin_complete_handles_empty_choices():
    """Gracefully handle a response with empty choices."""
    respx.post("https://api.devin.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"model": "devin", "choices": [{}]})
    )
    provider = DevinProvider(api_key="test-key")
    response = provider.complete([Message(role="user", content="Hi")])
    assert response.text == ""
    assert response.model == "devin"


@respx.mock
def test_devin_complete_custom_base_url():
    """A custom base URL should be used for requests."""
    respx.post("https://custom.devin.example/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=DEVIN_CHAT_RESPONSE)
    )
    provider = DevinProvider(api_key="test-key", base_url="https://custom.devin.example")
    response = provider.complete([Message(role="user", content="Hi")])
    assert response.text == "Hello from Devin!"


def test_devin_validate_no_key():
    provider = DevinProvider(api_key="")
    with pytest.raises(ValueError, match="DEVIN_API_KEY"):
        provider.validate()


def test_devin_validate_with_key():
    """validate() should not raise when an API key is provided."""
    provider = DevinProvider(api_key="test-key")
    provider.validate()


def test_devin_capabilities_no_embeddings():
    provider = DevinProvider(api_key="test-key")
    caps = provider.capabilities()
    assert caps.embeddings is False
    assert caps.streaming is False


def test_devin_capabilities_full():
    """Verify all capability fields for completeness."""
    provider = DevinProvider(api_key="test-key")
    caps = provider.capabilities()
    assert caps.embeddings is False
    assert caps.streaming is False
    assert caps.function_calling is False
    assert caps.vision is False
    assert caps.max_context_tokens == 128_000
    assert caps.effort_levels == []


def test_devin_available_models():
    provider = DevinProvider(api_key="test-key")
    models = provider.available_models()
    assert "devin" in models


def test_devin_name():
    provider = DevinProvider(api_key="test-key")
    assert provider.name == "devin"


# ---------------------------------------------------------------------------
# Stream fallback
# ---------------------------------------------------------------------------

@respx.mock
def test_stream_fallback_to_complete():
    """Providers without native streaming fall back to complete()."""
    respx.post("https://api.devin.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=DEVIN_CHAT_RESPONSE)
    )
    provider = DevinProvider(api_key="test-key")
    messages = [Message(role="user", content="Hi")]
    chunks = list(provider.stream(messages))
    assert chunks == ["Hello from Devin!"]


@respx.mock
def test_stream_fallback_preserves_effort():
    """Stream fallback should forward the effort level to complete()."""
    respx.post("https://api.devin.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=DEVIN_CHAT_RESPONSE)
    )
    provider = DevinProvider(api_key="test-key")
    messages = [Message(role="user", content="Hi")]
    chunks = list(provider.stream(messages, effort=Effort.HIGH))
    assert len(chunks) == 1
    assert chunks[0] == "Hello from Devin!"


# ---------------------------------------------------------------------------
# Effort degradation
# ---------------------------------------------------------------------------

def test_effort_levels_empty_degrades_gracefully():
    """Providers with empty effort_levels still accept any effort value."""
    provider = DevinProvider(api_key="test-key")
    caps = provider.capabilities()
    assert caps.effort_levels == []


@respx.mock
def test_devin_accepts_all_effort_levels_without_error():
    """Even without effort support, all effort values should be accepted."""
    respx.post("https://api.devin.ai/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=DEVIN_CHAT_RESPONSE)
    )
    provider = DevinProvider(api_key="test-key")
    messages = [Message(role="user", content="Hi")]
    for effort in Effort:
        response = provider.complete(messages, effort=effort)
        assert response.text == "Hello from Devin!"


# ---------------------------------------------------------------------------
# Anthropic provider
# ---------------------------------------------------------------------------

ANTHROPIC_MESSAGES_RESPONSE = {
    "id": "msg_test",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-latest",
    "content": [{"type": "text", "text": "Hello from Claude!"}],
    "usage": {"input_tokens": 12, "output_tokens": 6},
}

ANTHROPIC_TOOL_USE_RESPONSE = {
    "id": "msg_tool",
    "type": "message",
    "role": "assistant",
    "model": "claude-3-5-sonnet-latest",
    "content": [
        {"type": "text", "text": "Let me remember that."},
        {
            "type": "tool_use",
            "id": "toolu_abc123",
            "name": "remember",
            "input": {"fact": "user prefers dark mode"},
        },
    ],
    "usage": {"input_tokens": 20, "output_tokens": 10},
}


@respx.mock
def test_anthropic_complete_returns_normalized_response():
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_MESSAGES_RESPONSE)
    )
    provider = AnthropicProvider(api_key="sk-ant-test")
    messages = [Message(role="user", content="Hi")]
    response = provider.complete(messages, effort=Effort.MEDIUM)

    assert isinstance(response, Response)
    assert response.text == "Hello from Claude!"
    assert response.model == "claude-3-5-sonnet-latest"
    assert response.usage.prompt_tokens == 12
    assert response.usage.completion_tokens == 6
    assert response.usage.total_tokens == 18
    assert response.effort == Effort.MEDIUM


@respx.mock
def test_anthropic_sends_auth_and_version_headers():
    """Anthropic uses x-api-key and anthropic-version, not Authorization."""
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_MESSAGES_RESPONSE)
    )
    provider = AnthropicProvider(api_key="my-secret-key")
    provider.complete([Message(role="user", content="Hi")])

    headers = route.calls[0].request.headers
    assert headers["x-api-key"] == "my-secret-key"
    assert headers["anthropic-version"] == "2023-06-01"
    assert "authorization" not in headers


@respx.mock
def test_anthropic_hoists_system_message():
    """System messages are hoisted into the top-level system field."""
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_MESSAGES_RESPONSE)
    )
    provider = AnthropicProvider(api_key="sk-ant-test")
    messages = [
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Hello"),
    ]
    provider.complete(messages)

    sent = json.loads(route.calls[0].request.content)
    assert sent["system"] == "You are helpful."
    assert sent["messages"] == [{"role": "user", "content": "Hello"}]
    assert sent["max_tokens"] > 0


@respx.mock
def test_anthropic_complete_with_tool_use():
    """tool_use blocks are parsed into normalized ToolCall objects."""
    respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_TOOL_USE_RESPONSE)
    )
    provider = AnthropicProvider(api_key="sk-ant-test")
    tools = [
        ToolSpec(
            name="remember",
            description="Store a fact",
            parameters={"type": "object", "properties": {"fact": {"type": "string"}}},
        )
    ]
    messages = [Message(role="user", content="Remember that I like dark mode")]
    response = provider.complete(messages, tools=tools)

    assert response.text == "Let me remember that."
    assert len(response.tool_calls) == 1
    tc = response.tool_calls[0]
    assert isinstance(tc, ToolCall)
    assert tc.id == "toolu_abc123"
    assert tc.name == "remember"
    assert tc.arguments == {"fact": "user prefers dark mode"}
    assert tc.raw["type"] == "tool_use"


def test_anthropic_parse_tool_calls_directly():
    """_parse_tool_calls converts tool_use blocks; text blocks are skipped."""
    blocks = [
        {"type": "text", "text": "thinking..."},
        {
            "type": "tool_use",
            "id": "toolu_xyz",
            "name": "recall",
            "input": {"query": "dark mode"},
        },
    ]
    parsed = AnthropicProvider._parse_tool_calls(blocks)
    assert len(parsed) == 1
    assert isinstance(parsed[0], ToolCall)
    assert parsed[0].id == "toolu_xyz"
    assert parsed[0].name == "recall"
    assert parsed[0].arguments == {"query": "dark mode"}


@respx.mock
def test_anthropic_forwards_tools_in_payload():
    """Tools are forwarded in Anthropic's {name, description, input_schema} shape."""
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_MESSAGES_RESPONSE)
    )
    provider = AnthropicProvider(api_key="sk-ant-test")
    schema = {"type": "object", "properties": {"query": {"type": "string"}}}
    tools = [ToolSpec(name="recall", description="Recall memories", parameters=schema)]
    provider.complete([Message(role="user", content="hi")], tools=tools)

    sent = json.loads(route.calls[0].request.content)
    assert sent["tools"][0] == {
        "name": "recall",
        "description": "Recall memories",
        "input_schema": schema,
    }


@respx.mock
def test_anthropic_maps_tool_choice_required_to_any():
    """OpenAI-style 'required' maps to Anthropic's {'type': 'any'}."""
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_MESSAGES_RESPONSE)
    )
    provider = AnthropicProvider(api_key="sk-ant-test")
    tools = [
        ToolSpec(
            name="recall",
            description="Recall memories",
            parameters={"type": "object", "properties": {}},
        )
    ]
    provider.complete(
        [Message(role="user", content="hi")], tools=tools, tool_choice="required"
    )
    sent = json.loads(route.calls[0].request.content)
    assert sent["tool_choice"] == {"type": "any"}


@respx.mock
def test_anthropic_omits_tool_choice_without_tools():
    """tool_choice and tools are not sent when no tools are advertised."""
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(200, json=ANTHROPIC_MESSAGES_RESPONSE)
    )
    provider = AnthropicProvider(api_key="sk-ant-test")
    provider.complete([Message(role="user", content="hi")], tool_choice="required")
    sent = json.loads(route.calls[0].request.content)
    assert "tool_choice" not in sent
    assert "tools" not in sent


def test_anthropic_validate_no_key():
    provider = AnthropicProvider(api_key="")
    with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
        provider.validate()


def test_anthropic_validate_with_key():
    provider = AnthropicProvider(api_key="sk-ant-test")
    provider.validate()


def test_anthropic_capabilities():
    provider = AnthropicProvider(api_key="sk-ant-test")
    caps = provider.capabilities()
    assert caps.embeddings is False
    assert caps.function_calling is True
    assert caps.streaming is True
    assert caps.vision is True
    assert caps.max_context_tokens == 200_000
    assert caps.effort_levels == []


def test_anthropic_available_models():
    provider = AnthropicProvider(api_key="sk-ant-test")
    models = provider.available_models()
    assert "claude-3-5-sonnet-latest" in models


def test_anthropic_name():
    provider = AnthropicProvider(api_key="sk-ant-test")
    assert provider.name == "anthropic"


# ---------------------------------------------------------------------------
# is_local_url / ProviderProfile.is_local_endpoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "http://localhost:11434/v1",
        "http://127.0.0.1:8000/v1",
        "http://[::1]:8000/v1",
        "http://0.0.0.0:8080/v1",
        "http://mybox.local:5000/api",
        "http://10.0.0.5:8080/v1",
        "http://192.168.1.100:8000/v1",
        "http://172.16.0.1:9000/v1",
    ],
)
def test_is_local_url_true(url: str):
    assert is_local_url(url) is True


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com/v1",
        "https://api.anthropic.com",
        "https://api.devin.ai",
        "https://api.deepseek.com/v1",
    ],
)
def test_is_local_url_false(url: str):
    assert is_local_url(url) is False


def test_provider_profile_is_local_endpoint():
    local = ProviderProfile(
        base_url="http://localhost:11434/v1", model="llama3", api_key_env="X",
    )
    assert local.is_local_endpoint() is True

    remote = ProviderProfile(
        base_url="https://api.openai.com/v1", model="gpt-4o", api_key_env="Y",
    )
    assert remote.is_local_endpoint() is False


# ---------------------------------------------------------------------------
# Model cache: in-memory single-fetch
# ---------------------------------------------------------------------------


@respx.mock
def test_openai_available_models_single_fetch(tmp_path: Path):
    """The httpx network call happens exactly once across multiple calls."""
    route = respx.get("https://api.openai.com/v1/models").mock(
        return_value=httpx.Response(
            200, json={"data": [{"id": "gpt-4o"}, {"id": "gpt-3.5-turbo"}]}
        )
    )

    cache_path = tmp_path / "models_cache.json"
    with patch("rickshaw.settings.load_model_cache", return_value={}), \
         patch("rickshaw.settings.save_model_cache") as mock_save:
        provider = OpenAIProvider(api_key="sk-test")
        m1 = provider.available_models()
        m2 = provider.available_models()
        m3 = provider.available_models()

    assert route.call_count == 1
    assert m1 == m2 == m3
    assert "gpt-4o" in m1
    mock_save.assert_called_once()


@respx.mock
def test_anthropic_available_models_single_fetch():
    """Static-list providers also cache in-memory (one fetcher call)."""
    with patch("rickshaw.settings.load_model_cache", return_value={}), \
         patch("rickshaw.settings.save_model_cache"):
        provider = AnthropicProvider(api_key="sk-ant-test")
        m1 = provider.available_models()
        m2 = provider.available_models()

    assert m1 == m2
    assert "claude-3-5-sonnet-latest" in m1


@respx.mock
def test_devin_available_models_single_fetch():
    """DevinProvider caches its static list in-memory."""
    with patch("rickshaw.settings.load_model_cache", return_value={}), \
         patch("rickshaw.settings.save_model_cache"):
        provider = DevinProvider(api_key="test-key")
        m1 = provider.available_models()
        m2 = provider.available_models()

    assert m1 == m2
    assert "devin" in m1


# ---------------------------------------------------------------------------
# Model cache: disk fallback on network failure
# ---------------------------------------------------------------------------


@respx.mock
def test_openai_disk_fallback_on_network_failure():
    """When the network fetch fails, fall back to the disk-cached list."""
    respx.get("https://api.openai.com/v1/models").mock(
        return_value=httpx.Response(500, json={"error": "down"})
    )

    cached_disk = {"openai:https://api.openai.com/v1": ["gpt-4o-cached"]}
    with patch("rickshaw.settings.load_model_cache", return_value=cached_disk), \
         patch("rickshaw.settings.save_model_cache"):
        provider = OpenAIProvider(api_key="sk-test")
        models = provider.available_models()

    assert models == ["gpt-4o-cached"]


# ---------------------------------------------------------------------------
# Model cache: no cache + remote API → instructive error
# ---------------------------------------------------------------------------


@respx.mock
def test_openai_no_cache_remote_raises():
    """No disk cache + remote endpoint failure → ConnectionError with guidance."""
    respx.get("https://api.openai.com/v1/models").mock(
        return_value=httpx.Response(500, json={"error": "down"})
    )

    with patch("rickshaw.settings.load_model_cache", return_value={}):
        provider = OpenAIProvider(api_key="sk-test")
        with pytest.raises(ConnectionError, match="connect to the internet"):
            provider.available_models()


# ---------------------------------------------------------------------------
# Model cache: local host → surfaces raw connection error
# ---------------------------------------------------------------------------


@respx.mock
def test_openai_local_host_surfaces_raw_error():
    """Local endpoint failure → ConnectionError naming the local server."""
    respx.get("http://localhost:11434/v1/models").mock(
        side_effect=httpx.ConnectError("connection refused")
    )

    with patch("rickshaw.settings.load_model_cache", return_value={}):
        provider = OpenAIProvider(
            api_key="sk-test", base_url="http://localhost:11434/v1"
        )
        with pytest.raises(ConnectionError, match="local inference server"):
            provider.available_models()


# ---------------------------------------------------------------------------
# Model cache: disk cache is written on successful fetch
# ---------------------------------------------------------------------------


@respx.mock
def test_openai_disk_cache_written_on_success():
    """A successful fetch persists the result to the disk cache."""
    respx.get("https://api.openai.com/v1/models").mock(
        return_value=httpx.Response(
            200, json={"data": [{"id": "gpt-4o"}]}
        )
    )

    saved: list[dict] = []

    def _capture_save(data: dict, path=None):
        saved.append(data)

    with patch("rickshaw.settings.load_model_cache", return_value={}), \
         patch("rickshaw.settings.save_model_cache", side_effect=_capture_save):
        provider = OpenAIProvider(api_key="sk-test")
        provider.available_models()

    assert len(saved) == 1
    assert saved[0]["openai:https://api.openai.com/v1"] == ["gpt-4o"]
