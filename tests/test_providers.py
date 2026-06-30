"""Tests for provider implementations (mocked HTTP)."""

import json

import httpx
import pytest
import respx

from rickshaw.providers.base import Effort, Message, Response
from rickshaw.providers.openai_provider import OpenAIProvider
from rickshaw.providers.devin_provider import DevinProvider


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


def test_openai_capabilities():
    provider = OpenAIProvider(api_key="sk-test")
    caps = provider.capabilities()
    assert caps.streaming is True
    assert caps.embeddings is True
    assert caps.max_context_tokens > 0


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


def test_devin_validate_no_key():
    provider = DevinProvider(api_key="")
    with pytest.raises(ValueError, match="DEVIN_API_KEY"):
        provider.validate()


def test_devin_capabilities_no_embeddings():
    provider = DevinProvider(api_key="test-key")
    caps = provider.capabilities()
    assert caps.embeddings is False
    assert caps.streaming is False


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


# ---------------------------------------------------------------------------
# Effort degradation
# ---------------------------------------------------------------------------

def test_effort_levels_empty_degrades_gracefully():
    """Providers with empty effort_levels still accept any effort value."""
    provider = DevinProvider(api_key="test-key")
    caps = provider.capabilities()
    assert caps.effort_levels == []
