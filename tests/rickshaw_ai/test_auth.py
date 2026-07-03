"""Auth resolution order and OAuth refresh-inside-modify semantics."""

from __future__ import annotations

import time

import httpx
import pytest
import respx

from rickshaw_ai import (
    ApiKeyCredential,
    AuthError,
    InMemoryCredentialStore,
    OAuthCredential,
)
from rickshaw_ai.auth.resolver import resolve_auth
from rickshaw_ai.registry import OAuthConfig, ProviderInfo

TOKEN_URL = "https://oauth.example/token"


def _provider(**over) -> ProviderInfo:
    base = dict(
        id="acme",
        base_url="https://api.acme.test",
        protocol="openai_compatible",
        env_keys=["ACME_API_KEY"],
        oauth=OAuthConfig(
            authorize_url="https://oauth.example/authorize",
            token_url=TOKEN_URL,
            client_id="cid",
        ),
    )
    base.update(over)
    return ProviderInfo(**base)


async def test_stored_api_key_beats_env(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "env-key")
    store = InMemoryCredentialStore({"acme": ApiKeyCredential(key="stored-key")})
    async with httpx.AsyncClient() as http:
        auth = await resolve_auth(_provider(), store, http)
    assert auth.headers["Authorization"] == "Bearer stored-key"


async def test_env_fallback_when_nothing_stored(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "env-key")
    store = InMemoryCredentialStore()
    async with httpx.AsyncClient() as http:
        auth = await resolve_auth(_provider(), store, http)
    assert auth.headers["Authorization"] == "Bearer env-key"


async def test_no_credential_raises(monkeypatch):
    monkeypatch.delenv("ACME_API_KEY", raising=False)
    store = InMemoryCredentialStore()
    async with httpx.AsyncClient() as http:
        with pytest.raises(AuthError, match="no credentials"):
            await resolve_auth(_provider(), store, http)


async def test_empty_stored_api_key_raises(monkeypatch):
    """A stored ApiKeyCredential with an empty key must raise AuthError."""
    monkeypatch.delenv("ACME_API_KEY", raising=False)
    store = InMemoryCredentialStore({"acme": ApiKeyCredential(key="")})
    async with httpx.AsyncClient() as http:
        with pytest.raises(AuthError, match="stored API key.*is empty"):
            await resolve_auth(_provider(), store, http)


async def test_api_key_injects_provider_env():
    store = InMemoryCredentialStore(
        {"acme": ApiKeyCredential(key="k", env={"CLOUDFLARE_ACCOUNT_ID": "acct"})}
    )
    async with httpx.AsyncClient() as http:
        auth = await resolve_auth(_provider(), store, http)
    assert auth.extra["CLOUDFLARE_ACCOUNT_ID"] == "acct"


async def test_valid_oauth_used_without_refresh():
    future = int((time.time() + 3600) * 1000)
    store = InMemoryCredentialStore(
        {"acme": OAuthCredential(access="live", refresh="r", expires=future)}
    )
    async with httpx.AsyncClient() as http:
        auth = await resolve_auth(_provider(), store, http)
    assert auth.headers["Authorization"] == "Bearer live"


@respx.mock
async def test_expired_oauth_refreshes_and_persists(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "env-key")  # must NOT be used
    past = int((time.time() - 10) * 1000)
    store = InMemoryCredentialStore(
        {"acme": OAuthCredential(access="stale", refresh="refresh-1", expires=past)}
    )
    route = respx.post(TOKEN_URL).mock(
        return_value=httpx.Response(
            200,
            json={"access_token": "fresh", "refresh_token": "refresh-2", "expires_in": 3600},
        )
    )
    async with httpx.AsyncClient() as http:
        auth = await resolve_auth(_provider(), store, http)

    assert auth.headers["Authorization"] == "Bearer fresh"
    assert route.called
    stored = await store.read("acme")
    assert isinstance(stored, OAuthCredential)
    assert stored.access == "fresh"
    assert stored.refresh == "refresh-2"


@respx.mock
async def test_failed_refresh_raises_and_does_not_fall_back_to_env(monkeypatch):
    monkeypatch.setenv("ACME_API_KEY", "env-key")  # must NOT be used
    past = int((time.time() - 10) * 1000)
    store = InMemoryCredentialStore(
        {"acme": OAuthCredential(access="stale", refresh="bad", expires=past)}
    )
    respx.post(TOKEN_URL).mock(return_value=httpx.Response(400, json={"error": "invalid_grant"}))
    async with httpx.AsyncClient() as http:
        with pytest.raises(AuthError):
            await resolve_auth(_provider(), store, http)
