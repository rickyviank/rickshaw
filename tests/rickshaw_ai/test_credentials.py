"""CredentialStore contract: serialized modify, single OAuth refresh, file store."""

from __future__ import annotations

import asyncio
import time

import httpx
import pytest
import respx

from rickshaw_ai import (
    ApiKeyCredential,
    FileCredentialStore,
    InMemoryCredentialStore,
    OAuthCredential,
)
from rickshaw_ai.auth.resolver import resolve_auth
from rickshaw_ai.registry import OAuthConfig, ProviderInfo

TOKEN_URL = "https://oauth.example/token"


def _provider() -> ProviderInfo:
    return ProviderInfo(
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


@respx.mock
async def test_concurrent_requests_trigger_single_refresh():
    """Two concurrent resolves against an expired token → exactly one refresh."""
    past = int((time.time() - 10) * 1000)
    store = InMemoryCredentialStore(
        {"acme": OAuthCredential(access="stale", refresh="r", expires=past)}
    )

    calls = {"n": 0}

    def _responder(request):
        calls["n"] += 1
        return httpx.Response(
            200,
            json={"access_token": f"fresh-{calls['n']}", "refresh_token": "r2", "expires_in": 3600},
        )

    respx.post(TOKEN_URL).mock(side_effect=_responder)

    async with httpx.AsyncClient() as http:
        results = await asyncio.gather(
            resolve_auth(_provider(), store, http),
            resolve_auth(_provider(), store, http),
        )

    assert calls["n"] == 1  # only one refresh despite two concurrent callers
    tokens = {r.headers["Authorization"] for r in results}
    assert tokens == {"Bearer fresh-1"}


async def test_modify_is_serialized_read_modify_write():
    store = InMemoryCredentialStore()

    async def _bump(current):
        n = 0 if current is None else int(current.key)
        await asyncio.sleep(0)  # yield to encourage interleaving
        return ApiKeyCredential(key=str(n + 1))

    await asyncio.gather(*[store.modify("p", _bump) for _ in range(20)])
    final = await store.read("p")
    assert final.key == "20"  # no lost updates


async def test_delete_removes_credential():
    store = InMemoryCredentialStore({"p": ApiKeyCredential(key="k")})
    await store.delete("p")
    assert await store.read("p") is None


async def test_file_store_round_trip(tmp_path):
    path = tmp_path / "auth.json"
    store = FileCredentialStore(path)

    def _set(_):
        return ApiKeyCredential(key="secret", env={"X": "y"})

    await store.modify("acme", _set)

    # A fresh store reading the same file sees the persisted credential.
    reopened = FileCredentialStore(path)
    cred = await reopened.read("acme")
    assert isinstance(cred, ApiKeyCredential)
    assert cred.key == "secret"
    assert cred.env == {"X": "y"}


async def test_file_store_persists_oauth_type(tmp_path):
    path = tmp_path / "auth.json"
    store = FileCredentialStore(path)

    def _set(_):
        return OAuthCredential(access="a", refresh="r", expires=123)

    await store.modify("acme", _set)
    cred = await FileCredentialStore(path).read("acme")
    assert isinstance(cred, OAuthCredential)
    assert cred.access == "a"


async def test_file_store_corrupt_json_logs_warning(tmp_path, caplog):
    """A corrupt credential file should log a warning, not crash."""
    path = tmp_path / "auth.json"
    path.write_text("{{{invalid json")
    store = FileCredentialStore(path)
    import logging

    with caplog.at_level(logging.WARNING, logger="rickshaw_ai.credentials.store"):
        cred = await store.read("acme")
    assert cred is None
    assert "corrupt or unreadable" in caplog.text


async def test_file_store_invalid_credential_logs_warning(tmp_path, caplog):
    """An unrecognized credential shape should log a warning per provider."""
    path = tmp_path / "auth.json"
    import json

    path.write_text(json.dumps({"bad_provider": {"not": "a credential"}}))
    store = FileCredentialStore(path)
    import logging

    with caplog.at_level(logging.WARNING, logger="rickshaw_ai.credentials.store"):
        cred = await store.read("bad_provider")
    assert cred is None
    assert "Skipping invalid credential" in caplog.text
