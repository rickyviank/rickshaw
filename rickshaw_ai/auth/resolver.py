"""Per-request auth resolution.

Resolution order (strict):

1. A stored credential for the provider — it *owns* the provider.
   - ``api_key`` → use its key (+ inject provider-scoped env/config).
   - ``oauth``   → use ``access`` if valid; otherwise refresh INSIDE
     ``store.modify`` (atomic across requests/processes). A failed refresh
     raises :class:`AuthError` — it does NOT fall back to env.
2. Environment variables (``ProviderInfo.env_keys``, in order) — only when no
   credential is stored at all.
3. Otherwise raise :class:`AuthError` explaining how to authenticate.
"""

from __future__ import annotations

import os

import httpx

from rickshaw_ai.auth.oauth import OAuthClient
from rickshaw_ai.credentials.store import CredentialStore
from rickshaw_ai.credentials.types import (
    ApiKeyCredential,
    Credential,
    OAuthCredential,
)
from rickshaw_ai.errors import AuthError
from rickshaw_ai.generate import ResolvedAuth
from rickshaw_ai.registry import ProviderInfo


def _apply_api_key(provider: ProviderInfo, key: str) -> dict[str, str]:
    return {provider.api_key_header: f"{provider.api_key_prefix}{key}"}


async def resolve_auth(
    provider: ProviderInfo,
    store: CredentialStore,
    http: httpx.AsyncClient,
) -> ResolvedAuth:
    """Resolve concrete auth for a request to *provider*."""
    stored = await store.read(provider.id)

    if stored is not None:
        return await _resolve_stored(provider, store, http, stored)

    # No stored credential — fall back to environment variables.
    for env_key in provider.env_keys:
        val = os.environ.get(env_key)
        if val:
            return ResolvedAuth(headers=_apply_api_key(provider, val))

    how = (
        f"set one of {provider.env_keys}"
        if provider.env_keys
        else "store a credential"
    )
    if provider.oauth is not None:
        how += f" or run login for provider {provider.id!r}"
    raise AuthError(
        f"no credentials for provider {provider.id!r}; {how}",
        provider_id=provider.id,
    )


async def _resolve_stored(
    provider: ProviderInfo,
    store: CredentialStore,
    http: httpx.AsyncClient,
    stored: Credential,
) -> ResolvedAuth:
    if isinstance(stored, ApiKeyCredential):
        if not stored.key:
            raise AuthError(
                f"stored API key for provider {provider.id!r} is empty; "
                f"set {', '.join(provider.env_keys) or 'a valid key'} or "
                f"re-authenticate",
                provider_id=provider.id,
            )
        return ResolvedAuth(
            headers=_apply_api_key(provider, stored.key),
            extra=dict(stored.env),
        )

    assert isinstance(stored, OAuthCredential)
    leeway = provider.oauth.refresh_leeway_seconds if provider.oauth else 60
    if not stored.is_expired(leeway_seconds=leeway):
        return ResolvedAuth(
            headers={"Authorization": f"Bearer {stored.access}"},
            extra=dict(stored.env),
        )

    if provider.oauth is None:
        raise AuthError(
            f"stored OAuth token for {provider.id!r} is expired and the provider "
            f"has no OAuth config to refresh it",
            provider_id=provider.id,
        )

    client = OAuthClient(config=provider.oauth, http=http)

    async def _refresh(current: Credential | None) -> Credential:
        # Re-check inside the lock: another caller may have refreshed already.
        if isinstance(current, OAuthCredential) and not current.is_expired(
            leeway_seconds=leeway
        ):
            return current
        if not isinstance(current, OAuthCredential):
            raise AuthError(
                f"credential for {provider.id!r} changed type during refresh",
                provider_id=provider.id,
            )
        return await client.refresh(current)

    refreshed = await store.modify(provider.id, _refresh)
    if not isinstance(refreshed, OAuthCredential):  # pragma: no cover - defensive
        raise AuthError(f"refresh produced no token for {provider.id!r}")
    return ResolvedAuth(
        headers={"Authorization": f"Bearer {refreshed.access}"},
        extra=dict(refreshed.env),
    )
