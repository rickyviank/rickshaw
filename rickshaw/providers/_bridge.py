"""Sync ⇄ async bridge between the harness and :mod:`rickshaw_ai`.

The harness provider interface is synchronous (it runs inside a Textual worker
thread), while ``rickshaw_ai`` is async. These helpers run a ``rickshaw_ai``
request to completion synchronously and translate between the harness's
dataclasses (:mod:`rickshaw.providers.base`) and ``rickshaw_ai``'s canonical
types.
"""

from __future__ import annotations

import asyncio
import os
import queue
import threading
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator

import httpx

from rickshaw.providers.base import (
    Effort,
    Message,
    Response,
    TokenUsage,
)
from rickshaw_ai._builtins import default_providers
from rickshaw_ai import (
    GenerateRequest,
    GenerateResult,
    Reasoning,
)
from rickshaw_ai import Message as AIMessage
from rickshaw_ai import TextBlock, Tool
from rickshaw_ai.credentials import ApiKeyCredential
from rickshaw_ai.credentials.store import CredentialStore, FileCredentialStore
from rickshaw_ai.credentials.types import Credential
from rickshaw_ai.providers import ProviderRuntime, adapter_for
from rickshaw_ai.registry import ModelInfo, ProviderInfo, RetryPolicy
from rickshaw_ai.streaming import StreamDone, TextDelta


# ---------------------------------------------------------------------------
# running coroutines / async generators from sync code
# ---------------------------------------------------------------------------


def run_sync(coro) -> Any:
    """Run *coro* to completion, tolerating an already-running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # A loop is already running on this thread — offload to a worker thread.
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.run(coro)).result()


def _iter_async(agen_factory: Callable[[], AsyncIterator]) -> Iterator:
    """Drive an async generator from sync code, yielding items as they arrive."""
    q: "queue.Queue[tuple[str, Any]]" = queue.Queue()

    def _worker() -> None:
        async def _run() -> None:
            try:
                async for item in agen_factory():
                    q.put(("item", item))
            except Exception as exc:  # noqa: BLE001 - re-raised on the main thread
                q.put(("error", exc))
            finally:
                q.put(("done", None))

        asyncio.run(_run())

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    while True:
        kind, value = q.get()
        if kind == "item":
            yield value
        elif kind == "error":
            raise value
        else:
            break
    thread.join()


# ---------------------------------------------------------------------------
# harness ⇄ rickshaw_ai conversions
# ---------------------------------------------------------------------------


def to_ai_messages(messages: list[Message]) -> list[AIMessage]:
    return [AIMessage(role=m.role, content=[TextBlock(text=m.content)]) for m in messages]


def to_ai_tools(tools) -> list[Tool]:
    return [
        Tool(
            name=t.name,
            description=t.description,
            parameters=t.parameters,
            category=getattr(t, "category", "general"),
            side_effect=getattr(t, "side_effect", True),
        )
        for t in (tools or [])
    ]


def to_response(
    result: GenerateResult,
    *,
    effort: Effort,
    tool_calls,
    fallback_model: str,
) -> Response:
    usage = TokenUsage(
        prompt_tokens=result.usage.input_tokens,
        completion_tokens=result.usage.output_tokens,
        total_tokens=result.usage.input_tokens + result.usage.output_tokens,
    )
    return Response(
        text=result.text,
        model=result.metadata.get("response_model", fallback_model),
        usage=usage,
        effort=effort,
        raw=result.metadata.get("raw", {}),
        tool_calls=tool_calls,
    )


# ---------------------------------------------------------------------------
# request execution
# ---------------------------------------------------------------------------


def credential_store_path() -> Path:
    return Path(
        os.environ.get("RICKSHAW_CREDENTIALS_PATH", "~/.rickshaw/credentials.json")
    ).expanduser()


def _builtin_info(provider_id: str):
    for provider in default_providers():
        if provider.id == provider_id:
            return provider
    return None


def _enrich(provider: ProviderInfo) -> ProviderInfo:
    builtin = _builtin_info(provider.id)
    if builtin is None:
        return provider
    return provider.model_copy(
        update={
            "oauth": builtin.oauth,
            "env_keys": builtin.env_keys or provider.env_keys,
            "auth_methods": builtin.auth_methods,
        }
    )


class _ResolveStore(CredentialStore):
    def __init__(
        self,
        file_store: FileCredentialStore,
        seed: dict[str, Credential] | None = None,
    ) -> None:
        self._file = file_store
        self._seed = seed

    async def read(self, provider_id: str) -> Credential | None:
        cred = await self._file.read(provider_id)
        if cred is not None:
            return cred
        if self._seed is not None:
            return self._seed.get(provider_id)
        return None

    async def modify(self, provider_id: str, fn):
        return await self._file.modify(provider_id, fn)

    async def delete(self, provider_id: str) -> None:
        await self._file.delete(provider_id)


def _runtime(provider: ProviderInfo, api_key: str, http: httpx.AsyncClient) -> ProviderRuntime:
    provider = _enrich(provider)
    file_store = FileCredentialStore(credential_store_path())
    seed = {provider.id: ApiKeyCredential(key=api_key)} if api_key else None
    store = _ResolveStore(file_store, seed)
    return ProviderRuntime(
        provider,
        adapter_for(provider.protocol),
        store=store,
        http=http,
        retry=RetryPolicy(max_retries=0),
    )


def has_stored_credential(provider_id: str) -> bool:
    try:
        return (
            run_sync(FileCredentialStore(credential_store_path()).read(provider_id))
            is not None
        )
    except Exception:
        return False


def generate(
    provider: ProviderInfo, model: ModelInfo, api_key: str, req: GenerateRequest
) -> GenerateResult:
    async def _run() -> GenerateResult:
        async with httpx.AsyncClient(timeout=120) as http:
            return await _runtime(provider, api_key, http).generate(req, model)

    return run_sync(_run())


def stream_text(
    provider: ProviderInfo, model: ModelInfo, api_key: str, req: GenerateRequest
) -> Iterator[str]:
    def _factory() -> AsyncIterator:
        async def _agen():
            async with httpx.AsyncClient(timeout=120) as http:
                async for event in _runtime(provider, api_key, http).stream(req, model):
                    yield event

        return _agen()

    for event in _iter_async(_factory):
        if isinstance(event, TextDelta) and event.text:
            yield event.text
        elif isinstance(event, StreamDone):
            break


__all__ = [
    "run_sync",
    "to_ai_messages",
    "to_ai_tools",
    "to_response",
    "credential_store_path",
    "has_stored_credential",
    "generate",
    "stream_text",
    "GenerateRequest",
    "Reasoning",
]
