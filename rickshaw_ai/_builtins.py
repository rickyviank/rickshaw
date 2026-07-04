"""The shipped provider collection — tool-calling models only.

Every :class:`ModelInfo` here has ``supports_tools=True``. Pricing is a
best-effort default (USD per 1M tokens) and can be overridden by the caller.
OAuth client ids/endpoints for the OAuth-first providers are the publicly known
values used by community tooling; they may drift and can be overridden.
"""

from __future__ import annotations

from rickshaw_ai.generate import Pricing
from rickshaw_ai.registry import ModelInfo, OAuthConfig, ProviderInfo


def _m(
    provider: str,
    model: str,
    *,
    ctx: int,
    max_out: int = 8192,
    vision: bool = False,
    image_out: bool = False,
    reasoning: bool = False,
    pin: float = 0.0,
    pout: float = 0.0,
    cache_read: float | None = None,
) -> ModelInfo:
    return ModelInfo(
        id=f"{provider}/{model}",
        provider_id=provider,
        model=model,
        context_window=ctx,
        max_output_tokens=max_out,
        supports_tools=True,
        supports_vision_input=vision,
        supports_image_output=image_out,
        supports_reasoning=reasoning,
        pricing=Pricing(input=pin, output=pout, cache_read=cache_read),
        modalities=["text", "image"] if vision else ["text"],
    )


# --- OAuth-first providers -------------------------------------------------

ANTHROPIC = ProviderInfo(
    id="anthropic",
    base_url="https://api.anthropic.com",
    protocol="anthropic",
    auth_methods=["oauth", "api_key"],
    env_keys=["ANTHROPIC_API_KEY"],
    api_key_header="x-api-key",
    api_key_prefix="",
    oauth=OAuthConfig(
        authorize_url="https://claude.ai/oauth/authorize",
        token_url="https://console.anthropic.com/v1/oauth/token",
        client_id="9d1c250a-e61b-44d9-88ed-5944d1962f5e",
        scopes=["user:profile", "user:inference"],
        redirect_uri="https://console.anthropic.com/oauth/code/callback",
        use_pkce=True,
    ),
    models=[
        _m("anthropic", "claude-opus-4-20250514", ctx=200_000, max_out=32000,
           vision=True, reasoning=True, pin=15, pout=75, cache_read=1.5),
        _m("anthropic", "claude-sonnet-4-20250514", ctx=200_000, max_out=64000,
           vision=True, reasoning=True, pin=3, pout=15, cache_read=0.3),
        _m("anthropic", "claude-3-5-sonnet-latest", ctx=200_000, max_out=8192,
           vision=True, pin=3, pout=15, cache_read=0.3),
        _m("anthropic", "claude-3-5-haiku-latest", ctx=200_000, max_out=8192,
           vision=True, pin=0.8, pout=4, cache_read=0.08),
    ],
)

OPENAI = ProviderInfo(
    id="openai",
    base_url="https://api.openai.com/v1",
    protocol="openai",
    auth_methods=["oauth", "api_key"],
    env_keys=["OPENAI_API_KEY"],
    oauth=OAuthConfig(
        authorize_url="https://auth.openai.com/oauth/authorize",
        token_url="https://auth.openai.com/oauth/token",
        client_id="app_EMoamEEZ73f0CkXaXp7hrann",
        scopes=["openid", "profile", "email", "offline_access"],
        redirect_uri="http://localhost:1455/auth/callback",
        use_pkce=True,
    ),
    models=[
        _m("openai", "gpt-4o", ctx=128_000, max_out=16384, vision=True,
           pin=2.5, pout=10, cache_read=1.25),
        _m("openai", "gpt-4o-mini", ctx=128_000, max_out=16384, vision=True,
           pin=0.15, pout=0.6, cache_read=0.075),
        _m("openai", "gpt-4.1", ctx=1_047_576, max_out=32768, vision=True,
           pin=2, pout=8, cache_read=0.5),
        _m("openai", "o3", ctx=200_000, max_out=100_000, vision=True,
           reasoning=True, pin=2, pout=8, cache_read=0.5),
        _m("openai", "o4-mini", ctx=200_000, max_out=100_000, vision=True,
           reasoning=True, pin=1.1, pout=4.4, cache_read=0.275),
    ],
)

# GitHub Copilot chat is OpenAI-compatible; auth is a GitHub OAuth device flow.
COPILOT = ProviderInfo(
    id="copilot",
    base_url="https://api.githubcopilot.com",
    protocol="openai_compatible",
    auth_methods=["oauth"],
    env_keys=["COPILOT_API_KEY"],
    oauth=OAuthConfig(
        authorize_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        device_code_url="https://github.com/login/device/code",
        client_id="Iv1.b507a08c87ecfe98",
        scopes=["read:user"],
        mode="device_code",
        use_pkce=False,
    ),
    models=[
        _m("copilot", "gpt-4o", ctx=128_000, vision=True, pin=0, pout=0),
        _m("copilot", "claude-3.5-sonnet", ctx=200_000, pin=0, pout=0),
    ],
)

# --- OpenAI-compatible fleet ----------------------------------------------

GROQ = ProviderInfo(
    id="groq",
    base_url="https://api.groq.com/openai/v1",
    protocol="openai_compatible",
    env_keys=["GROQ_API_KEY"],
    models=[
        _m("groq", "llama-3.3-70b-versatile", ctx=128_000, pin=0.59, pout=0.79),
        _m("groq", "moonshotai/kimi-k2-instruct", ctx=131_072, pin=1, pout=3),
    ],
)

XAI = ProviderInfo(
    id="xai",
    base_url="https://api.x.ai/v1",
    protocol="openai_compatible",
    env_keys=["XAI_API_KEY"],
    models=[
        _m("xai", "grok-3", ctx=131_072, pin=3, pout=15),
        _m("xai", "grok-3-mini", ctx=131_072, reasoning=True, pin=0.3, pout=0.5),
    ],
)

MISTRAL = ProviderInfo(
    id="mistral",
    base_url="https://api.mistral.ai/v1",
    protocol="openai_compatible",
    env_keys=["MISTRAL_API_KEY"],
    models=[
        _m("mistral", "mistral-large-latest", ctx=131_072, pin=2, pout=6),
        _m("mistral", "mistral-small-latest", ctx=131_072, pin=0.2, pout=0.6),
    ],
)

DEEPSEEK = ProviderInfo(
    id="deepseek",
    base_url="https://api.deepseek.com/v1",
    protocol="openai_compatible",
    env_keys=["DEEPSEEK_API_KEY"],
    models=[
        _m("deepseek", "deepseek-chat", ctx=64_000, pin=0.27, pout=1.1, cache_read=0.07),
        _m("deepseek", "deepseek-reasoner", ctx=64_000, reasoning=True,
           pin=0.55, pout=2.19, cache_read=0.14),
    ],
)

TOGETHER = ProviderInfo(
    id="together",
    base_url="https://api.together.xyz/v1",
    protocol="openai_compatible",
    env_keys=["TOGETHER_API_KEY"],
    models=[
        _m("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo", ctx=131_072,
           pin=0.88, pout=0.88),
    ],
)

FIREWORKS = ProviderInfo(
    id="fireworks",
    base_url="https://api.fireworks.ai/inference/v1",
    protocol="openai_compatible",
    env_keys=["FIREWORKS_API_KEY"],
    models=[
        _m("fireworks", "accounts/fireworks/models/llama-v3p3-70b-instruct",
           ctx=131_072, pin=0.9, pout=0.9),
    ],
)

# --- Native Google ---------------------------------------------------------

GOOGLE = ProviderInfo(
    id="google",
    base_url="https://generativelanguage.googleapis.com",
    protocol="google",
    env_keys=["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    api_key_header="x-goog-api-key",
    api_key_prefix="",
    models=[
        _m("google", "gemini-2.5-pro", ctx=1_048_576, max_out=65536, vision=True,
           reasoning=True, pin=1.25, pout=10),
        _m("google", "gemini-2.5-flash", ctx=1_048_576, max_out=65536, vision=True,
           reasoning=True, pin=0.3, pout=2.5),
        _m("google", "gemini-2.0-flash", ctx=1_048_576, max_out=8192, vision=True,
           pin=0.1, pout=0.4),
    ],
)

# --- Gateways --------------------------------------------------------------

OPENROUTER = ProviderInfo(
    id="openrouter",
    base_url="https://openrouter.ai/api/v1",
    protocol="openai_compatible",
    env_keys=["OPENROUTER_API_KEY"],
    models=[
        _m("openrouter", "anthropic/claude-sonnet-4", ctx=200_000, vision=True,
           reasoning=True, pin=3, pout=15),
        _m("openrouter", "openai/gpt-4o", ctx=128_000, vision=True, pin=2.5, pout=10),
    ],
)

# Cloudflare AI Gateway: base_url carries {CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}
# placeholders, substituted at request time from the credential's provider-scoped
# ``env`` (see ProviderRuntime).
CLOUDFLARE = ProviderInfo(
    id="cloudflare",
    base_url=(
        "https://gateway.ai.cloudflare.com/v1/"
        "{CLOUDFLARE_ACCOUNT_ID}/{CLOUDFLARE_GATEWAY_ID}/compat"
    ),
    protocol="openai_compatible",
    env_keys=["CLOUDFLARE_API_KEY"],
    models=[
        _m("cloudflare", "openai/gpt-4o", ctx=128_000, vision=True, pin=2.5, pout=10),
    ],
)


def default_providers() -> list[ProviderInfo]:
    """Return the built-in, tool-calling-only provider collection."""
    return [
        ANTHROPIC,
        OPENAI,
        COPILOT,
        GROQ,
        XAI,
        MISTRAL,
        DEEPSEEK,
        TOGETHER,
        FIREWORKS,
        GOOGLE,
        OPENROUTER,
        CLOUDFLARE,
    ]
