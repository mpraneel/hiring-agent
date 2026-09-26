"""Runtime configuration read from the environment.

Values are resolved on each call rather than cached at import time so that
tests can patch the environment with ``monkeypatch.setenv``.
"""

import os
from typing import Literal, Optional

from dotenv import load_dotenv

load_dotenv()

Provider = Literal["openai", "gemini"]

SUPPORTED_PROVIDERS: tuple[Provider, ...] = ("openai", "gemini")

DEFAULT_MODELS: dict[Provider, str] = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}


def get_provider() -> Provider:
    """Return the configured provider, falling back to openai when unrecognized."""
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        return "openai"
    return provider  # type: ignore[return-value]


def get_api_key() -> Optional[str]:
    """Return the configured API key, or None when it is unset or blank."""
    key = os.getenv("LLM_API_KEY", "").strip()
    return key or None


def get_model(provider: Optional[Provider] = None) -> str:
    """Return the model name for a provider, honouring an LLM_MODEL override."""
    override = os.getenv("LLM_MODEL", "").strip()
    if override:
        return override
    return DEFAULT_MODELS[provider or get_provider()]


def llm_enabled() -> bool:
    """Whether the LLM layers should be attempted at all."""
    return get_api_key() is not None
