"""Provider adapters for structured output.

Each provider is asked for JSON conforming to a Pydantic schema using its own
native mechanism, and the raw text is then validated with Pydantic regardless of
what the provider claims. Providers do occasionally return JSON that does not
satisfy the schema they were given, so trusting the claim is not an option.
"""

from __future__ import annotations

import json
import logging
from typing import Optional, Protocol, Type, TypeVar

from .. import config

logger = logging.getLogger(__name__)

ModelT = TypeVar("ModelT")


class StructuredProviderError(RuntimeError):
    """Raised when the provider call itself fails, as opposed to returning bad JSON."""


class StructuredProvider(Protocol):
    """Returns the raw JSON text a provider produced for a schema."""

    def generate_json(self, system: str, user: str, schema: Type[ModelT]) -> str:
        ...


class OpenAIStructuredProvider:
    """Structured output via the OpenAI chat completions parse helper."""

    def __init__(self, api_key: str, model: str) -> None:
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise StructuredProviderError("openai package not installed") from exc
        self._client = openai.OpenAI(api_key=api_key)
        self._model = model

    def generate_json(self, system: str, user: str, schema: Type[ModelT]) -> str:
        try:
            completion = self._client.chat.completions.parse(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format=schema,
                temperature=0,
            )
        except Exception as exc:
            raise StructuredProviderError(f"openai request failed: {exc}") from exc

        message = completion.choices[0].message
        if getattr(message, "refusal", None):
            raise StructuredProviderError(f"openai refused the request: {message.refusal}")

        # Prefer the already parsed object when the SDK supplies one, but fall
        # back to the raw content so validation happens in exactly one place.
        parsed = getattr(message, "parsed", None)
        if parsed is not None:
            return parsed.model_dump_json() if hasattr(parsed, "model_dump_json") else json.dumps(parsed)
        if not message.content:
            raise StructuredProviderError("openai returned an empty response")
        return message.content


class GeminiStructuredProvider:
    """Structured output via the google-genai response_schema config."""

    def __init__(self, api_key: str, model: str) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise StructuredProviderError("google-genai package not installed") from exc
        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def generate_json(self, system: str, user: str, schema: Type[ModelT]) -> str:
        from google.genai import types as genai_types

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=user,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0,
                ),
            )
        except Exception as exc:
            raise StructuredProviderError(f"gemini request failed: {exc}") from exc

        text = getattr(response, "text", None)
        if not text:
            raise StructuredProviderError("gemini returned an empty response")
        return text


def build_provider(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
) -> StructuredProvider:
    """Construct the configured structured-output provider."""
    name = (provider or config.get_provider()).lower()
    key = api_key or config.get_api_key()
    if not key:
        raise StructuredProviderError("LLM_API_KEY is not set")
    resolved_model = model or config.get_model(name)  # type: ignore[arg-type]

    if name == "openai":
        return OpenAIStructuredProvider(api_key=key, model=resolved_model)
    if name == "gemini":
        return GeminiStructuredProvider(api_key=key, model=resolved_model)
    raise StructuredProviderError(f"unsupported provider: {name}")
