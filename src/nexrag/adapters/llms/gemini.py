"""
GeminiLLM — wraps the Google Gemini API via the unified google-genai SDK.

Uses the current `from google import genai` SDK (not the deprecated
google-generativeai). The prompt string from DefaultPromptBuilder is split into a
system_instruction and user contents on the same "---" separator the other adapters
use. Retries transient failures (rate limits, server errors) with exponential backoff.

Requires: pip install "nexrag[gemini]"  (google-genai)
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from nexrag.core.interfaces.llm import BaseLLM
from nexrag.core.models.metrics import TokenUsage
from nexrag.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError

_BASE_BACKOFF = 1.0  # seconds; doubles each retry


class GeminiLLM(BaseLLM):
    """
    LLM adapter for Google Gemini (google-genai SDK).

    Args:
        model:       Model ID. e.g. "gemini-2.5-flash", "gemini-2.5-pro".
        api_key:     Gemini API key. If None, the SDK reads GOOGLE_API_KEY or
                     GEMINI_API_KEY from the environment.
        base_url:    Optional custom endpoint (proxy / Vertex-compatible gateway).
        temperature: Sampling temperature. Default 0.2.
        max_tokens:  Max tokens in the response (max_output_tokens). Default 1024.
        timeout:     Request timeout in seconds. Default 30.
        max_retries: Retry attempts after the first failure. 0 = no retries. Default 2.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 30,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries
        # A single genai.Client exposes both sync (.models) and async (.aio.models) APIs.
        self._client: Any = self._build_client(api_key, base_url)

    def generate(self, prompt: str) -> tuple[str, TokenUsage | None]:
        """
        Send the prompt and return (response_text, token_usage).

        Returns:
            (response_text, TokenUsage) — token_usage is None if the response
            does not include usage_metadata.

        Raises:
            LLMRateLimitError: On 429 rate limit (after retries exhausted).
            LLMTimeoutError:   On timeout.
            LLMError:          On any other failure.
        """
        system, contents = self._split_prompt(prompt)
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=self._build_config(system),
                )
                return response.text or "", self._extract_usage(response)
            except Exception as e:
                if self._is_retryable(e) and attempt < self._max_retries:
                    time.sleep(_BASE_BACKOFF * (2**attempt) + random.uniform(0, 1))
                    continue
                self._map_exception(e)
                raise  # unreachable — _map_exception always raises

        raise LLMError(
            "Retry loop exhausted without returning.", stage="llm", component="GeminiLLM"
        )  # unreachable

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream the response token by token.

        Note: streaming calls are not retried — a partial stream cannot be resumed.

        Yields:
            Response text chunks as they arrive from the API.
        """
        system, contents = self._split_prompt(prompt)
        try:
            stream = self._client.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=self._build_config(system),
            )
            for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            self._map_exception(e)

    async def async_generate(self, prompt: str) -> tuple[str, TokenUsage | None]:
        """Generate using the native async client (client.aio) — returns (text, token_usage)."""
        system, contents = self._split_prompt(prompt)
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.aio.models.generate_content(
                    model=self._model,
                    contents=contents,
                    config=self._build_config(system),
                )
                return response.text or "", self._extract_usage(response)
            except Exception as e:
                if self._is_retryable(e) and attempt < self._max_retries:
                    await asyncio.sleep(_BASE_BACKOFF * (2**attempt) + random.uniform(0, 1))
                    continue
                self._map_exception(e)
                raise
        raise LLMError("Retry loop exhausted.", stage="llm", component="GeminiLLM")

    async def async_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream using the native async client — tokens arrive live without buffering."""
        system, contents = self._split_prompt(prompt)
        try:
            stream = await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=self._build_config(system),
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            self._map_exception(e)

    # Private helpers

    def _build_client(self, api_key: str | None, base_url: str | None) -> Any:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise LLMError(
                "google-genai package is required for GeminiLLM. "
                'Install it: pip install "nexrag[gemini]" or pip install google-genai',
                stage="llm",
                component="GeminiLLM",
                cause=e,
            ) from e

        http_opts: dict[str, Any] = {}
        if base_url:
            http_opts["base_url"] = base_url
        if self._timeout:
            http_opts["timeout"] = self._timeout * 1000  # SDK expects milliseconds

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if http_opts:
            kwargs["http_options"] = types.HttpOptions(**http_opts)
        return genai.Client(**kwargs)

    def _build_config(self, system_instruction: str | None) -> Any:
        from google.genai import types

        return types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self._temperature,
            max_output_tokens=self._max_tokens,
        )

    @staticmethod
    def _split_prompt(prompt: str) -> tuple[str | None, str]:
        """Split prompt into (system_instruction, user_contents) on the --- separator."""
        sep = "\n\n---\n\n"
        if sep in prompt:
            system_part, user_part = prompt.split(sep, 1)
            return system_part.strip() or None, user_part.strip()
        return None, prompt.strip()

    @staticmethod
    def _extract_usage(response: Any) -> TokenUsage | None:
        um = getattr(response, "usage_metadata", None)
        if not um:
            return None
        prompt_tokens = getattr(um, "prompt_token_count", None) or 0
        completion_tokens = getattr(um, "candidates_token_count", None) or 0
        total_tokens = getattr(um, "total_token_count", None) or (prompt_tokens + completion_tokens)
        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        try:
            from google.genai import errors
        except ImportError:
            return False
        if isinstance(exc, errors.ServerError):
            return True
        if isinstance(exc, errors.APIError) and getattr(exc, "code", None) == 429:
            return True
        return False

    def _map_exception(self, exc: Exception) -> None:
        try:
            from google.genai import errors

            if isinstance(exc, errors.APIError):
                code = getattr(exc, "code", None)
                if code == 429:
                    raise LLMRateLimitError(
                        f"Gemini rate limit exceeded: {exc}",
                        stage="llm",
                        component="GeminiLLM",
                        cause=exc,
                    ) from exc
                if code in (401, 403):
                    raise LLMError(
                        "Gemini authentication failed. Check your GOOGLE_API_KEY / GEMINI_API_KEY.",
                        stage="llm",
                        component="GeminiLLM",
                        cause=exc,
                    ) from exc
        except ImportError:
            pass

        if self._looks_like_timeout(exc):
            raise LLMTimeoutError(
                f"Gemini request timed out after {self._timeout}s.",
                stage="llm",
                component="GeminiLLM",
                cause=exc,
            ) from exc

        raise LLMError(
            f"Gemini API call failed: {exc}",
            stage="llm",
            component="GeminiLLM",
            cause=exc,
        ) from exc

    @staticmethod
    def _looks_like_timeout(exc: Exception) -> bool:
        name = type(exc).__name__.lower()
        return "timeout" in name or "timed out" in str(exc).lower()
