"""
GroqLLM — wraps the Groq Chat Completions API.

The `groq` SDK mirrors the OpenAI Chat Completions client shape (same
`.chat.completions.create()` surface and exception hierarchy), so this adapter
follows the same prompt-splitting and retry conventions as OpenAILLM. Groq's
value proposition is inference speed (LPU-backed), not a different API shape.
Retries transient failures (rate limits, server errors) with exponential backoff.

Requires: pip install "nexrag[groq]"  (groq)
"""

from __future__ import annotations

import random
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from nexrag.core.interfaces.llm import BaseLLM
from nexrag.core.models.metrics import TokenUsage
from nexrag.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError

_BASE_BACKOFF = 1.0  # seconds; doubles each retry


class GroqLLM(BaseLLM):
    """
    LLM adapter for the Groq Chat Completions API.

    Args:
        model:       Model ID. e.g. "llama-3.3-70b-versatile", "llama-3.1-8b-instant".
        api_key:     Groq API key. If None, reads GROQ_API_KEY from env.
        base_url:    Optional custom endpoint (proxy).
        temperature: Sampling temperature. Default 0.2.
        max_tokens:  Max tokens in the response. Default 1024.
        timeout:     Request timeout in seconds. Default 30.
        max_retries: Retry attempts after the first failure. 0 = no retries. Default 2.
    """

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
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
        self._client: Any = self._build_client(api_key, base_url)
        self._async_client: Any = self._build_async_client(api_key, base_url)

    def generate(self, prompt: str) -> tuple[str, TokenUsage | None]:
        """
        Send the prompt and return (response_text, token_usage).

        The prompt is split at the first "---" separator line into system and
        user messages. If no separator is found, the whole prompt is the user message.

        Returns:
            (response_text, TokenUsage) — token_usage is None if the API response
            does not include usage data.

        Raises:
            LLMRateLimitError: On 429 rate limit (after retries exhausted).
            LLMTimeoutError:   On timeout.
            LLMError:          On any other failure.
        """
        messages = self._build_messages(prompt)
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    timeout=self._timeout,
                )
                text = response.choices[0].message.content or ""
                usage = None
                if response.usage:
                    usage = TokenUsage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                return text, usage
            except Exception as e:
                if self._is_retryable(e) and attempt < self._max_retries:
                    time.sleep(_BASE_BACKOFF * (2**attempt) + random.uniform(0, 1))
                    continue
                self._map_exception(e)
                raise  # unreachable — _map_exception always raises

        raise LLMError(
            "Retry loop exhausted without returning.", stage="llm", component="GroqLLM"
        )  # unreachable

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream the response token by token.

        Note: streaming calls are not retried — a partial stream cannot be resumed.

        Yields:
            Response text chunks as they arrive from the API.
        """
        messages = self._build_messages(prompt)
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            self._map_exception(e)

    async def async_generate(self, prompt: str) -> tuple[str, TokenUsage | None]:
        """Generate using AsyncGroq — native async, returns (text, token_usage)."""
        messages = self._build_messages(prompt)
        for attempt in range(self._max_retries + 1):
            try:
                response = await self._async_client.chat.completions.create(
                    model=self._model,
                    messages=messages,
                    temperature=self._temperature,
                    max_tokens=self._max_tokens,
                    timeout=self._timeout,
                )
                text = response.choices[0].message.content or ""
                usage = None
                if response.usage:
                    usage = TokenUsage(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                    )
                return text, usage
            except Exception as e:
                if self._is_retryable(e) and attempt < self._max_retries:
                    import asyncio

                    await asyncio.sleep(_BASE_BACKOFF * (2**attempt) + random.uniform(0, 1))
                    continue
                self._map_exception(e)
                raise
        raise LLMError("Retry loop exhausted.", stage="llm", component="GroqLLM")

    async def async_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream using AsyncGroq — tokens arrive live without buffering."""
        messages = self._build_messages(prompt)
        try:
            stream = await self._async_client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
                stream=True,
            )
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            self._map_exception(e)

    # Private helpers

    def _build_client(self, api_key: str | None, base_url: str | None) -> Any:
        try:
            import groq
        except ImportError as e:
            raise LLMError(
                "groq package is required for GroqLLM. "
                'Install it: pip install "nexrag[groq]" or pip install groq',
                stage="llm",
                component="GroqLLM",
                cause=e,
            ) from e

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return groq.Groq(**kwargs)

    def _build_async_client(self, api_key: str | None, base_url: str | None) -> Any:
        try:
            import groq
        except ImportError as e:
            raise LLMError(
                "groq package is required for GroqLLM. "
                'Install it: pip install "nexrag[groq]" or pip install groq',
                stage="llm",
                component="GroqLLM",
                cause=e,
            ) from e
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return groq.AsyncGroq(**kwargs)

    @staticmethod
    def _build_messages(prompt: str) -> list[dict[str, str]]:
        """Split prompt into system + user messages on the --- separator."""
        sep = "\n\n---\n\n"
        if sep in prompt:
            system_part, user_part = prompt.split(sep, 1)
            return [
                {"role": "system", "content": system_part.strip()},
                {"role": "user", "content": user_part.strip()},
            ]
        return [{"role": "user", "content": prompt.strip()}]

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        try:
            import groq as _groq

            return isinstance(
                exc,
                _groq.RateLimitError | _groq.InternalServerError | _groq.APIConnectionError,
            )
        except ImportError:
            return False

    def _map_exception(self, exc: Exception) -> None:
        try:
            import groq as _groq

            if isinstance(exc, _groq.RateLimitError):
                raise LLMRateLimitError(
                    f"Groq rate limit exceeded: {exc}",
                    stage="llm",
                    component="GroqLLM",
                    cause=exc,
                ) from exc
            if isinstance(exc, _groq.APITimeoutError):
                raise LLMTimeoutError(
                    f"Groq request timed out after {self._timeout}s.",
                    stage="llm",
                    component="GroqLLM",
                    cause=exc,
                ) from exc
            if isinstance(exc, _groq.AuthenticationError):
                raise LLMError(
                    "Groq authentication failed. Check your GROQ_API_KEY.",
                    stage="llm",
                    component="GroqLLM",
                    cause=exc,
                ) from exc
        except ImportError:
            pass

        raise LLMError(
            f"Groq API call failed: {exc}",
            stage="llm",
            component="GroqLLM",
            cause=exc,
        ) from exc
