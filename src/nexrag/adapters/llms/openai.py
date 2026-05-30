"""
OpenAILLM — wraps the OpenAI Chat Completions API.

Supports any OpenAI-compatible endpoint via base_url (Azure OpenAI, local proxies,
OpenAI-compatible servers). The prompt string from DefaultPromptBuilder is split
into system and user messages automatically.
Retries transient failures (rate limits, server errors) with exponential backoff.

Requires: pip install "nexrag[openai]"  (openai)
"""

from __future__ import annotations

import random
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

from nexrag.core.interfaces.llm import BaseLLM
from nexrag.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError

_BASE_BACKOFF = 1.0  # seconds; doubles each retry


class OpenAILLM(BaseLLM):
    """
    LLM adapter for OpenAI Chat Completions.

    Args:
        model:       Model ID. e.g. "gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo".
        api_key:     OpenAI API key. If None, reads OPENAI_API_KEY from env.
        base_url:    Optional custom endpoint (Azure OpenAI, local proxy).
        temperature: Sampling temperature. Default 0.2.
        max_tokens:  Max tokens in the response. Default 1024.
        timeout:     Request timeout in seconds. Default 30.
        max_retries: Retry attempts after the first failure. 0 = no retries. Default 2.
    """

    def __init__(
        self,
        model: str = "gpt-4o",
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

    def generate(self, prompt: str) -> str:
        """
        Send the prompt and return the complete response.

        The prompt is split at the first "---" separator line into system and
        user messages. If no separator is found, the whole prompt is the user message.

        Returns:
            The assistant response text.

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
                return response.choices[0].message.content or ""
            except Exception as e:
                if self._is_retryable(e) and attempt < self._max_retries:
                    time.sleep(_BASE_BACKOFF * (2**attempt) + random.uniform(0, 1))
                    continue
                self._map_exception(e)
                raise  # unreachable — _map_exception always raises

        raise LLMError(
            "Retry loop exhausted without returning.", stage="llm", component="OpenAILLM"
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
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            self._map_exception(e)

    async def async_generate(self, prompt: str) -> str:
        """Generate using AsyncOpenAI — native async, no thread pool needed."""
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
                return response.choices[0].message.content or ""
            except Exception as e:
                if self._is_retryable(e) and attempt < self._max_retries:
                    import asyncio

                    await asyncio.sleep(_BASE_BACKOFF * (2**attempt) + random.uniform(0, 1))
                    continue
                self._map_exception(e)
                raise
        raise LLMError("Retry loop exhausted.", stage="llm", component="OpenAILLM")

    async def async_stream(self, prompt: str) -> AsyncIterator[str]:
        """Stream using AsyncOpenAI — tokens arrive live without buffering."""
        messages = self._build_messages(prompt)
        try:
            async with await self._async_client.chat.completions.stream(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                timeout=self._timeout,
            ) as stream:
                async for chunk in stream:
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except Exception as e:
            self._map_exception(e)

    # Private helpers

    def _build_client(self, api_key: str | None, base_url: str | None) -> Any:
        try:
            import openai
        except ImportError as e:
            raise LLMError(
                "openai package is required for OpenAILLM. "
                'Install it: pip install "nexrag[openai]" or pip install openai',
                stage="llm",
                component="OpenAILLM",
                cause=e,
            ) from e

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return openai.OpenAI(**kwargs)

    def _build_async_client(self, api_key: str | None, base_url: str | None) -> Any:
        try:
            import openai
        except ImportError as e:
            raise LLMError(
                "openai package is required for OpenAILLM. "
                'Install it: pip install "nexrag[openai]" or pip install openai',
                stage="llm",
                component="OpenAILLM",
                cause=e,
            ) from e
        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return openai.AsyncOpenAI(**kwargs)

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
            import openai as _openai

            return isinstance(
                exc,
                _openai.RateLimitError | _openai.InternalServerError | _openai.APIConnectionError,
            )
        except ImportError:
            return False

    def _map_exception(self, exc: Exception) -> None:
        try:
            import openai as _openai

            if isinstance(exc, _openai.RateLimitError):
                raise LLMRateLimitError(
                    f"OpenAI rate limit exceeded: {exc}",
                    stage="llm",
                    component="OpenAILLM",
                    cause=exc,
                ) from exc
            if isinstance(exc, _openai.APITimeoutError):
                raise LLMTimeoutError(
                    f"OpenAI request timed out after {self._timeout}s.",
                    stage="llm",
                    component="OpenAILLM",
                    cause=exc,
                ) from exc
            if isinstance(exc, _openai.AuthenticationError):
                raise LLMError(
                    "OpenAI authentication failed. Check your OPENAI_API_KEY.",
                    stage="llm",
                    component="OpenAILLM",
                    cause=exc,
                ) from exc
        except ImportError:
            pass

        raise LLMError(
            f"OpenAI API call failed: {exc}",
            stage="llm",
            component="OpenAILLM",
            cause=exc,
        ) from exc
