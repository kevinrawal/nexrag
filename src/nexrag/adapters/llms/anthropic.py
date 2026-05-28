"""
AnthropicLLM — wraps the Anthropic Messages API.

Key difference from OpenAI: system prompt and user messages are separate parameters
in the Anthropic API. The \n\n---\n\n separator maps directly to system= and messages=.
Retries transient failures (rate limits, server errors) with exponential backoff.

Requires: pip install "nexrag[anthropic]"  (anthropic)
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterator
from typing import Any

from nexrag.core.interfaces.llm import BaseLLM
from nexrag.exceptions import LLMError, LLMRateLimitError, LLMTimeoutError

_BASE_BACKOFF = 1.0  # seconds; doubles each retry


class AnthropicLLM(BaseLLM):
    """
    LLM adapter for the Anthropic Messages API.

    Args:
        model:       Claude model ID. e.g. "claude-3-5-sonnet-20241022", "claude-3-haiku-20240307".
        api_key:     Anthropic API key. If None, reads ANTHROPIC_API_KEY from env.
        base_url:    Optional custom endpoint (proxy).
        temperature: Sampling temperature. Default 0.2.
        max_tokens:  Max tokens in the response. Default 1024.
        timeout:     Request timeout in seconds. Default 30.
        max_retries: Retry attempts after the first failure. 0 = no retries. Default 2.
    """

    def __init__(
        self,
        model: str = "claude-3-5-sonnet-20241022",
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

    def generate(self, prompt: str) -> str:
        """
        Send the prompt and return the complete response.

        The prompt is split at the first "---" separator into system and user messages.
        If no separator, the whole prompt is the user message (no system message).

        Returns:
            The assistant response text.

        Raises:
            LLMRateLimitError: On 429 rate limit (after retries exhausted).
            LLMTimeoutError:   On timeout.
            LLMError:          On any other failure.
        """
        system, messages = self._build_messages(prompt)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if system:
            kwargs["system"] = system

        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.messages.create(**kwargs)
                return response.content[0].text or ""
            except Exception as e:
                if self._is_retryable(e) and attempt < self._max_retries:
                    time.sleep(_BASE_BACKOFF * (2**attempt) + random.uniform(0, 1))
                    continue
                self._map_exception(e)
                raise  # unreachable — _map_exception always raises

        raise LLMError(
            "Retry loop exhausted without returning.", stage="llm", component="AnthropicLLM"
        )  # unreachable

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream the response token by token.

        Note: streaming calls are not retried — a partial stream cannot be resumed.

        Yields:
            Response text chunks as they arrive.
        """
        system, messages = self._build_messages(prompt)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": self._max_tokens,
            "temperature": self._temperature,
        }
        if system:
            kwargs["system"] = system

        try:
            with self._client.messages.stream(**kwargs) as stream:
                for text in stream.text_stream:
                    if text:
                        yield text
        except Exception as e:
            self._map_exception(e)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_client(self, api_key: str | None, base_url: str | None) -> Any:
        try:
            import anthropic  # type: ignore[import-not-found]
        except ImportError as e:
            raise LLMError(
                "anthropic package is required for AnthropicLLM. "
                'Install it: pip install "nexrag[anthropic]" or pip install anthropic',
                stage="llm",
                component="AnthropicLLM",
                cause=e,
            ) from e

        kwargs: dict[str, Any] = {"timeout": self._timeout}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return anthropic.Anthropic(**kwargs)

    @staticmethod
    def _build_messages(prompt: str) -> tuple[str | None, list[dict[str, str]]]:
        """Split prompt into (system, user_messages) on the --- separator."""
        sep = "\n\n---\n\n"
        if sep in prompt:
            system_part, user_part = prompt.split(sep, 1)
            return system_part.strip() or None, [{"role": "user", "content": user_part.strip()}]
        return None, [{"role": "user", "content": prompt.strip()}]

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        try:
            import anthropic as _anthropic

            return isinstance(
                exc,
                _anthropic.RateLimitError
                | _anthropic.InternalServerError
                | _anthropic.APIConnectionError,
            )
        except ImportError:
            return False

    def _map_exception(self, exc: Exception) -> None:
        try:
            import anthropic as _anthropic

            if isinstance(exc, _anthropic.RateLimitError):
                raise LLMRateLimitError(
                    f"Anthropic rate limit exceeded: {exc}",
                    stage="llm",
                    component="AnthropicLLM",
                    cause=exc,
                ) from exc
            if isinstance(exc, _anthropic.APITimeoutError):
                raise LLMTimeoutError(
                    f"Anthropic request timed out after {self._timeout}s.",
                    stage="llm",
                    component="AnthropicLLM",
                    cause=exc,
                ) from exc
            if isinstance(exc, _anthropic.AuthenticationError):
                raise LLMError(
                    "Anthropic authentication failed. Check your ANTHROPIC_API_KEY.",
                    stage="llm",
                    component="AnthropicLLM",
                    cause=exc,
                ) from exc
        except ImportError:
            pass

        raise LLMError(
            f"Anthropic API call failed: {exc}",
            stage="llm",
            component="AnthropicLLM",
            cause=exc,
        ) from exc
