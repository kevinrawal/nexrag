"""
OllamaLLM — wraps the Ollama local inference server.

Drop-in replacement for OpenAILLM for local testing without API keys.
Ollama exposes a REST API at http://localhost:11434.

Sync methods use the ollama Python SDK.
Async methods use httpx.AsyncClient for true non-blocking I/O — avoids
thread-pool saturation under concurrent load (10+ simultaneous queries).

Requires: pip install "nexrag[ollama]"  (ollama, httpx)
           and Ollama running locally: https://ollama.com
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Any

from nexrag.core.interfaces.llm import BaseLLM
from nexrag.core.models.metrics import TokenUsage
from nexrag.exceptions import LLMError, LLMTimeoutError


class OllamaLLM(BaseLLM):
    """
    LLM adapter for Ollama local inference.

    Args:
        model:       Ollama model name. e.g. "llama3.2", "mistral", "phi3".
        base_url:    Ollama server URL. Default "http://localhost:11434".
        temperature: Sampling temperature. Default 0.2.
        max_tokens:  Max tokens in response. Default 1024.
        timeout:     Request timeout in seconds. Default 60.
        max_retries: Maximum retries on transient HTTP failures for async methods. Default 2.
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 60,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout
        self._max_retries = max_retries

    def generate(self, prompt: str) -> tuple[str, TokenUsage | None]:
        """
        Send the prompt to Ollama and return (response_text, None).

        Ollama does not reliably expose token counts across all models, so
        token_usage is always None. Use OpenAILLM or AnthropicLLM when you
        need token tracking.

        Returns:
            (response_text, None)

        Raises:
            LLMTimeoutError: On timeout.
            LLMError:        On connection failure or API error.
        """
        client = self._build_client()
        messages = self._build_messages(prompt)
        try:
            response = client.chat(
                model=self._model,
                messages=messages,
                options={
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                },
            )
            return str(response["message"]["content"]), None
        except Exception as e:
            self._map_exception(e)
            raise  # unreachable

    def stream(self, prompt: str) -> Iterator[str]:
        """
        Stream the Ollama response token by token.
        """
        client = self._build_client()
        messages = self._build_messages(prompt)
        try:
            for chunk in client.chat(
                model=self._model,
                messages=messages,
                options={
                    "temperature": self._temperature,
                    "num_predict": self._max_tokens,
                },
                stream=True,
            ):
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
        except Exception as e:
            self._map_exception(e)

    async def async_generate(self, prompt: str) -> tuple[str, TokenUsage | None]:
        """
        Native async generate using httpx.AsyncClient.

        Uses true async I/O instead of asyncio.to_thread() so the event loop
        is never blocked — safe under 20+ concurrent queries.

        Returns:
            (response_text, None)

        Raises:
            LLMTimeoutError: On timeout.
            LLMError:        On connection failure or API error.
        """
        import httpx

        messages = self._build_messages(prompt)
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": self._temperature, "num_predict": self._max_tokens},
        }

        async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
            for attempt in range(self._max_retries + 1):
                try:
                    resp = await client.post("/api/chat", json=payload)
                    resp.raise_for_status()
                    data = resp.json()
                    return str(data["message"]["content"]), None
                except httpx.TimeoutException as e:
                    raise LLMTimeoutError(
                        f"Ollama request timed out after {self._timeout}s. "
                        f"Is Ollama running at {self._base_url}?",
                        stage="llm",
                        component="OllamaLLM",
                        cause=e,
                    ) from e
                except httpx.HTTPError as e:
                    if attempt < self._max_retries:
                        await asyncio.sleep(2**attempt)
                        continue
                    self._map_exception(e)
                    raise  # unreachable

        raise LLMError(  # unreachable
            "Retry loop exhausted without returning.",
            stage="llm",
            component="OllamaLLM",
        )

    async def async_stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Native async streaming using httpx.AsyncClient with server-sent events.

        Yields tokens as they arrive — no buffering, no thread-pool involvement.

        Yields:
            Response text tokens as they stream from Ollama.

        Raises:
            LLMTimeoutError: On timeout.
            LLMError:        On connection failure or API error.
        """
        import httpx

        messages = self._build_messages(prompt)
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": self._temperature, "num_predict": self._max_tokens},
        }

        try:
            async with httpx.AsyncClient(base_url=self._base_url, timeout=self._timeout) as client:
                async with client.stream("POST", "/api/chat", json=payload) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.strip():
                            continue
                        import json as _json

                        try:
                            data = _json.loads(line)
                        except _json.JSONDecodeError:
                            continue
                        content = data.get("message", {}).get("content", "")
                        if content:
                            yield content
                        if data.get("done"):
                            break
        except httpx.TimeoutException as e:
            raise LLMTimeoutError(
                f"Ollama streaming timed out after {self._timeout}s. "
                f"Is Ollama running at {self._base_url}?",
                stage="llm",
                component="OllamaLLM",
                cause=e,
            ) from e
        except httpx.HTTPError as e:
            self._map_exception(e)

    # Private helpers

    def _build_client(self) -> Any:
        try:
            import ollama  # type: ignore[import-not-found]

            return ollama.Client(host=self._base_url, timeout=self._timeout)
        except ImportError as e:
            raise LLMError(
                "ollama package is required for OllamaLLM. "
                'Install it: pip install "nexrag[ollama]" or pip install ollama',
                stage="llm",
                component="OllamaLLM",
                cause=e,
            ) from e

    @staticmethod
    def _build_messages(prompt: str) -> list[dict[str, str]]:
        sep = "\n\n---\n\n"
        if sep in prompt:
            system_part, user_part = prompt.split(sep, 1)
            return [
                {"role": "system", "content": system_part.strip()},
                {"role": "user", "content": user_part.strip()},
            ]
        return [{"role": "user", "content": prompt.strip()}]

    def _map_exception(self, exc: Exception) -> None:
        msg = str(exc).lower()
        if "timeout" in msg or "timed out" in msg:
            raise LLMTimeoutError(
                f"Ollama request timed out after {self._timeout}s. "
                f"Is Ollama running at {self._base_url}?",
                stage="llm",
                component="OllamaLLM",
                cause=exc,
            ) from exc
        if "connection" in msg or "refused" in msg:
            raise LLMError(
                f"Cannot connect to Ollama at {self._base_url}. "
                f"Start Ollama with: ollama serve",
                stage="llm",
                component="OllamaLLM",
                cause=exc,
            ) from exc
        raise LLMError(
            f"Ollama API call failed: {exc}",
            stage="llm",
            component="OllamaLLM",
            cause=exc,
        ) from exc
