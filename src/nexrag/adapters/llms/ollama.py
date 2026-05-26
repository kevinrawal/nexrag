"""
OllamaLLM — wraps the Ollama local inference server.

Drop-in replacement for OpenAILLM for local testing without API keys.
Ollama exposes an OpenAI-compatible API at http://localhost:11434.

Requires: pip install "nexrag[ollama]"  (ollama)
           and Ollama running locally: https://ollama.com
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from nexrag.core.interfaces.llm import BaseLLM
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
    """

    def __init__(
        self,
        model: str = "llama3.2",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        max_tokens: int = 1024,
        timeout: int = 60,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    def generate(self, prompt: str) -> str:
        """
        Send the prompt to Ollama and return the complete response.

        Returns:
            The model response text.

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
            return str(response["message"]["content"])
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
