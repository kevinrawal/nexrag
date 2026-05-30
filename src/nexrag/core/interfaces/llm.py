"""
BaseLLM — contract for all LLM adapters.

The LLM adapter takes the assembled prompt string and returns the model's
response. Local and cloud providers both implement this same interface —
the pipeline doesn't know or care which one it's talking to.

Built-in adapters (Phase 1):
    OpenAILLM, AnthropicLLM, OllamaLLM

Local via Ollama:
    Runs as a local HTTP server with an OpenAI-compatible API.
    NexRAG calls it identically to any cloud provider.

Custom implementation pattern:
    class InternalModelAdapter(BaseLLM):
        def generate(self, prompt: str) -> str: ...
        def stream(self, prompt: str) -> Iterator[str]: ...
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator


class BaseLLM(ABC):
    """Abstract base class for all NexRAG LLM adapters."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """
        Send a prompt and return the complete response as a string.

        Args:
            prompt: The assembled prompt string from PromptBuilder.

        Returns:
            The LLM's full response text.

        Raises:
            LLMTimeoutError:    If the call exceeds the configured timeout.
            LLMRateLimitError:  If the provider returns a rate limit error.
            LLMError:           For all other LLM failures.
        """

    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """
        Send a prompt and yield response tokens as they arrive.

        Args:
            prompt: The assembled prompt string from PromptBuilder.

        Yields:
            Response text chunks (tokens or small strings) as they stream.

        Raises:
            LLMTimeoutError:    If the stream connection times out.
            LLMRateLimitError:  If the provider returns a rate limit error.
            LLMError:           For all other LLM failures.
        """

    async def async_generate(self, prompt: str) -> str:
        """
        Async variant of generate(). Default: runs sync generate() in a thread pool.
        Override with a native async client (AsyncOpenAI, AsyncAnthropic) for true async I/O.
        """
        return await asyncio.to_thread(self.generate, prompt)

    async def async_stream(self, prompt: str) -> AsyncIterator[str]:
        """
        Async variant of stream(). Default: collects all tokens from sync stream() in a
        thread, then yields them. Adapters with native async clients should override this
        to yield tokens as they arrive without buffering.
        """
        tokens: list[str] = await asyncio.to_thread(list, self.stream(prompt))
        for token in tokens:
            yield token
