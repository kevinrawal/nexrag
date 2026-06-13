"""
BaseLLM — contract for all LLM adapters.

The LLM adapter takes the assembled prompt string and returns both the model's
response text and token usage (when the provider exposes it).

generate() returns a tuple[str, TokenUsage | None] so callers get both the
answer and usage in a single call — no separate generate_with_usage() needed.

Built-in adapters (Phase 1):
    OpenAILLM, AnthropicLLM, OllamaLLM

Local via Ollama:
    Runs as a local HTTP server with an OpenAI-compatible API.
    NexRAG calls it identically to any cloud provider.

Custom implementation pattern:
    class InternalModelAdapter(BaseLLM):
        def generate(self, prompt: str) -> tuple[str, TokenUsage | None]: ...
        def stream(self, prompt: str) -> Iterator[str]: ...
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator

from nexrag.core.models.metrics import TokenUsage


class BaseLLM(ABC):
    """Abstract base class for all NexRAG LLM adapters."""

    @property
    def model_name(self) -> str | None:
        """
        Identifier for the model this adapter calls, surfaced in RunMetrics and
        observability events.

        Default: reads a ``_model`` attribute if the adapter sets one (all built-in
        adapters do). Custom adapters that store the model under a different name
        should override this property so their model shows up in metrics.
        """
        return getattr(self, "_model", None)

    @abstractmethod
    def generate(self, prompt: str) -> tuple[str, TokenUsage | None]:
        """
        Send a prompt and return (response_text, token_usage).

        token_usage is None when the provider does not expose token counts
        (e.g. Ollama with some models). The pipeline always receives the tuple;
        callers that only need the text use result[0].

        Args:
            prompt: The assembled prompt string from PromptBuilder.

        Returns:
            (response_text, TokenUsage | None)

        Raises:
            LLMTimeoutError:    If the call exceeds the configured timeout.
            LLMRateLimitError:  If the provider returns a rate limit error.
            LLMError:           For all other LLM failures.
        """

    @abstractmethod
    def stream(self, prompt: str) -> Iterator[str]:
        """
        Send a prompt and yield response tokens as they arrive.

        Streaming does not return token usage — most provider streaming APIs
        do not expose usage mid-stream. Use generate() when you need usage counts.

        Args:
            prompt: The assembled prompt string from PromptBuilder.

        Yields:
            Response text chunks (tokens or small strings) as they stream.

        Raises:
            LLMTimeoutError:    If the stream connection times out.
            LLMRateLimitError:  If the provider returns a rate limit error.
            LLMError:           For all other LLM failures.
        """

    async def async_generate(self, prompt: str) -> tuple[str, TokenUsage | None]:
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
