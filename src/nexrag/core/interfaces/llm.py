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

from abc import ABC, abstractmethod
from collections.abc import Iterator


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
