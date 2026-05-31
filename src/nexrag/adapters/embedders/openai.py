"""
OpenAIEmbedder — wraps the OpenAI Embeddings API.

Supports any OpenAI-compatible endpoint via base_url (Azure OpenAI, local proxies).
Batches embed() calls to respect token limits and avoid oversized requests.
Retries transient failures (rate limits, server errors) with exponential backoff.

Requires: pip install "nexrag[openai]"  (openai)
"""

from __future__ import annotations

import random
import time
from typing import Any

from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.exceptions import EmbedderError

_BASE_BACKOFF = 1.0  # seconds; doubles each retry


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedding adapter for the OpenAI Embeddings API.

    Args:
        model:       Model name. e.g. "text-embedding-3-small", "text-embedding-ada-002".
        api_key:     OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        base_url:    Optional custom endpoint (Azure OpenAI, local proxy).
        batch_size:  Max texts per API call. Default 100.
        max_retries: Retry attempts after the first failure. 0 = no retries. Default 2.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 100,
        max_retries: int = 2,
    ) -> None:
        self._model = model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._client: Any = self._build_client(api_key, base_url)
        self._dimensions: int | None = None

    # BaseEmbedder properties

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._dimensions = len(self.embed_query(" "))
        return self._dimensions

    # Public methods

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: Strings to embed. Empty list returns [].

        Returns:
            One vector per input text, in the same order.

        Raises:
            EmbedderError: On API failure or unexpected response shape.
        """
        if not texts:
            return []

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            all_embeddings.extend(self._call_api(batch))

        if self._dimensions is None and all_embeddings:
            self._dimensions = len(all_embeddings[0])

        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Returns:
            A single embedding vector.

        Raises:
            EmbedderError: On API failure.
        """
        result = self._call_api([text])
        if not result:
            raise EmbedderError(
                "OpenAI embeddings API returned an empty response.",
                stage="embedder",
                component="OpenAIEmbedder",
            )
        if self._dimensions is None:
            self._dimensions = len(result[0])
        return result[0]

    # Private helpers

    def _build_client(self, api_key: str | None, base_url: str | None) -> Any:
        try:
            import openai
        except ImportError as e:
            raise EmbedderError(
                "openai package is required for OpenAIEmbedder. "
                'Install it: pip install "nexrag[openai]" or pip install openai',
                stage="embedder",
                component="OpenAIEmbedder",
                cause=e,
            ) from e

        kwargs: dict[str, Any] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        return openai.OpenAI(**kwargs)

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.embeddings.create(
                    input=texts,
                    model=self._model,
                )
            except Exception as e:
                if self._is_retryable(e) and attempt < self._max_retries:
                    time.sleep(_BASE_BACKOFF * (2**attempt) + random.uniform(0, 1))
                    continue
                self._map_exception(e)
                raise  # unreachable — _map_exception always raises

            data = response.data
            if len(data) != len(texts):
                raise EmbedderError(
                    f"OpenAI returned {len(data)} embeddings for {len(texts)} texts. "
                    f"Expected exactly one embedding per input.",
                    stage="embedder",
                    component="OpenAIEmbedder",
                )

            data.sort(key=lambda d: d.index)
            return [item.embedding for item in data]

        raise EmbedderError(  # unreachable
            "Retry loop exhausted without returning.",
            stage="embedder",
            component="OpenAIEmbedder",
        )

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
                raise EmbedderError(
                    f"OpenAI rate limit exceeded: {exc}",
                    stage="embedder",
                    component="OpenAIEmbedder",
                    cause=exc,
                ) from exc
            if isinstance(exc, _openai.AuthenticationError):
                raise EmbedderError(
                    "OpenAI authentication failed. Check your OPENAI_API_KEY.",
                    stage="embedder",
                    component="OpenAIEmbedder",
                    cause=exc,
                ) from exc
        except ImportError:
            pass

        raise EmbedderError(
            f"OpenAI embeddings API call failed: {exc}",
            stage="embedder",
            component="OpenAIEmbedder",
            cause=exc,
        ) from exc
