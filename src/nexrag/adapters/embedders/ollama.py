"""
OllamaEmbedder — wraps the Ollama local embeddings API.

Calls ollama.Client.embeddings() once per text (Ollama has no batch endpoint).
The sync path is sequential. The async path fires all requests concurrently
(bounded by max_concurrent_requests) via asyncio.gather() + asyncio.to_thread(),
which gives a ~10x throughput improvement for large ingestion batches.

Requires: pip install "nexrag[ollama]"  (ollama)
           and Ollama running locally: https://ollama.com
"""

from __future__ import annotations

import asyncio
from typing import Any

from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.exceptions import EmbedderError


class OllamaEmbedder(BaseEmbedder):
    """
    Embedding adapter for the Ollama local inference server.

    Args:
        model:                  Ollama model name. e.g. "nomic-embed-text", "mxbai-embed-large".
        base_url:               Ollama server URL. Default "http://localhost:11434".
        batch_size:             Accepted for config compatibility; Ollama embeds one text at a time.
        max_concurrent_requests: Maximum concurrent async HTTP calls in async_embed().
                                 Prevents overwhelming the local Ollama server. Default 10.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
        batch_size: int = 100,
        max_concurrent_requests: int = 10,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._batch_size = batch_size
        self._max_concurrent_requests = max_concurrent_requests
        self._client: Any = self._build_client()
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
        Embed a list of texts sequentially. Each text is one HTTP call.
        For concurrent throughput, use async_embed() instead.

        Returns:
            One vector per input text, in the same order.

        Raises:
            EmbedderError: On connection failure, model not found, or API error.
        """
        if not texts:
            return []

        result = [self._call_api(text) for text in texts]

        if self._dimensions is None and result:
            self._dimensions = len(result[0])

        return result

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Raises:
            EmbedderError: On connection failure, model not found, or API error.
        """
        vec = self._call_api(text)
        if self._dimensions is None:
            self._dimensions = len(vec)
        return vec

    async def async_embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts concurrently using asyncio.gather().

        Fires up to max_concurrent_requests calls simultaneously via
        asyncio.to_thread(), so Ollama's CPU is kept busy without spawning
        unbounded threads. For 512 texts at ~50ms/call with
        max_concurrent_requests=10, this completes in ~2.6s vs ~25s sequential.

        Returns:
            One vector per input text, in the same order.

        Raises:
            EmbedderError: On connection failure, model not found, or API error.
        """
        if not texts:
            return []

        sem = asyncio.Semaphore(self._max_concurrent_requests)

        async def _one(text: str) -> list[float]:
            async with sem:
                return await asyncio.to_thread(self._call_api, text)

        results = await asyncio.gather(*[_one(t) for t in texts])

        if self._dimensions is None and results:
            self._dimensions = len(results[0])

        return list(results)

    # Private helpers

    def _build_client(self) -> Any:
        try:
            import ollama

            return ollama.Client(host=self._base_url)
        except ImportError as e:
            raise EmbedderError(
                "ollama package is required for OllamaEmbedder. "
                'Install it: pip install "nexrag[ollama]" or pip install ollama',
                stage="embedder",
                component="OllamaEmbedder",
                cause=e,
            ) from e

    def _call_api(self, text: str) -> list[float]:
        try:
            response = self._client.embeddings(model=self._model, prompt=text)
            return list(response["embedding"])
        except EmbedderError:
            raise
        except Exception as e:
            self._map_exception(e)
            raise  # unreachable — _map_exception always raises

    def _map_exception(self, exc: Exception) -> None:
        msg = str(exc).lower()

        if "connection" in msg or "refused" in msg:
            raise EmbedderError(
                f"Cannot connect to Ollama at {self._base_url}. " "Start Ollama with: ollama serve",
                stage="embedder",
                component="OllamaEmbedder",
                cause=exc,
            ) from exc

        if "model" in msg and ("not found" in msg or "pull" in msg):
            raise EmbedderError(
                f"Ollama model {self._model!r} not found. "
                f"Pull it first: ollama pull {self._model}",
                stage="embedder",
                component="OllamaEmbedder",
                cause=exc,
            ) from exc

        raise EmbedderError(
            f"Ollama embeddings API call failed: {exc}",
            stage="embedder",
            component="OllamaEmbedder",
            cause=exc,
        ) from exc
