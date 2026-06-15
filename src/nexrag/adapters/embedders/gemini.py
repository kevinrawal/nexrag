"""
GeminiEmbedder — wraps the Google Gemini Embeddings API via the google-genai SDK.

Uses the current `from google import genai` SDK (not the deprecated
google-generativeai). Batches embed() calls and retries transient failures with
exponential backoff. Dimensionality is detected lazily on first use.

Requires: pip install "nexrag[gemini]"  (google-genai)
"""

from __future__ import annotations

import os
import random
import time
from typing import Any

from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.exceptions import EmbedderError

_BASE_BACKOFF = 1.0  # seconds; doubles each retry


class GeminiEmbedder(BaseEmbedder):
    """
    Embedding adapter for the Google Gemini Embeddings API.

    Args:
        model:       Model name. e.g. "gemini-embedding-001", "text-embedding-004".
        api_key:     Gemini API key. If None, the SDK reads GOOGLE_API_KEY / GEMINI_API_KEY.
        base_url:    Optional custom endpoint (proxy / gateway).
        batch_size:  Max texts per API call. Default 100.
        max_retries: Retry attempts after the first failure. 0 = no retries. Default 2.
    """

    def __init__(
        self,
        model: str = "gemini-embedding-001",
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
        """Embed a batch of texts, returning one vector per input in order."""
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
        """Embed a single query string."""
        result = self._call_api([text])
        if not result:
            raise EmbedderError(
                "Gemini embeddings API returned an empty response.",
                stage="embedder",
                component="GeminiEmbedder",
            )
        if self._dimensions is None:
            self._dimensions = len(result[0])
        return result[0]

    # Private helpers

    def _build_client(self, api_key: str | None, base_url: str | None) -> Any:
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise EmbedderError(
                "google-genai package is required for GeminiEmbedder. "
                'Install it: pip install "nexrag[gemini]" or pip install google-genai',
                stage="embedder",
                component="GeminiEmbedder",
                cause=e,
            ) from e

        kwargs: dict[str, Any] = {}
        key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if key:
            kwargs["api_key"] = key
        if base_url:
            kwargs["http_options"] = types.HttpOptions(base_url=base_url)
        return genai.Client(**kwargs)

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.models.embed_content(model=self._model, contents=texts)
            except Exception as e:
                if self._is_retryable(e) and attempt < self._max_retries:
                    time.sleep(_BASE_BACKOFF * (2**attempt) + random.uniform(0, 1))
                    continue
                self._map_exception(e)
                raise  # unreachable — _map_exception always raises

            embeddings = response.embeddings or []
            if len(embeddings) != len(texts):
                raise EmbedderError(
                    f"Gemini returned {len(embeddings)} embeddings for {len(texts)} texts. "
                    f"Expected exactly one embedding per input.",
                    stage="embedder",
                    component="GeminiEmbedder",
                )
            return [list(e.values) for e in embeddings]

        raise EmbedderError(  # unreachable
            "Retry loop exhausted without returning.",
            stage="embedder",
            component="GeminiEmbedder",
        )

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        try:
            from google.genai import errors
        except ImportError:
            return False
        if isinstance(exc, errors.ServerError):
            return True
        return isinstance(exc, errors.APIError) and getattr(exc, "code", None) == 429

    def _map_exception(self, exc: Exception) -> None:
        try:
            from google.genai import errors

            if isinstance(exc, errors.APIError):
                code = getattr(exc, "code", None)
                if code == 429:
                    raise EmbedderError(
                        f"Gemini rate limit exceeded: {exc}",
                        stage="embedder",
                        component="GeminiEmbedder",
                        cause=exc,
                    ) from exc
                if code in (401, 403):
                    raise EmbedderError(
                        "Gemini authentication failed. Check your GOOGLE_API_KEY / GEMINI_API_KEY.",
                        stage="embedder",
                        component="GeminiEmbedder",
                        cause=exc,
                    ) from exc
        except ImportError:
            pass

        raise EmbedderError(
            f"Gemini embeddings API call failed: {exc}",
            stage="embedder",
            component="GeminiEmbedder",
            cause=exc,
        ) from exc
