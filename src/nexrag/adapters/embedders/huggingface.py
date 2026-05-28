"""
HuggingFaceEmbedder — wraps the HuggingFace Inference API via huggingface_hub.

Calls InferenceClient.feature_extraction() in batches.
If base_url is set, targets a Dedicated Inference Endpoint instead of the shared API.

Requires: pip install "nexrag[huggingface]"  (huggingface-hub)
"""

from __future__ import annotations

from typing import Any

from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.exceptions import EmbedderError


class HuggingFaceEmbedder(BaseEmbedder):
    """
    Embedding adapter for the HuggingFace Inference API.

    Args:
        model:      HuggingFace model ID. e.g. "sentence-transformers/all-MiniLM-L6-v2".
        api_key:    HuggingFace token. If None, reads HF_TOKEN from env.
        base_url:   Optional Dedicated Endpoint URL (overrides the shared Inference API).
        batch_size: Number of texts per API call. Default 32.
    """

    def __init__(
        self,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 32,
    ) -> None:
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._batch_size = batch_size
        self._client: Any = self._build_client()
        self._dimensions: int | None = None

    # ── BaseEmbedder properties ───────────────────────────────────────────────

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            self._dimensions = len(self.embed_query(" "))
        return self._dimensions

    # ── Public methods ────────────────────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts, sending up to batch_size texts per API call.

        Returns:
            One vector per input text, in the same order.

        Raises:
            EmbedderError: On auth failure, model not found, or API error.
        """
        if not texts:
            return []

        results: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            results.extend(self._call_api(batch))

        if self._dimensions is None and results:
            self._dimensions = len(results[0])

        return results

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Raises:
            EmbedderError: On auth failure, model not found, or API error.
        """
        vec = self._call_api([text])[0]
        if self._dimensions is None:
            self._dimensions = len(vec)
        return vec

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_client(self) -> Any:
        try:
            from huggingface_hub import InferenceClient
        except ImportError as e:
            raise EmbedderError(
                "huggingface_hub package is required for HuggingFaceEmbedder. "
                'Install it: pip install "nexrag[huggingface]" or pip install huggingface-hub',
                stage="embedder",
                component="HuggingFaceEmbedder",
                cause=e,
            ) from e

        kwargs: dict[str, Any] = {"model": self._model}
        if self._api_key:
            kwargs["token"] = self._api_key
        if self._base_url:
            kwargs["base_url"] = self._base_url
        return InferenceClient(**kwargs)

    def _call_api(self, texts: list[str]) -> list[list[float]]:
        try:
            response = self._client.feature_extraction(text=texts)
            # response is a numpy ndarray (N, dim) or nested list
            if hasattr(response, "tolist"):
                vectors: list[Any] = response.tolist()
            else:
                vectors = list(response)
            return [list(v) for v in vectors]
        except EmbedderError:
            raise
        except Exception as e:
            self._map_exception(e)
            raise  # unreachable — _map_exception always raises

    def _map_exception(self, exc: Exception) -> None:
        msg = str(exc).lower()

        if any(k in msg for k in ("401", "unauthorized", "invalid token", "authentication")):
            raise EmbedderError(
                "HuggingFace authentication failed. Check your api_key / HF_TOKEN.",
                stage="embedder",
                component="HuggingFaceEmbedder",
                cause=exc,
            ) from exc

        if "model" in msg and any(k in msg for k in ("not found", "404", "does not exist")):
            raise EmbedderError(
                f"HuggingFace model {self._model!r} not found or not accessible.",
                stage="embedder",
                component="HuggingFaceEmbedder",
                cause=exc,
            ) from exc

        raise EmbedderError(
            f"HuggingFace feature_extraction failed: {exc}",
            stage="embedder",
            component="HuggingFaceEmbedder",
            cause=exc,
        ) from exc
