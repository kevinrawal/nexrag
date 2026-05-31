"""
CrossEncoderReranker — local cross-encoder reranking via sentence-transformers.

Scores (query, chunk) pairs jointly using a cross-encoder model. The cross-encoder
reads both query and document together, producing a much more accurate relevance
score than embedding-based cosine similarity.

Requires: pip install "nexrag[cross-encoder]"  (sentence-transformers)

Recommended models:
    cross-encoder/ms-marco-MiniLM-L-6-v2  (fast, English)
    cross-encoder/ms-marco-electra-base   (higher quality, slower)
    BAAI/bge-reranker-base                (multilingual)
"""

from __future__ import annotations

from typing import Any

from nexrag.core.interfaces.reranker import BaseReranker
from nexrag.core.models.chunk import ScoredChunk
from nexrag.exceptions import RetrieverError


class CrossEncoderReranker(BaseReranker):
    """
    Reranker using a local sentence-transformers CrossEncoder model.

    Args:
        model:  HuggingFace model ID for the cross-encoder.
                e.g. "cross-encoder/ms-marco-MiniLM-L-6-v2"
        top_n:  How many chunks to return after reranking. Default 5.
        device: PyTorch device. None = auto-detect (GPU if available, else CPU).
    """

    def __init__(
        self,
        model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n: int = 5,
        device: str | None = None,
    ) -> None:
        super().__init__(top_n)
        self._model_name = model
        self._device = device
        self._encoder: Any = self._build_encoder()

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_n: int,
    ) -> list[ScoredChunk]:
        """
        Rerank chunks using a local cross-encoder model.

        Args:
            query:  The user's question.
            chunks: Retrieved chunks to rerank.
            top_n:  Number of chunks to return after reranking.

        Returns:
            Re-scored ScoredChunk list in new relevance order, trimmed to top_n.

        Raises:
            RetrieverError: If the model inference fails.
        """
        if not chunks:
            return []

        pairs = [[query, sc.chunk.text] for sc in chunks]

        try:
            scores = self._encoder.predict(pairs)
        except Exception as e:
            raise RetrieverError(
                f"CrossEncoderReranker model inference failed: {e}",
                stage="reranker",
                component="CrossEncoderReranker",
                cause=e,
            ) from e

        scored_with_idx = sorted(enumerate(scores.tolist()), key=lambda x: x[1], reverse=True)

        results: list[ScoredChunk] = []
        for rank, (idx, score) in enumerate(scored_with_idx[: min(top_n, len(chunks))], start=1):
            original = chunks[idx]
            results.append(
                ScoredChunk(
                    chunk=original.chunk,
                    score=float(score),
                    rank=rank,
                )
            )

        return results

    def _build_encoder(self) -> Any:
        try:
            from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
        except ImportError as e:
            raise RetrieverError(
                "sentence-transformers package is required for CrossEncoderReranker. "
                'Install it: pip install "nexrag[cross-encoder]" or pip install sentence-transformers',
                stage="reranker",
                component="CrossEncoderReranker",
                cause=e,
            ) from e

        kwargs: dict[str, Any] = {}
        if self._device is not None:
            kwargs["device"] = self._device

        return CrossEncoder(self._model_name, **kwargs)
