"""
CohereReranker — wraps the Cohere Rerank API.

Sends the query and chunk texts to Cohere's rerank endpoint, which returns
relevance scores from a cross-encoder model. Chunks are reordered by the
Cohere relevance score and trimmed to top_n.

Requires: pip install "nexrag[cohere]"  (cohere)
"""

from __future__ import annotations

from typing import Any

from nexrag.core.interfaces.reranker import BaseReranker
from nexrag.core.models.chunk import ScoredChunk
from nexrag.exceptions import RetrieverError


class CohereReranker(BaseReranker):
    """
    Reranker adapter using the Cohere Rerank API.

    Args:
        model:   Cohere rerank model ID. e.g. "rerank-english-v3.0".
        api_key: Cohere API key. If None, reads COHERE_API_KEY from env.
        top_n:   How many chunks to return after reranking. Default 5.
    """

    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        api_key: str | None = None,
        top_n: int = 5,
    ) -> None:
        super().__init__(top_n)
        self._model = model
        self._api_key = api_key
        self._client: Any = self._build_client()

    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_n: int,
    ) -> list[ScoredChunk]:
        """
        Rerank chunks using Cohere Rerank API.

        Args:
            query:  The user's question.
            chunks: Retrieved chunks to rerank.
            top_n:  Number of chunks to return after reranking.

        Returns:
            Re-scored ScoredChunk list in new relevance order, trimmed to top_n.

        Raises:
            RetrieverError: If the Cohere API call fails.
        """
        if not chunks:
            return []

        documents = [sc.chunk.text for sc in chunks]

        try:
            response = self._client.rerank(
                model=self._model,
                query=query,
                documents=documents,
                top_n=min(top_n, len(chunks)),
            )
        except Exception as e:
            raise RetrieverError(
                f"CohereReranker API call failed: {e}",
                stage="reranker",
                component="CohereReranker",
                cause=e,
            ) from e

        results: list[ScoredChunk] = []
        for rank, result in enumerate(response.results, start=1):
            original = chunks[result.index]
            results.append(
                ScoredChunk(
                    chunk=original.chunk,
                    score=float(result.relevance_score),
                    rank=rank,
                )
            )

        return results

    def _build_client(self) -> Any:
        try:
            import cohere  # type: ignore[import-not-found]
        except ImportError as e:
            raise RetrieverError(
                "cohere package is required for CohereReranker. "
                'Install it: pip install "nexrag[cohere]" or pip install cohere',
                stage="reranker",
                component="CohereReranker",
                cause=e,
            ) from e

        if self._api_key:
            return cohere.Client(api_key=self._api_key)
        return cohere.Client()
