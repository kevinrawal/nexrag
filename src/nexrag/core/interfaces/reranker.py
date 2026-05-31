"""
BaseReranker — contract for all post-retrieval reranking adapters.

The Reranker sits between the Retriever and PromptBuilder in the query pipeline.
It receives the retrieved chunks and the original query, then re-scores and
re-ranks them using a more precise relevance model.

Why rerank?
    Vector similarity (cosine) is a fast but approximate relevance signal.
    A cross-encoder reads query + chunk together and produces a much more
    accurate relevance score. The standard production pattern:
        1. Retrieve a large candidate set (e.g. top_k=50 via dense search)
        2. Rerank to find the truly most relevant subset (e.g. top_n=5)
        3. Pass only top_n to the LLM prompt

V1 implementations: CohereReranker, CrossEncoderReranker
Future: ColBERTReranker, FlashRankReranker, custom class via class_path

The reranker is optional — no reranker config in nexrag.yaml means the
pipeline is unchanged from V1. Adding a reranker block enables the stage.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from nexrag.core.models.chunk import ScoredChunk


class BaseReranker(ABC):
    """Abstract base class for all NexRAG reranker adapters."""

    def __init__(self, top_n: int) -> None:
        self._top_n = top_n

    @property
    def top_n(self) -> int:
        """Number of chunks to return after reranking."""
        return self._top_n

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_n: int,
    ) -> list[ScoredChunk]:
        """
        Re-score and re-rank chunks by relevance to the query.

        Args:
            query:  The original user query string.
            chunks: Chunks to rerank, as returned by the retriever.
            top_n:  How many chunks to return after reranking. Must be <= len(chunks).

        Returns:
            List of ScoredChunk in new relevance order, trimmed to top_n.
            Scores reflect the reranker's relevance estimate (adapter-specific scale).

        Raises:
            RetrieverError: If the reranker API/model call fails.
        """

    async def async_rerank(
        self,
        query: str,
        chunks: list[ScoredChunk],
        top_n: int,
    ) -> list[ScoredChunk]:
        """
        Async variant of rerank(). Default: runs sync rerank() in a thread pool.
        Override for adapters that use a native async API.
        """
        return await asyncio.to_thread(self.rerank, query, chunks, top_n)
