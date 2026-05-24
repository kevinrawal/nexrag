"""
BaseRetriever — contract for all retrieval strategies.

The Retriever sits between the VectorDB and the PromptBuilder in the query pipeline.
It takes a raw query string, embeds it, queries the VectorDB, and returns
ranked ScoredChunks.

Why is this a separate interface from BaseVectorDB?
    VectorDB speaks vectors. Retriever speaks strings.
    This separation means:
    - In V2, a HybridRetriever can call both a DenseRetriever and a
      SparseRetriever and fuse results — without touching the VectorDB interface.
    - Retrieval strategy (dense, sparse, hybrid, graph) is swappable
      without changing how the VectorDB adapter works.

V1 implementation: DenseRetriever (cosine similarity via ChromaDB).
Future: SparseRetriever (BM25), HybridRetriever, GraphRetriever.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nexrag.core.models.chunk import ScoredChunk


class BaseRetriever(ABC):
    """Abstract base class for all NexRAG retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int,
        collection: str,
        score_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query:           Raw query string. The retriever embeds it internally.
            top_k:           Maximum number of chunks to return.
            collection:      Which vector collection to search.
            score_threshold: Minimum similarity score. Chunks below this are dropped.
            filters:         Optional metadata filters.

        Returns:
            List of ScoredChunks ordered by score descending.
            May be shorter than top_k if fewer results pass the score_threshold.

        Raises:
            RetrieverError: If retrieval fails.
        """
