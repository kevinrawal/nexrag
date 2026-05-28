"""
BaseRetriever — contract for all retrieval strategies.

The Retriever sits between the VectorDB and the PromptBuilder in the query pipeline.
It receives both the raw query string and a pre-computed query embedding from the
QueryPipeline, then returns ranked ScoredChunks.

Why pass both query and query_embedding?
    - DenseRetriever uses query_embedding for cosine similarity search.
    - Future SparseRetriever (BM25) uses query string for keyword matching.
    - Future HybridRetriever uses both and fuses results.
    The QueryPipeline owns the single embedder instance; no retriever duplicates it.

Why is this a separate interface from BaseVectorDB?
    VectorDB speaks vectors. Retriever speaks retrieval strategy.
    In V2, a HybridRetriever can call both a dense and a sparse path and
    fuse results without touching the VectorDB interface at all.

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
        query_embedding: list[float],
        top_k: int,
        collection: str,
        score_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """
        Retrieve the most relevant chunks for a query.

        The QueryPipeline embeds the query before calling this method.
        Both the raw string and the embedding are provided so that any
        retrieval strategy can use whichever is appropriate.

        Args:
            query:           Raw query string (for sparse/keyword retrievers).
            query_embedding: Pre-computed query vector (for dense retrievers).
                             Produced by the pipeline's embedder — same model
                             used at ingestion time.
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
