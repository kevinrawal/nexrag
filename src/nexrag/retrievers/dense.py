"""
DenseRetriever — cosine similarity retrieval via a BaseVectorDB.

This is NexRAG's default retriever for V1 naive RAG. It:
  1. Receives the pre-computed query embedding from QueryPipeline.
  2. Delegates vector search to the injected BaseVectorDB.
  3. Applies score_threshold filtering on results.

No third-party dependencies — depends only on nexrag.core interfaces.
"""

from __future__ import annotations

from typing import Any

from nexrag.core.interfaces.retriever import BaseRetriever
from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.models.chunk import ScoredChunk
from nexrag.exceptions import RetrieverError, VectorDBError


class DenseRetriever(BaseRetriever):
    """
    Retrieves chunks by cosine similarity against a vector database.

    Args:
        vector_db: Any BaseVectorDB implementation. ChromaDBAdapter in V1.
    """

    def __init__(self, vector_db: BaseVectorDB) -> None:
        self._vector_db = vector_db

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
        Retrieve the top-k most similar chunks for the query embedding.

        Args:
            query:           Raw query string (not used by this retriever;
                             included in signature for interface compliance).
            query_embedding: Pre-computed query vector from the pipeline embedder.
            top_k:           Maximum chunks to return before score filtering.
            collection:      Vector collection to search.
            score_threshold: Minimum similarity score (0.0–1.0). Default 0.0.
            filters:         Metadata filters applied at the DB level.

        Returns:
            List of ScoredChunks, ordered by score descending, filtered by threshold.

        Raises:
            RetrieverError: If the vector DB query fails or returns bad data.
        """
        if not query_embedding:
            raise RetrieverError(
                "query_embedding is empty. The embedder must return a non-empty vector.",
                stage="retriever",
                component="DenseRetriever",
            )

        try:
            results = self._vector_db.query(
                embedding=query_embedding,
                top_k=top_k,
                collection_name=collection,
                filters=filters,
            )
        except VectorDBError:
            raise
        except Exception as e:
            raise RetrieverError(
                f"DenseRetriever vector search failed: {e}",
                stage="retriever",
                component="DenseRetriever",
                cause=e,
            ) from e

        if score_threshold > 0.0:
            results = [r for r in results if r.score >= score_threshold]

        return results
