"""
BaseVectorDB — contract for all vector database adapters.

The VectorDB is shared between both pipelines:
    - Ingestion pipeline writes to it via upsert().
    - Query pipeline reads from it via query().

V1 implementation: ChromaDB (local in-process or remote HTTP).
Future: Pinecone, Weaviate, Qdrant, PGVector.

NexRAG manages data operations only.
NexRAG never manages: server provisioning, credentials rotation, scaling, billing.

Collection awareness:
    Every method accepts a collection_name parameter. In V1 this always
    matches the single active collection from config. In V2+ the router
    passes different collection names for cross-collection queries.
    The interface is collection-aware from day one so V2 requires no changes here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nexrag.core.models.chunk import Chunk, ScoredChunk


class BaseVectorDB(ABC):
    """Abstract base class for all NexRAG vector database adapters."""

    @abstractmethod
    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        collection_name: str,
    ) -> None:
        """
        Write chunks and their embeddings to the vector DB.

        Chunks are keyed by content_hash. Calling upsert with the same
        chunk hash overwrites the existing record (idempotent).

        Args:
            chunks:          Chunks to store.
            embeddings:      Vectors, one per chunk, in the same order.
            collection_name: Target collection.

        Raises:
            VectorDBUpsertError: If the write fails.
        """

    @abstractmethod
    def query(
        self,
        embedding: list[float],
        top_k: int,
        collection_name: str,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """
        Semantic search against the collection.

        Args:
            embedding:       Query vector.
            top_k:           Maximum number of results to return.
            collection_name: Collection to search.
            filters:         Optional metadata filters (e.g. {"vendor": "Acme"}).

        Returns:
            List of ScoredChunks ordered by score descending (most relevant first).

        Raises:
            VectorDBError: If the query fails.
        """

    @abstractmethod
    def delete(self, ids: list[str], collection_name: str) -> None:
        """
        Delete chunks by their content_hash IDs.

        Used by the idempotency check to remove stale chunks before re-inserting.

        Args:
            ids:             content_hash values of chunks to delete.
            collection_name: Collection to delete from.

        Raises:
            VectorDBError: If deletion fails.
        """

    @abstractmethod
    def count(self, collection_name: str) -> int:
        """Return the number of chunks stored in the collection."""

    @abstractmethod
    def get_collection_metadata(self, collection_name: str) -> dict[str, Any]:
        """
        Retrieve metadata stored at the collection level.

        Used by the fingerprint check to read the embedding model info
        that was stored on first ingestion.

        Returns:
            Dict of collection-level metadata. Empty dict if none stored.
        """

    @abstractmethod
    def set_collection_metadata(self, collection_name: str, metadata: dict[str, Any]) -> None:
        """
        Store metadata at the collection level.

        Called on first ingestion to persist the embedding model fingerprint.
        """
