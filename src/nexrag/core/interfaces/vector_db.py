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

import asyncio
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

    def get_all(self, collection_name: str, limit: int | None = None) -> list[Chunk]:
        """
        Return all chunks stored in the collection.

        Used by BM25Retriever to build a keyword index over the full corpus.
        Not abstract — existing adapters that don't need BM25 don't have to implement it.

        Args:
            collection_name: Collection to fetch from.
            limit:           Optional cap on number of chunks returned.

        Returns:
            List of Chunk objects (no scores). Empty list if collection is empty.

        Raises:
            NotImplementedError: If the adapter does not implement this method.
        """
        raise NotImplementedError(
            f"{type(self).__name__}.get_all() is not implemented. "
            "Required for BM25/hybrid retrieval (issue #10)."
        )

    async def async_get_all(self, collection_name: str, limit: int | None = None) -> list[Chunk]:
        """Async variant of get_all(). Default: runs sync get_all() in a thread pool."""
        return await asyncio.to_thread(self.get_all, collection_name, limit)

    def list_collections(self) -> list[str]:
        """
        Return all collection names in this vector DB instance.
        Not abstract — existing adapters don't need to implement it immediately.
        Required for multi-collection routing (#12) and async_list_collections().
        """
        raise NotImplementedError(
            f"{type(self).__name__}.list_collections() is not implemented. "
            "Required for multi-collection routing (issue #12)."
        )

    # Async variants — default via asyncio.to_thread. Override with a native async client for true async I/O.

    async def async_upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        collection_name: str,
    ) -> None:
        """Async variant of upsert(). Default: runs sync upsert() in a thread pool."""
        await asyncio.to_thread(self.upsert, chunks, embeddings, collection_name)

    async def async_query(
        self,
        embedding: list[float],
        top_k: int,
        collection_name: str,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """Async variant of query(). Default: runs sync query() in a thread pool."""
        return await asyncio.to_thread(self.query, embedding, top_k, collection_name, filters)

    async def async_list_collections(self) -> list[str]:
        """Async variant of list_collections(). Default: runs sync version in a thread pool."""
        return await asyncio.to_thread(self.list_collections)
