"""
ChromaDBAdapter — ChromaDB vector store, three modes.

  - persistent (default): data written to disk at the configured path.
    Survives container restarts when the path is a mounted volume.
  - memory: ephemeral in-process store. Fast, no disk I/O. Ideal for tests/CI.
  - server: connects to a remote ChromaDB HTTP server via HttpClient.
    Requires host (and optionally port, default 8000).

Production deployment:
    Use mode="server" with a dedicated ChromaDB container. Example nexrag.yaml:

        vector_db:
          provider: chroma
          default_collection: docs
          collections:
            docs:
              mode: server
              host: chroma.internal
              port: 8000

Requires: pip install "nexrag[chromadb]"  (chromadb)
"""

from __future__ import annotations

import time
from typing import Any

from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.models.chunk import Chunk, ScoredChunk
from nexrag.exceptions import VectorDBConnectionError, VectorDBError, VectorDBUpsertError

_NEXRAG_META_KEY = "__nexrag_collection_metadata__"

# Chunk struct fields stored in every ChromaDB metadata row.
# Stripped back out when reconstructing Chunk objects so chunk.metadata
# stays clean (document + user metadata only, no structural noise).
_CHUNK_STRUCT_KEYS: frozenset[str] = frozenset({"chunk_index", "total_chunks", "parent_doc_id"})


class ChromaDBAdapter(BaseVectorDB):
    """
    Vector database adapter backed by ChromaDB.

    Args:
        path:              Filesystem path for persistent storage. Relative to CWD.
                           Ignored when mode is not "persistent".
        mode:              "persistent" (default), "memory", or "server".
                           "memory"  — EphemeralClient, data lost on exit.
                           "server"  — HttpClient connecting to host:port.
        host:              Remote ChromaDB hostname. Required when mode="server".
        port:              Remote ChromaDB port. Default 8000.
        upsert_batch_size: Max chunks per ChromaDB upsert call. Default 500.
        query_batch_size:  Reserved for future multi-query batching. Default 100.
        max_retries:       Connection attempts before raising. Default 3.
        retry_delay:       Base seconds between retries (doubles each attempt). Default 1.0.
    """

    def __init__(
        self,
        path: str | None = None,
        mode: str = "persistent",
        host: str | None = None,
        port: int | None = None,
        upsert_batch_size: int = 500,
        query_batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._path = path
        self._mode = mode
        self._host = host
        self._port = port
        self._upsert_batch_size = upsert_batch_size
        self._query_batch_size = query_batch_size
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client: Any = self._connect()

    # BaseVectorDB implementation

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        collection_name: str,
    ) -> None:
        if not chunks:
            return

        # Deduplicate by content_hash — same text must not appear twice in one batch.
        seen: set[str] = set()
        deduped_chunks: list[Chunk] = []
        deduped_embeddings: list[list[float]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if chunk.content_hash not in seen:
                seen.add(chunk.content_hash)
                deduped_chunks.append(chunk)
                deduped_embeddings.append(embedding)
        chunks, embeddings = deduped_chunks, deduped_embeddings

        collection = self._get_or_create(collection_name)

        ids = [chunk.content_hash for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        # Merge struct fields into stored metadata so they survive the round-trip.
        # chunk.metadata only contains document/user fields; struct fields are
        # separate dataclass attributes that would otherwise be lost in ChromaDB.
        metadatas = [
            self._serialize_metadata(
                {
                    **chunk.metadata,
                    "chunk_index": chunk.chunk_index,
                    "total_chunks": chunk.total_chunks,
                    "parent_doc_id": chunk.parent_doc_id,
                }
            )
            for chunk in chunks
        ]

        try:
            batch = self._upsert_batch_size
            for i in range(0, len(ids), batch):
                collection.upsert(
                    ids=ids[i : i + batch],
                    embeddings=embeddings[i : i + batch],
                    documents=documents[i : i + batch],
                    metadatas=metadatas[i : i + batch],
                )
        except Exception as e:
            raise VectorDBUpsertError(
                f"ChromaDB upsert failed for collection '{collection_name}': {e}",
                stage="index_writer",
                component="ChromaDBAdapter",
                cause=e,
            ) from e

    def query(
        self,
        embedding: list[float],
        top_k: int,
        collection_name: str,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        collection = self._get_or_create(collection_name)

        where = self._build_where(filters)

        try:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=min(top_k, max(1, collection.count())),
                where=where if where else None,
                include=["documents", "metadatas", "distances", "embeddings"],
            )
        except Exception as e:
            raise VectorDBError(
                f"ChromaDB query failed on collection '{collection_name}': {e}",
                stage="retriever",
                component="ChromaDBAdapter",
                cause=e,
            ) from e

        return self._build_scored_chunks(results)

    def delete(self, ids: list[str], collection_name: str) -> None:
        if not ids:
            return
        collection = self._get_or_create(collection_name)
        try:
            collection.delete(ids=ids)
        except Exception as e:
            raise VectorDBError(
                f"ChromaDB delete failed on collection '{collection_name}': {e}",
                stage="index_writer",
                component="ChromaDBAdapter",
                cause=e,
            ) from e

    def count(self, collection_name: str) -> int:
        try:
            return int(self._get_or_create(collection_name).count())
        except Exception as e:
            raise VectorDBError(
                f"ChromaDB count failed on collection '{collection_name}': {e}",
                stage="pipeline",
                component="ChromaDBAdapter",
                cause=e,
            ) from e

    def list_collections(self) -> list[str]:
        try:
            return [c.name for c in self._client.list_collections()]
        except Exception as e:
            raise VectorDBError(
                f"ChromaDB list_collections failed: {e}",
                stage="pipeline",
                component="ChromaDBAdapter",
                cause=e,
            ) from e

    def get_collection_metadata(self, collection_name: str) -> dict[str, Any]:
        collection = self._get_or_create(collection_name)
        meta = collection.metadata or {}
        raw = meta.get(_NEXRAG_META_KEY, "")
        if not raw:
            return {}
        try:
            import json

            return json.loads(raw)  # type: ignore[no-any-return]
        except Exception:
            return {}

    def set_collection_metadata(self, collection_name: str, metadata: dict[str, Any]) -> None:
        collection = self._get_or_create(collection_name)
        try:
            import json

            # Exclude hnsw: prefixed keys — ChromaDB rejects modifying distance function.
            existing = {
                k: v for k, v in (collection.metadata or {}).items() if not k.startswith("hnsw:")
            }
            existing[_NEXRAG_META_KEY] = json.dumps(metadata)
            collection.modify(metadata=existing)
        except Exception as e:
            raise VectorDBError(
                f"ChromaDB failed to set collection metadata on '{collection_name}': {e}",
                stage="fingerprint_check",
                component="ChromaDBAdapter",
                cause=e,
            ) from e

    # Private helpers

    def _connect(self) -> Any:
        try:
            import chromadb
        except ImportError as e:
            raise VectorDBConnectionError(
                "chromadb package is required for ChromaDBAdapter. "
                'Install it: pip install "nexrag[chromadb]" or pip install chromadb',
                stage="pipeline",
                component="ChromaDBAdapter",
                cause=e,
            ) from e

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return self._connect_once(chromadb)
            except VectorDBConnectionError as exc:
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2**attempt))

        raise last_exc  # type: ignore[misc]

    def _connect_once(self, chromadb: Any) -> Any:
        try:
            if self._mode == "memory":
                return chromadb.EphemeralClient()

            if self._mode == "server":
                host = self._host or "localhost"
                port = self._port or 8000
                return chromadb.HttpClient(host=host, port=port)

            path = self._path or ".nexrag/chroma"
            return chromadb.PersistentClient(path=path)
        except Exception as e:
            raise VectorDBConnectionError(
                f"ChromaDB connection failed (mode={self._mode!r}, host={self._host!r}, path={self._path!r}): {e}",
                stage="pipeline",
                component="ChromaDBAdapter",
                cause=e,
            ) from e

    def _get_or_create(self, name: str) -> Any:
        try:
            return self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorDBConnectionError(
                f"Failed to get or create ChromaDB collection '{name}': {e}",
                stage="pipeline",
                component="ChromaDBAdapter",
                cause=e,
            ) from e

    @staticmethod
    def _serialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """ChromaDB metadata values must be str/int/float/bool — flatten others."""
        result: dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, str | int | float | bool):
                result[k] = v
            else:
                result[k] = str(v)
        return result

    @staticmethod
    def _build_where(filters: dict[str, Any] | None) -> dict[str, Any]:
        """
        Build a ChromaDB where clause from a filter dict.

        Scalar values are wrapped with $eq. Operator dicts are passed through
        unchanged, enabling range and list operators:
            {"year": 2024}                  → {"year": {"$eq": 2024}}
            {"year": {"$gte": 2023}}        → {"year": {"$gte": 2023}}
            {"source": {"$in": ["a","b"]}}  → {"source": {"$in": ["a","b"]}}

        Supported ChromaDB operators: $eq $ne $gt $gte $lt $lte $in $nin
        """
        if not filters:
            return {}

        def _wrap(v: Any) -> Any:
            return v if isinstance(v, dict) else {"$eq": v}

        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {key: _wrap(value)}
        return {"$and": [{k: _wrap(v)} for k, v in filters.items()]}

    @staticmethod
    def _build_scored_chunks(results: dict[str, Any]) -> list[ScoredChunk]:
        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        scored: list[ScoredChunk] = []
        for rank, (text, meta, dist) in enumerate(
            zip(documents, metadatas, distances, strict=True), start=1
        ):
            # ChromaDB cosine distance → similarity score (0-1, higher = more similar).
            score = max(0.0, 1.0 - dist)

            # meta is None when the row was stored without metadata (ChromaDB behaviour).
            meta = meta or {}
            chunk = Chunk(
                text=text,
                chunk_index=meta.get("chunk_index", 0),
                total_chunks=meta.get("total_chunks", 1),
                parent_doc_id=meta.get("parent_doc_id", ""),
                # Strip struct keys — keep chunk.metadata as document/user metadata only.
                metadata={k: v for k, v in meta.items() if k not in _CHUNK_STRUCT_KEYS},
            )
            scored.append(ScoredChunk(chunk=chunk, score=score, rank=rank))

        return scored
