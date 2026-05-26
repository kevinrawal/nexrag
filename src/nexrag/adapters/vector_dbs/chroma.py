"""
ChromaDBAdapter — local in-process ChromaDB vector store.

Supports two modes:
  - persistent (default): data written to disk at the configured path.
    Survives container restarts when the path is a mounted volume.
  - memory: ephemeral in-process store. Fast, no disk I/O. Ideal for tests/CI.

Requires: pip install "nexrag[chromadb]"  (chromadb)
"""

from __future__ import annotations

from typing import Any

from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.models.chunk import Chunk, ScoredChunk
from nexrag.exceptions import VectorDBConnectionError, VectorDBError, VectorDBUpsertError

_NEXRAG_META_KEY = "__nexrag_collection_metadata__"


class ChromaDBAdapter(BaseVectorDB):
    """
    Vector database adapter backed by ChromaDB.

    Args:
        path:  Filesystem path for persistent storage. Relative to CWD.
               Ignored when mode="memory".
        mode:  "persistent" (default) or "memory".
               "memory" uses chromadb.EphemeralClient — data is lost on process exit.
    """

    def __init__(
        self,
        path: str | None = None,
        mode: str = "persistent",
    ) -> None:
        self._path = path
        self._mode = mode
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
        metadatas = [self._serialize_metadata(chunk.metadata) for chunk in chunks]

        try:
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
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

        try:
            if self._mode == "memory":
                return chromadb.EphemeralClient()

            path = self._path or ".nexrag/chroma"
            return chromadb.PersistentClient(path=path)
        except Exception as e:
            raise VectorDBConnectionError(
                f"ChromaDB connection failed (mode={self._mode!r}, path={self._path!r}): {e}",
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
            if isinstance(v, (str, int, float, bool)):
                result[k] = v
            else:
                result[k] = str(v)
        return result

    @staticmethod
    def _build_where(filters: dict[str, Any] | None) -> dict[str, Any]:
        if not filters:
            return {}
        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {key: {"$eq": value}}
        return {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}

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

            chunk = Chunk(
                text=text,
                chunk_index=meta.get("chunk_index", 0),
                total_chunks=meta.get("total_chunks", 1),
                parent_doc_id=meta.get("parent_doc_id", ""),
                metadata=dict(meta.items()),
            )
            scored.append(ScoredChunk(chunk=chunk, score=score, rank=rank))

        return scored
