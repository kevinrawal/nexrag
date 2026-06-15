"""
PineconeVectorDB — Pinecone serverless vector store.

Mapping: each NexRAG *collection* is a Pinecone *namespace* inside a single index.
The index is created lazily on the first upsert, using the embedder's vector
dimension (so no dimension needs to be declared in config).

The embedding-model fingerprint (written by the ingestion fingerprint check, which
runs before any upsert) is stored as one record per collection in a reserved
namespace ("__nexrag_meta__"), keeping the data namespaces' counts clean.

Config (under vector_db.params):
    index_name: required — the Pinecone index name.
    api_key:    Pinecone API key. If omitted, reads PINECONE_API_KEY from env.
    cloud:      Serverless cloud. Default "aws".
    region:     Serverless region. Default "us-east-1".
    metric:     Distance metric. Default "cosine".

Note: Pinecone metadata is capped at ~40 KB per vector; chunk text is stored in
metadata so very large chunks may exceed it. Use smaller chunks with Pinecone.

Requires: pip install "nexrag[pinecone]"  (pinecone)
"""

from __future__ import annotations

import os
import time
from typing import Any

from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.models.chunk import Chunk, ScoredChunk
from nexrag.exceptions import VectorDBConnectionError, VectorDBError, VectorDBUpsertError

# Reserved namespace holding one fingerprint record per collection.
_META_NAMESPACE = "__nexrag_meta__"
# Metadata key under which each vector stores its chunk text (Pinecone has no
# separate "document" field like ChromaDB).
_TEXT_KEY = "__nexrag_text__"

# Chunk struct fields stored in every vector's metadata, stripped back out when
# reconstructing Chunk objects so chunk.metadata stays document/user-only.
_CHUNK_STRUCT_KEYS: frozenset[str] = frozenset({"chunk_index", "total_chunks", "parent_doc_id"})

# Pinecone caps query top_k / page sizes; used for the metadata-scan idempotency path.
_MAX_FILTER_RESULTS = 10000


class PineconeVectorDB(BaseVectorDB):
    """
    Vector database adapter backed by Pinecone (serverless). Collection = namespace.

    Args:
        index_name:        Pinecone index name (shared by all collections/namespaces).
        api_key:           Pinecone API key. If None, reads PINECONE_API_KEY from env.
        cloud:             Serverless cloud provider. Default "aws".
        region:            Serverless region. Default "us-east-1".
        metric:            Distance metric. Default "cosine".
        upsert_batch_size: Max vectors per upsert call. Default 100.
        query_batch_size:  Reserved for future multi-query batching. Default 100.
        max_retries:       Connection attempts before raising. Default 3.
        retry_delay:       Base seconds between retries (doubles each attempt). Default 1.0.
    """

    def __init__(
        self,
        index_name: str,
        api_key: str | None = None,
        cloud: str = "aws",
        region: str = "us-east-1",
        metric: str = "cosine",
        upsert_batch_size: int = 100,
        query_batch_size: int = 100,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        self._index_name = index_name
        self._cloud = cloud
        self._region = region
        self._metric = metric
        self._upsert_batch_size = max(1, min(upsert_batch_size, 1000))
        self._query_batch_size = query_batch_size
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._client: Any = self._connect(api_key)
        self._index: Any = None  # resolved lazily
        self._dimension: int | None = None

    # BaseVectorDB implementation

    def upsert(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        collection_name: str,
    ) -> None:
        if not chunks:
            return

        # Deduplicate by row_id (document-scoped) — same as the ChromaDB adapter.
        seen: set[str] = set()
        deduped_chunks: list[Chunk] = []
        deduped_embeddings: list[list[float]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if chunk.row_id not in seen:
                seen.add(chunk.row_id)
                deduped_chunks.append(chunk)
                deduped_embeddings.append(embedding)

        index = self._get_index(dimension=len(deduped_embeddings[0]))

        vectors = [
            {
                "id": chunk.row_id,
                "values": embedding,
                "metadata": self._serialize_metadata(
                    {
                        **chunk.metadata,
                        _TEXT_KEY: chunk.text,
                        "chunk_index": chunk.chunk_index,
                        "total_chunks": chunk.total_chunks,
                        "parent_doc_id": chunk.parent_doc_id,
                    }
                ),
            }
            for chunk, embedding in zip(deduped_chunks, deduped_embeddings, strict=True)
        ]

        try:
            batch = self._upsert_batch_size
            for i in range(0, len(vectors), batch):
                index.upsert(vectors=vectors[i : i + batch], namespace=collection_name)
        except Exception as e:
            raise VectorDBUpsertError(
                f"Pinecone upsert failed for collection '{collection_name}': {e}",
                stage="index_writer",
                component="PineconeVectorDB",
                cause=e,
            ) from e

    def query(
        self,
        embedding: list[float],
        top_k: int,
        collection_name: str,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        index = self._get_index()
        if index is None:
            return []

        try:
            response = index.query(
                vector=embedding,
                top_k=top_k,
                namespace=collection_name,
                filter=self._build_filter(filters),
                include_metadata=True,
            )
        except Exception as e:
            raise VectorDBError(
                f"Pinecone query failed on collection '{collection_name}': {e}",
                stage="retriever",
                component="PineconeVectorDB",
                cause=e,
            ) from e

        scored: list[ScoredChunk] = []
        for rank, match in enumerate(self._matches(response), start=1):
            meta = dict(self._match_metadata(match))
            scored.append(
                ScoredChunk(
                    chunk=self._chunk_from_metadata(meta),
                    score=float(self._match_score(match)),
                    rank=rank,
                )
            )
        return scored

    def delete(self, ids: list[str], collection_name: str) -> None:
        if not ids:
            return
        index = self._get_index()
        if index is None:
            return
        try:
            index.delete(ids=ids, namespace=collection_name)
        except Exception as e:
            raise VectorDBError(
                f"Pinecone delete failed on collection '{collection_name}': {e}",
                stage="index_writer",
                component="PineconeVectorDB",
                cause=e,
            ) from e

    def count(self, collection_name: str) -> int:
        index = self._get_index()
        if index is None:
            return 0
        try:
            stats = index.describe_index_stats()
        except Exception as e:
            raise VectorDBError(
                f"Pinecone count failed on collection '{collection_name}': {e}",
                stage="pipeline",
                component="PineconeVectorDB",
                cause=e,
            ) from e
        ns = self._namespaces(stats).get(collection_name)
        return int(self._vector_count(ns)) if ns is not None else 0

    def get_ids_by_metadata(self, filters: dict[str, Any], collection_name: str) -> list[str]:
        index = self._get_index()
        if index is None:
            return []
        dimension = self._get_dimension()
        if dimension is None:
            return []
        # Pinecone has no "scan by filter" API; query with a unit vector + the metadata
        # filter returns matching ids (similarity is irrelevant here). Capped at _MAX_FILTER_RESULTS.
        probe = [0.0] * dimension
        probe[0] = 1.0
        try:
            response = index.query(
                vector=probe,
                top_k=_MAX_FILTER_RESULTS,
                namespace=collection_name,
                filter=self._build_filter(filters),
                include_metadata=False,
                include_values=False,
            )
        except Exception as e:
            raise VectorDBError(
                f"Pinecone get_ids_by_metadata failed on collection '{collection_name}': {e}",
                stage="idempotency_check",
                component="PineconeVectorDB",
                cause=e,
            ) from e
        return [self._match_id(m) for m in self._matches(response)]

    def get_all(
        self, collection_name: str, limit: int | None = None, offset: int | None = None
    ) -> list[Chunk]:
        index = self._get_index()
        if index is None:
            return []

        try:
            ids = self._list_ids(index, collection_name, limit=limit, offset=offset)
            if not ids:
                return []
            chunks: list[Chunk] = []
            for i in range(0, len(ids), self._query_batch_size):
                fetched = index.fetch(
                    ids=ids[i : i + self._query_batch_size], namespace=collection_name
                )
                for vec in self._fetched_vectors(fetched).values():
                    meta = self._vector_metadata(vec)
                    if meta:
                        chunks.append(self._chunk_from_metadata(dict(meta)))
            return chunks
        except Exception as e:
            raise VectorDBError(
                f"Pinecone get_all failed on collection '{collection_name}': {e}",
                stage="retriever",
                component="PineconeVectorDB",
                cause=e,
            ) from e

    def list_collections(self) -> list[str]:
        index = self._get_index()
        if index is None:
            return []
        try:
            stats = index.describe_index_stats()
        except Exception as e:
            raise VectorDBError(
                f"Pinecone list_collections failed: {e}",
                stage="pipeline",
                component="PineconeVectorDB",
                cause=e,
            ) from e
        return [name for name in self._namespaces(stats) if name != _META_NAMESPACE]

    # TODO: rename to get_nexrag_metadata — fetches only NexRAG's fingerprint vector from _META_NAMESPACE, not all Pinecone index metadata
    def get_collection_metadata(self, collection_name: str) -> dict[str, Any]:
        """Return NexRAG's internal fingerprint metadata stored as a vector in _META_NAMESPACE. Not a general Pinecone index metadata accessor."""
        index = self._get_index()
        if index is None:
            return {}
        try:
            fetched = index.fetch(ids=[collection_name], namespace=_META_NAMESPACE)
        except Exception:
            return {}
        vectors = self._fetched_vectors(fetched)
        vec = vectors.get(collection_name)
        if vec is None:
            return {}
        return dict(self._vector_metadata(vec) or {})

    # TODO: rename to set_nexrag_metadata — upserts only NexRAG's fingerprint vector into _META_NAMESPACE, not a general Pinecone index metadata setter
    def set_collection_metadata(self, collection_name: str, metadata: dict[str, Any]) -> None:
        """Persist NexRAG's internal fingerprint metadata as a vector in _META_NAMESPACE. May create the Pinecone index on demand using embedding_dimensions from metadata."""
        # Runs during the fingerprint check, before any data upsert — so the index may
        # not exist yet. The fingerprint payload carries embedding_dimensions, which we
        # use to create the index on demand.
        dimension = metadata.get("embedding_dimensions")
        index = self._get_index(dimension=dimension if isinstance(dimension, int) else None)
        if index is None:
            raise VectorDBError(
                "Cannot persist collection metadata: Pinecone index does not exist and no "
                "embedding dimension was available to create it.",
                stage="fingerprint_check",
                component="PineconeVectorDB",
            )
        placeholder = [0.0] * self._require_dimension()
        placeholder[0] = 1.0
        try:
            index.upsert(
                vectors=[
                    {
                        "id": collection_name,
                        "values": placeholder,
                        "metadata": self._serialize_metadata(metadata),
                    }
                ],
                namespace=_META_NAMESPACE,
            )
        except Exception as e:
            raise VectorDBError(
                f"Pinecone failed to set collection metadata on '{collection_name}': {e}",
                stage="fingerprint_check",
                component="PineconeVectorDB",
                cause=e,
            ) from e

    # Private helpers

    def _connect(self, api_key: str | None) -> Any:
        try:
            from pinecone import Pinecone
        except ImportError as e:
            raise VectorDBConnectionError(
                "pinecone package is required for PineconeVectorDB. "
                'Install it: pip install "nexrag[pinecone]" or pip install pinecone',
                stage="pipeline",
                component="PineconeVectorDB",
                cause=e,
            ) from e

        key = api_key or os.environ.get("PINECONE_API_KEY")
        if not key:
            raise VectorDBConnectionError(
                "Pinecone API key is required. Set vector_db.params.api_key or the "
                "PINECONE_API_KEY environment variable.",
                stage="pipeline",
                component="PineconeVectorDB",
            )

        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                return Pinecone(api_key=key)
            except Exception as exc:  # noqa: BLE001 — retried below, re-raised if exhausted
                last_exc = exc
                if attempt < self._max_retries - 1:
                    time.sleep(self._retry_delay * (2**attempt))
        raise VectorDBConnectionError(
            f"Failed to connect to Pinecone: {last_exc}",
            stage="pipeline",
            component="PineconeVectorDB",
            cause=last_exc,
        )

    def _get_index(self, dimension: int | None = None) -> Any:
        """Return the index handle, creating the index if needed and a dimension is known."""
        if self._index is not None:
            return self._index

        if self._index_exists():
            self._index = self._client.Index(self._index_name)
            return self._index

        if dimension is None:
            return None  # nothing ingested yet and no dimension to create with

        self._create_index(dimension)
        self._dimension = dimension
        self._index = self._client.Index(self._index_name)
        return self._index

    def _index_exists(self) -> bool:
        try:
            return bool(self._client.has_index(self._index_name))
        except AttributeError:
            names = [getattr(i, "name", i) for i in self._client.list_indexes()]
            return self._index_name in names

    def _create_index(self, dimension: int) -> None:
        from pinecone import ServerlessSpec

        try:
            self._client.create_index(
                name=self._index_name,
                dimension=dimension,
                metric=self._metric,
                spec=ServerlessSpec(cloud=self._cloud, region=self._region),
            )
        except Exception as e:
            # Tolerate a concurrent creation that already made the index.
            if self._index_exists():
                return
            raise VectorDBConnectionError(
                f"Pinecone failed to create index '{self._index_name}': {e}",
                stage="pipeline",
                component="PineconeVectorDB",
                cause=e,
            ) from e
        self._wait_until_ready()

    def _wait_until_ready(self, timeout: float = 60.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                desc = self._client.describe_index(self._index_name)
            except Exception:  # noqa: BLE001 — transient during provisioning
                time.sleep(1.0)
                continue
            if self._index_ready(desc):
                return
            time.sleep(1.0)

    def _get_dimension(self) -> int | None:
        if self._dimension is not None:
            return self._dimension
        try:
            desc = self._client.describe_index(self._index_name)
        except Exception:
            return None
        dim = getattr(desc, "dimension", None)
        if dim is None and isinstance(desc, dict):
            dim = desc.get("dimension")
        if isinstance(dim, int):
            self._dimension = dim
        return self._dimension

    def _require_dimension(self) -> int:
        dim = self._get_dimension()
        if dim is None:
            raise VectorDBError(
                "Pinecone index dimension is unknown.",
                stage="fingerprint_check",
                component="PineconeVectorDB",
            )
        return dim

    def _list_ids(
        self, index: Any, namespace: str, limit: int | None, offset: int | None
    ) -> list[str]:
        ids: list[str] = []
        skip = offset or 0
        for page in index.list(namespace=namespace):
            # index.list yields a page (list of ids) per iteration.
            page_ids = page if isinstance(page, list) else [page]
            for vid in page_ids:
                if skip > 0:
                    skip -= 1
                    continue
                ids.append(vid)
                if limit is not None and len(ids) >= limit:
                    return ids
        return ids

    @staticmethod
    def _serialize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
        """Pinecone metadata values must be str/number/bool/list[str] — flatten others."""
        result: dict[str, Any] = {}
        for k, v in metadata.items():
            if isinstance(v, str | int | float | bool):
                result[k] = v
            elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                result[k] = v
            else:
                result[k] = str(v)
        return result

    @staticmethod
    def _build_filter(filters: dict[str, Any] | None) -> dict[str, Any] | None:
        """
        Build a Pinecone metadata filter. Scalars wrap with $eq; operator dicts pass
        through. Mirrors the ChromaDB adapter's semantics.
        """
        if not filters:
            return None

        def _wrap(v: Any) -> Any:
            return v if isinstance(v, dict) else {"$eq": v}

        if len(filters) == 1:
            key, value = next(iter(filters.items()))
            return {key: _wrap(value)}
        return {"$and": [{k: _wrap(v)} for k, v in filters.items()]}

    @classmethod
    def _chunk_from_metadata(cls, meta: dict[str, Any]) -> Chunk:
        text = meta.pop(_TEXT_KEY, "")
        chunk_index = meta.get("chunk_index", 0)
        total_chunks = meta.get("total_chunks", 1)
        parent_doc_id = meta.get("parent_doc_id", "")
        clean = {k: v for k, v in meta.items() if k not in _CHUNK_STRUCT_KEYS}
        return Chunk(
            text=str(text),
            chunk_index=int(chunk_index),
            total_chunks=int(total_chunks),
            parent_doc_id=str(parent_doc_id),
            metadata=clean,
        )

    @staticmethod
    def _index_ready(desc: Any) -> bool:
        status = getattr(desc, "status", None)
        if status is None and isinstance(desc, dict):
            status = desc.get("status")
        if status is None:
            return False
        ready = getattr(status, "ready", None)
        if ready is None and isinstance(status, dict):
            ready = status.get("ready")
        return bool(ready)

    # Response shape adapters — Pinecone SDK objects support both attribute and item access.

    @staticmethod
    def _matches(response: Any) -> list[Any]:
        matches = getattr(response, "matches", None)
        if matches is None and isinstance(response, dict):
            matches = response.get("matches")
        return list(matches or [])

    @staticmethod
    def _match_metadata(match: Any) -> dict[str, Any]:
        meta = getattr(match, "metadata", None)
        if meta is None and isinstance(match, dict):
            meta = match.get("metadata")
        return meta or {}

    @staticmethod
    def _match_score(match: Any) -> float:
        score = getattr(match, "score", None)
        if score is None and isinstance(match, dict):
            score = match.get("score")
        return float(score or 0.0)

    @staticmethod
    def _match_id(match: Any) -> str:
        vid = getattr(match, "id", None)
        if vid is None and isinstance(match, dict):
            vid = match.get("id")
        return str(vid)

    @staticmethod
    def _namespaces(stats: Any) -> dict[str, Any]:
        ns = getattr(stats, "namespaces", None)
        if ns is None and isinstance(stats, dict):
            ns = stats.get("namespaces")
        return dict(ns or {})

    @staticmethod
    def _vector_count(ns: Any) -> int:
        count = getattr(ns, "vector_count", None)
        if count is None and isinstance(ns, dict):
            count = ns.get("vector_count")
        return int(count or 0)

    @staticmethod
    def _fetched_vectors(fetched: Any) -> dict[str, Any]:
        vectors = getattr(fetched, "vectors", None)
        if vectors is None and isinstance(fetched, dict):
            vectors = fetched.get("vectors")
        return dict(vectors or {})

    @staticmethod
    def _vector_metadata(vec: Any) -> dict[str, Any]:
        meta = getattr(vec, "metadata", None)
        if meta is None and isinstance(vec, dict):
            meta = vec.get("metadata")
        return meta or {}
