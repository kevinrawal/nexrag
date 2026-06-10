"""
IngestionPipeline orchestrates the full ingestion flow.

Two entry points:

    ingest(data, loader?)
        For any data — bytes, dict, list, raw text, or any structure
        your loader accepts.
        A loader must be available: either passed here or set at
        construction. The loader converts data into Documents.
        Raises PipelineError if no loader is configured.

    ingest_documents(documents)
        User has already produced Documents (fetched + parsed externally).
        Skips the loader stage entirely. Pipeline starts from Sanitizer.
        Use this when your data comes from S3, an API, a database, or
        anywhere that doesn't fit a simple file path.

After Documents are in hand, both entry points run the same stages:

    Documents → Sanitizer → Chunker → Embedder
              → FingerprintCheck → IdempotencyCheck → VectorDB
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.interfaces.observer import BaseObserver, NoOpObserver
from nexrag.core.interfaces.sanitizer import BaseSanitizer, PassthroughSanitizer
from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.models.chunk import Chunk
from nexrag.core.models.document import Document
from nexrag.core.models.event import PipelineEvent
from nexrag.core.models.metrics import RunMetrics
from nexrag.exceptions import (
    ChunkError,
    ConfigError,
    EmbedderError,
    EmbedderMismatchError,
    LoaderError,
    PipelineError,
    SanitizerError,
    VectorDBError,
)


class IngestionPipeline:
    """
    Orchestrates the NexRAG ingestion pipeline.

    Args:
        chunker:     Splits documents into chunks.
        embedder:    Produces embedding vectors.
        vector_db:   Stores chunks and vectors.
        collection:  Target collection name.
        loader:      Optional. Required for ingest(). Not needed for ingest_documents().
        sanitizer:   Optional. Defaults to PassthroughSanitizer.
        on_conflict: "overwrite" | "skip" | "append". Defaults to "overwrite".
        observer:    Optional. Defaults to NoOpObserver.
    """

    def __init__(
        self,
        chunker: BaseChunker,
        embedder: BaseEmbedder,
        vector_db: BaseVectorDB,
        collection: str,
        loader: BaseLoader | None = None,
        sanitizer: BaseSanitizer | None = None,
        on_conflict: str = "overwrite",
        observer: BaseObserver | None = None,
        valid_collections: frozenset[str] | None = None,
    ) -> None:
        self._loader = loader
        self._sanitizer = sanitizer or PassthroughSanitizer()
        self._chunker = chunker
        self._embedder = embedder
        self._vector_db = vector_db
        self._collection = collection
        self._on_conflict = on_conflict
        self._observer = observer or NoOpObserver()
        self._valid_collections: frozenset[str] = (
            valid_collections if valid_collections is not None else frozenset([collection])
        )

    # Public API

    def ingest(
        self,
        data: Any,
        loader: BaseLoader | None = None,
        metadata: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> IngestionResult:
        """
        Ingest any data by parsing it through a Loader first.

        The loader receives data as-is. NexRAG takes over from Documents onward.

        Args:
            data:     Anything the loader accepts (bytes, str, tuple, etc.).
            loader:   Optional loader override for this call. Falls back to the
                      loader passed at pipeline construction. If neither is set,
                      raises PipelineError.
            metadata: Optional metadata merged into every Document produced by
                      the loader. Keys in this dict overwrite loader-set defaults,
                      so passing metadata={"source": "contract-456"} correctly
                      sets the idempotency key regardless of loader defaults.
                      Example: {"source": "s3://bucket/file.pdf", "tenant": "acme"}

        Returns:
            IngestionResult with counts and pipeline_id.

        Raises:
            PipelineError: If no loader is configured, or any stage fails.
        """
        pipeline_id = str(uuid.uuid4())
        started_at = time.monotonic()
        active_collection = self._resolve_collection(collection, pipeline_id)

        active_loader = loader or self._loader
        if active_loader is None:
            raise PipelineError(
                "No loader configured. Pass a loader to ingest() or set one "
                "at pipeline construction. Use ingest_documents() to skip the "
                "loader stage entirely.",
                stage="loader",
                component="ingestion",
                pipeline_id=pipeline_id,
            )

        try:
            documents = self._run_loader(active_loader, data, pipeline_id)
            if metadata:
                documents = [doc.with_metadata(metadata) for doc in documents]
            return self._run_from_documents(documents, pipeline_id, started_at, active_collection)
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(
                f"Unexpected error during ingestion: {e}",
                stage="pipeline",
                component="ingestion",
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

    def ingest_documents(
        self,
        documents: list[Document],
        collection: str | None = None,
    ) -> IngestionResult:
        """
        Ingest pre-built Documents, skipping the loader stage entirely.

        Use this when you have already fetched and parsed your data externally —
        from S3, an API, a database, Notion, Confluence, or anywhere else.
        You produce the Documents. NexRAG handles the rest.

        Every Document should have metadata["source"] set to a stable
        identifier — the idempotency check uses it to detect re-ingestion.

        Args:
            documents: Already-built Document objects. Must not be empty.

        Returns:
            IngestionResult with counts and pipeline_id.

        Raises:
            PipelineError: If documents is empty or any stage fails.
        """
        pipeline_id = str(uuid.uuid4())
        started_at = time.monotonic()
        active_collection = self._resolve_collection(collection, pipeline_id)

        if not documents:
            raise PipelineError(
                "ingest_documents() received an empty list. Provide at least one Document.",
                stage="pipeline",
                component="ingestion",
                pipeline_id=pipeline_id,
            )

        try:
            return self._run_from_documents(documents, pipeline_id, started_at, active_collection)
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(
                f"Unexpected error during ingestion: {e}",
                stage="pipeline",
                component="ingestion",
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

    # Shared pipeline from Documents onward

    def _run_from_documents(
        self,
        documents: list[Document],
        pipeline_id: str,
        started_at: float,
        collection: str,
    ) -> IngestionResult:
        """
        Common path for both ingest() and ingest_documents().
        Receives Documents and runs all remaining stages.
        """
        stage_latencies: dict[str, float] = {}
        documents = _stabilise_doc_ids(documents)

        t = time.monotonic()
        chunks = self._run_sanitizer_and_chunker(documents, pipeline_id)
        stage_latencies["chunker"] = (time.monotonic() - t) * 1000

        t = time.monotonic()
        embeddings = self._run_embedder(chunks, pipeline_id)
        stage_latencies["embedder"] = (time.monotonic() - t) * 1000

        t = time.monotonic()
        self._run_fingerprint_check(collection, pipeline_id)
        stage_latencies["fingerprint_check"] = (time.monotonic() - t) * 1000

        t = time.monotonic()
        chunks_to_write, embeddings_to_write = self._run_idempotency_check(
            chunks, embeddings, documents, collection, pipeline_id
        )
        stage_latencies["idempotency_check"] = (time.monotonic() - t) * 1000

        t = time.monotonic()
        written = self._run_vector_db_write(
            chunks_to_write, embeddings_to_write, collection, pipeline_id
        )
        stage_latencies["index_writer"] = (time.monotonic() - t) * 1000

        latency_ms = (time.monotonic() - started_at) * 1000

        metrics = RunMetrics(
            pipeline_id=pipeline_id,
            total_latency_ms=latency_ms,
            stage_latencies=stage_latencies,
            chunks_written=written,
        )
        self._emit(
            pipeline_id,
            "pipeline",
            "completed",
            metadata={
                "total_latency_ms": round(latency_ms, 2),
                "stage_latencies": {k: round(v, 2) for k, v in stage_latencies.items()},
                "chunks_written": written,
            },
        )
        return IngestionResult(
            pipeline_id=pipeline_id,
            documents_loaded=len(documents),
            chunks_produced=len(chunks),
            chunks_written=written,
            latency_ms=latency_ms,
            collection_used=collection,
            metrics=metrics,
        )

    # Stage runners

    def _run_loader(
        self,
        loader: BaseLoader,
        data: Any,
        pipeline_id: str,
    ) -> list[Document]:
        self._emit(pipeline_id, "loader", "started")
        t = time.monotonic()
        try:
            documents = loader.load(data)
        except LoaderError:
            raise
        except Exception as e:
            raise PipelineError(
                f"Loader '{type(loader).__name__}' raised an unexpected error.",
                stage="loader",
                component=type(loader).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

        if not documents:
            raise PipelineError(
                f"Loader '{type(loader).__name__}' returned an empty list. "
                f"Raise LoaderError inside load() if the data cannot be parsed.",
                stage="loader",
                component=type(loader).__name__,
                pipeline_id=pipeline_id,
            )

        self._emit(
            pipeline_id,
            "loader",
            "completed",
            t,
            {"document_count": len(documents)},
        )
        return documents

    def _run_sanitizer_and_chunker(
        self,
        documents: list[Document],
        pipeline_id: str,
    ) -> list[Chunk]:
        all_chunks: list[Chunk] = []

        for document in documents:
            self._emit(pipeline_id, "sanitizer", "started")
            t = time.monotonic()
            try:
                clean_doc = self._sanitizer.sanitize(document)
            except SanitizerError:
                raise
            except Exception as e:
                raise PipelineError(
                    f"Sanitizer failed on document '{document.doc_id}'.",
                    stage="sanitizer",
                    component=type(self._sanitizer).__name__,
                    pipeline_id=pipeline_id,
                    cause=e,
                ) from e
            self._emit(pipeline_id, "sanitizer", "completed", t)

            self._emit(pipeline_id, "chunker", "started")
            t = time.monotonic()
            try:
                chunks = self._chunker.chunk(clean_doc)
            except ChunkError:
                raise
            except Exception as e:
                raise PipelineError(
                    f"Chunker failed on document '{document.doc_id}'.",
                    stage="chunker",
                    component=type(self._chunker).__name__,
                    pipeline_id=pipeline_id,
                    cause=e,
                ) from e
            self._emit(
                pipeline_id,
                "chunker",
                "completed",
                t,
                {"chunk_count": len(chunks)},
            )
            all_chunks.extend(chunks)

        return all_chunks

    def _run_embedder(
        self,
        chunks: list[Chunk],
        pipeline_id: str,
    ) -> list[list[float]]:
        self._emit(pipeline_id, "embedder", "started")
        t = time.monotonic()
        try:
            embeddings = self._embedder.embed([chunk.text for chunk in chunks])
        except EmbedderError:
            raise
        except Exception as e:
            raise PipelineError(
                "Embedder failed during batch embedding.",
                stage="embedder",
                component=type(self._embedder).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

        if len(embeddings) != len(chunks):
            raise PipelineError(
                f"Embedder returned {len(embeddings)} vectors for {len(chunks)} chunks. "
                f"Must return exactly one vector per input text.",
                stage="embedder",
                component=type(self._embedder).__name__,
                pipeline_id=pipeline_id,
            )

        self._emit(
            pipeline_id,
            "embedder",
            "completed",
            t,
            {
                "model": self._embedder.model_name,
                "chunk_count": len(chunks),
                "dimensions": len(embeddings[0]) if embeddings else 0,
            },
        )
        return embeddings

    # TODO: can we optimize this part?
    # instead of collecting all metadata["source"] and then querying the DB for each,
    # can we do a single query with a $in filter on source?
    def _run_fingerprint_check(self, collection: str, pipeline_id: str) -> None:
        self._emit(pipeline_id, "fingerprint_check", "started")
        t = time.monotonic()

        try:
            stored = self._vector_db.get_collection_metadata(collection)
        except VectorDBError:
            raise

        current_fingerprint = _compute_fingerprint(
            self._embedder.model_name, self._embedder.dimensions
        )

        if not stored:
            self._vector_db.set_collection_metadata(
                collection,
                {
                    "embedding_model": self._embedder.model_name,
                    "embedding_dimensions": self._embedder.dimensions,
                    "fingerprint": current_fingerprint,
                },
            )
        else:
            stored_fingerprint = stored.get("fingerprint")
            if stored_fingerprint and stored_fingerprint != current_fingerprint:
                raise EmbedderMismatchError(
                    stored_model=stored.get("embedding_model", "unknown"),
                    configured_model=self._embedder.model_name,
                    collection=collection,
                    stage="fingerprint_check",
                    pipeline_id=pipeline_id,
                )

        self._emit(pipeline_id, "fingerprint_check", "completed", t)

    def _run_idempotency_check(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        documents: list[Document],
        collection: str,
        pipeline_id: str,
    ) -> tuple[list[Chunk], list[list[float]]]:
        """
        Apply on_conflict rules per document source.

        Uses metadata["source"] from each Document to identify existing chunks.
        Documents without metadata["source"] are always written (no dedup possible).
        """
        self._emit(pipeline_id, "idempotency_check", "started")
        t = time.monotonic()

        if self._on_conflict == "append":
            self._emit(
                pipeline_id,
                "idempotency_check",
                "completed",
                t,
                {"action": "append", "chunks_to_write": len(chunks)},
            )
            return chunks, embeddings

        # Collect all source identifiers from the documents.
        sources = {doc.metadata.get("source") for doc in documents if doc.metadata.get("source")}

        if not sources:
            # No sources set — cannot do dedup. Write everything.
            self._emit(
                pipeline_id,
                "idempotency_check",
                "completed",
                t,
                {
                    "action": "insert",
                    "reason": "no source in metadata",
                    "chunks_to_write": len(chunks),
                },
            )
            return chunks, embeddings

        # For each source, fetch existing chunk IDs by metadata (not by vector similarity).
        existing_hashes: set[str] = set()
        for source in sources:
            try:
                existing_ids = self._vector_db.get_ids_by_metadata(
                    filters={"source": source},
                    collection_name=collection,
                )
                existing_hashes.update(existing_ids)
            except VectorDBError:
                existing_hashes = set()

        incoming_hashes = {chunk.content_hash for chunk in chunks}

        if self._on_conflict == "skip":
            if existing_hashes:
                self._emit(
                    pipeline_id,
                    "idempotency_check",
                    "completed",
                    t,
                    {"action": "skipped", "reason": "source already exists"},
                )
                return [], []
            self._emit(
                pipeline_id,
                "idempotency_check",
                "completed",
                t,
                {"action": "insert", "chunks_to_write": len(chunks)},
            )
            return chunks, embeddings

        # on_conflict == "overwrite"
        if existing_hashes == incoming_hashes:
            self._emit(
                pipeline_id,
                "idempotency_check",
                "completed",
                t,
                {"action": "skipped", "reason": "all hashes match"},
            )
            return [], []

        if existing_hashes:
            try:
                self._vector_db.delete(list(existing_hashes), collection)
            except VectorDBError:
                raise

        self._emit(
            pipeline_id,
            "idempotency_check",
            "completed",
            t,
            {"action": "overwrite", "chunks_to_write": len(chunks)},
        )
        return chunks, embeddings

    def _run_vector_db_write(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        collection: str,
        pipeline_id: str,
    ) -> int:
        if not chunks:
            return 0

        self._emit(pipeline_id, "index_writer", "started")
        t = time.monotonic()
        try:
            self._vector_db.upsert(chunks, embeddings, collection)
        except VectorDBError:
            raise
        except Exception as e:
            raise PipelineError(
                "VectorDB write failed.",
                stage="index_writer",
                component=type(self._vector_db).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e
        self._emit(
            pipeline_id,
            "index_writer",
            "completed",
            t,
            {"chunks_written": len(chunks)},
        )
        return len(chunks)

    # Helpers

    def _resolve_collection(self, collection: str | None, pipeline_id: str) -> str:
        if collection is None:
            return self._collection
        if collection not in self._valid_collections:
            raise ConfigError(
                f"Collection '{collection}' is not configured. "
                f"Available: {sorted(self._valid_collections)}. "
                "Add it to vector_db.collections in nexrag.yaml.",
                stage="config",
                component="ingestion",
            )
        return collection

    def _emit(
        self,
        pipeline_id: str,
        stage: str,
        status: str,
        started: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        duration_ms = (time.monotonic() - started) * 1000 if started else 0.0
        event = PipelineEvent(
            pipeline_id=pipeline_id,
            stage=stage,  # type: ignore[arg-type]
            status=status,  # type: ignore[arg-type]
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._observer.emit(event)


# Result object


@dataclass(frozen=True)
class IngestionResult:
    """
    Returned by IngestionPipeline.ingest() and ingest_documents().

    Attributes:
        pipeline_id:      Unique ID for this run — correlate with logs.
        documents_loaded: How many Documents entered the pipeline.
        chunks_produced:  Total chunks after chunking all documents.
        chunks_written:   Chunks actually written (0 if skipped by idempotency).
        latency_ms:       Total wall-clock time for the full pipeline.
        collection_used:  Which vector collection was written to.
    """

    pipeline_id: str
    documents_loaded: int
    chunks_produced: int
    chunks_written: int
    latency_ms: float
    collection_used: str = ""
    metrics: RunMetrics | None = field(default=None)

    def __repr__(self) -> str:
        return (
            f"IngestionResult(pipeline_id={self.pipeline_id!r}, "
            f"docs={self.documents_loaded}, chunks={self.chunks_produced}, "
            f"written={self.chunks_written}, latency_ms={self.latency_ms:.1f}, "
            f"collection={self.collection_used!r})"
        )


# Utilities


def _compute_fingerprint(model_name: str, dimensions: int) -> str:
    """Stable hash of model identity. Detects embedding model changes."""
    return hashlib.sha256(f"{model_name}:{dimensions}".encode()).hexdigest()


def _stabilise_doc_ids(documents: list[Document]) -> list[Document]:
    """
    Derive a deterministic doc_id from metadata["source"] when available.

    Without this, Document generates a fresh UUID on every instantiation, so
    parent_doc_id stored in chunks changes on every re-ingest of the same file.
    A stable doc_id means parent_doc_id in ChromaDB is consistent across ingestions,
    which makes per-document chunk queries and citation attribution reliable.

    Documents without a source keep their random UUID — they are ephemeral by
    design (no source = no idempotency, always written fresh).
    """
    out: list[Document] = []
    for doc in documents:
        source = doc.metadata.get("source")
        if source:
            stable_id = hashlib.sha256(str(source).encode()).hexdigest()[:32]
            if doc.doc_id != stable_id:
                doc = Document(content=doc.content, metadata=doc.metadata, doc_id=stable_id)
        out.append(doc)
    return out
