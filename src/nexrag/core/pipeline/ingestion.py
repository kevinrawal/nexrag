"""
IngestionPipeline orchestrates the full ingestion flow.

Two entry points:

    ingest(data, loader?)
        For any data — file path, bytes, dict, list, raw text.
        If loader is provided, uses it to parse data into Documents.
        If loader is None, data must be a file path string and the
        pipeline uses the loader configured in nexrag.yaml.

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
from dataclasses import dataclass
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
from nexrag.exceptions import (
    ChunkError,
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
    ) -> None:
        self._loader = loader
        self._sanitizer = sanitizer or PassthroughSanitizer()
        self._chunker = chunker
        self._embedder = embedder
        self._vector_db = vector_db
        self._collection = collection
        self._on_conflict = on_conflict
        self._observer = observer or NoOpObserver()

    # Public API

    def ingest(
        self,
        data: Any,
        loader: BaseLoader | None = None,
    ) -> IngestionResult:
        """
        Ingest any data by parsing it through a Loader first.

        The loader receives data as-is — file path, bytes, dict, list,
        raw text, or any other type the loader accepts.
        The loader's job is to parse it into Documents.
        NexRAG takes over from Documents onward.

        Args:
            data:   Anything the loader accepts.
            loader: Optional loader override for this call.
                    Falls back to the loader passed at pipeline construction.
                    If neither is set, raises PipelineError.

        Returns:
            IngestionResult with counts and pipeline_id.

        Raises:
            PipelineError: If no loader is configured, or any stage fails.
        """
        pipeline_id = str(uuid.uuid4())
        started_at = time.monotonic()

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
            return self._run_from_documents(documents, pipeline_id, started_at)
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

    # TODO: can we combine this with the above ingest() method and just detect if the first argument is Documents or raw data?
    # Maybe cleaner to keep separate for now, since they have different requirements and error cases.
    def ingest_documents(
        self,
        documents: list[Document],
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

        if not documents:
            raise PipelineError(
                "ingest_documents() received an empty list. " "Provide at least one Document.",
                stage="pipeline",
                component="ingestion",
                pipeline_id=pipeline_id,
            )

        try:
            return self._run_from_documents(documents, pipeline_id, started_at)
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
    ) -> IngestionResult:
        """
        Common path for both ingest() and ingest_documents().
        Receives Documents and runs all remaining stages.
        """
        chunks = self._run_sanitizer_and_chunker(documents, pipeline_id)
        embeddings = self._run_embedder(chunks, pipeline_id)
        self._run_fingerprint_check(pipeline_id)
        chunks_to_write, embeddings_to_write = self._run_idempotency_check(
            chunks, embeddings, documents, pipeline_id
        )
        written = self._run_vector_db_write(chunks_to_write, embeddings_to_write, pipeline_id)

        latency_ms = (time.monotonic() - started_at) * 1000
        return IngestionResult(
            pipeline_id=pipeline_id,
            documents_loaded=len(documents),
            chunks_produced=len(chunks),
            chunks_written=written,
            latency_ms=latency_ms,
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
                "chunk_count": len(chunks),
                "dimensions": len(embeddings[0]) if embeddings else 0,
            },
        )
        return embeddings

    def _run_fingerprint_check(self, pipeline_id: str) -> None:
        self._emit(pipeline_id, "fingerprint_check", "started")
        t = time.monotonic()

        try:
            stored = self._vector_db.get_collection_metadata(self._collection)
        except VectorDBError:
            raise

        current_fingerprint = _compute_fingerprint(
            self._embedder.model_name, self._embedder.dimensions
        )

        if not stored:
            self._vector_db.set_collection_metadata(
                self._collection,
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
                    collection=self._collection,
                    stage="fingerprint_check",
                    pipeline_id=pipeline_id,
                )

        self._emit(pipeline_id, "fingerprint_check", "completed", t)

    def _run_idempotency_check(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        documents: list[Document],
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

        # For each source, find existing chunks in the DB.
        existing_hashes: set[str] = set()
        for source in sources:
            try:
                existing = self._vector_db.query(
                    embedding=embeddings[0],
                    top_k=10_000,
                    collection_name=self._collection,
                    filters={"source": source},
                )
                existing_hashes.update(sc.chunk.content_hash for sc in existing)
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
                self._vector_db.delete(list(existing_hashes), self._collection)
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
        pipeline_id: str,
    ) -> int:
        if not chunks:
            return 0

        self._emit(pipeline_id, "index_writer", "started")
        t = time.monotonic()
        try:
            self._vector_db.upsert(chunks, embeddings, self._collection)
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
    """

    pipeline_id: str
    documents_loaded: int
    chunks_produced: int
    chunks_written: int
    latency_ms: float

    def __repr__(self) -> str:
        return (
            f"IngestionResult(pipeline_id={self.pipeline_id!r}, "
            f"docs={self.documents_loaded}, chunks={self.chunks_produced}, "
            f"written={self.chunks_written}, latency_ms={self.latency_ms:.1f})"
        )


# Utilities


def _compute_fingerprint(model_name: str, dimensions: int) -> str:
    """Stable hash of model identity. Detects embedding model changes."""
    return hashlib.sha256(f"{model_name}:{dimensions}".encode()).hexdigest()
