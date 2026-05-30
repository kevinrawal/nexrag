"""
AsyncIngestionPipeline — async version of IngestionPipeline.

Enabled by setting `mode: async` in nexrag.yaml. The embedding step runs in
parallel batches (main async win). The DB operations (fingerprint check,
idempotency check, write) are serialized per collection via a narrow asyncio.Lock
to prevent TOCTOU races between concurrent ingestions.

Lock scope is intentionally narrow:
    - Loader, sanitizer, chunker, and embedder stages run WITHOUT the lock.
    - Only the fingerprint check → idempotency check → upsert sequence is locked.

This means:
    - Concurrent ingestions into the SAME collection embed in parallel; DB ops serialize.
    - Concurrent ingestions into DIFFERENT collections are fully parallel end-to-end.
    - CPU-bound adapters (HuggingFace) won't gain true parallelism due to Python's GIL.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import defaultdict
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
from nexrag.core.pipeline.ingestion import IngestionResult, _compute_fingerprint
from nexrag.exceptions import (
    ChunkError,
    EmbedderError,
    EmbedderMismatchError,
    LoaderError,
    NexRAGError,
    PipelineError,
    SanitizerError,
    VectorDBError,
)

_DEFAULT_EMBED_BATCH_SIZE = 50


class AsyncIngestionPipeline:
    """
    Async orchestrator for the NexRAG ingestion pipeline.

    Args:
        chunker:          Splits documents into chunks.
        embedder:         Produces embedding vectors (async_embed called in parallel batches).
        vector_db:        Stores chunks and vectors (async_upsert used).
        collection:       Target collection name.
        loader:           Optional. Required for aingest(). Not needed for aingest_documents().
        sanitizer:        Optional. Defaults to PassthroughSanitizer.
        on_conflict:      "overwrite" | "skip" | "append". Defaults to "overwrite".
        observer:         Optional. Defaults to NoOpObserver.
        embed_batch_size: How many chunks to embed per parallel batch. Default: 50.
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
        embed_batch_size: int = _DEFAULT_EMBED_BATCH_SIZE,
    ) -> None:
        self._loader = loader
        self._sanitizer = sanitizer or PassthroughSanitizer()
        self._chunker = chunker
        self._embedder = embedder
        self._vector_db = vector_db
        self._collection = collection
        self._on_conflict = on_conflict
        self._observer = observer or NoOpObserver()
        self._embed_batch_size = embed_batch_size
        # Per-collection locks: serializes only the DB read-check + write sequence.
        self._collection_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def ingest(
        self,
        data: Any,
        loader: BaseLoader | None = None,
    ) -> IngestionResult:
        """
        Sync entry point — wraps aingest() in asyncio.run() for non-async callers.

        Raises NexRAGError if called from inside a running event loop; use
        async_ingest() / aingest() there instead.
        """
        self._assert_no_running_loop()
        return asyncio.run(self.aingest(data, loader))

    def ingest_documents(self, documents: list[Document]) -> IngestionResult:
        """Sync entry point for pre-built Documents — wraps aingest_documents()."""
        self._assert_no_running_loop()
        return asyncio.run(self.aingest_documents(documents))

    async def aingest(
        self,
        data: Any,
        loader: BaseLoader | None = None,
    ) -> IngestionResult:
        """
        Async ingest: parse data through a loader, then run the pipeline.

        Args:
            data:   Anything the loader accepts.
            loader: Optional loader override for this call.

        Returns:
            IngestionResult with counts and pipeline_id.
        """
        pipeline_id = str(uuid.uuid4())
        started_at = time.monotonic()

        active_loader = loader or self._loader
        if active_loader is None:
            raise PipelineError(
                "No loader configured. Pass a loader to aingest() or set one "
                "at pipeline construction. Use aingest_documents() to skip the loader stage.",
                stage="loader",
                component="async_ingestion",
                pipeline_id=pipeline_id,
            )

        try:
            documents = await self._run_loader(active_loader, data, pipeline_id)
            return await self._run_from_documents(documents, pipeline_id, started_at)
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(
                f"Unexpected error during async ingestion: {e}",
                stage="pipeline",
                component="async_ingestion",
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

    async def aingest_documents(
        self,
        documents: list[Document],
    ) -> IngestionResult:
        """
        Async ingest pre-built Documents, skipping the loader stage.

        Args:
            documents: Already-built Document objects. Must not be empty.

        Returns:
            IngestionResult with counts and pipeline_id.
        """
        pipeline_id = str(uuid.uuid4())
        started_at = time.monotonic()

        if not documents:
            raise PipelineError(
                "aingest_documents() received an empty list. Provide at least one Document.",
                stage="pipeline",
                component="async_ingestion",
                pipeline_id=pipeline_id,
            )

        try:
            return await self._run_from_documents(documents, pipeline_id, started_at)
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(
                f"Unexpected error during async ingestion: {e}",
                stage="pipeline",
                component="async_ingestion",
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

    # Shared pipeline from Documents onward

    async def _run_from_documents(
        self,
        documents: list[Document],
        pipeline_id: str,
        started_at: float,
    ) -> IngestionResult:
        # Stages before DB: run WITHOUT the lock (safe, pure compute or read-only).
        chunks = await self._run_sanitizer_and_chunker(documents, pipeline_id)
        embeddings = await self._run_embedder(chunks, pipeline_id)

        # Narrow lock: only the DB read-check + write sequence is serialized per collection.
        async with self._collection_locks[self._collection]:
            await self._run_fingerprint_check(pipeline_id)
            chunks_to_write, embeddings_to_write = await self._run_idempotency_check(
                chunks, embeddings, documents, pipeline_id
            )
            written = await self._run_vector_db_write(
                chunks_to_write, embeddings_to_write, pipeline_id
            )

        latency_ms = (time.monotonic() - started_at) * 1000
        return IngestionResult(
            pipeline_id=pipeline_id,
            documents_loaded=len(documents),
            chunks_produced=len(chunks),
            chunks_written=written,
            latency_ms=latency_ms,
        )

    # Stage runners

    async def _run_loader(
        self,
        loader: BaseLoader,
        data: Any,
        pipeline_id: str,
    ) -> list[Document]:
        await self._emit(pipeline_id, "loader", "started")
        t = time.monotonic()
        try:
            documents = await asyncio.to_thread(loader.load, data)
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
                f"Loader '{type(loader).__name__}' returned an empty list.",
                stage="loader",
                component=type(loader).__name__,
                pipeline_id=pipeline_id,
            )

        await self._emit(pipeline_id, "loader", "completed", t, {"document_count": len(documents)})
        return documents

    async def _run_sanitizer_and_chunker(
        self,
        documents: list[Document],
        pipeline_id: str,
    ) -> list[Chunk]:
        all_chunks: list[Chunk] = []

        for document in documents:
            await self._emit(pipeline_id, "sanitizer", "started")
            t = time.monotonic()
            try:
                clean_doc = await asyncio.to_thread(self._sanitizer.sanitize, document)
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
            await self._emit(pipeline_id, "sanitizer", "completed", t)

            await self._emit(pipeline_id, "chunker", "started")
            t = time.monotonic()
            try:
                chunks = await asyncio.to_thread(self._chunker.chunk, clean_doc)
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
            await self._emit(pipeline_id, "chunker", "completed", t, {"chunk_count": len(chunks)})
            all_chunks.extend(chunks)

        return all_chunks

    async def _run_embedder(
        self,
        chunks: list[Chunk],
        pipeline_id: str,
    ) -> list[list[float]]:
        """
        Embed chunks in parallel batches — the main async performance win.
        Each batch runs as a separate async task via async_embed().
        For cloud API adapters, this means N concurrent API calls instead of N sequential ones.
        """
        await self._emit(pipeline_id, "embedder", "started")
        t = time.monotonic()

        texts = [chunk.text for chunk in chunks]
        batch_size = self._embed_batch_size
        batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]

        try:
            batch_results = await asyncio.gather(
                *[self._embedder.async_embed(batch) for batch in batches]
            )
        except EmbedderError:
            raise
        except Exception as e:
            raise PipelineError(
                "Embedder failed during async batch embedding.",
                stage="embedder",
                component=type(self._embedder).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

        embeddings = [emb for batch in batch_results for emb in batch]

        if len(embeddings) != len(chunks):
            raise PipelineError(
                f"Embedder returned {len(embeddings)} vectors for {len(chunks)} chunks.",
                stage="embedder",
                component=type(self._embedder).__name__,
                pipeline_id=pipeline_id,
            )

        await self._emit(
            pipeline_id,
            "embedder",
            "completed",
            t,
            {"chunk_count": len(chunks), "dimensions": len(embeddings[0]) if embeddings else 0},
        )
        return embeddings

    async def _run_fingerprint_check(self, pipeline_id: str) -> None:
        await self._emit(pipeline_id, "fingerprint_check", "started")
        t = time.monotonic()

        try:
            stored = await asyncio.to_thread(
                self._vector_db.get_collection_metadata, self._collection
            )
        except VectorDBError:
            raise

        current_fingerprint = _compute_fingerprint(
            self._embedder.model_name, self._embedder.dimensions
        )

        if not stored:
            await asyncio.to_thread(
                self._vector_db.set_collection_metadata,
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

        await self._emit(pipeline_id, "fingerprint_check", "completed", t)

    async def _run_idempotency_check(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        documents: list[Document],
        pipeline_id: str,
    ) -> tuple[list[Chunk], list[list[float]]]:
        await self._emit(pipeline_id, "idempotency_check", "started")
        t = time.monotonic()

        if self._on_conflict == "append":
            await self._emit(
                pipeline_id,
                "idempotency_check",
                "completed",
                t,
                {"action": "append", "chunks_to_write": len(chunks)},
            )
            return chunks, embeddings

        sources = {doc.metadata.get("source") for doc in documents if doc.metadata.get("source")}

        if not sources:
            await self._emit(
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

        existing_hashes: set[str] = set()
        for source in sources:
            try:
                existing = await self._vector_db.async_query(
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
                await self._emit(
                    pipeline_id,
                    "idempotency_check",
                    "completed",
                    t,
                    {"action": "skipped", "reason": "source already exists"},
                )
                return [], []
            await self._emit(
                pipeline_id,
                "idempotency_check",
                "completed",
                t,
                {"action": "insert", "chunks_to_write": len(chunks)},
            )
            return chunks, embeddings

        # on_conflict == "overwrite"
        if existing_hashes == incoming_hashes:
            await self._emit(
                pipeline_id,
                "idempotency_check",
                "completed",
                t,
                {"action": "skipped", "reason": "all hashes match"},
            )
            return [], []

        if existing_hashes:
            try:
                await asyncio.to_thread(
                    self._vector_db.delete, list(existing_hashes), self._collection
                )
            except VectorDBError:
                raise

        await self._emit(
            pipeline_id,
            "idempotency_check",
            "completed",
            t,
            {"action": "overwrite", "chunks_to_write": len(chunks)},
        )
        return chunks, embeddings

    async def _run_vector_db_write(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        pipeline_id: str,
    ) -> int:
        if not chunks:
            return 0

        await self._emit(pipeline_id, "index_writer", "started")
        t = time.monotonic()
        try:
            await self._vector_db.async_upsert(chunks, embeddings, self._collection)
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
        await self._emit(
            pipeline_id, "index_writer", "completed", t, {"chunks_written": len(chunks)}
        )
        return len(chunks)

    # Helpers

    async def _emit(
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
        await self._observer.async_emit(event)

    def _assert_no_running_loop(self) -> None:
        try:
            asyncio.get_running_loop()
            raise NexRAGError(
                "This method cannot be called from inside a running event loop. "
                "Use 'await nexrag.async_ingest()' inside async contexts."
            )
        except RuntimeError:
            pass
