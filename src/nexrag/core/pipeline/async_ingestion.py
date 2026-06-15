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

from nexrag.core.guards.apply import apply_ingestion_guards
from nexrag.core.guards.chain import GuardChain
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
from nexrag.core.pipeline.ingestion import IngestionResult, _compute_fingerprint, _stabilise_doc_ids
from nexrag.exceptions import (
    ChunkError,
    ConfigError,
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
        valid_collections: frozenset[str] | None = None,
        ingestion_guards: GuardChain | None = None,
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
        self._valid_collections: frozenset[str] = (
            valid_collections if valid_collections is not None else frozenset([collection])
        )
        self._ingestion_guards = ingestion_guards
        # Per-collection locks: serializes only the DB read-check + write sequence.
        self._collection_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    def ingest(
        self,
        data: Any,
        loader: BaseLoader | None = None,
        metadata: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> IngestionResult:
        """
        Sync entry point — wraps aingest() in asyncio.run() for non-async callers.

        Raises NexRAGError if called from inside a running event loop; use
        async_ingest() / aingest() there instead.
        """
        self._assert_no_running_loop()
        return asyncio.run(self.aingest(data, loader, metadata, collection))

    def ingest_documents(
        self, documents: list[Document], collection: str | None = None
    ) -> IngestionResult:
        """Sync entry point for pre-built Documents — wraps aingest_documents()."""
        self._assert_no_running_loop()
        return asyncio.run(self.aingest_documents(documents, collection))

    async def aingest(
        self,
        data: Any,
        loader: BaseLoader | None = None,
        metadata: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> IngestionResult:
        """
        Async ingest: parse data through a loader, then run the pipeline.

        Args:
            data:     Anything the loader accepts.
            loader:   Optional loader override for this call.
            metadata: Optional metadata merged into every Document after loading.
                      Keys here overwrite loader-set defaults.
                      Example: {"source": "contract-456", "tenant": "acme"}

        Returns:
            IngestionResult with counts and pipeline_id.
        """
        pipeline_id = str(uuid.uuid4())
        started_at = time.monotonic()
        active_collection = self._resolve_collection(collection, pipeline_id)

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
            if metadata:
                documents = [doc.with_metadata(metadata) for doc in documents]
            return await self._run_from_documents(
                documents, pipeline_id, started_at, active_collection
            )
        except Exception as e:
            await self._emit_failed(pipeline_id, "pipeline", started_at, e)
            if isinstance(e, PipelineError):
                raise
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
        collection: str | None = None,
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
        active_collection = self._resolve_collection(collection, pipeline_id)

        if not documents:
            raise PipelineError(
                "aingest_documents() received an empty list. Provide at least one Document.",
                stage="pipeline",
                component="async_ingestion",
                pipeline_id=pipeline_id,
            )

        try:
            return await self._run_from_documents(
                documents, pipeline_id, started_at, active_collection
            )
        except Exception as e:
            await self._emit_failed(pipeline_id, "pipeline", started_at, e)
            if isinstance(e, PipelineError):
                raise
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
        collection: str,
    ) -> IngestionResult:
        stage_latencies: dict[str, float] = {}
        documents = _stabilise_doc_ids(documents)

        # Stages before DB: run WITHOUT the lock (safe, pure compute or read-only).
        t = time.monotonic()
        chunks = await self._run_sanitizer_and_chunker(documents, pipeline_id)
        stage_latencies["chunker"] = (time.monotonic() - t) * 1000

        t = time.monotonic()
        embeddings = await self._run_embedder(chunks, pipeline_id)
        stage_latencies["embedder"] = (time.monotonic() - t) * 1000

        # Narrow lock: only the DB read-check + write sequence is serialized per collection.
        async with self._collection_locks[collection]:
            t = time.monotonic()
            await self._run_fingerprint_check(collection, pipeline_id)
            stage_latencies["fingerprint_check"] = (time.monotonic() - t) * 1000

            t = time.monotonic()
            chunks_to_write, embeddings_to_write = await self._run_idempotency_check(
                chunks, embeddings, documents, collection, pipeline_id
            )
            stage_latencies["idempotency_check"] = (time.monotonic() - t) * 1000

            t = time.monotonic()
            written = await self._run_vector_db_write(
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
        await self._emit(
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
        except Exception as e:
            await self._emit_failed(pipeline_id, "loader", t, e)
            if isinstance(e, LoaderError):
                raise
            raise PipelineError(
                f"Loader '{type(loader).__name__}' raised an unexpected error.",
                stage="loader",
                component=type(loader).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

        if not documents:
            err = PipelineError(
                f"Loader '{type(loader).__name__}' returned an empty list.",
                stage="loader",
                component=type(loader).__name__,
                pipeline_id=pipeline_id,
            )
            await self._emit_failed(pipeline_id, "loader", t, err)
            raise err

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
            except Exception as e:
                await self._emit_failed(pipeline_id, "sanitizer", t, e)
                if isinstance(e, SanitizerError):
                    raise
                raise PipelineError(
                    f"Sanitizer failed on document '{document.doc_id}'.",
                    stage="sanitizer",
                    component=type(self._sanitizer).__name__,
                    pipeline_id=pipeline_id,
                    cause=e,
                ) from e
            await self._emit(pipeline_id, "sanitizer", "completed", t)

            clean_doc = apply_ingestion_guards(
                self._ingestion_guards, clean_doc, pipeline_id=pipeline_id
            )

            await self._emit(pipeline_id, "chunker", "started")
            t = time.monotonic()
            try:
                chunks = await asyncio.to_thread(self._chunker.chunk, clean_doc)
            except Exception as e:
                await self._emit_failed(pipeline_id, "chunker", t, e)
                if isinstance(e, ChunkError):
                    raise
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
        except Exception as e:
            await self._emit_failed(pipeline_id, "embedder", t, e)
            if isinstance(e, EmbedderError):
                raise
            raise PipelineError(
                "Embedder failed during async batch embedding.",
                stage="embedder",
                component=type(self._embedder).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

        embeddings = [emb for batch in batch_results for emb in batch]

        if len(embeddings) != len(chunks):
            err = PipelineError(
                f"Embedder returned {len(embeddings)} vectors for {len(chunks)} chunks.",
                stage="embedder",
                component=type(self._embedder).__name__,
                pipeline_id=pipeline_id,
            )
            await self._emit_failed(pipeline_id, "embedder", t, err)
            raise err

        await self._emit(
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

    async def _run_fingerprint_check(self, collection: str, pipeline_id: str) -> None:
        await self._emit(pipeline_id, "fingerprint_check", "started")
        t = time.monotonic()

        current_fingerprint = _compute_fingerprint(
            self._embedder.model_name, self._embedder.dimensions
        )

        try:
            stored = await asyncio.to_thread(self._vector_db.get_collection_metadata, collection)

            if not stored:
                await asyncio.to_thread(
                    self._vector_db.set_collection_metadata,
                    collection,
                    {
                        "embedding_model": self._embedder.model_name,
                        "embedding_dimensions": self._embedder.dimensions,
                        "fingerprint": current_fingerprint,
                    },
                )
                # Best-effort compare-and-set: re-read and confirm OUR fingerprint
                # won. The per-collection lock serializes tasks in THIS process, but
                # a separate process racing the first ingest with a different embedder
                # is not covered — detect that here rather than silently mixing models.
                stored = await asyncio.to_thread(
                    self._vector_db.get_collection_metadata, collection
                )

            stored_fingerprint = stored.get("fingerprint")
            if stored_fingerprint and stored_fingerprint != current_fingerprint:
                raise EmbedderMismatchError(
                    stored_model=stored.get("embedding_model", "unknown"),
                    configured_model=self._embedder.model_name,
                    collection=collection,
                    stage="fingerprint_check",
                    pipeline_id=pipeline_id,
                )
        except Exception as e:
            await self._emit_failed(pipeline_id, "fingerprint_check", t, e)
            raise

        await self._emit(pipeline_id, "fingerprint_check", "completed", t)

    async def _run_idempotency_check(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        documents: list[Document],
        collection: str,
        pipeline_id: str,
    ) -> tuple[list[Chunk], list[list[float]]]:
        """
        Apply on_conflict rules independently per document source.

        Same per-source semantics as the sync pipeline: each source's skip /
        overwrite decision is made from its own existing rows only, a per-source
        VectorDB lookup failure never clears the others' state (the source is
        written and a failed event emitted), and chunks without metadata["source"]
        are always written.
        """
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

        # Bucket (chunk, embedding) pairs by source, preserving order. The "" key
        # holds chunks without a source — always written (no dedup possible).
        buckets: dict[str, list[tuple[Chunk, list[float]]]] = {}
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            source = chunk.metadata.get("source") or ""
            buckets.setdefault(source, []).append((chunk, embedding))

        chunks_to_write: list[Chunk] = []
        embeddings_to_write: list[list[float]] = []
        actions: dict[str, int] = {"insert": 0, "overwrite": 0, "skipped": 0, "failed": 0}

        for source, pairs in buckets.items():
            src_chunks = [c for c, _ in pairs]
            src_embeddings = [e for _, e in pairs]

            if not source:
                chunks_to_write.extend(src_chunks)
                embeddings_to_write.extend(src_embeddings)
                actions["insert"] += 1
                continue

            try:
                existing_ids = set(
                    await self._vector_db.async_get_ids_by_metadata(
                        filters={"source": source},
                        collection_name=collection,
                    )
                )
            except VectorDBError as e:
                await self._emit_failed(pipeline_id, "idempotency_check", t, e)
                chunks_to_write.extend(src_chunks)
                embeddings_to_write.extend(src_embeddings)
                actions["failed"] += 1
                continue

            incoming_ids = {c.row_id for c in src_chunks}

            if self._on_conflict == "skip":
                if existing_ids:
                    actions["skipped"] += 1
                    continue
                chunks_to_write.extend(src_chunks)
                embeddings_to_write.extend(src_embeddings)
                actions["insert"] += 1
                continue

            # on_conflict == "overwrite"
            if existing_ids == incoming_ids:
                actions["skipped"] += 1
                continue
            if existing_ids:
                try:
                    await asyncio.to_thread(self._vector_db.delete, list(existing_ids), collection)
                except VectorDBError as e:
                    await self._emit_failed(pipeline_id, "idempotency_check", t, e)
                    raise
            chunks_to_write.extend(src_chunks)
            embeddings_to_write.extend(src_embeddings)
            actions["overwrite"] += 1

        await self._emit(
            pipeline_id,
            "idempotency_check",
            "completed",
            t,
            {
                "sources": len(buckets),
                "sources_inserted": actions["insert"],
                "sources_overwritten": actions["overwrite"],
                "sources_skipped": actions["skipped"],
                "sources_failed_lookup": actions["failed"],
                "chunks_to_write": len(chunks_to_write),
            },
        )
        return chunks_to_write, embeddings_to_write

    async def _run_vector_db_write(
        self,
        chunks: list[Chunk],
        embeddings: list[list[float]],
        collection: str,
        pipeline_id: str,
    ) -> int:
        if not chunks:
            return 0

        await self._emit(pipeline_id, "index_writer", "started")
        t = time.monotonic()
        try:
            await self._vector_db.async_upsert(chunks, embeddings, collection)
        except Exception as e:
            await self._emit_failed(pipeline_id, "index_writer", t, e)
            if isinstance(e, VectorDBError):
                raise
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

    def _resolve_collection(self, collection: str | None, pipeline_id: str) -> str:
        if collection is None:
            return self._collection
        if collection not in self._valid_collections:
            raise ConfigError(
                f"Collection '{collection}' is not configured. "
                f"Available: {sorted(self._valid_collections)}. "
                "Add it to vector_db.collections in nexrag.yaml.",
                stage="config",
                component="async_ingestion",
            )
        return collection

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

    async def _emit_failed(
        self,
        pipeline_id: str,
        stage: str,
        started: float | None,
        exc: BaseException,
    ) -> None:
        """Emit a terminal 'failed' event for a stage, with error context."""
        await self._emit(
            pipeline_id,
            stage,
            "failed",
            started,
            {"error_type": type(exc).__name__, "message": str(exc)},
        )

    def _assert_no_running_loop(self) -> None:
        try:
            asyncio.get_running_loop()
            raise NexRAGError(
                "This method cannot be called from inside a running event loop. "
                "Use 'await nexrag.async_ingest()' inside async contexts."
            )
        except RuntimeError:
            pass
