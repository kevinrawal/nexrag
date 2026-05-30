"""
NexRAG — Framework-agnostic RAG pipeline SDK.

Public API:
    NexRAG.from_config("nexrag.yaml")  →  NexRAG
    pipeline.ingest("resume.pdf")      →  IngestionResult
    pipeline.query("What skills?")     →  PipelineResult

All other symbols are implementation detail. Import from nexrag, not from internals.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.models.result import PipelineResult
from nexrag.core.pipeline.async_ingestion import AsyncIngestionPipeline
from nexrag.core.pipeline.async_query import AsyncQueryPipeline
from nexrag.core.pipeline.ingestion import IngestionPipeline, IngestionResult
from nexrag.core.pipeline.query import QueryPipeline
from nexrag.exceptions import NexRAGError

__version__ = "0.2.0"
__all__ = [
    "NexRAG",
    "PipelineResult",
    "IngestionResult",
    "NexRAGError",
    "__version__",
]


class NexRAG:
    """
    The NexRAG pipeline facade.

    Instantiate via from_config() — do not call __init__ directly.

        pipeline = NexRAG.from_config("nexrag.yaml")
        pipeline.ingest("my_document.pdf")
        result = pipeline.query("What does the document say about X?")
    """

    def __init__(
        self,
        ingestion: IngestionPipeline | AsyncIngestionPipeline,
        query: QueryPipeline | AsyncQueryPipeline,
    ) -> None:
        self._ingestion = ingestion
        self._query = query

    @classmethod
    def from_config(cls, path: str | Path = "nexrag.yaml") -> NexRAG:
        """
        Load nexrag.yaml, resolve all components, wire both pipelines.

        Args:
            path: Path to the YAML config file. Resolved relative to CWD.

        Returns:
            A fully wired NexRAG instance ready to ingest and query.

        Raises:
            ConfigError:          If the YAML is missing, invalid, or fails validation.
            ClassResolutionError: If a custom class_path cannot be imported.
        """
        from nexrag._factory import wire
        from nexrag.core.config.loader import load_config

        config = load_config(path)
        ingestion, query = wire(config)
        return cls(ingestion=ingestion, query=query)

    # Public pipeline methods

    def ingest(self, data: Any, loader: BaseLoader | None = None) -> IngestionResult:
        """
        Ingest data through the configured loader, chunker, embedder, and vector DB.

        Args:
            data:   Anything the loader accepts — file path, bytes, string, etc.
            loader: Optional loader override for this call only.
                    If not given, uses the loader from nexrag.yaml.

        Returns:
            IngestionResult with document count, chunk count, and latency.
        """
        return self._ingestion.ingest(data, loader)

    def ingest_batch(
        self,
        sources: list[str | Path],
        loader: BaseLoader | None = None,
    ) -> list[IngestionResult]:
        """
        Ingest multiple sources in sequence.

        Args:
            sources: File paths or data items to ingest.
            loader:  Optional loader override applied to every item in this batch.

        Returns:
            One IngestionResult per source, in the same order.
        """
        return [self.ingest(source, loader=loader) for source in sources]

    def ingest_documents(self, documents: list[Any]) -> IngestionResult:
        """
        Ingest pre-built Document objects, skipping the loader stage.

        Use this when your data comes from S3, an API, a database, or anywhere
        that doesn't fit a file path. You produce the Documents; NexRAG handles the rest.

        Args:
            documents: List of nexrag.core.models.document.Document objects.

        Returns:
            IngestionResult with counts and pipeline_id.
        """
        return self._ingestion.ingest_documents(documents)

    def query(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """
        Run the full query pipeline: embed → retrieve → prompt → LLM → result.

        Args:
            text:             The user's question as a plain string.
            collection:       Override the default collection for this query.
            top_k:            Override the top_k retrieval count.
            score_threshold:  Override the minimum similarity score.
            metadata_filter:  Optional key-value metadata filters, e.g. {"year": 2024}.

        Returns:
            PipelineResult with answer, sources, scores, and latency.
        """
        return self._query.run(
            text,
            collection=collection,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

    def stream_query(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """
        Stream the LLM response token by token (sync).

        All pipeline stages run synchronously except the LLM generation step,
        which yields tokens as they arrive. Compatible with any sync caller.

        Args:
            text:             The user's question.
            collection:       Override the default collection.
            top_k:            Override the top_k retrieval count.
            score_threshold:  Override the minimum similarity score.
            metadata_filter:  Optional metadata filters.

        Yields:
            Response text tokens as they stream from the LLM.
        """
        return self._query.stream(
            text,
            collection=collection,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

    async def astream_query(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Stream the LLM response token by token (async).

        With mode: async — pre-LLM stages use native async clients; tokens arrive
        live as they stream from the LLM.

        With mode: sync — the sync stream() runs in a thread pool (non-blocking for
        the event loop) and tokens are yielded after the full stream completes in the
        thread. Configure mode: async for true live token streaming.

        Args:
            text:             The user's question.
            collection:       Override the default collection.
            top_k:            Override the top_k retrieval count.
            score_threshold:  Override the minimum similarity score.
            metadata_filter:  Optional metadata filters.

        Yields:
            Response text tokens as they stream from the LLM.
        """
        if isinstance(self._query, AsyncQueryPipeline):
            async for token in self._query.astream(
                text,
                collection=collection,
                top_k=top_k,
                score_threshold=score_threshold,
                metadata_filter=metadata_filter,
            ):
                yield token
        else:
            tokens = await asyncio.to_thread(
                list,
                self._query.stream(
                    text,
                    collection=collection,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    metadata_filter=metadata_filter,
                ),
            )
            for token in tokens:
                yield token

    async def async_query(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """
        Async variant of query(). Use inside async frameworks (FastAPI, Starlette).

        When mode: async is configured, all pipeline stages use native async clients.
        When mode: sync is configured, stages run via asyncio.to_thread (non-blocking
        for the event loop but no true async parallelism).

        Returns:
            PipelineResult with answer, sources, scores, and latency.
        """
        if isinstance(self._query, AsyncQueryPipeline):
            return await self._query.arun(
                text,
                collection=collection,
                top_k=top_k,
                score_threshold=score_threshold,
                metadata_filter=metadata_filter,
            )

        return await asyncio.to_thread(
            self._query.run,
            text,
            collection=collection,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )

    async def async_ingest(
        self,
        data: Any,
        loader: Any | None = None,
    ) -> IngestionResult:
        """
        Async variant of ingest(). Use inside async frameworks (FastAPI, Starlette).

        When mode: async is configured, embedding runs in parallel batches.
        When mode: sync is configured, runs ingest() via asyncio.to_thread.

        Returns:
            IngestionResult with document count, chunk count, and latency.
        """
        if isinstance(self._ingestion, AsyncIngestionPipeline):
            return await self._ingestion.aingest(data, loader)

        return await asyncio.to_thread(self._ingestion.ingest, data, loader)
