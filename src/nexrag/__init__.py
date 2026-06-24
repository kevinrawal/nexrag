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

from nexrag._factory import wire
from nexrag.core.config.loader import load_config
from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.models.metrics import RunMetrics
from nexrag.core.models.result import PipelineResult
from nexrag.core.observability.runner import EvaluationRunner, NoOpEvaluationRunner
from nexrag.core.pipeline.async_ingestion import AsyncIngestionPipeline
from nexrag.core.pipeline.async_query import AsyncQueryPipeline
from nexrag.core.pipeline.ingestion import IngestionPipeline, IngestionResult
from nexrag.core.pipeline.query import QueryPipeline
from nexrag.exceptions import GuardrailBlockedError, GuardrailError, NexRAGError

__version__ = "0.5.0"
__all__ = [
    "NexRAG",
    "PipelineResult",
    "IngestionResult",
    "RunMetrics",
    "NexRAGError",
    "GuardrailError",
    "GuardrailBlockedError",
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
        retriever: Any | None = None,
        evaluation_runner: EvaluationRunner | NoOpEvaluationRunner | None = None,
    ) -> None:
        self._ingestion = ingestion
        self._query = query
        self._retriever = retriever
        self._evaluation_runner = evaluation_runner

    @classmethod
    def from_config(cls, path: str | Path = "nexrag.yaml") -> NexRAG:
        """
        Load a yaml, resolve all components, wire pipelines.

        Args:
            path: Path to the YAML config file. Resolved relative to CWD.

        Returns:
            A fully wired NexRAG instance ready to ingest and query.

        Raises:
            ConfigError:          If the YAML is missing, invalid, or fails validation.
            ClassResolutionError: If a custom class_path cannot be imported.
        """

        config = load_config(path)
        ingestion, query, retriever, eval_runner = wire(config)
        return cls(
            ingestion=ingestion, query=query, retriever=retriever, evaluation_runner=eval_runner
        )

    def _notify_ingest(self, collection: str) -> None:
        """Notify the retriever to invalidate cache for this collection, if supported."""
        if self._retriever is not None and hasattr(self._retriever, "invalidate_cache"):
            self._retriever.invalidate_cache(collection)

    # Evaluation

    @property
    def evaluation_runner(self) -> EvaluationRunner | NoOpEvaluationRunner | None:
        """The active evaluation runner (may be NoOpEvaluationRunner when disabled)."""
        return self._evaluation_runner

    def add_evaluator(
        self,
        evaluator: Any,
        sample_rate: float = 1.0,
    ) -> None:
        """
        Attach a custom evaluator at runtime.

        The evaluator is added to the active runner and will fire on future
        ``query()`` / ``async_query()`` calls at the given sample rate.

        Requires ``observability.evaluations.enabled: true`` in config (so the
        pipeline starts with a real ``EvaluationRunner``).  If evaluations are
        disabled (the default), call this after enabling them via YAML, or
        upgrade to a real runner by enabling evaluations and reloading config.

        Args:
            evaluator:   Instance of a :class:`~nexrag.core.interfaces.evaluator.BaseEvaluator`
                         subclass.
            sample_rate: Fraction of queries on which to run this evaluator (0.0–1.0).
                         Defaults to 1.0 (every query).

        Raises:
            RuntimeError: If evaluations are disabled (``evaluation_runner`` is a
                          ``NoOpEvaluationRunner``).
        """
        runner = self._evaluation_runner
        if runner is None or isinstance(runner, NoOpEvaluationRunner):
            raise RuntimeError(
                "Evaluations are disabled. Set observability.evaluations.enabled: true "
                "in your config, then call NexRAG.from_config() again."
            )
        runner.add(evaluator, sample_rate)

    # Public pipeline methods

    def ingest(
        self,
        data: Any,
        loader: BaseLoader | None = None,
        metadata: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> IngestionResult:
        """
        Ingest data through the configured loader, chunker, embedder, and vector DB.

        Args:
            data:       Anything the loader accepts — bytes, str, tuple, etc.
            loader:     Optional loader override for this call only.
                        If not given, uses the loader from nexrag.yaml.
            metadata:   Optional metadata merged into every Document after loading.
                        Keys here overwrite loader-set defaults, so you can set
                        the idempotency source and any domain fields in one call:
                        metadata={"source": "contract-456", "tenant": "acme"}
            collection: Override the default collection for this ingest call.
                        Must be one of the collections defined in nexrag.yaml.

        Returns:
            IngestionResult with document count, chunk count, latency, and collection_used.
        """
        result = self._ingestion.ingest(data, loader, metadata=metadata, collection=collection)
        self._notify_ingest(result.collection_used)
        return result

    def ingest_batch(
        self,
        sources: list[Any],
        loader: BaseLoader | None = None,
        metadata: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> list[IngestionResult]:
        """
        Ingest multiple items, one after another.

        Each source is run through a full, independent ``ingest()`` call — loader →
        chunk → embed → fingerprint check → idempotency check → write. Processing
        is **sequential**: items are not parallelised, embeddings are not batched
        across items, and every item pays its own fingerprint read and idempotency
        lookups. This is a convenience wrapper, not a throughput optimisation; for
        large batches into one collection, prefer building Documents yourself and
        calling ``ingest_documents()`` once. (A single-pass batched ingest is
        planned for a future release.)

        Args:
            sources:    Data items to ingest (bytes, str, tuples, etc.).
            loader:     Optional loader override applied to every item in this batch.
            metadata:   Optional metadata merged into every Document for every item.
            collection: Override the default collection for every item in this batch.

        Returns:
            One IngestionResult per source, in the same order.

        Raises:
            PipelineError: Propagated from the first item that fails — items after
                           it are not processed (fail-fast).
        """
        return [
            self.ingest(source, loader=loader, metadata=metadata, collection=collection)
            for source in sources
        ]

    def ingest_documents(
        self,
        documents: list[Any],
        collection: str | None = None,
    ) -> IngestionResult:
        """
        Ingest pre-built Document objects, skipping the loader stage.

        Use this when your data comes from S3, an API, a database, or anywhere
        that doesn't fit a file path. You produce the Documents; NexRAG handles the rest.

        Args:
            documents:  List of nexrag.core.models.document.Document objects.
            collection: Override the default collection for this ingest call.

        Returns:
            IngestionResult with counts, pipeline_id, and collection_used.
        """
        result = self._ingestion.ingest_documents(documents, collection=collection)
        self._notify_ingest(result.collection_used)
        return result

    def query(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        auth_context: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """
        Run the full query pipeline: embed → retrieve → prompt → LLM → result.

        Args:
            text:             The user's question as a plain string.
            collection:       Override the default collection for this query.
            top_k:            Override the top_k retrieval count.
            score_threshold:  Override the minimum similarity score.
            metadata_filter:  Optional key-value metadata filters, e.g. {"year": 2024}.
            auth_context:     Per-request principal info (e.g. {"tenant": "acme"}) for the
                              access-control guard, which turns it into a retrieval filter.

        Returns:
            PipelineResult with answer, sources, scores, and latency.

        Raises:
            GuardrailBlockedError: If an input or output guard blocks the request.
        """
        return self._query.run(
            text,
            collection=collection,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
            auth_context=auth_context,
        )

    def stream_query(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        auth_context: dict[str, Any] | None = None,
    ) -> Iterator[str | RunMetrics]:
        """
        Stream the LLM response token by token (sync).

        All pipeline stages run synchronously. Tokens are yielded as they arrive
        from the LLM. The final item yielded is always a RunMetrics object with
        full per-stage latencies and chunk count.

        Note: when an output guard chain is configured, tokens are buffered and the
        guarded answer is emitted as a single chunk (output guards cannot edit a
        stream that has already been sent).

        Args:
            text:             The user's question.
            collection:       Override the default collection.
            top_k:            Override the top_k retrieval count.
            score_threshold:  Override the minimum similarity score.
            metadata_filter:  Optional metadata filters.
            auth_context:     Per-request principal info for the access-control guard.

        Yields:
            Response text tokens followed by a final RunMetrics object.
            Use isinstance(item, RunMetrics) to separate tokens from metrics.
        """
        return self._query.stream(
            text,
            collection=collection,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
            auth_context=auth_context,
        )

    async def astream_query(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
        auth_context: dict[str, Any] | None = None,
    ) -> AsyncIterator[str | RunMetrics]:
        """
        Stream the LLM response token by token (async).

        With mode: async — pre-LLM stages use native async clients; tokens arrive
        live as they stream from the LLM.

        With mode: sync — the sync stream() runs in a thread pool (non-blocking for
        the event loop) and tokens are yielded after the full stream completes in the
        thread. Configure mode: async for true live token streaming.

        The final item yielded is always a RunMetrics object with full per-stage
        latencies and chunk count.

        Args:
            text:             The user's question.
            collection:       Override the default collection.
            top_k:            Override the top_k retrieval count.
            score_threshold:  Override the minimum similarity score.
            metadata_filter:  Optional metadata filters.

        Yields:
            Response text tokens followed by a final RunMetrics object.
            Use isinstance(item, RunMetrics) to separate tokens from metrics.
        """
        if isinstance(self._query, AsyncQueryPipeline):
            async for token in self._query.astream(
                text,
                collection=collection,
                top_k=top_k,
                score_threshold=score_threshold,
                metadata_filter=metadata_filter,
                auth_context=auth_context,
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
                    auth_context=auth_context,
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
        auth_context: dict[str, Any] | None = None,
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
                auth_context=auth_context,
            )

        return await asyncio.to_thread(
            self._query.run,
            text,
            collection=collection,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
            auth_context=auth_context,
        )

    async def async_ingest(
        self,
        data: Any,
        loader: Any | None = None,
        metadata: dict[str, Any] | None = None,
        collection: str | None = None,
    ) -> IngestionResult:
        """
        Async variant of ingest(). Use inside async frameworks (FastAPI, Starlette).

        When mode: async is configured, embedding runs in parallel batches.
        When mode: sync is configured, runs ingest() via asyncio.to_thread.

        Args:
            data:     Anything the loader accepts.
            loader:   Optional loader override for this call.
            metadata: Optional metadata merged into every Document after loading.

        Returns:
            IngestionResult with document count, chunk count, and latency.
        """
        if isinstance(self._ingestion, AsyncIngestionPipeline):
            result = await self._ingestion.aingest(data, loader, metadata, collection)
        else:
            result = await asyncio.to_thread(
                self._ingestion.ingest, data, loader, metadata, collection
            )
        self._notify_ingest(result.collection_used)
        return result
