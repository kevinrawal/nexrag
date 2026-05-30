"""
AsyncQueryPipeline — async version of QueryPipeline.

Enabled by setting `mode: async` in nexrag.yaml. All stage helpers call
async ABC variants, allowing true non-blocking I/O with native async clients
(AsyncOpenAI, AsyncAnthropic) and non-blocking event loop use in async frameworks.

Concurrent reads (multiple simultaneous queries) are safe — no locking needed.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.core.interfaces.llm import BaseLLM
from nexrag.core.interfaces.observer import BaseObserver, NoOpObserver
from nexrag.core.interfaces.prompt_builder import BasePromptBuilder
from nexrag.core.interfaces.retriever import BaseRetriever
from nexrag.core.models.chunk import ScoredChunk
from nexrag.core.models.event import PipelineEvent
from nexrag.core.models.result import PipelineResult, Source
from nexrag.exceptions import (
    EmbedderError,
    LLMError,
    NexRAGError,
    PipelineError,
    PromptError,
    RetrieverError,
)


class AsyncQueryPipeline:
    """
    Async orchestrator for the NexRAG query pipeline.

    Stages (in order):
        1. Embedder       — async embeds the user query string → vector
        2. Retriever      — async semantic search → list[ScoredChunk]
        3. PromptBuilder  — assembles prompt (CPU-bound, runs in thread)
        4. LLM            — async generates response from prompt
        5. ResponseBuilder— wraps everything into a PipelineResult

    Args:
        embedder:        Embeds the user query (async_embed_query used).
        retriever:       Retrieves relevant chunks (async_retrieve used).
        prompt_builder:  Assembles the final prompt string.
        llm:             Generates the answer (async_generate used).
        collection:      Which vector collection to query.
        top_k:           Maximum chunks to retrieve. Default: 5.
        score_threshold: Minimum similarity score for retrieved chunks. Default: 0.0.
        observer:        Optional. Defaults to NoOpObserver.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        retriever: BaseRetriever,
        prompt_builder: BasePromptBuilder,
        llm: BaseLLM,
        collection: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        observer: BaseObserver | None = None,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._collection = collection
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._observer = observer or NoOpObserver()

    def run(
        self,
        query: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """
        Sync entry point — wraps arun() in asyncio.run() for non-async callers.

        Raises NexRAGError if called from inside a running event loop; use
        async_query() / arun() there instead.
        """
        self._assert_no_running_loop()
        return asyncio.run(
            self.arun(
                query,
                collection=collection,
                top_k=top_k,
                score_threshold=score_threshold,
                metadata_filter=metadata_filter,
            )
        )

    def stream(
        self,
        query: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        """
        Sync streaming — collects tokens from astream() via asyncio.run().

        Note: tokens are buffered until the full stream completes before yielding.
        For true live token streaming, use astream_query() with mode: async.
        Raises NexRAGError if called from inside a running event loop.
        """
        self._assert_no_running_loop()

        async def _collect() -> list[str]:
            return [
                t
                async for t in self.astream(
                    query,
                    collection=collection,
                    top_k=top_k,
                    score_threshold=score_threshold,
                    metadata_filter=metadata_filter,
                )
            ]

        return iter(asyncio.run(_collect()))

    async def arun(
        self,
        query: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """
        Run the full async query pipeline for a user query.

        Args:
            query:           The user's question as a plain string.
            collection:      Override the default collection for this query.
            top_k:           Override default top_k for this query.
            score_threshold: Override default score_threshold for this query.
            metadata_filter: Optional metadata filters applied during retrieval.

        Returns:
            PipelineResult with answer, sources, scores, and latency.

        Raises:
            PipelineError: Wraps any stage-level error with full context.
        """
        pipeline_id = str(uuid.uuid4())
        started_at = time.monotonic()

        active_collection = collection or self._collection
        active_top_k = top_k if top_k is not None else self._top_k
        active_threshold = score_threshold if score_threshold is not None else self._score_threshold

        try:
            query_embedding = await self._run_query_embedder(query, pipeline_id)
            chunks = await self._run_retriever(
                query,
                query_embedding,
                active_collection,
                active_top_k,
                active_threshold,
                metadata_filter,
                pipeline_id,
            )
            prompt = await self._run_prompt_builder(query, chunks, pipeline_id)
            answer = await self._run_llm(prompt, pipeline_id)
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(
                f"Unexpected error during async query pipeline: {e}",
                stage="pipeline",
                component="async_query",
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

        latency_ms = (time.monotonic() - started_at) * 1000
        return await self._build_result(
            answer=answer,
            query=query,
            chunks=chunks,
            collection=active_collection,
            latency_ms=latency_ms,
            pipeline_id=pipeline_id,
        )

    async def astream(
        self,
        query: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """
        Async streaming variant. Pre-LLM stages run via async ABCs,
        then tokens are yielded live from llm.async_stream().

        Yields:
            Response text tokens as they arrive from the LLM.
        """
        pipeline_id = str(uuid.uuid4())
        active_collection = collection or self._collection
        active_top_k = top_k if top_k is not None else self._top_k
        active_threshold = score_threshold if score_threshold is not None else self._score_threshold

        query_embedding = await self._run_query_embedder(query, pipeline_id)
        chunks = await self._run_retriever(
            query,
            query_embedding,
            active_collection,
            active_top_k,
            active_threshold,
            metadata_filter,
            pipeline_id,
        )
        prompt = await self._run_prompt_builder(query, chunks, pipeline_id)

        await self._emit(pipeline_id, "llm", "started")
        t = time.monotonic()
        try:
            async for token in self._llm.async_stream(prompt):
                yield token
        except LLMError:
            raise
        except Exception as e:
            raise PipelineError(
                "LLM failed during async streaming.",
                stage="llm",
                component=type(self._llm).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e
        await self._emit(pipeline_id, "llm", "completed", t)

    # Stage runners

    async def _run_query_embedder(self, query: str, pipeline_id: str) -> list[float]:
        await self._emit(pipeline_id, "embedder", "started")
        t = time.monotonic()
        try:
            embedding = await self._embedder.async_embed_query(query)
        except EmbedderError:
            raise
        except Exception as e:
            raise PipelineError(
                "Embedder failed while embedding the query.",
                stage="embedder",
                component=type(self._embedder).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e
        await self._emit(pipeline_id, "embedder", "completed", t, {"dimensions": len(embedding)})
        return embedding

    async def _run_retriever(
        self,
        query: str,
        query_embedding: list[float],
        collection: str,
        top_k: int,
        score_threshold: float,
        metadata_filter: dict[str, Any] | None,
        pipeline_id: str,
    ) -> list[ScoredChunk]:
        await self._emit(pipeline_id, "retriever", "started")
        t = time.monotonic()
        try:
            chunks = await self._retriever.async_retrieve(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                collection=collection,
                score_threshold=score_threshold,
                filters=metadata_filter,
            )
        except RetrieverError:
            raise
        except Exception as e:
            raise PipelineError(
                "Retriever failed during async semantic search.",
                stage="retriever",
                component=type(self._retriever).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e
        await self._emit(
            pipeline_id,
            "retriever",
            "completed",
            t,
            {"chunks_retrieved": len(chunks), "collection": collection},
        )
        return chunks

    async def _run_prompt_builder(
        self, query: str, chunks: list[ScoredChunk], pipeline_id: str
    ) -> str:
        await self._emit(pipeline_id, "prompt_builder", "started")
        t = time.monotonic()
        try:
            prompt = self._prompt_builder.build(query, chunks)
        except PromptError:
            raise
        except Exception as e:
            raise PipelineError(
                "PromptBuilder failed while assembling the prompt.",
                stage="prompt_builder",
                component=type(self._prompt_builder).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e
        await self._emit(
            pipeline_id, "prompt_builder", "completed", t, {"prompt_length": len(prompt)}
        )
        return prompt

    async def _run_llm(self, prompt: str, pipeline_id: str) -> str:
        await self._emit(pipeline_id, "llm", "started")
        t = time.monotonic()
        try:
            answer = await self._llm.async_generate(prompt)
        except LLMError:
            raise
        except Exception as e:
            raise PipelineError(
                "LLM failed to generate a response.",
                stage="llm",
                component=type(self._llm).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e
        await self._emit(pipeline_id, "llm", "completed", t, {"response_length": len(answer)})
        return answer

    async def _build_result(
        self,
        answer: str,
        query: str,
        chunks: list[ScoredChunk],
        collection: str,
        latency_ms: float,
        pipeline_id: str,
    ) -> PipelineResult:
        await self._emit(pipeline_id, "response_builder", "started")
        t = time.monotonic()

        sources = [
            Source(
                content=sc.chunk.text,
                source=sc.chunk.metadata.get("source", ""),
                metadata=sc.chunk.metadata,
                score=sc.score,
                rank=sc.rank,
            )
            for sc in chunks
        ]
        scores = [sc.score for sc in chunks]

        result = PipelineResult(
            answer=answer,
            query=query,
            sources=sources,
            scores=scores,
            collection_used=collection,
            latency_ms=latency_ms,
            pipeline_id=pipeline_id,
        )

        await self._emit(pipeline_id, "response_builder", "completed", t)
        return result

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
                "Use 'await nexrag.async_query()' or 'async for token in nexrag.astream_query()' inside async contexts."
            )
        except RuntimeError:
            pass
