"""
QueryPipeline orchestrates the full query flow:

    User Query → Embedder → Retriever → Reranker → PromptBuilder → LLM → ResponseBuilder
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from typing import Any

from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.core.interfaces.llm import BaseLLM
from nexrag.core.interfaces.observer import BaseObserver, NoOpObserver
from nexrag.core.interfaces.prompt_builder import BasePromptBuilder
from nexrag.core.interfaces.reranker import BaseReranker
from nexrag.core.interfaces.retriever import BaseRetriever
from nexrag.core.models.chunk import ScoredChunk
from nexrag.core.models.event import PipelineEvent
from nexrag.core.models.metrics import RunMetrics, TokenUsage
from nexrag.core.models.result import PipelineResult, Source
from nexrag.exceptions import (
    EmbedderError,
    LLMError,
    PipelineError,
    PromptError,
    RetrieverError,
)


class QueryPipeline:
    """
    Orchestrates the NexRAG query pipeline.

    Stages (in order):
        1. Embedder       — embeds the user query string → vector
        2. Retriever      — semantic search → list[ScoredChunk]
        3. Reranker        — optional, re-scores and re-ranks retrieved chunks
        4. PromptBuilder  — assembles prompt from query + chunks
        5. LLM            — generates response from prompt
        6. ResponseBuilder— wraps everything into a PipelineResult

    Args:
        embedder:        Embeds the user query. Must be the same model used
                         during ingestion — enforced by the fingerprint check
                         in IngestionPipeline.
        retriever:       Retrieves relevant chunks from the vector DB.
        prompt_builder:  Assembles the final prompt string.
        llm:             Generates the answer.
        collection:      Which vector collection to query.
        top_k:           Maximum chunks to retrieve. Default: 5.
        score_threshold: Minimum similarity score for retrieved chunks. Default: 0.0.
        observer:        Optional. Defaults to NoOpObserver.
        reranker:        Optional. Re-scores and re-ranks retrieved chunks.
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
        reranker: BaseReranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._retriever = retriever
        self._prompt_builder = prompt_builder
        self._llm = llm
        self._collection = collection
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._observer = observer or NoOpObserver()
        self._reranker = reranker

    # Public API

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
        Run the full query pipeline for a user query.

        Args:
            query:           The user's question as a plain string.
            collection:      Override the default collection for this query.
            top_k:           Override default top_k for this query.
            score_threshold: Override default score_threshold for this query.
            metadata_filter: Optional metadata filters applied during retrieval.
                             e.g. {"vendor": "Acme", "year": 2024}

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

        stage_latencies: dict[str, float] = {}
        try:
            t = time.monotonic()
            query_embedding = self._run_query_embedder(query, pipeline_id)
            stage_latencies["embedder"] = (time.monotonic() - t) * 1000

            t = time.monotonic()
            chunks = self._run_retriever(
                query,
                query_embedding,
                active_collection,
                active_top_k,
                active_threshold,
                metadata_filter,
                pipeline_id,
            )
            stage_latencies["retriever"] = (time.monotonic() - t) * 1000

            if self._reranker is not None:
                t = time.monotonic()
                chunks = self._run_reranker(query, chunks, pipeline_id)
                stage_latencies["reranker"] = (time.monotonic() - t) * 1000

            t = time.monotonic()
            prompt = self._run_prompt_builder(query, chunks, pipeline_id)
            stage_latencies["prompt_builder"] = (time.monotonic() - t) * 1000

            t = time.monotonic()
            answer, token_usage = self._run_llm(prompt, pipeline_id)
            stage_latencies["llm"] = (time.monotonic() - t) * 1000
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(
                f"Unexpected error during query pipeline: {e}",
                stage="pipeline",
                component="query",
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

        latency_ms = (time.monotonic() - started_at) * 1000

        return self._build_result(
            answer=answer,
            query=query,
            chunks=chunks,
            collection=active_collection,
            latency_ms=latency_ms,
            pipeline_id=pipeline_id,
            token_usage=token_usage,
            stage_latencies=stage_latencies,
        )

    def stream(
        self,
        query: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> Iterator[str | RunMetrics]:
        """
        Run the query pipeline and stream LLM response tokens as they arrive.

        All stages before the LLM (embed, retrieve, rerank, prompt_builder) run
        synchronously and are timed identically to run(). Tokens are yielded as they
        stream from the LLM. The final item yielded is always a RunMetrics object
        carrying full per-stage latencies and chunk count.

        Args:
            query:           The user's question as a plain string.
            collection:      Override the default collection for this call.
            top_k:           Override default top_k for this call.
            score_threshold: Override default score_threshold for this call.
            metadata_filter: Optional metadata filters applied during retrieval.

        Yields:
            Response text tokens as they arrive from the LLM, followed by a
            final RunMetrics object. Use isinstance(item, RunMetrics) to detect it.

        Raises:
            PipelineError: Wraps any stage-level error with full context.
        """
        pipeline_id = str(uuid.uuid4())
        started_at = time.monotonic()
        active_collection = collection or self._collection
        active_top_k = top_k if top_k is not None else self._top_k
        active_threshold = score_threshold if score_threshold is not None else self._score_threshold

        stage_latencies: dict[str, float] = {}
        try:
            t = time.monotonic()
            query_embedding = self._run_query_embedder(query, pipeline_id)
            stage_latencies["embedder"] = (time.monotonic() - t) * 1000

            t = time.monotonic()
            chunks = self._run_retriever(
                query,
                query_embedding,
                active_collection,
                active_top_k,
                active_threshold,
                metadata_filter,
                pipeline_id,
            )
            stage_latencies["retriever"] = (time.monotonic() - t) * 1000

            if self._reranker is not None:
                t = time.monotonic()
                chunks = self._run_reranker(query, chunks, pipeline_id)
                stage_latencies["reranker"] = (time.monotonic() - t) * 1000

            t = time.monotonic()
            prompt = self._run_prompt_builder(query, chunks, pipeline_id)
            stage_latencies["prompt_builder"] = (time.monotonic() - t) * 1000
        except PipelineError:
            raise
        except Exception as e:
            raise PipelineError(
                f"Unexpected error during streaming pipeline: {e}",
                stage="pipeline",
                component="query",
                pipeline_id=pipeline_id,
                cause=e,
            ) from e

        self._emit(pipeline_id, "llm", "started")
        t = time.monotonic()
        try:
            yield from self._llm.stream(prompt)
        except LLMError:
            raise
        except Exception as e:
            raise PipelineError(
                "LLM failed during streaming.",
                stage="llm",
                component=type(self._llm).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e
        stage_latencies["llm"] = (time.monotonic() - t) * 1000
        self._emit(pipeline_id, "llm", "completed", t)

        latency_ms = (time.monotonic() - started_at) * 1000
        yield RunMetrics(
            pipeline_id=pipeline_id,
            total_latency_ms=latency_ms,
            stage_latencies=stage_latencies,
            token_usage=None,
            model=getattr(self._llm, "_model", None),
            chunks_retrieved=len(chunks),
        )

    # Stage runners

    def _run_query_embedder(self, query: str, pipeline_id: str) -> list[float]:
        self._emit(pipeline_id, "embedder", "started")
        t = time.monotonic()
        try:
            embedding = self._embedder.embed_query(query)
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
        self._emit(
            pipeline_id,
            "embedder",
            "completed",
            t,
            {"model": self._embedder.model_name, "dimensions": len(embedding)},
        )
        return embedding

    def _run_retriever(
        self,
        query: str,
        query_embedding: list[float],
        collection: str,
        top_k: int,
        score_threshold: float,
        metadata_filter: dict[str, Any] | None,
        pipeline_id: str,
    ) -> list[ScoredChunk]:
        self._emit(pipeline_id, "retriever", "started")
        t = time.monotonic()
        try:
            chunks = self._retriever.retrieve(
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
                "Retriever failed during semantic search.",
                stage="retriever",
                component=type(self._retriever).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e
        self._emit(
            pipeline_id,
            "retriever",
            "completed",
            t,
            {"chunks_retrieved": len(chunks), "collection": collection},
        )
        return chunks

    def _run_reranker(
        self,
        query: str,
        chunks: list[ScoredChunk],
        pipeline_id: str,
    ) -> list[ScoredChunk]:
        assert self._reranker is not None
        self._emit(pipeline_id, "reranker", "started")
        t = time.monotonic()
        top_n = min(self._reranker.top_n, len(chunks))
        try:
            reranked = self._reranker.rerank(query, chunks, top_n)
        except Exception as e:
            raise PipelineError(
                "Reranker failed to re-score chunks.",
                stage="reranker",
                component=type(self._reranker).__name__,
                pipeline_id=pipeline_id,
                cause=e,
            ) from e
        self._emit(
            pipeline_id,
            "reranker",
            "completed",
            t,
            {"chunks_in": len(chunks), "chunks_out": len(reranked), "top_n": top_n},
        )
        return reranked

    def _run_prompt_builder(self, query: str, chunks: list[ScoredChunk], pipeline_id: str) -> str:
        self._emit(pipeline_id, "prompt_builder", "started")
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
        self._emit(
            pipeline_id,
            "prompt_builder",
            "completed",
            t,
            {"prompt_length": len(prompt)},
        )
        return prompt

    def _run_llm(self, prompt: str, pipeline_id: str) -> tuple[str, TokenUsage | None]:
        self._emit(pipeline_id, "llm", "started")
        t = time.monotonic()
        try:
            answer, token_usage = self._llm.generate(prompt)
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
        meta: dict[str, Any] = {
            "model": getattr(self._llm, "_model", None),
            "response_length": len(answer),
        }
        if token_usage is not None:
            meta["token_usage"] = {
                "prompt_tokens": token_usage.prompt_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "total_tokens": token_usage.total_tokens,
            }
        self._emit(pipeline_id, "llm", "completed", t, meta)
        return answer, token_usage

    # Response assembly

    def _build_result(
        self,
        answer: str,
        query: str,
        chunks: list[ScoredChunk],
        collection: str,
        latency_ms: float,
        pipeline_id: str,
        token_usage: TokenUsage | None = None,
        stage_latencies: dict[str, float] | None = None,
    ) -> PipelineResult:
        """
        Wrap all pipeline outputs into a structured PipelineResult.
        Never return a raw string to the caller.

        Source.source is pulled from chunk.metadata["source"] — the opaque
        identifier the Loader set at ingestion time (file path, URL, S3 URI,
        page ID, etc.). Falls back to empty string if not set.
        """
        self._emit(pipeline_id, "response_builder", "started")
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

        sl = stage_latencies or {}
        metrics = RunMetrics(
            pipeline_id=pipeline_id,
            total_latency_ms=latency_ms,
            stage_latencies=sl,
            token_usage=token_usage,
            model=getattr(self._llm, "_model", None),
            chunks_retrieved=len(chunks),
        )
        self._emit(pipeline_id, "response_builder", "completed", t)
        meta: dict[str, Any] = {
            "total_latency_ms": round(latency_ms, 2),
            "chunks_retrieved": len(chunks),
        }
        if token_usage is not None:
            meta["token_usage"] = {
                "prompt_tokens": token_usage.prompt_tokens,
                "completion_tokens": token_usage.completion_tokens,
                "total_tokens": token_usage.total_tokens,
            }
        self._emit(pipeline_id, "pipeline", "completed", metadata=meta)

        result = PipelineResult(
            answer=answer,
            query=query,
            sources=sources,
            scores=scores,
            collection_used=collection,
            latency_ms=latency_ms,
            pipeline_id=pipeline_id,
            token_usage=token_usage,
            metrics=metrics,
        )

        return result

    # Helper

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
