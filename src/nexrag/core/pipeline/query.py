"""
QueryPipeline orchestrates the full query flow:

    User Query → Embedder → Retriever → PromptBuilder → LLM → ResponseBuilder
"""

from __future__ import annotations

import time
import uuid
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
        3. PromptBuilder  — assembles prompt from query + chunks
        4. LLM            — generates response from prompt
        5. ResponseBuilder— wraps everything into a PipelineResult

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

        try:
            chunks = self._run_retriever(
                query,
                active_collection,
                active_top_k,
                active_threshold,
                metadata_filter,
                pipeline_id,
            )
            prompt = self._run_prompt_builder(query, chunks, pipeline_id)
            answer = self._run_llm(prompt, pipeline_id)
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
        )

    # Stage runners

    def _run_retriever(
        self,
        query: str,
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

    def _run_llm(self, prompt: str, pipeline_id: str) -> str:
        self._emit(pipeline_id, "llm", "started")
        t = time.monotonic()
        try:
            answer = self._llm.generate(prompt)
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
        self._emit(pipeline_id, "llm", "completed", t, {"response_length": len(answer)})
        return answer

    # Response assembly

    def _build_result(
        self,
        answer: str,
        query: str,
        chunks: list[ScoredChunk],
        collection: str,
        latency_ms: float,
        pipeline_id: str,
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

        result = PipelineResult(
            answer=answer,
            query=query,
            sources=sources,
            scores=scores,
            collection_used=collection,
            latency_ms=latency_ms,
            pipeline_id=pipeline_id,
        )

        self._emit(pipeline_id, "response_builder", "completed", t)
        return result

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
