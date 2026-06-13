"""
Observability: every stage failure emits exactly one `failed` event, plus a
pipeline-level `failed` event — in both ingestion and query pipelines, and on
the streaming path (where a mid-stream error must NOT yield a RunMetrics).

Covers v0.3.3 issues #3 (failed-event coverage) and #4 (stream finally-yield).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexrag.core.models.metrics import RunMetrics
from nexrag.core.pipeline.ingestion import IngestionPipeline
from nexrag.core.pipeline.query import QueryPipeline
from nexrag.exceptions import EmbedderError, LLMError, PipelineError


class _RecordingObserver:
    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)


def _failed(observer: _RecordingObserver, stage: str) -> list:
    return [e for e in observer.events if e.stage == stage and e.status == "failed"]


# --- Ingestion -------------------------------------------------------------


def _ingestion_with_failing_embedder(observer):
    embedder = MagicMock()
    embedder.model_name = "mock-model"
    embedder.dimensions = 4
    embedder.embed.side_effect = EmbedderError("boom", stage="embedder", component="MockEmbedder")
    chunker = MagicMock()
    chunk = MagicMock()
    chunk.text = "text"
    chunk.metadata = {"source": "s"}
    chunker.chunk.return_value = [chunk]
    sanitizer = MagicMock()
    sanitizer.sanitize.side_effect = lambda d: d

    return IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_db=MagicMock(),
        collection="c",
        sanitizer=sanitizer,
        observer=observer,
    )


class TestIngestionFailedEvents:
    def test_embedder_failure_emits_stage_and_pipeline_failed(self):
        from nexrag.core.models.document import Document

        observer = _RecordingObserver()
        pipe = _ingestion_with_failing_embedder(observer)

        # EmbedderError is wrapped into PipelineError at the facade boundary.
        with pytest.raises(PipelineError):
            pipe.ingest_documents([Document(content="hello", metadata={"source": "s"})])

        emb_failed = _failed(observer, "embedder")
        assert len(emb_failed) == 1
        assert emb_failed[0].metadata.get("error_type") == "EmbedderError"
        assert emb_failed[0].metadata.get("message")

        assert len(_failed(observer, "pipeline")) == 1


# --- Query -----------------------------------------------------------------


def _make_query_pipeline(observer, *, retriever_error=None, stream_tokens=None):
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    embedder.model_name = "mock"

    scored = MagicMock()
    scored.chunk = MagicMock(text="t", metadata={"source": "s"})
    scored.score = 0.9
    scored.rank = 1
    retriever = MagicMock()
    if retriever_error is not None:
        retriever.retrieve.side_effect = retriever_error
    else:
        retriever.retrieve.return_value = [scored]

    prompt_builder = MagicMock()
    prompt_builder.build.return_value = "prompt"

    llm = MagicMock()
    llm.generate.return_value = ("answer", None)
    if stream_tokens is not None:
        llm.stream.return_value = iter(stream_tokens)

    return QueryPipeline(
        embedder=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        collection="c",
        observer=observer,
    )


class TestQueryFailedEvents:
    def test_retriever_failure_emits_stage_and_pipeline_failed(self):
        observer = _RecordingObserver()
        err = RuntimeError("retriever down")
        pipe = _make_query_pipeline(observer, retriever_error=err)

        with pytest.raises(PipelineError):
            pipe.run("q")

        assert len(_failed(observer, "retriever")) == 1
        assert len(_failed(observer, "pipeline")) == 1


# --- Streaming (#4) --------------------------------------------------------


def _stream_then_raise(tokens, exc):
    def _gen():
        yield from tokens
        raise exc

    return _gen()


class TestStreamFailurePath:
    def test_midstream_error_raises_and_yields_no_metrics(self):
        observer = _RecordingObserver()
        pipe = _make_query_pipeline(observer)
        pipe._llm.stream.return_value = _stream_then_raise(
            ["a", "b"], LLMError("mid-stream", stage="llm", component="MockLLM")
        )

        received: list = []
        with pytest.raises(LLMError):
            for item in pipe.stream("q"):
                received.append(item)

        # Tokens arrived, but NO RunMetrics was yielded on the failure path.
        assert received == ["a", "b"]
        assert not any(isinstance(x, RunMetrics) for x in received)
        # Failure-path metrics travel via the failed events instead.
        assert len(_failed(observer, "llm")) == 1
        assert len(_failed(observer, "pipeline")) == 1

    def test_early_break_does_not_raise_runtime_error(self):
        observer = _RecordingObserver()
        pipe = _make_query_pipeline(observer, stream_tokens=["a", "b", "c", "d"])

        # Abandon the iterator after the first token — must not raise
        # "generator ignored GeneratorExit".
        gen = pipe.stream("q")
        first = next(gen)
        assert first == "a"
        gen.close()  # triggers GeneratorExit inside the generator

    def test_success_path_still_ends_with_run_metrics(self):
        observer = _RecordingObserver()
        pipe = _make_query_pipeline(observer, stream_tokens=["a", "b"])
        items = list(pipe.stream("q"))
        assert items[:-1] == ["a", "b"]
        assert isinstance(items[-1], RunMetrics)
        assert not _failed(observer, "llm")
