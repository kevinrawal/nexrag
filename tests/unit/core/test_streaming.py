"""Tests for QueryPipeline.stream() — issue #9.

Async streaming (astream) lives on AsyncQueryPipeline and is tested in
test_async_pipelines.py, consistent with how async ingestion is separated.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from nexrag.core.pipeline.query import QueryPipeline


def _make_query_pipeline(stream_tokens: list[str] | None = None) -> QueryPipeline:
    """Build a QueryPipeline with all dependencies mocked."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]

    chunk = MagicMock()
    chunk.text = "Some chunk text"
    chunk.metadata = {"source": "test.pdf"}
    scored = MagicMock()
    scored.chunk = chunk
    scored.score = 0.9
    scored.rank = 1

    retriever = MagicMock()
    retriever.retrieve.return_value = [scored]

    prompt_builder = MagicMock()
    prompt_builder.build.return_value = "System\n\n---\n\nUser question"

    llm = MagicMock()
    llm.generate.return_value = "Full answer"
    tokens = stream_tokens if stream_tokens is not None else ["Hello", " world"]
    llm.stream.return_value = iter(tokens)

    observer = MagicMock()
    observer.emit = MagicMock()

    return QueryPipeline(
        embedder=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        collection="test_collection",
        observer=observer,
    )


class TestQueryPipelineStream:
    def test_stream_yields_tokens(self):
        pipeline = _make_query_pipeline(["Hello", " world", "!"])
        tokens = list(pipeline.stream("What is NexRAG?"))
        assert tokens == ["Hello", " world", "!"]

    def test_stream_concatenated_equals_generate(self):
        pipeline = _make_query_pipeline(["Full", " answer"])
        streamed = "".join(pipeline.stream("q"))
        assert streamed == "Full answer"

    def test_stream_fires_llm_started_event(self):
        pipeline = _make_query_pipeline()
        list(pipeline.stream("q"))
        emit_calls = [c.args[0] for c in pipeline._observer.emit.call_args_list]
        llm_events = [e for e in emit_calls if e.stage == "llm"]
        assert any(e.status == "started" for e in llm_events)

    def test_stream_fires_llm_completed_event(self):
        pipeline = _make_query_pipeline()
        list(pipeline.stream("q"))
        emit_calls = [c.args[0] for c in pipeline._observer.emit.call_args_list]
        llm_events = [e for e in emit_calls if e.stage == "llm"]
        assert any(e.status == "completed" for e in llm_events)

    def test_stream_fires_all_stage_events(self):
        pipeline = _make_query_pipeline()
        list(pipeline.stream("q"))
        emit_calls = [c.args[0] for c in pipeline._observer.emit.call_args_list]
        stages_seen = {e.stage for e in emit_calls}
        assert {"embedder", "retriever", "prompt_builder", "llm"}.issubset(stages_seen)

    def test_stream_empty_tokens(self):
        pipeline = _make_query_pipeline([])
        tokens = list(pipeline.stream("q"))
        assert tokens == []

    def test_stream_single_token(self):
        pipeline = _make_query_pipeline(["only one"])
        tokens = list(pipeline.stream("q"))
        assert tokens == ["only one"]

    def test_stream_calls_llm_stream_not_generate(self):
        pipeline = _make_query_pipeline()
        list(pipeline.stream("q"))
        pipeline._llm.stream.assert_called_once()
        pipeline._llm.generate.assert_not_called()

    def test_stream_passes_collection_override(self):
        pipeline = _make_query_pipeline()
        list(pipeline.stream("q", collection="other_collection"))
        call_kwargs = pipeline._retriever.retrieve.call_args.kwargs
        assert call_kwargs["collection"] == "other_collection"

    def test_stream_passes_top_k_override(self):
        pipeline = _make_query_pipeline()
        list(pipeline.stream("q", top_k=10))
        call_kwargs = pipeline._retriever.retrieve.call_args.kwargs
        assert call_kwargs["top_k"] == 10
