"""Tests for QueryPipeline.stream() — issue #9 / #25."""

from __future__ import annotations

from unittest.mock import MagicMock

from nexrag.core.models.metrics import RunMetrics
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


def _str_tokens(stream_items: list) -> list[str]:
    return [item for item in stream_items if isinstance(item, str)]


def _stream_metrics(stream_items: list) -> RunMetrics:
    metrics = [item for item in stream_items if isinstance(item, RunMetrics)]
    assert len(metrics) == 1, f"expected exactly 1 RunMetrics, got {len(metrics)}"
    return metrics[0]


class TestQueryPipelineStream:
    def test_stream_yields_tokens(self):
        pipeline = _make_query_pipeline(["Hello", " world", "!"])
        items = list(pipeline.stream("What is NexRAG?"))
        assert _str_tokens(items) == ["Hello", " world", "!"]

    def test_stream_concatenated_equals_generate(self):
        pipeline = _make_query_pipeline(["Full", " answer"])
        items = list(pipeline.stream("q"))
        assert "".join(_str_tokens(items)) == "Full answer"

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
        items = list(pipeline.stream("q"))
        assert _str_tokens(items) == []

    def test_stream_single_token(self):
        pipeline = _make_query_pipeline(["only one"])
        items = list(pipeline.stream("q"))
        assert _str_tokens(items) == ["only one"]

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


class TestQueryPipelineStreamMetrics:
    def test_last_item_is_run_metrics(self):
        pipeline = _make_query_pipeline(["Hello", " world"])
        items = list(pipeline.stream("q"))
        assert isinstance(items[-1], RunMetrics)

    def test_metrics_yielded_even_with_no_tokens(self):
        pipeline = _make_query_pipeline([])
        items = list(pipeline.stream("q"))
        assert len(items) == 1
        assert isinstance(items[0], RunMetrics)

    def test_metrics_has_all_stage_latencies(self):
        pipeline = _make_query_pipeline(["tok"])
        items = list(pipeline.stream("q"))
        m = _stream_metrics(items)
        assert "embedder" in m.stage_latencies
        assert "retriever" in m.stage_latencies
        assert "prompt_builder" in m.stage_latencies
        assert "llm" in m.stage_latencies

    def test_metrics_chunks_retrieved(self):
        pipeline = _make_query_pipeline(["tok"])
        items = list(pipeline.stream("q"))
        m = _stream_metrics(items)
        assert m.chunks_retrieved == 1  # mock retriever returns 1 chunk

    def test_metrics_total_latency_positive(self):
        pipeline = _make_query_pipeline(["tok"])
        items = list(pipeline.stream("q"))
        m = _stream_metrics(items)
        assert m.total_latency_ms >= 0.0

    def test_metrics_pipeline_id_is_uuid(self):
        import uuid

        pipeline = _make_query_pipeline(["tok"])
        items = list(pipeline.stream("q"))
        m = _stream_metrics(items)
        uuid.UUID(m.pipeline_id)  # raises ValueError if not a valid UUID

    def test_metrics_token_usage_is_none(self):
        pipeline = _make_query_pipeline(["tok"])
        items = list(pipeline.stream("q"))
        m = _stream_metrics(items)
        assert m.token_usage is None  # not available from streaming LLMs

    def test_tokens_come_before_metrics(self):
        pipeline = _make_query_pipeline(["a", "b", "c"])
        items = list(pipeline.stream("q"))
        # all strings must come before the RunMetrics
        metrics_idx = next(i for i, item in enumerate(items) if isinstance(item, RunMetrics))
        for item in items[:metrics_idx]:
            assert isinstance(item, str)
