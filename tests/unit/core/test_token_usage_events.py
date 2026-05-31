"""Tests for token usage in PipelineEvents and PipelineResult — issue #4."""

from unittest.mock import MagicMock

from nexrag.core.models.chunk import Chunk, ScoredChunk
from nexrag.core.models.event import PipelineEvent
from nexrag.core.models.metrics import TokenUsage
from nexrag.core.pipeline.query import QueryPipeline


def _make_scored_chunk() -> ScoredChunk:
    chunk = Chunk(
        text="text",
        chunk_index=0,
        total_chunks=1,
        parent_doc_id="d1",
        metadata={"source": "test.pdf"},
    )
    return ScoredChunk(chunk=chunk, score=0.9, rank=1)


def _make_pipeline(token_usage: TokenUsage | None = None) -> QueryPipeline:
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2]
    embedder.model_name = "text-embedding-3-small"

    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_scored_chunk()]

    prompt_builder = MagicMock()
    prompt_builder.build.return_value = "prompt"

    llm = MagicMock()
    llm.generate.return_value = ("answer", token_usage)
    llm._model = "gpt-4o"

    observer = MagicMock()
    return QueryPipeline(
        embedder=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        collection="test",
        observer=observer,
    )


def _get_events(pipeline: QueryPipeline) -> list[PipelineEvent]:
    return [call.args[0] for call in pipeline._observer.emit.call_args_list]


class TestTokenUsageInEvents:
    def test_llm_completed_event_has_token_usage(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150)
        pipeline = _make_pipeline(token_usage=usage)
        pipeline.run("query")

        events = _get_events(pipeline)
        llm_completed = next(
            (e for e in events if e.stage == "llm" and e.status == "completed"), None
        )
        assert llm_completed is not None
        assert "token_usage" in llm_completed.metadata
        assert llm_completed.metadata["token_usage"]["total_tokens"] == 150
        assert llm_completed.metadata["token_usage"]["prompt_tokens"] == 100
        assert llm_completed.metadata["token_usage"]["completion_tokens"] == 50

    def test_llm_completed_event_has_model(self):
        pipeline = _make_pipeline()
        pipeline.run("query")

        events = _get_events(pipeline)
        llm_completed = next(
            (e for e in events if e.stage == "llm" and e.status == "completed"), None
        )
        assert llm_completed is not None
        assert llm_completed.metadata.get("model") == "gpt-4o"

    def test_embedder_completed_event_has_model(self):
        pipeline = _make_pipeline()
        pipeline.run("query")

        events = _get_events(pipeline)
        embedder_completed = next(
            (e for e in events if e.stage == "embedder" and e.status == "completed"), None
        )
        assert embedder_completed is not None
        assert embedder_completed.metadata.get("model") == "text-embedding-3-small"

    def test_pipeline_result_token_usage_populated(self):
        usage = TokenUsage(prompt_tokens=200, completion_tokens=80, total_tokens=280)
        pipeline = _make_pipeline(token_usage=usage)
        result = pipeline.run("query")

        assert result.token_usage is not None
        assert result.token_usage.total_tokens == 280

    def test_pipeline_result_token_usage_none_when_no_usage(self):
        pipeline = _make_pipeline(token_usage=None)
        result = pipeline.run("query")
        assert result.token_usage is None

    def test_no_token_usage_event_still_has_model(self):
        pipeline = _make_pipeline(token_usage=None)
        pipeline.run("query")

        events = _get_events(pipeline)
        llm_completed = next(
            (e for e in events if e.stage == "llm" and e.status == "completed"), None
        )
        assert llm_completed is not None
        assert "token_usage" not in llm_completed.metadata
        assert llm_completed.metadata.get("model") == "gpt-4o"

    def test_pipeline_summary_event_emitted(self):
        pipeline = _make_pipeline()
        pipeline.run("query")

        events = _get_events(pipeline)
        summary = next(
            (e for e in events if e.stage == "pipeline" and e.status == "completed"), None
        )
        assert summary is not None
        assert "total_latency_ms" in summary.metadata
