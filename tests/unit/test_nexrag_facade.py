"""Unit tests for the NexRAG facade (ingest_batch, cache, rate limit, sessions)."""

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from nexrag import NexRAG
from nexrag.core.models.result import PipelineResult
from nexrag.core.pipeline.ingestion import IngestionResult
from nexrag.core.runtime import QueryRuntime
from nexrag.defaults.context_strategy import WindowStrategy
from nexrag.defaults.query_cache import InMemoryQueryCache
from nexrag.defaults.rate_limiter import TokenBucketRateLimiter
from nexrag.defaults.session_store import InMemorySessionStore
from nexrag.exceptions import LLMRateLimitError


def _make_pipeline() -> NexRAG:
    ingestion = MagicMock()
    query = MagicMock()
    return NexRAG(ingestion=ingestion, query=query)


def _make_result(chunks: int = 3, collection: str = "default") -> IngestionResult:
    r = MagicMock(spec=IngestionResult)
    r.chunks_written = chunks
    r.collection_used = collection
    return r


def _query_result(answer: str = "answer", collection: str = "default") -> PipelineResult:
    return PipelineResult(
        answer=answer,
        query="q",
        sources=[],
        scores=[],
        collection_used=collection,
        latency_ms=1.0,
        pipeline_id="pid",
    )


def _pipeline_with_runtime(runtime: QueryRuntime) -> NexRAG:
    ingestion = MagicMock()
    query = MagicMock()
    query.default_collection = "default"
    query.run.return_value = _query_result()
    return NexRAG(ingestion=ingestion, query=query, runtime=runtime)


class TestIngestBatch:
    """ingest_batch delegates to the pipeline's batched path (single embed call)."""

    def test_returns_results_from_pipeline_batch(self):
        pipeline = _make_pipeline()
        r1, r2 = _make_result(3), _make_result(5)
        pipeline._ingestion.ingest_batch.return_value = [r1, r2]
        results = pipeline.ingest_batch(["a.pdf", "b.pdf"])
        assert results == [r1, r2]

    def test_delegates_once_not_per_source(self):
        # The whole point of B1: one batched call, not one ingest() per source.
        pipeline = _make_pipeline()
        pipeline._ingestion.ingest_batch.return_value = [_make_result()]
        pipeline.ingest_batch(["a.pdf", "b.txt", "c.md"])
        pipeline._ingestion.ingest_batch.assert_called_once_with(
            ["a.pdf", "b.txt", "c.md"], loader=None, metadata=None, collection=None
        )
        pipeline._ingestion.ingest.assert_not_called()

    def test_empty_list_returns_empty_without_calling_pipeline(self):
        pipeline = _make_pipeline()
        pipeline._ingestion.ingest_batch.return_value = []
        results = pipeline.ingest_batch([])
        assert results == []

    def test_path_objects_accepted(self):
        pipeline = _make_pipeline()
        pipeline._ingestion.ingest_batch.return_value = [_make_result()]
        results = pipeline.ingest_batch([Path("doc.pdf")])
        assert len(results) == 1

    def test_loader_metadata_collection_forwarded(self):
        pipeline = _make_pipeline()
        pipeline._ingestion.ingest_batch.return_value = [_make_result(collection="docs")]
        mock_loader = MagicMock()
        pipeline.ingest_batch(
            ["a.pdf", "b.pdf"], loader=mock_loader, metadata={"t": "x"}, collection="docs"
        )
        pipeline._ingestion.ingest_batch.assert_called_once_with(
            ["a.pdf", "b.pdf"], loader=mock_loader, metadata={"t": "x"}, collection="docs"
        )

    def test_invalidates_cache_for_collection_after_batch(self):
        pipeline = _make_pipeline()
        cache = MagicMock()
        pipeline._runtime = pipeline._runtime.__class__(cache=cache)
        pipeline._ingestion.ingest_batch.return_value = [_make_result(collection="docs")]
        pipeline.ingest_batch(["a.pdf"])
        cache.invalidate.assert_called_once_with("docs")


class TestQueryCache:
    def test_second_identical_query_is_cached(self):
        pipeline = _pipeline_with_runtime(QueryRuntime(cache=InMemoryQueryCache()))
        pipeline.query("hello")
        pipeline.query("hello")
        assert pipeline._query.run.call_count == 1  # second served from cache

    def test_different_queries_both_run(self):
        pipeline = _pipeline_with_runtime(QueryRuntime(cache=InMemoryQueryCache()))
        pipeline.query("hello")
        pipeline.query("world")
        assert pipeline._query.run.call_count == 2

    def test_no_cache_runs_every_time(self):
        pipeline = _pipeline_with_runtime(QueryRuntime())
        pipeline.query("hello")
        pipeline.query("hello")
        assert pipeline._query.run.call_count == 2

    def test_ingest_invalidates_query_cache(self):
        pipeline = _pipeline_with_runtime(QueryRuntime(cache=InMemoryQueryCache()))
        pipeline._ingestion.ingest.return_value = _make_result(collection="default")
        pipeline.query("hello")
        pipeline.ingest("doc.txt")  # invalidates "default"
        pipeline.query("hello")
        assert pipeline._query.run.call_count == 2  # cache busted by ingest


class TestRateLimit:
    def test_within_burst_allowed(self):
        rt = QueryRuntime(rate_limiter=TokenBucketRateLimiter(requests_per_minute=60, burst=3))
        pipeline = _pipeline_with_runtime(rt)
        pipeline.query("a")
        pipeline.query("b")
        pipeline.query("c")  # 3 allowed

    def test_over_limit_raises(self):
        rt = QueryRuntime(rate_limiter=TokenBucketRateLimiter(requests_per_minute=60, burst=1))
        pipeline = _pipeline_with_runtime(rt)
        pipeline.query("a")
        with pytest.raises(LLMRateLimitError):
            pipeline.query("b")


class TestSessions:
    def _session_pipeline(self) -> NexRAG:
        rt = QueryRuntime(
            session_store=InMemorySessionStore(),
            context_strategy=WindowStrategy(max_turns=6),
        )
        return _pipeline_with_runtime(rt)

    def test_query_session_records_turns(self):
        pipeline = self._session_pipeline()
        pipeline.query_session("first question", session_id="s1")
        history = pipeline._runtime.session_store.get_history("s1")
        assert [t.role for t in history] == ["user", "assistant"]
        assert history[0].content == "first question"

    def test_history_passed_to_pipeline_on_second_turn(self):
        pipeline = self._session_pipeline()
        pipeline.query_session("first", session_id="s1")
        pipeline.query_session("second", session_id="s1")
        # The second run() call should receive the first turn pair as history.
        _, kwargs = pipeline._query.run.call_args
        assert kwargs["history"]
        assert any(t.content == "first" for t in kwargs["history"])

    def test_sessions_isolated_by_id(self):
        pipeline = self._session_pipeline()
        pipeline.query_session("a", session_id="s1")
        pipeline.query_session("b", session_id="s2")
        assert len(pipeline._runtime.session_store.get_history("s1")) == 2
        assert len(pipeline._runtime.session_store.get_history("s2")) == 2

    def test_clear_session(self):
        pipeline = self._session_pipeline()
        pipeline.query_session("a", session_id="s1")
        pipeline.clear_session("s1")
        assert pipeline._runtime.session_store.get_history("s1") == []

    def test_query_session_disabled_raises(self):
        pipeline = _pipeline_with_runtime(QueryRuntime())  # no session store
        with pytest.raises(RuntimeError):
            pipeline.query_session("q", session_id="s1")

    def test_delete_turns_disabled_raises(self):
        pipeline = _pipeline_with_runtime(QueryRuntime())
        with pytest.raises(RuntimeError):
            pipeline.delete_turns("s1", before=0.0)


class TestAsyncFacade:
    """Async entry points honour cache, rate limit, and sessions (sync-mode pipeline)."""

    def test_async_query_uses_cache(self):
        pipeline = _pipeline_with_runtime(QueryRuntime(cache=InMemoryQueryCache()))

        async def run():
            await pipeline.async_query("hello")
            await pipeline.async_query("hello")

        asyncio.run(run())
        assert pipeline._query.run.call_count == 1

    def test_async_query_rate_limited(self):
        rt = QueryRuntime(rate_limiter=TokenBucketRateLimiter(requests_per_minute=60, burst=1))
        pipeline = _pipeline_with_runtime(rt)

        async def run():
            await pipeline.async_query("a")
            with pytest.raises(LLMRateLimitError):
                await pipeline.async_query("b")

        asyncio.run(run())

    def test_async_query_session_records_turns(self):
        rt = QueryRuntime(session_store=InMemorySessionStore(), context_strategy=WindowStrategy())
        pipeline = _pipeline_with_runtime(rt)

        asyncio.run(pipeline.async_query_session("hi", session_id="s1"))
        history = pipeline._runtime.session_store.get_history("s1")
        assert [t.role for t in history] == ["user", "assistant"]

    def test_async_ingest_batch_delegates_and_invalidates(self):
        pipeline = _make_pipeline()
        cache = MagicMock()
        pipeline._runtime = QueryRuntime(cache=cache)
        pipeline._ingestion.ingest_batch = MagicMock(return_value=[_make_result(collection="docs")])
        # Sync-mode ingestion pipeline → facade runs ingest_batch via to_thread.
        results = asyncio.run(pipeline.async_ingest_batch(["a.pdf"]))
        assert len(results) == 1
        cache.invalidate.assert_called_once_with("docs")
