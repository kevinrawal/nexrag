"""Tests for AsyncQueryPipeline and AsyncIngestionPipeline — issues #3 and #14."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from nexrag.core.models.result import PipelineResult
from nexrag.core.pipeline.async_ingestion import AsyncIngestionPipeline
from nexrag.core.pipeline.async_query import AsyncQueryPipeline
from nexrag.core.pipeline.ingestion import IngestionResult
from nexrag.exceptions import PipelineError

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_scored_chunk(text: str = "chunk text", score: float = 0.9) -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    chunk.metadata = {"source": "test.pdf"}
    chunk.content_hash = "abc123"
    scored = MagicMock()
    scored.chunk = chunk
    scored.score = score
    scored.rank = 1
    return scored


def _make_async_query_pipeline(answer: str = "The answer.") -> AsyncQueryPipeline:
    embedder = MagicMock()
    embedder.async_embed_query = AsyncMock(return_value=[0.1, 0.2, 0.3])
    embedder.model_name = "test-embedding-model"

    retriever = MagicMock()
    retriever.async_retrieve = AsyncMock(return_value=[_make_scored_chunk()])

    prompt_builder = MagicMock()
    prompt_builder.build.return_value = "assembled prompt"

    llm = MagicMock()
    llm.async_generate = AsyncMock(return_value=(answer, None))
    llm._model = "test-model"

    async def _async_stream(_prompt):
        for t in answer.split():
            yield t

    llm.async_stream = _async_stream

    observer = MagicMock()
    observer.async_emit = AsyncMock()

    return AsyncQueryPipeline(
        embedder=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        collection="test_collection",
        observer=observer,
    )


def _make_chunk(text: str = "chunk text") -> MagicMock:
    chunk = MagicMock()
    chunk.text = text
    chunk.content_hash = "abc123"
    chunk.metadata = {"source": "test.pdf"}
    return chunk


def _make_document(source: str = "test.pdf") -> MagicMock:
    doc = MagicMock()
    doc.doc_id = "doc1"
    doc.metadata = {"source": source}
    return doc


def _make_async_ingestion_pipeline() -> AsyncIngestionPipeline:
    chunk = _make_chunk()

    chunker = MagicMock()
    chunker.chunk.return_value = [chunk]

    embedder = MagicMock()
    embedder.model_name = "text-embedding-3-small"
    embedder.dimensions = 3
    embedder.async_embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    scored = MagicMock()
    scored.chunk = chunk

    vector_db = MagicMock()
    vector_db.get_collection_metadata.return_value = {}
    vector_db.set_collection_metadata.return_value = None
    vector_db.query.return_value = []
    vector_db.async_upsert = AsyncMock()
    vector_db.async_query = AsyncMock(return_value=[])
    vector_db.async_get_ids_by_metadata = AsyncMock(return_value=[])

    sanitizer = MagicMock()
    sanitizer.sanitize.return_value = MagicMock()

    loader = MagicMock()
    loader.load.return_value = [_make_document()]

    observer = MagicMock()
    observer.async_emit = AsyncMock()

    return AsyncIngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_db=vector_db,
        collection="test_collection",
        loader=loader,
        sanitizer=sanitizer,
        observer=observer,
        embed_batch_size=50,
    )


# ── AsyncQueryPipeline tests ──────────────────────────────────────────────────


class TestAsyncQueryPipeline:
    def test_arun_returns_pipeline_result(self):
        pipeline = _make_async_query_pipeline("The answer.")

        result = asyncio.run(pipeline.arun("What is NexRAG?"))

        assert isinstance(result, PipelineResult)
        assert result.answer == "The answer."
        assert result.query == "What is NexRAG?"

    def test_arun_uses_async_embedder(self):
        pipeline = _make_async_query_pipeline()

        asyncio.run(pipeline.arun("q"))

        pipeline._embedder.async_embed_query.assert_awaited_once_with("q")

    def test_arun_uses_async_retriever(self):
        pipeline = _make_async_query_pipeline()

        asyncio.run(pipeline.arun("q"))

        pipeline._retriever.async_retrieve.assert_awaited_once()

    def test_arun_uses_async_llm(self):
        pipeline = _make_async_query_pipeline("answer")

        asyncio.run(pipeline.arun("q"))

        pipeline._llm.async_generate.assert_awaited_once()

    def test_arun_fires_all_stage_events(self):
        pipeline = _make_async_query_pipeline()

        asyncio.run(pipeline.arun("q"))

        emit_calls = [c.args[0] for c in pipeline._observer.async_emit.call_args_list]
        stages = {e.stage for e in emit_calls}
        assert {"embedder", "retriever", "prompt_builder", "llm", "response_builder"}.issubset(
            stages
        )

    def test_arun_collection_override(self):
        pipeline = _make_async_query_pipeline()

        asyncio.run(pipeline.arun("q", collection="custom"))

        call_kwargs = pipeline._retriever.async_retrieve.call_args.kwargs
        assert call_kwargs["collection"] == "custom"

    def test_arun_top_k_override(self):
        pipeline = _make_async_query_pipeline()

        asyncio.run(pipeline.arun("q", top_k=20))

        call_kwargs = pipeline._retriever.async_retrieve.call_args.kwargs
        assert call_kwargs["top_k"] == 20

    def test_arun_result_has_sources(self):
        pipeline = _make_async_query_pipeline()

        result = asyncio.run(pipeline.arun("q"))

        assert len(result.sources) == 1
        assert result.sources[0].source == "test.pdf"

    def test_astream_yields_tokens(self):
        pipeline = _make_async_query_pipeline("The answer")

        async def collect():
            return [item async for item in pipeline.astream("q")]

        items = asyncio.run(collect())
        tokens = [item for item in items if isinstance(item, str)]
        assert tokens == ["The", "answer"]

    def test_astream_last_item_is_run_metrics(self):
        from nexrag.core.models.metrics import RunMetrics

        pipeline = _make_async_query_pipeline("word")

        async def collect():
            return [item async for item in pipeline.astream("q")]

        items = asyncio.run(collect())
        assert isinstance(items[-1], RunMetrics)

    def test_astream_metrics_has_all_stage_latencies(self):
        from nexrag.core.models.metrics import RunMetrics

        pipeline = _make_async_query_pipeline("word")

        async def collect():
            return [item async for item in pipeline.astream("q")]

        items = asyncio.run(collect())
        m = next(item for item in items if isinstance(item, RunMetrics))
        assert "embedder" in m.stage_latencies
        assert "retriever" in m.stage_latencies
        assert "prompt_builder" in m.stage_latencies
        assert "llm" in m.stage_latencies

    def test_astream_fires_llm_events(self):
        pipeline = _make_async_query_pipeline("word")

        async def run():
            async for _ in pipeline.astream("q"):
                pass

        asyncio.run(run())
        emit_calls = [c.args[0] for c in pipeline._observer.async_emit.call_args_list]
        llm_events = [e for e in emit_calls if e.stage == "llm"]
        statuses = {e.status for e in llm_events}
        assert "started" in statuses
        assert "completed" in statuses


# ── AsyncIngestionPipeline tests ──────────────────────────────────────────────


class TestAsyncIngestionPipeline:
    def test_aingest_documents_returns_ingestion_result(self):
        pipeline = _make_async_ingestion_pipeline()
        docs = [_make_document()]

        result = asyncio.run(pipeline.aingest_documents(docs))

        assert isinstance(result, IngestionResult)
        assert result.documents_loaded == 1

    def test_aingest_documents_chunks_written(self):
        pipeline = _make_async_ingestion_pipeline()
        docs = [_make_document()]

        result = asyncio.run(pipeline.aingest_documents(docs))

        assert result.chunks_produced == 1

    def test_aingest_calls_async_embed(self):
        pipeline = _make_async_ingestion_pipeline()
        docs = [_make_document()]

        asyncio.run(pipeline.aingest_documents(docs))

        pipeline._embedder.async_embed.assert_awaited()

    def test_aingest_calls_async_upsert(self):
        pipeline = _make_async_ingestion_pipeline()
        docs = [_make_document()]

        asyncio.run(pipeline.aingest_documents(docs))

        pipeline._vector_db.async_upsert.assert_awaited_once()

    def test_aingest_fires_embedder_events(self):
        pipeline = _make_async_ingestion_pipeline()

        asyncio.run(pipeline.aingest_documents([_make_document()]))

        emit_calls = [c.args[0] for c in pipeline._observer.async_emit.call_args_list]
        embedder_events = [e for e in emit_calls if e.stage == "embedder"]
        assert any(e.status == "started" for e in embedder_events)
        assert any(e.status == "completed" for e in embedder_events)

    def test_aingest_empty_documents_raises(self):
        pipeline = _make_async_ingestion_pipeline()

        with pytest.raises(PipelineError):
            asyncio.run(pipeline.aingest_documents([]))

    def test_parallel_embedding_batches(self):
        """Verify that multiple embedding batches are gathered in parallel."""
        chunks = [_make_chunk(f"chunk {i}") for i in range(5)]

        chunker = MagicMock()
        chunker.chunk.return_value = chunks

        call_count = 0

        async def mock_async_embed(texts):
            nonlocal call_count
            call_count += 1
            return [[0.1, 0.2, 0.3] for _ in texts]

        embedder = MagicMock()
        embedder.model_name = "test-model"
        embedder.dimensions = 3
        embedder.async_embed = mock_async_embed

        vector_db = MagicMock()
        vector_db.get_collection_metadata.return_value = {}
        vector_db.set_collection_metadata.return_value = None
        vector_db.async_upsert = AsyncMock()
        vector_db.async_query = AsyncMock(return_value=[])
        vector_db.async_get_ids_by_metadata = AsyncMock(return_value=[])

        sanitizer = MagicMock()
        sanitizer.sanitize.return_value = MagicMock()

        observer = MagicMock()
        observer.async_emit = AsyncMock()

        pipeline = AsyncIngestionPipeline(
            chunker=chunker,
            embedder=embedder,
            vector_db=vector_db,
            collection="test",
            sanitizer=sanitizer,
            observer=observer,
            embed_batch_size=2,  # 5 chunks / batch_size=2 → 3 batches
        )

        asyncio.run(pipeline.aingest_documents([_make_document()]))

        assert call_count == 3  # ceil(5/2) = 3 batches

    def test_collection_lock_serializes_concurrent_same_collection(self):
        """Two concurrent ingestions into the same collection must not overlap DB ops."""
        order: list[str] = []

        async def tracked_upsert(_chunks, _embeddings, _collection_name):
            order.append("upsert_start")
            await asyncio.sleep(0)  # yield to allow other coroutine to run
            order.append("upsert_end")

        pipeline = _make_async_ingestion_pipeline()
        pipeline._vector_db.async_upsert = tracked_upsert

        async def run_two_concurrent():
            await asyncio.gather(
                pipeline.aingest_documents([_make_document()]),
                pipeline.aingest_documents([_make_document()]),
            )

        asyncio.run(run_two_concurrent())

        # With the lock, upserts must not interleave: start1, end1, start2, end2
        assert order == ["upsert_start", "upsert_end", "upsert_start", "upsert_end"]
