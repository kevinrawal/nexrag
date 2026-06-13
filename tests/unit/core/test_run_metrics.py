"""Tests for RunMetrics — issue #13."""

from unittest.mock import MagicMock

from nexrag.core.models.chunk import Chunk, ScoredChunk
from nexrag.core.models.document import Document
from nexrag.core.models.metrics import RunMetrics, TokenUsage
from nexrag.core.pipeline.ingestion import IngestionPipeline
from nexrag.core.pipeline.query import QueryPipeline


def _make_scored_chunk(text: str = "chunk") -> ScoredChunk:
    chunk = Chunk(
        text=text,
        chunk_index=0,
        total_chunks=1,
        parent_doc_id="d1",
        metadata={"source": "test.pdf"},
    )
    return ScoredChunk(chunk=chunk, score=0.9, rank=1)


def _make_query_pipeline() -> QueryPipeline:
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2]
    embedder.model_name = "embed-model"

    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_scored_chunk(), _make_scored_chunk("chunk2")]

    prompt_builder = MagicMock()
    prompt_builder.build.return_value = "prompt"

    llm = MagicMock()
    llm.generate.return_value = ("answer text", TokenUsage(100, 50, 150))
    llm.model_name = "gpt-4o"

    observer = MagicMock()
    return QueryPipeline(
        embedder=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        collection="test",
        observer=observer,
    )


def _make_ingestion_pipeline() -> IngestionPipeline:
    chunk = MagicMock()
    chunk.text = "chunk"
    chunk.content_hash = "abc"
    chunk.metadata = {"source": "test.pdf"}

    chunker = MagicMock()
    chunker.chunk.return_value = [chunk]

    embedder = MagicMock()
    embedder.model_name = "embed-model"
    embedder.dimensions = 3
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    vector_db = MagicMock()
    vector_db.get_collection_metadata.return_value = {}
    vector_db.query.return_value = []

    sanitizer = MagicMock()
    doc = Document(content="test content", metadata={"source": "test.pdf"})
    sanitizer.sanitize.return_value = doc

    return IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_db=vector_db,
        collection="test",
        sanitizer=sanitizer,
    )


class TestRunMetrics:
    def test_query_result_has_metrics(self):
        pipeline = _make_query_pipeline()
        result = pipeline.run("What?")
        assert result.metrics is not None
        assert isinstance(result.metrics, RunMetrics)

    def test_metrics_total_latency_non_negative(self):
        pipeline = _make_query_pipeline()
        result = pipeline.run("query")
        assert result.metrics.total_latency_ms >= 0.0

    def test_metrics_stage_latencies_populated(self):
        pipeline = _make_query_pipeline()
        result = pipeline.run("query")
        stages = result.metrics.stage_latencies
        assert "embedder" in stages
        assert "retriever" in stages
        assert "llm" in stages
        assert all(v >= 0 for v in stages.values())

    def test_metrics_token_usage_from_llm(self):
        pipeline = _make_query_pipeline()
        result = pipeline.run("query")
        assert result.metrics.token_usage is not None
        assert result.metrics.token_usage.total_tokens == 150
        assert result.metrics.token_usage.prompt_tokens == 100
        assert result.metrics.token_usage.completion_tokens == 50

    def test_metrics_model_populated(self):
        pipeline = _make_query_pipeline()
        result = pipeline.run("query")
        assert result.metrics.model == "gpt-4o"

    def test_metrics_chunks_retrieved_matches_sources(self):
        pipeline = _make_query_pipeline()
        result = pipeline.run("query")
        assert result.metrics.chunks_retrieved == len(result.sources)

    def test_ingestion_result_has_metrics(self):
        pipeline = _make_ingestion_pipeline()
        doc = Document(content="test content", metadata={"source": "test.pdf"})
        result = pipeline.ingest_documents([doc])
        assert result.metrics is not None
        assert isinstance(result.metrics, RunMetrics)

    def test_ingestion_metrics_stage_latencies(self):
        pipeline = _make_ingestion_pipeline()
        doc = Document(content="test content", metadata={"source": "test.pdf"})
        result = pipeline.ingest_documents([doc])
        stages = result.metrics.stage_latencies
        assert "chunker" in stages
        assert "embedder" in stages
        assert all(v >= 0 for v in stages.values())

    def test_ingestion_metrics_chunks_written(self):
        pipeline = _make_ingestion_pipeline()
        doc = Document(content="test content", metadata={"source": "test.pdf"})
        result = pipeline.ingest_documents([doc])
        assert result.metrics.chunks_written == result.chunks_written

    def test_run_metrics_as_dict(self):
        metrics = RunMetrics(
            pipeline_id="test-id",
            total_latency_ms=123.4,
            stage_latencies={"embedder": 10.0, "llm": 80.0},
            token_usage=TokenUsage(100, 50, 150),
            model="gpt-4o",
            chunks_retrieved=3,
        )
        d = metrics.as_dict()
        assert d["pipeline_id"] == "test-id"
        assert d["total_latency_ms"] == 123.4
        assert d["token_usage"]["total_tokens"] == 150
        assert d["model"] == "gpt-4o"
        assert d["chunks_retrieved"] == 3
