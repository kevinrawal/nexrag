"""
Integration test: ChromaDB (memory) → DenseRetriever → DefaultPromptBuilder → mock LLM.

Verifies the full query pipeline wires correctly. Embedder and LLM are mocked.
ChromaDB runs in-process memory mode. This is safe to run in CI without API keys.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from nexrag.adapters.vector_dbs.chroma import ChromaDBAdapter
from nexrag.chunkers.recursive import RecursiveChunker
from nexrag.core.pipeline.ingestion import IngestionPipeline
from nexrag.core.pipeline.query import QueryPipeline
from nexrag.defaults.prompt_builder import DefaultPromptBuilder
from nexrag.loaders.raw import RawTextLoader
from nexrag.retrievers.dense import DenseRetriever


def _varied_text(n_sentences: int = 20) -> str:
    return " ".join(f"NexRAG sentence {i} covers topic {i % 5}." for i in range(n_sentences))


def _mock_embedder(dims: int = 4):
    embedder = MagicMock()
    embedder.model_name = "mock-model"
    embedder.dimensions = dims
    embedder.embed.side_effect = lambda texts: [[0.5] * dims for _ in texts]
    embedder.embed_query.return_value = [0.5] * dims
    return embedder


def _mock_llm(answer: str = "The answer is 42."):
    llm = MagicMock()
    llm.generate.return_value = answer
    return llm


@pytest.fixture
def col():
    return f"docs_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def query_pipeline(col):
    vector_db = ChromaDBAdapter(mode="memory")
    embedder = _mock_embedder()
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0, min_chunk_size=1)
    loader = RawTextLoader()

    ingestion = IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_db=vector_db,
        collection=col,
        loader=loader,
    )
    ingestion.ingest(_varied_text(20))

    retriever = DenseRetriever(vector_db=vector_db)
    prompt_builder = DefaultPromptBuilder()
    llm = _mock_llm()

    pipeline = QueryPipeline(
        embedder=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        collection=col,
        top_k=3,
        score_threshold=0.0,
    )
    return pipeline


class TestQueryE2E:
    def test_query_returns_pipeline_result(self, query_pipeline):
        from nexrag.core.models.result import PipelineResult

        result = query_pipeline.run("What is NexRAG?")
        assert isinstance(result, PipelineResult)

    def test_query_answer_is_string(self, query_pipeline):
        result = query_pipeline.run("What is NexRAG?")
        assert isinstance(result.answer, str)
        assert result.answer == "The answer is 42."

    def test_query_has_sources(self, query_pipeline):
        result = query_pipeline.run("What is NexRAG?")
        assert len(result.sources) > 0

    def test_query_pipeline_id_is_set(self, query_pipeline):
        result = query_pipeline.run("What is NexRAG?")
        assert result.pipeline_id

    def test_query_latency_is_non_negative(self, query_pipeline):
        result = query_pipeline.run("What is NexRAG?")
        assert result.latency_ms >= 0

    def test_collection_override(self, query_pipeline, col):
        result = query_pipeline.run("q", collection=col)
        assert result.answer

    def test_top_k_override(self, query_pipeline):
        result = query_pipeline.run("What is NexRAG?", top_k=1)
        assert len(result.sources) <= 1

    def test_embedder_called_with_query(self, query_pipeline):
        query_pipeline.run("test query")
        query_pipeline._embedder.embed_query.assert_called_with("test query")
