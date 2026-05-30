"""
Integration test: RawTextLoader → RecursiveChunker → mock embedder → ChromaDB (memory).

No real API calls. ChromaDB runs in-process memory mode. This verifies the full
ingestion pipeline wires correctly end-to-end.
"""

import uuid
from unittest.mock import MagicMock

import pytest

from nexrag.adapters.vector_dbs.chroma import ChromaDBAdapter
from nexrag.chunkers.recursive import RecursiveChunker
from nexrag.core.models.document import Document
from nexrag.core.pipeline.ingestion import IngestionPipeline
from nexrag.loaders.raw import RawTextLoader


def _varied_text(n_sentences: int = 30) -> str:
    """Generate n distinct sentences so every chunk has a unique content_hash."""
    return " ".join(f"This is sentence number {i} about topic {i % 7}." for i in range(n_sentences))


def _mock_embedder(dims: int = 4):
    embedder = MagicMock()
    embedder.model_name = "mock-model"
    embedder.dimensions = dims
    embedder.embed.side_effect = lambda texts: [[0.1] * dims for _ in texts]
    embedder.embed_query.return_value = [0.1] * dims
    return embedder


@pytest.fixture
def col():
    return f"col_{uuid.uuid4().hex[:10]}"


@pytest.fixture
def pipeline(col):
    loader = RawTextLoader()
    chunker = RecursiveChunker(chunk_size=200, chunk_overlap=0, min_chunk_size=1)
    embedder = _mock_embedder()
    vector_db = ChromaDBAdapter(mode="memory")

    return IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_db=vector_db,
        collection=col,
        loader=loader,
    )


class TestIngestionE2E:
    def test_ingest_raw_text_succeeds(self, pipeline):
        result = pipeline.ingest(_varied_text(30))
        assert result.documents_loaded == 1
        assert result.chunks_produced > 0
        assert result.chunks_written > 0

    def test_ingest_stores_chunks_in_db(self, pipeline, col):
        pipeline.ingest(_varied_text(30))
        count = pipeline._vector_db.count(col)
        assert count > 0

    def test_ingest_documents_directly(self, pipeline):
        docs = [
            Document(content=_varied_text(15), metadata={"source": "doc1.txt"}),
            Document(content=_varied_text(15), metadata={"source": "doc2.txt"}),
        ]
        result = pipeline.ingest_documents(docs)
        assert result.documents_loaded == 2
        assert result.chunks_produced > 0

    def test_second_ingest_same_content_is_skipped(self, pipeline):
        text = _varied_text(30)
        # Source must be set explicitly for idempotency to work — without it every
        # ingest always writes (safe default, but no deduplication).
        r1 = pipeline.ingest(text, metadata={"source": "doc-idempotency-test"})
        r2 = pipeline.ingest(text, metadata={"source": "doc-idempotency-test"})
        assert r1.chunks_written > 0
        assert r2.chunks_written == 0  # idempotency: all hashes match

    def test_pipeline_id_is_unique_per_run(self, pipeline):
        r1 = pipeline.ingest(_varied_text(10))
        r2 = pipeline.ingest(_varied_text(20))
        assert r1.pipeline_id != r2.pipeline_id

    def test_latency_is_positive(self, pipeline):
        result = pipeline.ingest(_varied_text(20))
        assert result.latency_ms >= 0

    def test_ingest_empty_documents_raises(self, pipeline):
        from nexrag.exceptions import PipelineError

        with pytest.raises(PipelineError):
            pipeline.ingest_documents([])
