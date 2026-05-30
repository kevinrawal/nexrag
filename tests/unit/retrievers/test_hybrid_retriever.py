"""Tests for HybridRetriever — issue #10."""

import uuid
from unittest.mock import MagicMock

import pytest

from nexrag.adapters.vector_dbs.chroma import ChromaDBAdapter
from nexrag.core.models.chunk import Chunk, ScoredChunk
from nexrag.retrievers.hybrid import HybridRetriever


def _make_chunk(text: str, idx: int = 0) -> Chunk:
    return Chunk(
        text=text,
        chunk_index=idx,
        total_chunks=1,
        parent_doc_id="doc1",
        metadata={"source": f"doc-{idx}.pdf"},
    )


def _make_scored(chunk: Chunk, score: float, rank: int = 1) -> ScoredChunk:
    return ScoredChunk(chunk=chunk, score=score, rank=rank)


class TestHybridRetrieverFusion:
    def test_alpha_1_favors_dense(self):
        """alpha=1.0 should produce results strongly ordered by dense score."""
        c1 = _make_chunk("semantic match text", 1)
        c2 = _make_chunk("keyword match word", 2)

        db = MagicMock()
        db.query.return_value = [
            _make_scored(c1, 0.95, 1),
            _make_scored(c2, 0.20, 2),
        ]
        db.get_all.return_value = [c1, c2]

        retriever = HybridRetriever(vector_db=db, alpha=1.0)
        results = retriever.retrieve("semantic match text", [0.1, 0.2], top_k=2, collection="col")

        assert len(results) > 0
        # With alpha=1.0, dense scores dominate — c1 (dense=0.95) should rank #1
        assert results[0].chunk.text == "semantic match text"

    def test_invalid_alpha_raises(self):
        db = MagicMock()
        with pytest.raises(ValueError, match="alpha"):
            HybridRetriever(vector_db=db, alpha=1.5)

    def test_empty_both_returns_empty(self):
        db = MagicMock()
        db.query.return_value = []
        db.get_all.return_value = []

        retriever = HybridRetriever(vector_db=db, alpha=0.7)
        results = retriever.retrieve("query", [0.1], top_k=5, collection="col")
        assert results == []

    def test_top_k_respected(self):
        chunks = [_make_chunk(f"text {i}", i) for i in range(10)]

        db = MagicMock()
        db.query.return_value = [
            _make_scored(c, 0.9 - i * 0.05, i + 1) for i, c in enumerate(chunks)
        ]
        db.get_all.return_value = chunks

        retriever = HybridRetriever(vector_db=db, alpha=0.7)
        results = retriever.retrieve("text", [1.0, 0.0], top_k=3, collection="col")
        assert len(results) <= 3

    def test_rank_starts_at_one(self):
        c1 = _make_chunk("first chunk", 1)
        c2 = _make_chunk("second chunk", 2)

        db = MagicMock()
        db.query.return_value = [_make_scored(c1, 0.9, 1), _make_scored(c2, 0.5, 2)]
        db.get_all.return_value = [c1, c2]

        retriever = HybridRetriever(vector_db=db, alpha=0.7)
        results = retriever.retrieve("chunk", [1.0], top_k=2, collection="col")
        ranks = {r.rank for r in results}
        assert 1 in ranks

    def test_scores_descending_order(self):
        chunks = [_make_chunk(f"doc {i}", i) for i in range(5)]

        db = MagicMock()
        db.query.return_value = [
            _make_scored(c, 0.9 - i * 0.1, i + 1) for i, c in enumerate(chunks)
        ]
        db.get_all.return_value = chunks

        retriever = HybridRetriever(vector_db=db, alpha=0.7)
        results = retriever.retrieve("doc", [1.0], top_k=5, collection="col")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_threshold_filters(self):
        chunks = [_make_chunk(f"text {i}", i) for i in range(4)]

        db = MagicMock()
        db.query.return_value = [
            _make_scored(c, 0.9 - i * 0.3, i + 1) for i, c in enumerate(chunks)
        ]
        db.get_all.return_value = chunks

        retriever = HybridRetriever(vector_db=db, alpha=0.7)
        results = retriever.retrieve("text", [1.0], top_k=10, collection="col", score_threshold=0.5)
        for r in results:
            assert r.score >= 0.5


class TestHybridRetrieverIntegration:
    """Integration tests using real ChromaDB memory instance."""

    @pytest.fixture
    def adapter(self):
        return ChromaDBAdapter(mode="memory")

    @pytest.fixture
    def col(self):
        return f"test_{uuid.uuid4().hex[:8]}"

    def test_hybrid_returns_results_from_real_db(self, adapter, col):
        chunks = [
            Chunk(
                text="machine learning models",
                chunk_index=0,
                total_chunks=1,
                parent_doc_id="d1",
                metadata={"source": "ml.pdf"},
            ),
            Chunk(
                text="deep neural networks",
                chunk_index=0,
                total_chunks=1,
                parent_doc_id="d2",
                metadata={"source": "nn.pdf"},
            ),
            Chunk(
                text="Python web development",
                chunk_index=0,
                total_chunks=1,
                parent_doc_id="d3",
                metadata={"source": "web.pdf"},
            ),
        ]
        embeddings = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]]
        adapter.upsert(chunks, embeddings, col)

        retriever = HybridRetriever(vector_db=adapter, alpha=0.6)
        results = retriever.retrieve("machine learning", [1.0, 0.0, 0.0], top_k=3, collection=col)
        assert len(results) > 0
        assert all(hasattr(r, "score") for r in results)
