"""Tests for BM25Retriever — issue #10."""

import uuid
from unittest.mock import MagicMock

import pytest

from nexrag.adapters.vector_dbs.chroma import ChromaDBAdapter
from nexrag.core.models.chunk import Chunk
from nexrag.retrievers.sparse.bm25 import BM25Retriever, _normalize_scores


def _make_chunk(text: str, source: str = "test.pdf") -> Chunk:
    return Chunk(
        text=text,
        chunk_index=0,
        total_chunks=1,
        parent_doc_id="doc1",
        metadata={"source": source},
    )


class TestNormalizeScores:
    def test_empty_returns_empty(self):
        assert _normalize_scores([]) == []

    def test_all_same_nonzero_returns_ones(self):
        result = _normalize_scores([5.0, 5.0, 5.0])
        assert all(r == 1.0 for r in result)

    def test_all_zero_returns_zeros(self):
        result = _normalize_scores([0.0, 0.0, 0.0])
        assert all(r == 0.0 for r in result)

    def test_range_normalized_to_0_1(self):
        result = _normalize_scores([0.0, 5.0, 10.0])
        assert result[0] == pytest.approx(0.0)
        assert result[1] == pytest.approx(0.5)
        assert result[2] == pytest.approx(1.0)


class TestBM25Retriever:
    @pytest.fixture
    def adapter(self):
        return ChromaDBAdapter(mode="memory")

    @pytest.fixture
    def col(self):
        return f"test_{uuid.uuid4().hex[:8]}"

    def test_empty_corpus_returns_empty(self, adapter, col):
        retriever = BM25Retriever(vector_db=adapter)
        results = retriever.retrieve("query", [], top_k=5, collection=col)
        assert results == []

    def test_basic_keyword_match(self, adapter, col):
        chunks = [
            _make_chunk("Python programming language"),
            _make_chunk("JavaScript web development"),
            _make_chunk("Python data science"),
        ]
        embeddings = [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]]
        adapter.upsert(chunks, embeddings, col)

        retriever = BM25Retriever(vector_db=adapter)
        results = retriever.retrieve("Python", [], top_k=3, collection=col)

        assert len(results) > 0
        # Python-related docs should rank above JavaScript
        python_ranks = [r.rank for r in results if "Python" in r.chunk.text]
        js_ranks = [r.rank for r in results if "JavaScript" in r.chunk.text]
        if python_ranks and js_ranks:
            assert min(python_ranks) < min(js_ranks)

    def test_top_k_respected(self, adapter, col):
        chunks = [_make_chunk(f"document about topic {i}") for i in range(10)]
        embeddings = [[float(i), 0.0] for i in range(10)]
        adapter.upsert(chunks, embeddings, col)

        retriever = BM25Retriever(vector_db=adapter)
        results = retriever.retrieve("document", [], top_k=3, collection=col)
        assert len(results) <= 3

    def test_results_are_scored_chunks(self, adapter, col):
        adapter.upsert([_make_chunk("some text")], [[1.0, 0.0]], col)
        retriever = BM25Retriever(vector_db=adapter)
        results = retriever.retrieve("some", [], top_k=5, collection=col)
        assert len(results) == 1
        assert hasattr(results[0], "chunk")
        assert hasattr(results[0], "score")
        assert hasattr(results[0], "rank")

    def test_rank_starts_at_one(self, adapter, col):
        chunks = [_make_chunk(f"text {i}") for i in range(3)]
        embeddings = [[float(i), 0.0] for i in range(3)]
        adapter.upsert(chunks, embeddings, col)

        retriever = BM25Retriever(vector_db=adapter)
        results = retriever.retrieve("text", [], top_k=3, collection=col)
        ranks = [r.rank for r in results]
        assert 1 in ranks

    def test_scores_in_descending_order(self, adapter, col):
        chunks = [
            _make_chunk("apple fruit food"),
            _make_chunk("car vehicle transport"),
            _make_chunk("apple cider vinegar"),
        ]
        embeddings = [[1.0, 0.0], [0.8, 0.1], [0.6, 0.2]]
        adapter.upsert(chunks, embeddings, col)

        retriever = BM25Retriever(vector_db=adapter)
        results = retriever.retrieve("apple", [], top_k=3, collection=col)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_bm25_not_installed_raises_retriever_error(self):
        import sys
        from unittest.mock import patch

        db = MagicMock()
        db.get_all.return_value = [_make_chunk("test")]
        retriever = BM25Retriever(vector_db=db)

        with patch.dict(sys.modules, {"rank_bm25": None}):
            from nexrag.exceptions import RetrieverError

            with pytest.raises(RetrieverError, match="rank_bm25"):
                retriever.retrieve("test", [], top_k=5, collection="col")
