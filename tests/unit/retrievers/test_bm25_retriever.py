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
        retriever = BM25Retriever(vector_db=adapter, cache_dir=None)
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

        retriever = BM25Retriever(vector_db=adapter, cache_dir=None)
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

        retriever = BM25Retriever(vector_db=adapter, cache_dir=None)
        results = retriever.retrieve("document", [], top_k=3, collection=col)
        assert len(results) <= 3

    def test_results_are_scored_chunks(self, adapter, col):
        adapter.upsert([_make_chunk("some text")], [[1.0, 0.0]], col)
        retriever = BM25Retriever(vector_db=adapter, cache_dir=None)
        results = retriever.retrieve("some", [], top_k=5, collection=col)
        assert len(results) == 1
        assert hasattr(results[0], "chunk")
        assert hasattr(results[0], "score")
        assert hasattr(results[0], "rank")

    def test_rank_starts_at_one(self, adapter, col):
        chunks = [_make_chunk(f"text {i}") for i in range(3)]
        embeddings = [[float(i), 0.0] for i in range(3)]
        adapter.upsert(chunks, embeddings, col)

        retriever = BM25Retriever(vector_db=adapter, cache_dir=None)
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

        retriever = BM25Retriever(vector_db=adapter, cache_dir=None)
        results = retriever.retrieve("apple", [], top_k=3, collection=col)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_rank_bm25_not_installed_raises_retriever_error(self):
        import sys
        from unittest.mock import patch

        db = MagicMock()
        db.get_all.return_value = [_make_chunk("test")]
        retriever = BM25Retriever(vector_db=db, cache_dir=None)

        with patch.dict(sys.modules, {"rank_bm25": None}):
            from nexrag.exceptions import RetrieverError

            with pytest.raises(RetrieverError, match="rank_bm25"):
                retriever.retrieve("test", [], top_k=5, collection="col")


class TestBM25RetrieverMetadataFilter:
    @pytest.fixture
    def adapter(self):
        return ChromaDBAdapter(mode="memory")

    @pytest.fixture
    def col(self):
        return f"filter_{uuid.uuid4().hex[:8]}"

    def test_filter_returns_only_matching_chunks(self, adapter, col):
        chunks = [
            _make_chunk("contract clause about payment", source="contract.pdf"),
            _make_chunk("invoice number and amount", source="invoice.pdf"),
            _make_chunk("contract renewal terms", source="contract.pdf"),
        ]
        adapter.upsert(chunks, [[float(i), 0.0] for i in range(3)], col)

        retriever = BM25Retriever(vector_db=adapter, cache_dir=None)
        results = retriever.retrieve(
            "contract",
            [],
            top_k=10,
            collection=col,
            filters={"source": "contract.pdf"},
        )

        assert len(results) == 2
        assert all(r.chunk.metadata["source"] == "contract.pdf" for r in results)

    def test_filter_no_match_returns_empty(self, adapter, col):
        chunks = [_make_chunk("some text", source="a.pdf")]
        adapter.upsert(chunks, [[1.0, 0.0]], col)

        retriever = BM25Retriever(vector_db=adapter, cache_dir=None)
        results = retriever.retrieve(
            "text",
            [],
            top_k=5,
            collection=col,
            filters={"source": "b.pdf"},
        )
        assert results == []

    def test_no_filter_returns_all_chunks(self, adapter, col):
        chunks = [
            _make_chunk("doc one alpha", source="a.pdf"),
            _make_chunk("doc two alpha", source="b.pdf"),
        ]
        adapter.upsert(chunks, [[1.0, 0.0], [0.9, 0.1]], col)

        retriever = BM25Retriever(vector_db=adapter, cache_dir=None)
        results = retriever.retrieve("alpha", [], top_k=10, collection=col)
        assert len(results) == 2

    def test_filter_ranks_are_contiguous_from_one(self, adapter, col):
        chunks = [
            _make_chunk("alpha beta", source="x.pdf"),
            _make_chunk("alpha gamma", source="y.pdf"),
            _make_chunk("delta epsilon", source="x.pdf"),
        ]
        adapter.upsert(chunks, [[float(i), 0.0] for i in range(3)], col)

        retriever = BM25Retriever(vector_db=adapter, cache_dir=None)
        results = retriever.retrieve(
            "alpha", [], top_k=10, collection=col, filters={"source": "x.pdf"}
        )
        ranks = [r.rank for r in results]
        assert ranks == list(range(1, len(ranks) + 1))


class TestBM25RetrieverCaching:
    def test_second_retrieve_uses_cache(self):
        db = MagicMock()
        chunks = [_make_chunk("hello world"), _make_chunk("goodbye world")]
        db.get_all.return_value = chunks

        retriever = BM25Retriever(vector_db=db, cache_dir=None)
        retriever.retrieve("hello", [], top_k=5, collection="col")
        retriever.retrieve("hello", [], top_k=5, collection="col")

        assert db.get_all.call_count == 1

    def test_invalidate_cache_triggers_rebuild(self):
        db = MagicMock()
        db.get_all.return_value = [_make_chunk("hello world")]

        retriever = BM25Retriever(vector_db=db, cache_dir=None)
        retriever.retrieve("hello", [], top_k=5, collection="col")
        retriever.invalidate_cache("col")
        retriever.retrieve("hello", [], top_k=5, collection="col")

        assert db.get_all.call_count == 2

    def test_invalidate_none_clears_all_collections(self):
        db = MagicMock()
        db.get_all.return_value = [_make_chunk("some text")]

        retriever = BM25Retriever(vector_db=db, cache_dir=None)
        retriever.retrieve("some", [], top_k=5, collection="col_a")
        retriever.retrieve("some", [], top_k=5, collection="col_b")
        assert db.get_all.call_count == 2

        retriever.invalidate_cache()  # clear all
        retriever.retrieve("some", [], top_k=5, collection="col_a")
        retriever.retrieve("some", [], top_k=5, collection="col_b")
        assert db.get_all.call_count == 4

    def test_different_collections_cached_independently(self):
        db = MagicMock()
        db.get_all.return_value = [_make_chunk("text")]

        retriever = BM25Retriever(vector_db=db, cache_dir=None)
        retriever.retrieve("text", [], top_k=5, collection="col_a")
        retriever.retrieve("text", [], top_k=5, collection="col_a")
        retriever.retrieve("text", [], top_k=5, collection="col_b")
        retriever.retrieve("text", [], top_k=5, collection="col_b")

        assert db.get_all.call_count == 2  # one build per collection

    def test_ttl_expiry_triggers_rebuild(self):
        db = MagicMock()
        db.get_all.return_value = [_make_chunk("text")]

        retriever = BM25Retriever(vector_db=db, cache_ttl=0.01, cache_dir=None)
        retriever.retrieve("text", [], top_k=5, collection="col")

        import time

        time.sleep(0.02)

        retriever.retrieve("text", [], top_k=5, collection="col")
        assert db.get_all.call_count == 2


class TestBM25RetrieverPagination:
    def test_corpus_fetched_in_pages(self, monkeypatch):
        from nexrag.retrievers.sparse import bm25 as bm25_module

        monkeypatch.setattr(bm25_module, "_FETCH_PAGE_SIZE", 2)

        chunks = [_make_chunk(f"doc number {i}") for i in range(5)]

        def _paged_get_all(collection, limit=None, offset=None):
            start = offset or 0
            return chunks[start : start + (limit or len(chunks))]

        db = MagicMock()
        db.get_all.side_effect = _paged_get_all
        db.count.return_value = 5

        retriever = BM25Retriever(vector_db=db, cache_dir=None)
        results = retriever.retrieve("doc", [], top_k=10, collection="col")

        # 5 chunks at page size 2 → offsets 0, 2, 4 (last page short → stop).
        assert len(results) == 5
        offsets = [c.kwargs.get("offset") for c in db.get_all.call_args_list]
        assert offsets == [0, 2, 4]


class TestBM25RetrieverDiskCache:
    def test_cold_process_reloads_from_disk_without_refetch(self, tmp_path):
        chunks = [_make_chunk("alpha beta"), _make_chunk("gamma delta")]
        db = MagicMock()
        db.get_all.return_value = chunks
        db.count.return_value = len(chunks)

        cache_dir = str(tmp_path / "bm25")

        # First retriever builds the index and persists it to disk.
        r1 = BM25Retriever(vector_db=db, cache_dir=cache_dir)
        r1.retrieve("alpha", [], top_k=5, collection="col")
        assert db.get_all.call_count == 1

        # A fresh retriever (cold L1) over the same disk cache must NOT refetch.
        db2 = MagicMock()
        db2.get_all.return_value = chunks
        db2.count.return_value = len(chunks)
        r2 = BM25Retriever(vector_db=db2, cache_dir=cache_dir)
        results = r2.retrieve("alpha", [], top_k=5, collection="col")

        assert db2.get_all.call_count == 0  # served from disk
        assert len(results) > 0

    def test_stale_disk_index_rebuilt_on_count_change(self, tmp_path):
        chunks = [_make_chunk("alpha beta"), _make_chunk("gamma delta")]
        db = MagicMock()
        db.get_all.return_value = chunks
        db.count.return_value = len(chunks)
        cache_dir = str(tmp_path / "bm25")

        r1 = BM25Retriever(vector_db=db, cache_dir=cache_dir)
        r1.retrieve("alpha", [], top_k=5, collection="col")

        # New process sees a different document count → disk index is stale.
        db2 = MagicMock()
        db2.get_all.return_value = chunks + [_make_chunk("epsilon")]
        db2.count.return_value = 3
        r2 = BM25Retriever(vector_db=db2, cache_dir=cache_dir)
        r2.retrieve("alpha", [], top_k=5, collection="col")

        assert db2.get_all.call_count == 1  # rebuilt, not trusted from disk

    def test_invalidate_cache_clears_disk(self, tmp_path):
        chunks = [_make_chunk("alpha beta")]
        db = MagicMock()
        db.get_all.return_value = chunks
        db.count.return_value = len(chunks)
        cache_dir = str(tmp_path / "bm25")

        retriever = BM25Retriever(vector_db=db, cache_dir=cache_dir)
        retriever.retrieve("alpha", [], top_k=5, collection="col")
        retriever.invalidate_cache("col")

        # Disk entry gone → a fresh retriever must rebuild from the DB.
        db2 = MagicMock()
        db2.get_all.return_value = chunks
        db2.count.return_value = len(chunks)
        r2 = BM25Retriever(vector_db=db2, cache_dir=cache_dir)
        r2.retrieve("alpha", [], top_k=5, collection="col")
        assert db2.get_all.call_count == 1
