import uuid
from unittest.mock import MagicMock, patch

import pytest

from nexrag.adapters.vector_dbs.chroma import ChromaDBAdapter
from nexrag.core.models.chunk import Chunk


def _make_chunk(text: str, index: int = 0, total: int = 1, doc_id: str = "doc1") -> Chunk:
    return Chunk(
        text=text,
        chunk_index=index,
        total_chunks=total,
        parent_doc_id=doc_id,
        metadata={"source": "test.pdf"},
    )


@pytest.fixture
def adapter():
    return ChromaDBAdapter(mode="memory")


@pytest.fixture
def col():
    """Unique collection name per test — prevents ChromaDB state leakage."""
    return f"test_{uuid.uuid4().hex[:10]}"


class TestChromaDBAdapter:
    def test_upsert_and_count(self, adapter, col):
        chunk = _make_chunk("Hello world")
        adapter.upsert([chunk], [[0.1, 0.2, 0.3]], col)
        assert adapter.count(col) == 1

    def test_upsert_empty_is_noop(self, adapter, col):
        adapter.upsert([], [], col)
        assert adapter.count(col) == 0

    def test_idempotent_upsert(self, adapter, col):
        chunk = _make_chunk("Same text")
        emb = [[0.1, 0.2]]
        adapter.upsert([chunk], emb, col)
        adapter.upsert([chunk], emb, col)
        assert adapter.count(col) == 1

    def test_query_returns_scored_chunks(self, adapter, col):
        chunk = _make_chunk("Relevant content")
        adapter.upsert([chunk], [[1.0, 0.0, 0.0]], col)
        results = adapter.query([1.0, 0.0, 0.0], top_k=5, collection_name=col)
        assert len(results) == 1
        assert results[0].score >= 0.0
        assert results[0].rank == 1

    def test_query_empty_collection_returns_empty(self, adapter, col):
        results = adapter.query([0.1, 0.2], top_k=5, collection_name=col)
        assert results == []

    def test_delete_removes_chunk(self, adapter, col):
        chunk = _make_chunk("To be deleted")
        adapter.upsert([chunk], [[0.1, 0.2]], col)
        assert adapter.count(col) == 1
        adapter.delete([chunk.content_hash], col)
        assert adapter.count(col) == 0

    def test_delete_empty_list_is_noop(self, adapter, col):
        adapter.delete([], col)

    def test_collection_metadata_roundtrip(self, adapter, col):
        meta = {"model": "text-embedding-3-small", "dims": 1536}
        adapter.set_collection_metadata(col, meta)
        retrieved = adapter.get_collection_metadata(col)
        assert retrieved["model"] == "text-embedding-3-small"
        assert retrieved["dims"] == 1536

    def test_get_collection_metadata_empty_collection(self, adapter, col):
        result = adapter.get_collection_metadata(col)
        assert result == {}

    def test_score_is_between_0_and_1(self, adapter, col):
        chunks = [_make_chunk(f"chunk {i}") for i in range(3)]
        embeddings = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
        adapter.upsert(chunks, embeddings, col)
        results = adapter.query([1.0, 0.0], top_k=3, collection_name=col)
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_serialize_metadata_flattens_non_primitives(self):
        meta = {"key": "val", "num": 1, "nested": {"a": 1}}
        result = ChromaDBAdapter._serialize_metadata(meta)
        assert result["nested"] == str({"a": 1})

    def test_build_where_single_filter(self):
        result = ChromaDBAdapter._build_where({"year": 2024})
        assert result == {"year": {"$eq": 2024}}

    def test_build_where_multi_filter(self):
        result = ChromaDBAdapter._build_where({"year": 2024, "source": "test.pdf"})
        assert "$and" in result

    def test_build_where_none_returns_empty(self):
        assert ChromaDBAdapter._build_where(None) == {}

    def test_multiple_collections_are_independent(self, adapter):
        col_a = f"col_a_{uuid.uuid4().hex[:8]}"
        col_b = f"col_b_{uuid.uuid4().hex[:8]}"
        c1 = _make_chunk("doc a", doc_id="d1")
        c2 = _make_chunk("doc b", doc_id="d2")
        adapter.upsert([c1], [[0.1, 0.2]], col_a)
        adapter.upsert([c2], [[0.3, 0.4]], col_b)
        assert adapter.count(col_a) == 1
        assert adapter.count(col_b) == 1


class TestChromaDBAdapterServerMode:
    def test_server_mode_calls_http_client(self):
        mock_http = MagicMock()
        with patch("chromadb.HttpClient", return_value=mock_http) as patched:
            adapter = ChromaDBAdapter(mode="server", host="chroma.internal", port=8000)
        patched.assert_called_once_with(host="chroma.internal", port=8000)
        assert adapter._client is mock_http

    def test_server_mode_default_host_and_port(self):
        with patch("chromadb.HttpClient") as patched:
            ChromaDBAdapter(mode="server")
        call_kwargs = patched.call_args
        assert call_kwargs.kwargs["host"] == "localhost"
        assert call_kwargs.kwargs["port"] == 8000

    def test_server_mode_custom_port(self):
        with patch("chromadb.HttpClient") as patched:
            ChromaDBAdapter(mode="server", host="chroma.internal", port=9000)
        assert patched.call_args.kwargs["port"] == 9000

    def test_server_mode_connection_error_raises_vector_db_connection_error(self):
        from nexrag.exceptions import VectorDBConnectionError

        with patch("chromadb.HttpClient", side_effect=Exception("connection refused")):
            with pytest.raises(VectorDBConnectionError):
                ChromaDBAdapter(mode="server", host="unreachable")
