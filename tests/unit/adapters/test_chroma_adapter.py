import uuid
from unittest.mock import MagicMock, patch

import pytest

from nexrag.adapters.vector_dbs.chroma import _CHUNK_STRUCT_KEYS, ChromaDBAdapter
from nexrag.core.models.chunk import Chunk


def _make_chunk(
    text: str,
    index: int = 0,
    total: int = 1,
    doc_id: str = "doc1",
    metadata: dict | None = None,
) -> Chunk:
    return Chunk(
        text=text,
        chunk_index=index,
        total_chunks=total,
        parent_doc_id=doc_id,
        metadata=metadata if metadata is not None else {"source": "test.pdf"},
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
        # Rows are keyed by the document-scoped row_id, not the bare content_hash.
        adapter.delete([chunk.row_id], col)
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


class TestChromaDBAdapterStructFieldsRoundTrip:
    """chunk_index / total_chunks / parent_doc_id must survive the ChromaDB round-trip."""

    def test_struct_fields_restored_after_query(self, adapter, col):
        chunk = _make_chunk("position test", index=2, total=5, doc_id="doc-xyz")
        adapter.upsert([chunk], [[1.0, 0.0]], col)
        result = adapter.query([1.0, 0.0], top_k=1, collection_name=col)[0].chunk
        assert result.chunk_index == 2
        assert result.total_chunks == 5
        assert result.parent_doc_id == "doc-xyz"

    def test_struct_fields_not_in_chunk_metadata_after_query(self, adapter, col):
        chunk = _make_chunk("clean metadata", index=1, total=3, doc_id="doc-abc")
        adapter.upsert([chunk], [[1.0, 0.0]], col)
        result = adapter.query([1.0, 0.0], top_k=1, collection_name=col)[0].chunk
        for key in _CHUNK_STRUCT_KEYS:
            assert key not in result.metadata

    def test_user_metadata_preserved_and_clean_after_query(self, adapter, col):
        chunk = _make_chunk(
            "meta test", index=0, total=1, metadata={"source": "s3://x", "tenant": "acme"}
        )
        adapter.upsert([chunk], [[1.0, 0.0]], col)
        result = adapter.query([1.0, 0.0], top_k=1, collection_name=col)[0].chunk
        assert result.metadata["source"] == "s3://x"
        assert result.metadata["tenant"] == "acme"
        for key in _CHUNK_STRUCT_KEYS:
            assert key not in result.metadata


class TestChromaDBAdapterEmptyMetadata:
    """Chunks with no document metadata must not crash on upsert or query."""

    def test_upsert_chunk_with_no_metadata_succeeds(self, adapter, col):
        chunk = _make_chunk("no metadata text", metadata={})
        adapter.upsert([chunk], [[0.5, 0.5]], col)
        assert adapter.count(col) == 1

    def test_query_chunk_with_no_metadata_no_crash(self, adapter, col):
        chunk = _make_chunk("sourceless text", metadata={})
        adapter.upsert([chunk], [[1.0, 0.0]], col)
        results = adapter.query([1.0, 0.0], top_k=1, collection_name=col)
        assert len(results) == 1
        assert results[0].chunk.text == "sourceless text"
        assert results[0].chunk.metadata == {}


class TestChromaDBAdapterFilterOperators:
    """Operator dicts are passed through; scalars are wrapped with $eq."""

    def test_scalar_filter_wraps_with_eq(self):
        assert ChromaDBAdapter._build_where({"year": 2024}) == {"year": {"$eq": 2024}}

    def test_operator_dict_passed_through(self):
        f = {"year": {"$gte": 2023}}
        assert ChromaDBAdapter._build_where(f) == {"year": {"$gte": 2023}}

    def test_in_operator_passed_through(self):
        f = {"source": {"$in": ["a.pdf", "b.pdf"]}}
        assert ChromaDBAdapter._build_where(f) == {"source": {"$in": ["a.pdf", "b.pdf"]}}

    def test_multi_filter_with_mixed_operators(self):
        result = ChromaDBAdapter._build_where({"year": {"$gte": 2020}, "source": "x.pdf"})
        assert "$and" in result
        clauses = result["$and"]
        assert {"year": {"$gte": 2020}} in clauses
        assert {"source": {"$eq": "x.pdf"}} in clauses

    def test_gte_filter_matches_in_real_chroma(self, adapter, col):
        chunks = [
            _make_chunk("old doc", metadata={"source": "old", "year": 2019}),
            _make_chunk("new doc", metadata={"source": "new", "year": 2024}),
        ]
        adapter.upsert(chunks, [[1.0, 0.0], [0.9, 0.1]], col)
        results = adapter.query(
            [1.0, 0.0],
            top_k=5,
            collection_name=col,
            filters={"year": {"$gte": 2020}},
        )
        assert len(results) == 1
        assert results[0].chunk.metadata["source"] == "new"

    def test_in_filter_matches_in_real_chroma(self, adapter, col):
        c1 = _make_chunk("alpha", metadata={"source": "alpha"})
        c2 = _make_chunk("beta", metadata={"source": "beta"})
        c3 = _make_chunk("gamma", metadata={"source": "gamma"})
        adapter.upsert([c1, c2, c3], [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2]], col)
        results = adapter.query(
            [1.0, 0.0],
            top_k=5,
            collection_name=col,
            filters={"source": {"$in": ["alpha", "gamma"]}},
        )
        sources = {r.chunk.metadata["source"] for r in results}
        assert sources == {"alpha", "gamma"}


class TestChromaDBAdapterBatchUpsert:
    """Large ingestions are split into upsert_batch_size slices."""

    def test_upsert_larger_than_batch_size_stores_all(self, col):
        adapter = ChromaDBAdapter(mode="memory", upsert_batch_size=10)
        chunks = [
            _make_chunk(f"chunk text {i}", metadata={"source": f"doc-{i}"}) for i in range(25)
        ]
        embeddings = [[float(i), 0.0] for i in range(25)]
        adapter.upsert(chunks, embeddings, col)
        assert adapter.count(col) == 25


class TestChromaDBAdapterListCollections:
    def test_list_collections_empty(self, adapter):
        names = adapter.list_collections()
        assert isinstance(names, list)

    def test_list_collections_returns_created_collections(self, adapter):
        col_a = f"col_a_{uuid.uuid4().hex[:8]}"
        col_b = f"col_b_{uuid.uuid4().hex[:8]}"
        adapter.upsert([_make_chunk("a")], [[0.1, 0.2]], col_a)
        adapter.upsert([_make_chunk("b")], [[0.3, 0.4]], col_b)
        names = adapter.list_collections()
        assert col_a in names
        assert col_b in names


class TestChromaDBAdapterRetry:
    def test_retry_on_connection_failure_then_success(self):
        import chromadb

        real_client = chromadb.EphemeralClient()
        call_count = 0

        def flaky_ephemeral():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("transient error")
            return real_client

        with patch("chromadb.EphemeralClient", side_effect=flaky_ephemeral):
            adapter = ChromaDBAdapter(mode="memory", max_retries=3, retry_delay=0.0)
        assert call_count == 3
        assert adapter._client is real_client

    def test_exhaust_retries_raises_connection_error(self):
        from nexrag.exceptions import VectorDBConnectionError

        with patch("chromadb.EphemeralClient", side_effect=Exception("always fails")):
            with pytest.raises(VectorDBConnectionError):
                ChromaDBAdapter(mode="memory", max_retries=2, retry_delay=0.0)


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
