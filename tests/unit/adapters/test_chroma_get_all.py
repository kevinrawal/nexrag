"""Tests for ChromaDBAdapter.get_all() — issue #10."""

import uuid

import pytest

from nexrag.adapters.vector_dbs.chroma import _CHUNK_STRUCT_KEYS, ChromaDBAdapter
from nexrag.core.models.chunk import Chunk


def _make_chunk(
    text: str, index: int = 0, total: int = 1, doc_id: str = "doc1", metadata: dict | None = None
) -> Chunk:
    return Chunk(
        text=text,
        chunk_index=index,
        total_chunks=total,
        parent_doc_id=doc_id,
        metadata=metadata or {"source": "test.pdf"},
    )


@pytest.fixture
def adapter():
    return ChromaDBAdapter(mode="memory")


@pytest.fixture
def col():
    return f"test_{uuid.uuid4().hex[:10]}"


class TestChromaGetAll:
    def test_empty_collection_returns_empty(self, adapter, col):
        result = adapter.get_all(col)
        assert result == []

    def test_returns_all_upserted_chunks(self, adapter, col):
        chunks = [_make_chunk(f"text {i}", metadata={"source": f"doc-{i}"}) for i in range(5)]
        embeddings = [[float(i), 0.0] for i in range(5)]
        adapter.upsert(chunks, embeddings, col)

        result = adapter.get_all(col)
        assert len(result) == 5

    def test_returned_chunks_have_correct_text(self, adapter, col):
        chunks = [
            _make_chunk("hello world"),
            _make_chunk("goodbye world"),
        ]
        adapter.upsert(chunks, [[1.0, 0.0], [0.0, 1.0]], col)

        result = adapter.get_all(col)
        texts = {c.text for c in result}
        assert texts == {"hello world", "goodbye world"}

    def test_struct_fields_preserved_after_get_all(self, adapter, col):
        chunk = _make_chunk("position test", index=2, total=5, doc_id="doc-xyz")
        adapter.upsert([chunk], [[1.0, 0.0]], col)

        result = adapter.get_all(col)
        assert len(result) == 1
        c = result[0]
        assert c.chunk_index == 2
        assert c.total_chunks == 5
        assert c.parent_doc_id == "doc-xyz"

    def test_struct_keys_not_in_chunk_metadata(self, adapter, col):
        chunk = _make_chunk("clean metadata", index=1, total=3, doc_id="doc-abc")
        adapter.upsert([chunk], [[1.0, 0.0]], col)

        result = adapter.get_all(col)
        for key in _CHUNK_STRUCT_KEYS:
            assert key not in result[0].metadata

    def test_user_metadata_preserved(self, adapter, col):
        chunk = _make_chunk("meta test", metadata={"source": "s3://bucket", "tenant": "acme"})
        adapter.upsert([chunk], [[1.0, 0.0]], col)

        result = adapter.get_all(col)
        assert result[0].metadata["source"] == "s3://bucket"
        assert result[0].metadata["tenant"] == "acme"

    def test_limit_respected(self, adapter, col):
        chunks = [_make_chunk(f"chunk {i}", metadata={"source": f"s{i}"}) for i in range(20)]
        embeddings = [[float(i), 0.0] for i in range(20)]
        adapter.upsert(chunks, embeddings, col)

        result = adapter.get_all(col, limit=5)
        assert len(result) <= 5

    def test_multiple_collections_independent(self, adapter):
        col_a = f"col_a_{uuid.uuid4().hex[:8]}"
        col_b = f"col_b_{uuid.uuid4().hex[:8]}"

        c1 = _make_chunk("doc a", metadata={"source": "a"})
        c2 = _make_chunk("doc b", metadata={"source": "b"})
        adapter.upsert([c1], [[0.1, 0.2]], col_a)
        adapter.upsert([c2], [[0.3, 0.4]], col_b)

        result_a = adapter.get_all(col_a)
        result_b = adapter.get_all(col_b)

        assert len(result_a) == 1
        assert result_a[0].text == "doc a"
        assert len(result_b) == 1
        assert result_b[0].text == "doc b"
