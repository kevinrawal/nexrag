import math
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nexrag.core.models.chunk import Chunk

# --- Stateful fake Pinecone SDK (no in-memory mode exists) -------------------


def _match_filter(meta: dict, filt: dict | None) -> bool:
    if not filt:
        return True
    for key, cond in filt.items():
        if key == "$and":
            return all(_match_filter(meta, sub) for sub in cond)
        if key == "$or":
            return any(_match_filter(meta, sub) for sub in cond)
        val = meta.get(key)
        if isinstance(cond, dict):
            for op, target in cond.items():
                if op == "$eq" and val != target:
                    return False
                if op == "$ne" and val == target:
                    return False
                if op == "$in" and val not in target:
                    return False
                if op == "$nin" and val in target:
                    return False
        elif val != cond:
            return False
    return True


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class _FakeIndex:
    def __init__(self, store: dict):
        self._store = store  # namespace -> {id: {"values": [...], "metadata": {...}}}

    def upsert(self, vectors, namespace=None):
        ns = self._store.setdefault(namespace, {})
        for v in vectors:
            ns[v["id"]] = {"values": v["values"], "metadata": v.get("metadata", {})}

    def query(
        self,
        vector,
        top_k,
        namespace=None,
        filter=None,  # noqa: A002 — mirrors Pinecone's parameter name
        include_metadata=False,
        include_values=False,
    ):
        ns = self._store.get(namespace, {})
        matches = []
        for vid, rec in ns.items():
            if not _match_filter(rec["metadata"], filter):
                continue
            meta = rec["metadata"] if include_metadata else None
            matches.append(
                SimpleNamespace(id=vid, score=_cosine(vector, rec["values"]), metadata=meta)
            )
        matches.sort(key=lambda m: m.score, reverse=True)
        return SimpleNamespace(matches=matches[:top_k])

    def delete(self, ids, namespace=None):
        ns = self._store.get(namespace, {})
        for i in ids:
            ns.pop(i, None)

    def describe_index_stats(self):
        return SimpleNamespace(
            namespaces={
                ns: SimpleNamespace(vector_count=len(recs)) for ns, recs in self._store.items()
            }
        )

    def fetch(self, ids, namespace=None):
        ns = self._store.get(namespace, {})
        return SimpleNamespace(
            vectors={
                i: SimpleNamespace(values=ns[i]["values"], metadata=ns[i]["metadata"])
                for i in ids
                if i in ns
            }
        )

    def list(self, namespace=None):
        yield list(self._store.get(namespace, {}).keys())


class _FakeClient:
    def __init__(self, store, dim_holder):
        self._store = store
        self._dim = dim_holder
        self._exists = False

    def has_index(self, name):
        return self._exists

    def create_index(self, name, dimension, metric, spec):
        self._exists = True
        self._dim["dim"] = dimension

    def describe_index(self, name):
        return SimpleNamespace(dimension=self._dim.get("dim"), status=SimpleNamespace(ready=True))

    def Index(self, name):  # noqa: N802 — mirrors Pinecone SDK casing
        return _FakeIndex(self._store)

    def list_indexes(self):
        return []


@pytest.fixture
def env():
    """Yields (adapter, client) backed by a stateful fake pinecone module."""
    from nexrag.adapters.vector_dbs.pinecone import PineconeVectorDB

    store: dict = {}
    dim_holder: dict = {}
    client = _FakeClient(store, dim_holder)
    fake_module = SimpleNamespace(
        Pinecone=lambda api_key=None: client,
        ServerlessSpec=lambda **kw: kw,
    )
    with patch.dict(sys.modules, {"pinecone": fake_module}):
        adapter = PineconeVectorDB(index_name="nexrag-test", api_key="test-key")
        yield adapter, client


def _make_chunk(text, index=0, total=1, doc_id="doc1", metadata=None):
    return Chunk(
        text=text,
        chunk_index=index,
        total_chunks=total,
        parent_doc_id=doc_id,
        metadata=metadata if metadata is not None else {"source": "test.pdf"},
    )


class TestPineconeVectorDB:
    def test_upsert_and_count(self, env):
        adapter, _ = env
        adapter.upsert([_make_chunk("Hello world")], [[0.1, 0.2, 0.3]], "docs")
        assert adapter.count("docs") == 1

    def test_upsert_empty_is_noop(self, env):
        adapter, _ = env
        adapter.upsert([], [], "docs")
        assert adapter.count("docs") == 0

    def test_idempotent_upsert_dedupes_by_row_id(self, env):
        adapter, _ = env
        chunk = _make_chunk("Same text")
        adapter.upsert([chunk, chunk], [[0.1, 0.2], [0.1, 0.2]], "docs")
        assert adapter.count("docs") == 1

    def test_query_returns_scored_chunk_with_text(self, env):
        adapter, _ = env
        adapter.upsert([_make_chunk("Relevant content")], [[1.0, 0.0, 0.0]], "docs")
        results = adapter.query([1.0, 0.0, 0.0], top_k=5, collection_name="docs")
        assert len(results) == 1
        assert results[0].rank == 1
        assert results[0].chunk.text == "Relevant content"
        assert results[0].chunk.metadata == {"source": "test.pdf"}
        assert results[0].score == pytest.approx(1.0)

    def test_query_before_any_upsert_returns_empty(self, env):
        adapter, _ = env
        assert adapter.query([0.1, 0.2], top_k=5, collection_name="docs") == []

    def test_namespace_isolation(self, env):
        adapter, _ = env
        adapter.upsert([_make_chunk("a")], [[1.0, 0.0]], "tenant_a")
        assert adapter.count("tenant_a") == 1
        assert adapter.count("tenant_b") == 0
        assert adapter.query([1.0, 0.0], top_k=5, collection_name="tenant_b") == []

    def test_delete_removes_chunk(self, env):
        adapter, _ = env
        chunk = _make_chunk("To be deleted")
        adapter.upsert([chunk], [[0.1, 0.2]], "docs")
        adapter.delete([chunk.row_id], "docs")
        assert adapter.count("docs") == 0

    def test_lazy_index_creation(self, env):
        adapter, client = env
        assert client.has_index("nexrag-test") is False
        adapter.upsert([_make_chunk("x")], [[0.1, 0.2]], "docs")
        assert client.has_index("nexrag-test") is True

    def test_collection_metadata_roundtrip(self, env):
        adapter, _ = env
        meta = {
            "embedding_model": "text-embedding-3-small",
            "embedding_dimensions": 3,
            "fingerprint": "fp",
        }
        adapter.set_collection_metadata("docs", meta)
        retrieved = adapter.get_collection_metadata("docs")
        assert retrieved["embedding_model"] == "text-embedding-3-small"
        assert retrieved["fingerprint"] == "fp"

    def test_get_collection_metadata_before_index_returns_empty(self, env):
        adapter, _ = env
        assert adapter.get_collection_metadata("docs") == {}

    def test_fingerprint_namespace_excluded_from_count_and_collections(self, env):
        adapter, _ = env
        adapter.set_collection_metadata(
            "docs", {"embedding_model": "m", "embedding_dimensions": 2, "fingerprint": "fp"}
        )
        adapter.upsert([_make_chunk("hello")], [[0.1, 0.2]], "docs")
        assert adapter.count("docs") == 1  # meta namespace not counted
        assert adapter.list_collections() == ["docs"]

    def test_get_all_reconstructs_chunks(self, env):
        adapter, _ = env
        chunks = [_make_chunk(f"chunk {i}", index=i, total=2) for i in range(2)]
        adapter.upsert(chunks, [[0.1, 0.2], [0.3, 0.4]], "docs")
        out = adapter.get_all("docs")
        assert {c.text for c in out} == {"chunk 0", "chunk 1"}
        assert all(c.metadata == {"source": "test.pdf"} for c in out)

    def test_get_ids_by_metadata(self, env):
        adapter, _ = env
        a = _make_chunk("a", doc_id="d1", metadata={"source": "a.pdf"})
        b = _make_chunk("b", doc_id="d2", metadata={"source": "b.pdf"})
        adapter.upsert([a, b], [[1.0, 0.0], [0.0, 1.0]], "docs")
        ids = adapter.get_ids_by_metadata({"source": "a.pdf"}, "docs")
        assert ids == [a.row_id]

    def test_build_filter_single(self):
        from nexrag.adapters.vector_dbs.pinecone import PineconeVectorDB

        assert PineconeVectorDB._build_filter({"year": 2024}) == {"year": {"$eq": 2024}}

    def test_build_filter_multi(self):
        from nexrag.adapters.vector_dbs.pinecone import PineconeVectorDB

        result = PineconeVectorDB._build_filter({"year": 2024, "source": "a.pdf"})
        assert "$and" in result

    def test_build_filter_none(self):
        from nexrag.adapters.vector_dbs.pinecone import PineconeVectorDB

        assert PineconeVectorDB._build_filter(None) is None

    def test_serialize_metadata_flattens_non_primitives(self):
        from nexrag.adapters.vector_dbs.pinecone import PineconeVectorDB

        out = PineconeVectorDB._serialize_metadata(
            {"s": "v", "n": 1, "lst": ["a", "b"], "obj": {"x": 1}}
        )
        assert out["s"] == "v"
        assert out["lst"] == ["a", "b"]
        assert out["obj"] == str({"x": 1})
