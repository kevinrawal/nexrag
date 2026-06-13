"""Tests for _MultiChromaAdapter — per-collection routing (issue #24)."""

import uuid
from unittest.mock import MagicMock

import pytest

from nexrag.adapters.vector_dbs.chroma import ChromaDBAdapter, _MultiChromaAdapter
from nexrag.core.models.chunk import Chunk


def _make_chunk(text: str, source: str = "test.pdf") -> Chunk:
    return Chunk(
        text=text,
        chunk_index=0,
        total_chunks=1,
        parent_doc_id="doc1",
        metadata={"source": source},
    )


def _col() -> str:
    return f"col_{uuid.uuid4().hex[:8]}"


class TestMultiChromaAdapterRouting:
    def test_collection_with_override_uses_its_own_adapter(self):
        default = MagicMock(spec=ChromaDBAdapter)
        override = MagicMock(spec=ChromaDBAdapter)
        adapter = _MultiChromaAdapter(
            default_adapter=default,
            collection_adapters={"special": override},
        )

        adapter.count("special")
        override.count.assert_called_once_with("special")
        default.count.assert_not_called()

    def test_collection_without_override_uses_default(self):
        default = MagicMock(spec=ChromaDBAdapter)
        override = MagicMock(spec=ChromaDBAdapter)
        adapter = _MultiChromaAdapter(
            default_adapter=default,
            collection_adapters={"special": override},
        )

        adapter.count("other")
        default.count.assert_called_once_with("other")
        override.count.assert_not_called()

    def test_upsert_routes_to_correct_adapter(self):
        default = MagicMock(spec=ChromaDBAdapter)
        override = MagicMock(spec=ChromaDBAdapter)
        adapter = _MultiChromaAdapter(
            default_adapter=default,
            collection_adapters={"resumes": override},
        )

        chunks = [_make_chunk("hello")]
        embeddings = [[1.0, 0.0]]
        adapter.upsert(chunks, embeddings, "resumes")

        override.upsert.assert_called_once_with(chunks, embeddings, "resumes")
        default.upsert.assert_not_called()

    def test_get_all_routes_correctly(self):
        default = MagicMock(spec=ChromaDBAdapter)
        override = MagicMock(spec=ChromaDBAdapter)
        override.get_all.return_value = [_make_chunk("resume text")]
        adapter = _MultiChromaAdapter(
            default_adapter=default,
            collection_adapters={"resumes": override},
        )

        result = adapter.get_all("resumes")
        assert result == [_make_chunk("resume text")]
        default.get_all.assert_not_called()

    def test_query_routes_correctly(self):
        default = MagicMock(spec=ChromaDBAdapter)
        override = MagicMock(spec=ChromaDBAdapter)
        override.query.return_value = []
        adapter = _MultiChromaAdapter(
            default_adapter=default,
            collection_adapters={"resumes": override},
        )

        adapter.query([0.1, 0.2], top_k=5, collection_name="resumes", filters={"x": "y"})
        override.query.assert_called_once_with([0.1, 0.2], 5, "resumes", {"x": "y"})

    def test_metadata_ops_route_correctly(self):
        default = MagicMock(spec=ChromaDBAdapter)
        override = MagicMock(spec=ChromaDBAdapter)
        override.get_collection_metadata.return_value = {"model": "test"}
        adapter = _MultiChromaAdapter(
            default_adapter=default,
            collection_adapters={"resumes": override},
        )

        meta = adapter.get_collection_metadata("resumes")
        assert meta == {"model": "test"}

        adapter.set_collection_metadata("resumes", {"model": "v2"})
        override.set_collection_metadata.assert_called_once_with("resumes", {"model": "v2"})


class TestMultiChromaAdapterEndToEnd:
    """Integration test: two collections on different in-memory adapters."""

    @pytest.fixture(autouse=True)
    def _require_chromadb(self):
        pytest.importorskip("chromadb")

    def test_isolated_collections_do_not_share_data(self):
        adapter_docs = ChromaDBAdapter(mode="memory")
        adapter_resumes = ChromaDBAdapter(mode="memory")

        multi = _MultiChromaAdapter(
            default_adapter=adapter_docs,
            collection_adapters={"resumes": adapter_resumes},
        )

        col_docs = _col()
        col_resumes = "resumes"

        chunk_doc = _make_chunk("python developer job posting", source="jobs.pdf")
        chunk_resume = _make_chunk("python developer resume", source="resume.pdf")

        multi.upsert([chunk_doc], [[1.0, 0.0]], col_docs)
        multi.upsert([chunk_resume], [[0.9, 0.1]], col_resumes)

        assert multi.count(col_docs) == 1
        assert multi.count(col_resumes) == 1

        # docs collection should NOT see the resume chunk
        docs_chunks = multi.get_all(col_docs)
        assert all("jobs.pdf" in c.metadata.get("source", "") for c in docs_chunks)


class TestFactoryMultiCollectionWiring:
    """Verify the factory builds _MultiChromaAdapter when collections differ."""

    @pytest.fixture(autouse=True)
    def _require_chromadb(self):
        pytest.importorskip("chromadb")

    def test_single_config_returns_plain_adapter(self):
        from nexrag._factory import _build_vector_db
        from nexrag.adapters.vector_dbs.chroma import _MultiChromaAdapter
        from nexrag.core.config.schema import CollectionConfig, VectorDBConfig

        cfg = VectorDBConfig(
            default_collection="docs",
            collections={"docs": CollectionConfig(mode="memory")},
        )
        adapter = _build_vector_db(cfg)
        assert not isinstance(adapter, _MultiChromaAdapter)
        assert isinstance(adapter, ChromaDBAdapter)

    def test_different_configs_returns_multi_adapter(self):
        from nexrag._factory import _build_vector_db
        from nexrag.adapters.vector_dbs.chroma import _MultiChromaAdapter
        from nexrag.core.config.schema import CollectionConfig, VectorDBConfig

        cfg = VectorDBConfig(
            default_collection="docs",
            collections={
                "docs": CollectionConfig(mode="memory"),
                "resumes": CollectionConfig(mode="persistent", path=".nexrag/resumes"),
            },
        )
        adapter = _build_vector_db(cfg)
        assert isinstance(adapter, _MultiChromaAdapter)

    def test_same_config_collections_share_adapter_instance(self):
        from nexrag._factory import _build_vector_db
        from nexrag.adapters.vector_dbs.chroma import _MultiChromaAdapter
        from nexrag.core.config.schema import CollectionConfig, VectorDBConfig

        cfg = VectorDBConfig(
            default_collection="docs",
            collections={
                "docs": CollectionConfig(mode="memory"),
                "articles": CollectionConfig(mode="memory"),
            },
        )
        # Both collections use mode=memory with no path — same config as default.
        # articles shares default config → no per-collection override needed.
        adapter = _build_vector_db(cfg)
        assert not isinstance(adapter, _MultiChromaAdapter)
