"""Tests for multi-collection routing on ingest — issue #12."""

from unittest.mock import MagicMock

import pytest

from nexrag.core.models.document import Document
from nexrag.core.pipeline.ingestion import IngestionPipeline, IngestionResult
from nexrag.exceptions import ConfigError


def _make_doc(source: str = "test.pdf") -> Document:
    return Document(content="test content", metadata={"source": source})


def _make_pipeline(
    collection: str = "default",
    valid_collections: frozenset | None = None,
) -> IngestionPipeline:
    chunker = MagicMock()
    chunk = MagicMock()
    chunk.text = "chunk text"
    chunk.content_hash = "abc123"
    chunk.metadata = {"source": "test.pdf"}
    chunker.chunk.return_value = [chunk]

    embedder = MagicMock()
    embedder.model_name = "text-embedding-3-small"
    embedder.dimensions = 3
    embedder.embed.return_value = [[0.1, 0.2, 0.3]]

    vector_db = MagicMock()
    vector_db.get_collection_metadata.return_value = {}
    vector_db.set_collection_metadata.return_value = None
    vector_db.query.return_value = []

    sanitizer = MagicMock()
    sanitizer.sanitize.return_value = _make_doc()

    return IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_db=vector_db,
        collection=collection,
        sanitizer=sanitizer,
        valid_collections=valid_collections or frozenset([collection]),
    )


class TestMultiCollectionRouting:
    def test_default_collection_used_when_no_override(self):
        pipeline = _make_pipeline(
            collection="default",
            valid_collections=frozenset(["default", "other"]),
        )
        result = pipeline.ingest_documents([_make_doc()])
        assert result.collection_used == "default"

    def test_explicit_collection_overrides_default(self):
        pipeline = _make_pipeline(
            collection="default",
            valid_collections=frozenset(["default", "resumes"]),
        )
        result = pipeline.ingest_documents([_make_doc()], collection="resumes")
        assert result.collection_used == "resumes"

    def test_unknown_collection_raises_config_error(self):
        pipeline = _make_pipeline(
            collection="default",
            valid_collections=frozenset(["default"]),
        )
        with pytest.raises(ConfigError, match="not configured"):
            pipeline.ingest_documents([_make_doc()], collection="nonexistent")

    def test_collection_used_in_vector_db_write(self):
        pipeline = _make_pipeline(
            collection="default",
            valid_collections=frozenset(["default", "resumes"]),
        )
        pipeline.ingest_documents([_make_doc()], collection="resumes")

        # The upsert call should target "resumes", not "default"
        upsert_call = pipeline._vector_db.upsert.call_args
        collection_arg = upsert_call[0][2]
        assert collection_arg == "resumes"

    def test_ingestion_result_collection_used_field(self):
        pipeline = _make_pipeline(
            collection="default",
            valid_collections=frozenset(["default", "docs"]),
        )
        result = pipeline.ingest_documents([_make_doc()], collection="docs")
        assert isinstance(result, IngestionResult)
        assert result.collection_used == "docs"

    def test_fingerprint_check_uses_correct_collection(self):
        pipeline = _make_pipeline(
            collection="default",
            valid_collections=frozenset(["default", "archive"]),
        )
        pipeline.ingest_documents([_make_doc()], collection="archive")

        get_meta_call = pipeline._vector_db.get_collection_metadata.call_args
        assert get_meta_call[0][0] == "archive"

    def test_result_has_metrics(self):
        pipeline = _make_pipeline()
        result = pipeline.ingest_documents([_make_doc()])
        assert result.metrics is not None
        assert result.metrics.total_latency_ms >= 0.0
