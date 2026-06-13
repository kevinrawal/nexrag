"""
Per-source idempotency + composite row-ID isolation.

Covers the v0.3.3 fixes:
  - on_conflict (skip/overwrite) decisions are made independently per source.
  - a VectorDB lookup failure for one source does not affect the others.
  - vector-DB rows are keyed by row_id (parent_doc_id + content_hash), so two
    documents containing identical text keep separate rows.

Uses a real in-memory ChromaDBAdapter so the dedup/delete paths run for real.
"""

from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

pytest.importorskip("chromadb")

from nexrag.adapters.vector_dbs.chroma import ChromaDBAdapter
from nexrag.chunkers.recursive import RecursiveChunker
from nexrag.core.models.document import Document
from nexrag.core.pipeline.ingestion import IngestionPipeline
from nexrag.exceptions import VectorDBError


def _mock_embedder(dims: int = 4):
    embedder = MagicMock()
    embedder.model_name = "mock-model"
    embedder.dimensions = dims
    embedder.embed.side_effect = lambda texts: [[0.1] * dims for _ in texts]
    embedder.embed_query.return_value = [0.1] * dims
    return embedder


def _make_pipeline(vector_db, col: str, on_conflict: str) -> IngestionPipeline:
    return IngestionPipeline(
        chunker=RecursiveChunker(chunk_size=200, chunk_overlap=0, min_chunk_size=1),
        embedder=_mock_embedder(),
        vector_db=vector_db,
        collection=col,
        on_conflict=on_conflict,
    )


@pytest.fixture
def col():
    return f"col_{uuid.uuid4().hex[:10]}"


# Short, single-chunk texts so chunk counts are easy to reason about.
TEXT_A = "Alpha document about contracts."
TEXT_B = "Beta document about invoices."
TEXT_B2 = "Beta document, revised, about invoices and refunds."


class TestPerSourceSkip:
    def test_new_source_written_when_other_source_already_exists(self, col):
        vdb = ChromaDBAdapter(mode="memory")
        pipe = _make_pipeline(vdb, col, on_conflict="skip")

        # A already ingested.
        pipe.ingest_documents([Document(content=TEXT_A, metadata={"source": "a"})])
        count_after_a = vdb.count(col)
        assert count_after_a == 1

        # Batch with existing A + brand-new B. Old behaviour dropped the whole
        # batch; new behaviour skips A and writes B.
        result = pipe.ingest_documents(
            [
                Document(content=TEXT_A, metadata={"source": "a"}),
                Document(content=TEXT_B, metadata={"source": "b"}),
            ]
        )

        assert result.chunks_written == 1  # only B
        assert vdb.count(col) == count_after_a + 1
        assert set(vdb.get_ids_by_metadata({"source": "b"}, col))  # B is present


class TestPerSourceOverwrite:
    def test_only_changed_source_is_rewritten(self, col):
        vdb = ChromaDBAdapter(mode="memory")
        pipe = _make_pipeline(vdb, col, on_conflict="overwrite")

        pipe.ingest_documents(
            [
                Document(content=TEXT_A, metadata={"source": "a"}),
                Document(content=TEXT_B, metadata={"source": "b"}),
            ]
        )
        ids_a_before = set(vdb.get_ids_by_metadata({"source": "a"}, col))
        ids_b_before = set(vdb.get_ids_by_metadata({"source": "b"}, col))
        assert vdb.count(col) == 2

        # Re-ingest A unchanged + B changed. Only B should be rewritten.
        result = pipe.ingest_documents(
            [
                Document(content=TEXT_A, metadata={"source": "a"}),
                Document(content=TEXT_B2, metadata={"source": "b"}),
            ]
        )

        assert result.chunks_written == 1  # only B2 (old code rewrote both → 2)

        ids_a_after = set(vdb.get_ids_by_metadata({"source": "a"}, col))
        ids_b_after = set(vdb.get_ids_by_metadata({"source": "b"}, col))
        assert ids_a_after == ids_a_before  # A untouched
        assert ids_b_after != ids_b_before  # B replaced
        assert vdb.count(col) == 2  # still one row per source


class _FlakyAdapter(ChromaDBAdapter):
    """ChromaDBAdapter that raises VectorDBError on lookup for one source."""

    def __init__(self, fail_source: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fail_source = fail_source

    def get_ids_by_metadata(self, filters, collection_name):  # type: ignore[override]
        if filters.get("source") == self._fail_source:
            raise VectorDBError(
                "simulated lookup failure",
                stage="idempotency_check",
                component="ChromaDBAdapter",
            )
        return super().get_ids_by_metadata(filters, collection_name)


class _RecordingObserver:
    def __init__(self) -> None:
        self.events: list = []

    def emit(self, event) -> None:
        self.events.append(event)


class TestPerSourceLookupFailureIsolation:
    def test_one_source_lookup_failure_does_not_affect_others(self, col):
        observer = _RecordingObserver()
        vdb = _FlakyAdapter(fail_source="a", mode="memory")
        pipe = IngestionPipeline(
            chunker=RecursiveChunker(chunk_size=200, chunk_overlap=0, min_chunk_size=1),
            embedder=_mock_embedder(),
            vector_db=vdb,
            collection=col,
            on_conflict="skip",
            observer=observer,
        )

        # Pre-ingest B via a non-flaky adapter sharing the same in-memory store is
        # not possible, so seed B directly through the flaky adapter (B never fails).
        pipe.ingest_documents([Document(content=TEXT_B, metadata={"source": "b"})])
        assert vdb.count(col) == 1

        # Batch: A's lookup raises (write A anyway), B exists (skip B).
        result = pipe.ingest_documents(
            [
                Document(content=TEXT_A, metadata={"source": "a"}),
                Document(content=TEXT_B, metadata={"source": "b"}),
            ]
        )

        assert result.chunks_written == 1  # A written; B skipped
        assert vdb.count(col) == 2

        failed = [
            e for e in observer.events if e.stage == "idempotency_check" and e.status == "failed"
        ]
        assert len(failed) == 1
        assert failed[0].metadata.get("error_type") == "VectorDBError"


class TestAsyncPerSource:
    def test_async_new_source_written_when_other_exists(self, col):
        from unittest.mock import AsyncMock

        from nexrag.core.pipeline.async_ingestion import AsyncIngestionPipeline

        embedder = MagicMock()
        embedder.model_name = "mock-model"
        embedder.dimensions = 4
        embedder.async_embed = AsyncMock(side_effect=lambda texts: [[0.1] * 4 for _ in texts])

        vdb = ChromaDBAdapter(mode="memory")
        pipe = AsyncIngestionPipeline(
            chunker=RecursiveChunker(chunk_size=200, chunk_overlap=0, min_chunk_size=1),
            embedder=embedder,
            vector_db=vdb,
            collection=col,
            on_conflict="skip",
        )

        # Sync wrappers run aingest_documents() via asyncio.run().
        pipe.ingest_documents([Document(content=TEXT_A, metadata={"source": "a"})])
        result = pipe.ingest_documents(
            [
                Document(content=TEXT_A, metadata={"source": "a"}),
                Document(content=TEXT_B, metadata={"source": "b"}),
            ]
        )
        assert result.chunks_written == 1  # only B
        assert vdb.count(col) == 2


class TestRowIdIsolation:
    def test_identical_text_in_two_documents_keeps_two_rows(self, col):
        vdb = ChromaDBAdapter(mode="memory")
        pipe = _make_pipeline(vdb, col, on_conflict="overwrite")

        shared = "This identical boilerplate paragraph appears in both documents."
        pipe.ingest_documents(
            [
                Document(content=shared, metadata={"source": "doc-a"}),
                Document(content=shared, metadata={"source": "doc-b"}),
            ]
        )

        # Bare-content_hash IDs would have collapsed these into one row.
        assert vdb.count(col) == 2
        assert len(vdb.get_ids_by_metadata({"source": "doc-a"}, col)) == 1
        assert len(vdb.get_ids_by_metadata({"source": "doc-b"}, col)) == 1

    def test_overwriting_one_document_keeps_the_others_shared_chunk(self, col):
        vdb = ChromaDBAdapter(mode="memory")
        pipe = _make_pipeline(vdb, col, on_conflict="overwrite")

        shared = "Shared disclaimer text used across multiple documents."
        pipe.ingest_documents(
            [
                Document(content=shared, metadata={"source": "doc-a"}),
                Document(content=shared, metadata={"source": "doc-b"}),
            ]
        )
        ids_b_before = set(vdb.get_ids_by_metadata({"source": "doc-b"}, col))

        # Overwrite doc-a with new content — doc-b's row must survive.
        pipe.ingest_documents(
            [Document(content="Doc A now has entirely new content.", metadata={"source": "doc-a"})]
        )

        ids_b_after = set(vdb.get_ids_by_metadata({"source": "doc-b"}, col))
        assert ids_b_after == ids_b_before
        assert len(ids_b_after) == 1
