"""Tests for batched ingestion — B1: a single embed() call across the whole batch."""

from unittest.mock import MagicMock

from nexrag.core.models.document import Document
from nexrag.core.pipeline.ingestion import IngestionPipeline


def _make_pipeline() -> IngestionPipeline:
    # Loader: one Document per source, source metadata = the input string.
    loader = MagicMock()
    loader.load.side_effect = lambda data: [
        Document(content=f"content {data}", metadata={"source": data})
    ]

    # Chunker: two chunks per document.
    def _chunk(doc):
        chunks = []
        for i in range(2):
            c = MagicMock()
            c.text = f"{doc.metadata['source']}-chunk{i}"
            c.metadata = {"source": doc.metadata["source"]}
            c.row_id = f"{doc.metadata['source']}-{i}"
            chunks.append(c)
        return chunks

    chunker = MagicMock()
    chunker.chunk.side_effect = _chunk

    embedder = MagicMock()
    embedder.model_name = "text-embedding-3-small"
    embedder.dimensions = 3
    embedder.embed.side_effect = lambda texts: [[0.1, 0.2, 0.3] for _ in texts]

    vector_db = MagicMock()
    vector_db.get_collection_metadata.return_value = {}
    vector_db.set_collection_metadata.return_value = None
    vector_db.get_ids_by_metadata.return_value = []

    sanitizer = MagicMock()
    sanitizer.sanitize.side_effect = lambda doc: doc

    return IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_db=vector_db,
        collection="default",
        loader=loader,
        sanitizer=sanitizer,
    )


class TestIngestBatch:
    def test_single_embed_call_for_whole_batch(self):
        pipeline = _make_pipeline()
        pipeline.ingest_batch(["a.txt", "b.txt", "c.txt"])
        # The whole point of B1: ONE embed() round-trip, not one per source.
        assert pipeline._embedder.embed.call_count == 1

    def test_embed_receives_all_chunks_across_sources(self):
        pipeline = _make_pipeline()
        pipeline.ingest_batch(["a.txt", "b.txt"])
        texts = pipeline._embedder.embed.call_args[0][0]
        # 2 sources x 2 chunks each = 4 texts in one call.
        assert len(texts) == 4
        assert "a.txt-chunk0" in texts
        assert "b.txt-chunk1" in texts

    def test_returns_one_result_per_source(self):
        pipeline = _make_pipeline()
        results = pipeline.ingest_batch(["a.txt", "b.txt", "c.txt"])
        assert len(results) == 3

    def test_per_source_counts(self):
        pipeline = _make_pipeline()
        results = pipeline.ingest_batch(["a.txt", "b.txt"])
        for r in results:
            assert r.documents_loaded == 1
            assert r.chunks_produced == 2
            assert r.chunks_written == 2

    def test_fingerprint_checked_once_for_batch(self):
        pipeline = _make_pipeline()
        pipeline.ingest_batch(["a.txt", "b.txt", "c.txt"])
        # Fingerprint runs once for the whole batch (not per source): on a fresh
        # collection that means exactly one set_collection_metadata write, even
        # though the CAS pattern reads metadata twice within that single check.
        assert pipeline._vector_db.set_collection_metadata.call_count == 1

    def test_empty_sources_returns_empty(self):
        pipeline = _make_pipeline()
        assert pipeline.ingest_batch([]) == []
        pipeline._embedder.embed.assert_not_called()

    def test_results_share_batch_pipeline_id(self):
        pipeline = _make_pipeline()
        results = pipeline.ingest_batch(["a.txt", "b.txt"])
        assert results[0].pipeline_id == results[1].pipeline_id

    def test_writes_per_source(self):
        pipeline = _make_pipeline()
        pipeline.ingest_batch(["a.txt", "b.txt"])
        # One upsert per source (writes stay per-source for correct on_conflict).
        assert pipeline._vector_db.upsert.call_count == 2
