"""Unit tests for the NexRAG facade (ingest_batch and related methods)."""

from pathlib import Path
from unittest.mock import MagicMock, call

from nexrag import NexRAG
from nexrag.core.pipeline.ingestion import IngestionResult


def _make_pipeline() -> NexRAG:
    ingestion = MagicMock()
    query = MagicMock()
    return NexRAG(ingestion=ingestion, query=query)


def _make_result(chunks: int = 3) -> IngestionResult:
    r = MagicMock(spec=IngestionResult)
    r.chunks_written = chunks
    return r


class TestIngestBatch:
    def test_returns_one_result_per_source(self):
        pipeline = _make_pipeline()
        pipeline._ingestion.ingest.side_effect = [_make_result(3), _make_result(5)]
        results = pipeline.ingest_batch(["a.pdf", "b.pdf"])
        assert len(results) == 2

    def test_results_are_in_order(self):
        pipeline = _make_pipeline()
        r1, r2, r3 = _make_result(1), _make_result(2), _make_result(3)
        pipeline._ingestion.ingest.side_effect = [r1, r2, r3]
        results = pipeline.ingest_batch(["x.pdf", "y.txt", "z.md"])
        assert results == [r1, r2, r3]

    def test_each_source_is_passed_to_ingest(self):
        pipeline = _make_pipeline()
        pipeline._ingestion.ingest.return_value = _make_result()
        pipeline.ingest_batch(["a.pdf", "b.txt"])
        pipeline._ingestion.ingest.assert_any_call("a.pdf", None)
        pipeline._ingestion.ingest.assert_any_call("b.txt", None)

    def test_empty_list_returns_empty(self):
        pipeline = _make_pipeline()
        results = pipeline.ingest_batch([])
        assert results == []
        pipeline._ingestion.ingest.assert_not_called()

    def test_path_objects_accepted(self):
        pipeline = _make_pipeline()
        pipeline._ingestion.ingest.return_value = _make_result()
        results = pipeline.ingest_batch([Path("doc.pdf")])
        assert len(results) == 1

    def test_loader_override_passed_to_each_ingest(self):
        pipeline = _make_pipeline()
        pipeline._ingestion.ingest.return_value = _make_result()
        mock_loader = MagicMock()
        pipeline.ingest_batch(["a.pdf", "b.pdf"], loader=mock_loader)
        pipeline._ingestion.ingest.assert_has_calls(
            [
                call("a.pdf", mock_loader),
                call("b.pdf", mock_loader),
            ]
        )
