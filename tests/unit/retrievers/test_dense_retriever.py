from unittest.mock import MagicMock

import pytest

from nexrag.core.models.chunk import Chunk, ScoredChunk
from nexrag.exceptions import RetrieverError, VectorDBError
from nexrag.retrievers.dense import DenseRetriever


def _make_scored_chunk(text: str, score: float, rank: int = 1) -> ScoredChunk:
    chunk = Chunk(
        text=text,
        chunk_index=0,
        total_chunks=1,
        parent_doc_id="doc1",
        metadata={},
    )
    return ScoredChunk(chunk=chunk, score=score, rank=rank)


def _make_retriever(results: list[ScoredChunk]) -> DenseRetriever:
    vector_db = MagicMock()
    vector_db.query.return_value = results
    return DenseRetriever(vector_db=vector_db)


class TestDenseRetriever:
    def test_returns_results_from_vector_db(self):
        expected = [_make_scored_chunk("hello", 0.9)]
        retriever = _make_retriever(expected)
        result = retriever.retrieve("q", [0.1, 0.2], top_k=5, collection="col")
        assert result == expected

    def test_passes_embedding_to_vector_db(self):
        retriever = _make_retriever([])
        retriever.retrieve("q", [0.3, 0.4], top_k=3, collection="col")
        call_kwargs = retriever._vector_db.query.call_args
        assert call_kwargs.kwargs["embedding"] == [0.3, 0.4]

    def test_passes_top_k_to_vector_db(self):
        retriever = _make_retriever([])
        retriever.retrieve("q", [0.1], top_k=10, collection="col")
        assert retriever._vector_db.query.call_args.kwargs["top_k"] == 10

    def test_passes_collection_to_vector_db(self):
        retriever = _make_retriever([])
        retriever.retrieve("q", [0.1], top_k=5, collection="my_col")
        assert retriever._vector_db.query.call_args.kwargs["collection_name"] == "my_col"

    def test_score_threshold_filters_results(self):
        results = [
            _make_scored_chunk("high", 0.9, rank=1),
            _make_scored_chunk("low", 0.3, rank=2),
        ]
        retriever = _make_retriever(results)
        filtered = retriever.retrieve("q", [0.1], top_k=5, collection="col", score_threshold=0.5)
        assert len(filtered) == 1
        assert filtered[0].score == 0.9

    def test_zero_threshold_returns_all(self):
        results = [_make_scored_chunk(f"chunk {i}", float(i) * 0.1, rank=i + 1) for i in range(5)]
        retriever = _make_retriever(results)
        out = retriever.retrieve("q", [0.1], top_k=5, collection="col", score_threshold=0.0)
        assert len(out) == 5

    def test_empty_embedding_raises_retriever_error(self):
        retriever = _make_retriever([])
        with pytest.raises(RetrieverError):
            retriever.retrieve("q", [], top_k=5, collection="col")

    def test_vector_db_error_propagates(self):
        retriever = _make_retriever([])
        retriever._vector_db.query.side_effect = VectorDBError(
            "db down", stage="retriever", component="mock"
        )
        with pytest.raises(VectorDBError):
            retriever.retrieve("q", [0.1], top_k=5, collection="col")

    def test_unexpected_exception_wrapped_in_retriever_error(self):
        retriever = _make_retriever([])
        retriever._vector_db.query.side_effect = RuntimeError("unexpected")
        with pytest.raises(RetrieverError):
            retriever.retrieve("q", [0.1], top_k=5, collection="col")

    def test_filters_forwarded_to_vector_db(self):
        retriever = _make_retriever([])
        filters = {"year": 2024}
        retriever.retrieve("q", [0.1], top_k=5, collection="col", filters=filters)
        assert retriever._vector_db.query.call_args.kwargs["filters"] == filters
