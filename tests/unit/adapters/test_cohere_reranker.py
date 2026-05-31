"""Tests for CohereReranker — issue #11."""

import sys
from unittest.mock import MagicMock, patch

import pytest

from nexrag.core.models.chunk import Chunk, ScoredChunk


def _make_scored_chunk(text: str, score: float = 0.5, rank: int = 1) -> ScoredChunk:
    chunk = Chunk(
        text=text,
        chunk_index=0,
        total_chunks=1,
        parent_doc_id="doc1",
        metadata={"source": "test.pdf"},
    )
    return ScoredChunk(chunk=chunk, score=score, rank=rank)


def _make_cohere_result(index: int, relevance_score: float) -> MagicMock:
    r = MagicMock()
    r.index = index
    r.relevance_score = relevance_score
    return r


def _make_cohere_response(results: list) -> MagicMock:
    resp = MagicMock()
    resp.results = results
    return resp


class TestCohereReranker:
    def _make_reranker(self, model="rerank-english-v3.0", top_n=5):
        mock_client = MagicMock()
        with patch("cohere.Client", return_value=mock_client):
            from nexrag.adapters.rerankers.cohere import CohereReranker

            reranker = CohereReranker(model=model, api_key="test-key", top_n=top_n)
        reranker._client = mock_client
        return reranker

    def test_reranks_correctly(self):
        from nexrag.adapters.rerankers.cohere import CohereReranker

        chunks = [
            _make_scored_chunk("low relevance text", 0.3, 1),
            _make_scored_chunk("highly relevant answer", 0.8, 2),
        ]
        mock_client = MagicMock()
        # Cohere says index=1 (second chunk) is most relevant
        mock_client.rerank.return_value = _make_cohere_response(
            [
                _make_cohere_result(1, 0.95),
                _make_cohere_result(0, 0.10),
            ]
        )

        reranker = CohereReranker.__new__(CohereReranker)
        reranker._model = "rerank-english-v3.0"
        reranker._top_n = 5
        reranker._client = mock_client

        results = reranker.rerank("query", chunks, top_n=2)

        assert len(results) == 2
        # Most relevant (index=1) should be rank 1
        assert results[0].chunk.text == "highly relevant answer"
        assert results[0].rank == 1

    def test_top_n_truncates(self):
        from nexrag.adapters.rerankers.cohere import CohereReranker

        chunks = [_make_scored_chunk(f"text {i}", rank=i + 1) for i in range(5)]
        mock_client = MagicMock()
        mock_client.rerank.return_value = _make_cohere_response(
            [_make_cohere_result(i, 0.9 - i * 0.1) for i in range(3)]
        )

        reranker = CohereReranker.__new__(CohereReranker)
        reranker._model = "rerank-english-v3.0"
        reranker._top_n = 3
        reranker._client = mock_client

        results = reranker.rerank("query", chunks, top_n=3)
        assert len(results) <= 3

    def test_empty_input_returns_empty(self):
        from nexrag.adapters.rerankers.cohere import CohereReranker

        mock_client = MagicMock()
        reranker = CohereReranker.__new__(CohereReranker)
        reranker._model = "rerank-english-v3.0"
        reranker._top_n = 5
        reranker._client = mock_client

        results = reranker.rerank("query", [], top_n=5)
        assert results == []
        mock_client.rerank.assert_not_called()

    def test_api_failure_raises_retriever_error(self):
        from nexrag.adapters.rerankers.cohere import CohereReranker
        from nexrag.exceptions import RetrieverError

        chunks = [_make_scored_chunk("text")]
        mock_client = MagicMock()
        mock_client.rerank.side_effect = Exception("API error")

        reranker = CohereReranker.__new__(CohereReranker)
        reranker._model = "rerank-english-v3.0"
        reranker._top_n = 5
        reranker._client = mock_client

        with pytest.raises(RetrieverError, match="CohereReranker API call failed"):
            reranker.rerank("query", chunks, top_n=1)

    def test_cohere_not_installed_raises_retriever_error(self):
        from nexrag.exceptions import RetrieverError

        with patch.dict(sys.modules, {"cohere": None}):
            import importlib

            from nexrag.adapters.rerankers import cohere as cohere_mod

            importlib.reload(cohere_mod)
            with pytest.raises((ImportError, RetrieverError)):
                cohere_mod.CohereReranker(model="m", top_n=5)
