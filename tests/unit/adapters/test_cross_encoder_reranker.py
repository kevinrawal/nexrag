"""Tests for CrossEncoderReranker — issue #11."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
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


class TestCrossEncoderReranker:
    def _make_reranker(
        self, scores: list[float], model: str = "cross-encoder/test", top_n: int = 5
    ):
        from nexrag.adapters.rerankers.cross_encoder import CrossEncoderReranker

        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = np.array(scores)

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model_name = model
        reranker._top_n = top_n
        reranker._device = None
        reranker._encoder = mock_encoder
        return reranker

    def test_reranks_by_model_scores(self):
        chunks = [
            _make_scored_chunk("low relevance text", 0.2, 1),
            _make_scored_chunk("high relevance answer", 0.9, 2),
            _make_scored_chunk("medium relevance doc", 0.5, 3),
        ]
        # Cross-encoder gives these scores: [0.1, 0.95, 0.4]
        reranker = self._make_reranker([0.1, 0.95, 0.4])
        results = reranker.rerank("query", chunks, top_n=3)

        # Index 1 (score 0.95) should be rank 1
        assert results[0].chunk.text == "high relevance answer"
        assert results[0].rank == 1

    def test_top_n_truncates_results(self):
        chunks = [_make_scored_chunk(f"text {i}", rank=i + 1) for i in range(5)]
        scores = [0.9, 0.8, 0.7, 0.6, 0.5]
        reranker = self._make_reranker(scores)
        results = reranker.rerank("query", chunks, top_n=3)
        assert len(results) == 3

    def test_empty_input_returns_empty(self):
        from nexrag.adapters.rerankers.cross_encoder import CrossEncoderReranker

        mock_encoder = MagicMock()
        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model_name = "test"
        reranker._top_n = 5
        reranker._device = None
        reranker._encoder = mock_encoder

        results = reranker.rerank("query", [], top_n=5)
        assert results == []
        mock_encoder.predict.assert_not_called()

    def test_model_inference_failure_raises_retriever_error(self):
        from nexrag.adapters.rerankers.cross_encoder import CrossEncoderReranker
        from nexrag.exceptions import RetrieverError

        mock_encoder = MagicMock()
        mock_encoder.predict.side_effect = RuntimeError("CUDA OOM")

        reranker = CrossEncoderReranker.__new__(CrossEncoderReranker)
        reranker._model_name = "test"
        reranker._top_n = 5
        reranker._device = None
        reranker._encoder = mock_encoder

        chunks = [_make_scored_chunk("text")]
        with pytest.raises(RetrieverError, match="CrossEncoderReranker model inference failed"):
            reranker.rerank("query", chunks, top_n=1)

    def test_scores_in_new_rank_order(self):
        chunks = [_make_scored_chunk(f"doc {i}", rank=i + 1) for i in range(4)]
        scores = [0.3, 0.9, 0.1, 0.7]
        reranker = self._make_reranker(scores)
        results = reranker.rerank("query", chunks, top_n=4)
        reranker_scores = [r.score for r in results]
        assert reranker_scores == sorted(reranker_scores, reverse=True)

    def test_sentence_transformers_not_installed_raises_error(self):
        from nexrag.exceptions import RetrieverError

        with patch.dict(sys.modules, {"sentence_transformers": None}):
            import importlib

            from nexrag.adapters.rerankers import cross_encoder as ce_mod

            importlib.reload(ce_mod)
            with pytest.raises((ImportError, RetrieverError)):
                ce_mod.CrossEncoderReranker(model="test")
