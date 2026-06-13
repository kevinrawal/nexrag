"""Tests for optional reranker stage in QueryPipeline — issue #11."""

from unittest.mock import MagicMock

import pytest

from nexrag.core.models.chunk import Chunk, ScoredChunk
from nexrag.core.models.result import PipelineResult
from nexrag.core.pipeline.query import QueryPipeline
from nexrag.exceptions import PipelineError


def _make_chunk(text: str = "chunk text", score: float = 0.9) -> ScoredChunk:
    chunk = Chunk(
        text=text,
        chunk_index=0,
        total_chunks=1,
        parent_doc_id="doc1",
        metadata={"source": "test.pdf"},
    )
    return ScoredChunk(chunk=chunk, score=score, rank=1)


def _make_pipeline(reranker=None) -> QueryPipeline:
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3]
    embedder.model_name = "test-model"

    retriever = MagicMock()
    retriever.retrieve.return_value = [_make_chunk("first"), _make_chunk("second")]

    prompt_builder = MagicMock()
    prompt_builder.build.return_value = "assembled prompt"

    llm = MagicMock()
    llm.generate.return_value = ("The answer.", None)
    llm.model_name = "gpt-test"

    observer = MagicMock()

    return QueryPipeline(
        embedder=embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        collection="test_collection",
        observer=observer,
        reranker=reranker,
    )


class TestRerankerPipelineStage:
    def test_without_reranker_pipeline_unchanged(self):
        pipeline = _make_pipeline(reranker=None)
        result = pipeline.run("What is NexRAG?")
        assert isinstance(result, PipelineResult)
        # reranker should not have been called
        assert pipeline._reranker is None

    def test_with_reranker_stage_fires(self):
        reranker = MagicMock()
        reranker.top_n = 1
        reranked_chunk = _make_chunk("reranked best chunk", 0.99)
        reranker.rerank.return_value = [reranked_chunk]

        pipeline = _make_pipeline(reranker=reranker)
        result = pipeline.run("What is NexRAG?")

        reranker.rerank.assert_called_once()
        assert isinstance(result, PipelineResult)

    def test_reranker_reduces_chunks_passed_to_prompt(self):
        reranker = MagicMock()
        reranker.top_n = 1
        single_chunk = _make_chunk("top result", 0.99)
        reranker.rerank.return_value = [single_chunk]

        pipeline = _make_pipeline(reranker=reranker)
        pipeline.run("query")

        # PromptBuilder should have received only 1 chunk after reranking
        call_args = pipeline._prompt_builder.build.call_args
        chunks_passed = call_args[0][1]
        assert len(chunks_passed) == 1
        assert chunks_passed[0].chunk.text == "top result"

    def test_reranker_event_emitted(self):
        reranker = MagicMock()
        reranker.top_n = 2
        reranker.rerank.return_value = [_make_chunk("reranked")]

        pipeline = _make_pipeline(reranker=reranker)
        pipeline.run("query")

        # Check that "reranker" stage events were emitted
        emitted_stages = [call.args[0].stage for call in pipeline._observer.emit.call_args_list]
        assert "reranker" in emitted_stages

    def test_reranker_failure_raises_pipeline_error(self):
        reranker = MagicMock()
        reranker.top_n = 2
        reranker.rerank.side_effect = RuntimeError("model crash")

        pipeline = _make_pipeline(reranker=reranker)
        with pytest.raises(PipelineError, match="Reranker failed"):
            pipeline.run("query")

    def test_top_n_capped_at_available_chunks(self):
        reranker = MagicMock()
        reranker.top_n = 100  # More than available
        reranker.rerank.return_value = [_make_chunk("only result")]

        pipeline = _make_pipeline(reranker=reranker)
        pipeline.run("query")

        # top_n passed to rerank should be min(100, len(chunks))
        call_kwargs = reranker.rerank.call_args
        top_n_passed = call_kwargs[0][2] if len(call_kwargs[0]) > 2 else call_kwargs[1].get("top_n")
        assert top_n_passed == 2  # only 2 chunks from retriever
