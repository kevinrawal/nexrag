import pytest

from nexrag.core.models.chunk import Chunk, ScoredChunk
from nexrag.defaults.prompt_builder import DefaultPromptBuilder
from nexrag.exceptions import PromptError


def _scored(text: str, score: float = 0.9, rank: int = 1) -> ScoredChunk:
    chunk = Chunk(
        text=text,
        chunk_index=0,
        total_chunks=1,
        parent_doc_id="doc1",
        metadata={},
    )
    return ScoredChunk(chunk=chunk, score=score, rank=rank)


class TestDefaultPromptBuilder:
    def test_build_returns_string(self):
        builder = DefaultPromptBuilder()
        result = builder.build("What is RAG?", [_scored("RAG is ...")])
        assert isinstance(result, str)

    def test_build_contains_query(self):
        builder = DefaultPromptBuilder()
        result = builder.build("What is RAG?", [_scored("context")])
        assert "What is RAG?" in result

    def test_build_contains_chunk_text(self):
        builder = DefaultPromptBuilder()
        result = builder.build("q", [_scored("Chunk content here")])
        assert "Chunk content here" in result

    def test_separator_present(self):
        builder = DefaultPromptBuilder()
        result = builder.build("q", [_scored("ctx")])
        assert "\n\n---\n\n" in result

    def test_numbered_format_has_brackets(self):
        builder = DefaultPromptBuilder(context_format="numbered")
        result = builder.build("q", [_scored("a"), _scored("b", rank=2)])
        assert "[1]" in result
        assert "[2]" in result

    def test_labeled_format_has_source_prefix(self):
        builder = DefaultPromptBuilder(context_format="labeled")
        result = builder.build("q", [_scored("a"), _scored("b", rank=2)])
        assert "Source 1:" in result
        assert "Source 2:" in result

    def test_plain_format_no_labels(self):
        builder = DefaultPromptBuilder(context_format="plain")
        result = builder.build("q", [_scored("a")])
        assert "[1]" not in result
        assert "Source 1:" not in result

    def test_no_chunks_returns_no_documents_message(self):
        builder = DefaultPromptBuilder()
        result = builder.build("q", [])
        assert "no relevant documents" in result

    def test_empty_query_raises_prompt_error(self):
        builder = DefaultPromptBuilder()
        with pytest.raises(PromptError):
            builder.build("", [_scored("ctx")])

    def test_whitespace_query_raises_prompt_error(self):
        builder = DefaultPromptBuilder()
        with pytest.raises(PromptError):
            builder.build("   ", [_scored("ctx")])

    def test_custom_system_prompt(self):
        builder = DefaultPromptBuilder(system="Custom system message.")
        result = builder.build("q", [_scored("ctx")])
        assert "Custom system message." in result

    def test_system_prompt_property(self):
        builder = DefaultPromptBuilder(system="Hello system.")
        assert builder.system_prompt == "Hello system."

    def test_multiple_chunks_all_included(self):
        chunks = [_scored(f"chunk {i}", rank=i + 1) for i in range(5)]
        builder = DefaultPromptBuilder()
        result = builder.build("q", chunks)
        for i in range(5):
            assert f"chunk {i}" in result
