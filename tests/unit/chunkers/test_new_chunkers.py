from unittest.mock import MagicMock, patch

import pytest

from nexrag import _factory
from nexrag.core.config.schema import ChunkerConfig
from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.models.document import Document


def _doc(content: str, metadata: dict | None = None) -> Document:
    return Document(content=content, metadata=metadata or {"source": "t.md"})


# --- FixedChunker (now wired into the factory) ------------------------------


class TestFixedChunker:
    def test_splits_into_fixed_windows(self):
        from nexrag.chunkers.fixed import FixedChunker

        chunks = FixedChunker(chunk_size=10, chunk_overlap=2, min_chunk_size=1).chunk(
            _doc("a" * 35)
        )
        assert len(chunks) > 1
        assert all(c.parent_doc_id == chunks[0].parent_doc_id for c in chunks)


# --- SentenceChunker / SentenceWindowChunker --------------------------------


class TestSentenceChunker:
    def test_packs_sentences_without_splitting(self):
        from nexrag.chunkers.sentence import SentenceChunker

        text = "First sentence here. Second one follows. Third arrives now. Fourth ends it."
        chunks = SentenceChunker(chunk_size=40, chunk_overlap=0, min_chunk_size=1).chunk(_doc(text))
        assert len(chunks) > 1
        # No chunk splits a sentence — every chunk ends on terminal punctuation.
        assert all(c.text.rstrip()[-1] in ".!?" for c in chunks)

    def test_single_sentence_one_chunk(self):
        from nexrag.chunkers.sentence import SentenceChunker

        chunks = SentenceChunker(min_chunk_size=1).chunk(_doc("Only one sentence."))
        assert len(chunks) == 1


class TestSentenceWindowChunker:
    def test_window_includes_neighbors_and_core_metadata(self):
        from nexrag.chunkers.sentence import SentenceWindowChunker

        chunks = SentenceWindowChunker(window_size=1, min_chunk_size=1).chunk(
            _doc("A one. B two. C three.")
        )
        assert len(chunks) == 3
        # Middle chunk's window spans all three sentences; its core is the middle one.
        assert chunks[1].metadata["window_core"] == "B two."
        assert "A one." in chunks[1].text and "C three." in chunks[1].text


# --- MarkdownChunker --------------------------------------------------------


class TestMarkdownChunker:
    def test_splits_on_headings_and_records_header_path(self):
        from nexrag.chunkers.markdown import MarkdownChunker

        md = "# Title\nIntro paragraph.\n\n## Setup\nInstall instructions here.\n"
        chunks = MarkdownChunker(min_chunk_size=1).chunk(_doc(md))
        paths = [c.metadata.get("header_path") for c in chunks]
        assert "Title" in paths
        assert "Title > Setup" in paths


# --- CodeChunker (regex fallback when tree-sitter absent) -------------------


class TestCodeChunker:
    def test_splits_python_on_definitions(self):
        from nexrag.chunkers.code import CodeChunker

        code = "import os\n\ndef foo():\n    return 1\n\ndef bar():\n    return 2\n"
        chunks = CodeChunker(language="python", min_chunk_size=1).chunk(_doc(code))
        assert len(chunks) >= 2
        joined = "\n".join(c.text for c in chunks)
        assert "def foo" in joined and "def bar" in joined
        assert all(c.metadata["language"] == "python" for c in chunks)


# --- TokenChunker -----------------------------------------------------------


class TestTokenChunker:
    def test_splits_by_token_windows(self):
        pytest.importorskip("tiktoken")
        from nexrag.chunkers.token import TokenChunker

        text = "word " * 50
        chunks = TokenChunker(chunk_size=10, chunk_overlap=2, min_chunk_size=1).chunk(_doc(text))
        assert len(chunks) > 1


# --- SemanticChunker (mocked embedder) --------------------------------------


class TestSemanticChunker:
    def test_breaks_on_low_similarity(self):
        from nexrag.chunkers.semantic import SemanticChunker

        embedder = MagicMock()
        # buffer_size=0 → embed receives the 4 raw sentences; craft a big jump in the middle.
        embedder.embed.return_value = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
        chunker = SemanticChunker(embedder=embedder, threshold=0.5, buffer_size=0, min_chunk_size=1)
        chunks = chunker.chunk(_doc("A one. B two. C three. D four."))
        assert len(chunks) == 2

    def test_single_sentence_one_chunk_no_embed(self):
        from nexrag.chunkers.semantic import SemanticChunker

        embedder = MagicMock()
        chunks = SemanticChunker(embedder=embedder, min_chunk_size=1).chunk(_doc("Just one."))
        assert len(chunks) == 1
        embedder.embed.assert_not_called()


# --- PropositionChunker (mocked LLM) ----------------------------------------


class TestPropositionChunker:
    def test_one_chunk_per_proposition(self):
        from nexrag.chunkers.proposition import PropositionChunker

        llm = MagicMock()
        llm.generate.return_value = ("Prop one.\n- Prop two.\n2. Prop three.", None)
        chunks = PropositionChunker(llm=llm, min_chunk_size=1).chunk(_doc("Some source text."))
        assert [c.text for c in chunks] == ["Prop one.", "Prop two.", "Prop three."]


# --- Factory: nested component sub-configs resolved INDEPENDENTLY ------------


class DummyChunker(BaseChunker):
    def __init__(self, embedder=None, llm=None, **kwargs):
        self.embedder = embedder
        self.llm = llm

    def chunk(self, document):
        return []


class TestChunkerFactoryNesting:
    def test_semantic_resolves_its_own_embedder(self):
        sentinel = MagicMock()
        cfg = ChunkerConfig(
            strategy="semantic",
            embedder={"provider": "openai", "model": "text-embedding-3-small"},
        )
        with patch.object(_factory, "_build_embedder", return_value=sentinel) as m:
            chunker = _factory._build_chunker(cfg)
        # Resolved from the chunker's OWN nested config, not the pipeline embedder.
        assert m.call_args.args[0] is cfg.embedder
        assert chunker._embedder is sentinel

    def test_proposition_resolves_its_own_llm(self):
        sentinel = MagicMock()
        cfg = ChunkerConfig(
            strategy="proposition",
            llm={"provider": "openai", "model": "gpt-4o"},
        )
        with patch.object(_factory, "_build_llm", return_value=sentinel) as m:
            chunker = _factory._build_chunker(cfg)
        assert m.call_args.args[0] is cfg.llm
        assert chunker._llm is sentinel

    def test_custom_chunker_receives_nested_components(self):
        emb, llm = MagicMock(), MagicMock()
        cfg = ChunkerConfig.model_validate(
            {
                "strategy": "custom",
                "class": f"{__name__}.DummyChunker",
                "embedder": {"provider": "openai", "model": "text-embedding-3-small"},
                "llm": {"provider": "openai", "model": "gpt-4o"},
            }
        )
        with (
            patch.object(_factory, "_build_embedder", return_value=emb),
            patch.object(_factory, "_build_llm", return_value=llm),
        ):
            chunker = _factory._build_chunker(cfg)
        assert chunker.embedder is emb
        assert chunker.llm is llm

    def test_semantic_without_embedder_rejected_at_config(self):
        with pytest.raises(ValueError, match="chunker.embedder is required"):
            ChunkerConfig(strategy="semantic")

    def test_proposition_without_llm_rejected_at_config(self):
        with pytest.raises(ValueError, match="chunker.llm is required"):
            ChunkerConfig(strategy="proposition")
