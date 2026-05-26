import pytest

from nexrag.chunkers.recursive import RecursiveChunker
from nexrag.core.models.document import Document
from nexrag.exceptions import ChunkError


def _doc(content: str) -> Document:
    return Document(content=content)


class TestRecursiveChunker:
    def test_short_text_returns_single_chunk(self):
        chunker = RecursiveChunker(chunk_size=200, min_chunk_size=1)
        doc = _doc("Short text that fits in one chunk.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text == "Short text that fits in one chunk."

    def test_chunk_indices_are_correct(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0, min_chunk_size=1)
        doc = _doc("a" * 200)
        chunks = chunker.chunk(doc)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.total_chunks == len(chunks)

    def test_parent_doc_id_propagated(self):
        doc = _doc("Some text " * 20)
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0, min_chunk_size=1)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.parent_doc_id == doc.doc_id

    def test_metadata_propagated(self):
        doc = Document(content="Text " * 30, metadata={"source": "test.pdf", "year": 2024})
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0, min_chunk_size=1)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test.pdf"
            assert chunk.metadata["year"] == 2024

    def test_chunk_size_respected(self):
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=0, min_chunk_size=1)
        doc = _doc("word " * 200)
        chunks = chunker.chunk(doc)
        for chunk in chunks:
            assert len(chunk.text) <= 100 + 20  # small tolerance for word boundaries

    def test_paragraph_split_preferred(self):
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        chunker = RecursiveChunker(chunk_size=30, chunk_overlap=0, min_chunk_size=5)
        doc = _doc(text)
        chunks = chunker.chunk(doc)
        texts = [c.text for c in chunks]
        assert any("First paragraph" in t for t in texts)
        assert any("Second paragraph" in t for t in texts)

    def test_content_hash_unique_per_chunk(self):
        # Build content where every 50-char window is distinct.
        words = [f"uniqueword{i}" for i in range(200)]
        doc = _doc(" ".join(words))
        chunker = RecursiveChunker(chunk_size=80, chunk_overlap=0, min_chunk_size=1)
        chunks = chunker.chunk(doc)
        hashes = [c.content_hash for c in chunks]
        assert len(hashes) == len(set(hashes)), "All chunk hashes should be unique"

    def test_empty_content_raises_chunk_error(self):
        chunker = RecursiveChunker()
        with pytest.raises(ChunkError):
            chunker.chunk(_doc(""))

    def test_whitespace_only_raises_chunk_error(self):
        chunker = RecursiveChunker()
        with pytest.raises(ChunkError):
            chunker.chunk(_doc("   \n\n   "))

    def test_overlap_too_large_raises_chunk_error(self):
        with pytest.raises(ChunkError):
            RecursiveChunker(chunk_size=100, chunk_overlap=100)

    def test_all_below_min_size_raises_chunk_error(self):
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=0, min_chunk_size=10000)
        with pytest.raises(ChunkError, match="min_chunk_size"):
            chunker.chunk(_doc("Short."))

    def test_char_split_fallback(self):
        # A long word with no separators — must fall back to character splitting
        chunker = RecursiveChunker(chunk_size=10, chunk_overlap=0, min_chunk_size=1)
        doc = _doc("A" * 50)
        chunks = chunker.chunk(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.text) <= 10
