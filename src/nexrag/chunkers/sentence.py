"""
SentenceChunker and SentenceWindowChunker — sentence-boundary splitting.

SentenceChunker greedily packs whole sentences into chunks up to chunk_size
characters, never splitting mid-sentence, with a character-budget overlap.

SentenceWindowChunker emits one chunk per sentence but expands each chunk's text
to include N neighbouring sentences on each side (the "window"), giving the
retriever local context. The core sentence is preserved in metadata.

No external dependencies — sentences are detected with a lightweight regex.
"""

from __future__ import annotations

import re
from typing import Any

from nexrag.chunkers._util import assemble_chunks, require_content
from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.models.chunk import Chunk
from nexrag.core.models.document import Document
from nexrag.exceptions import ChunkError

# Split after ., !, or ? followed by whitespace. Good enough without a heavy NLP dep.
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on terminal punctuation. Never returns empties."""
    parts = _SENTENCE_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


class SentenceChunker(BaseChunker):
    """
    Packs whole sentences into chunks up to chunk_size characters.

    A single sentence longer than chunk_size becomes its own chunk (never split
    mid-sentence). Overlap carries trailing sentences whose combined length fits
    within chunk_overlap characters into the next chunk.

    Args:
        chunk_size:     Soft maximum characters per chunk. Default 512.
        chunk_overlap:  Characters of trailing-sentence overlap. Default 64.
        min_chunk_size: Chunks shorter than this are dropped. Default 50.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ChunkError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size}).",
                stage="chunker",
                component="SentenceChunker",
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        text = require_content(document, "SentenceChunker")
        sentences = split_sentences(text)

        chunks: list[str] = []
        current: list[str] = []
        current_len = 0
        for sentence in sentences:
            add_len = len(sentence) + (1 if current else 0)
            if current and current_len + add_len > self._chunk_size:
                chunks.append(" ".join(current))
                current, current_len = self._overlap_tail(current)
            current.append(sentence)
            current_len = len(" ".join(current))
        if current:
            chunks.append(" ".join(current))

        return assemble_chunks(
            chunks, document, min_chunk_size=self._min_chunk_size, component="SentenceChunker"
        )

    def _overlap_tail(self, sentences: list[str]) -> tuple[list[str], int]:
        """Return the trailing sentences (and their length) that fit in chunk_overlap chars."""
        if self._chunk_overlap <= 0:
            return [], 0
        tail: list[str] = []
        length = 0
        for sentence in reversed(sentences):
            add = len(sentence) + (1 if tail else 0)
            if length + add > self._chunk_overlap:
                break
            tail.insert(0, sentence)
            length += add
        return tail, len(" ".join(tail)) if tail else 0


class SentenceWindowChunker(BaseChunker):
    """
    One chunk per sentence, expanded with N neighbouring sentences as context.

    The chunk text is the window (core sentence ± window_size neighbours), so the
    retriever matches and returns with surrounding context. The core sentence is
    stored in metadata["window_core"].

    Note: NexRAG embeds and returns the same chunk text, so the window text is what
    gets embedded. True "retrieve tight, expand on return" (embed only the core,
    expand on retrieval) would require a pipeline hook and is a future enhancement.

    Args:
        window_size:    Number of neighbouring sentences on EACH side. Default 1.
        min_chunk_size: Windows shorter than this are dropped. Default 1.
    """

    def __init__(self, window_size: int = 1, min_chunk_size: int = 1) -> None:
        if window_size < 0:
            raise ChunkError(
                f"window_size ({window_size}) must be >= 0.",
                stage="chunker",
                component="SentenceWindowChunker",
            )
        self._window_size = window_size
        self._min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        text = require_content(document, "SentenceWindowChunker")
        sentences = split_sentences(text)

        windows: list[str] = []
        metas: list[dict[str, Any]] = []
        n = len(sentences)
        for i, core in enumerate(sentences):
            start = max(0, i - self._window_size)
            end = min(n, i + self._window_size + 1)
            windows.append(" ".join(sentences[start:end]))
            metas.append({"window_core": core, "window_size": self._window_size})

        return assemble_chunks(
            windows,
            document,
            min_chunk_size=self._min_chunk_size,
            component="SentenceWindowChunker",
            per_chunk_metadata=metas,
        )
