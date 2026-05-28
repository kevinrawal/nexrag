"""
RecursiveChunker — splits documents using a cascading separator strategy.

Tries separators in order: paragraph → newline → sentence → word → character.
Merges small pieces into chunks of ~chunk_size with chunk_overlap overlap.

This is the recommended default chunker for most use cases because it:
  - Preserves semantic boundaries (prefers paragraph/sentence breaks)
  - Handles any text length (falls back to character splitting if needed)
  - Produces consistent chunk sizes with configurable overlap

No external dependencies.
"""

from __future__ import annotations

from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.models.chunk import Chunk
from nexrag.core.models.document import Document
from nexrag.exceptions import ChunkError

_DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


class RecursiveChunker(BaseChunker):
    """
    Recursively splits text using a separator cascade, then merges with overlap.

    Args:
        chunk_size:    Target maximum character length per chunk. Default: 512.
        chunk_overlap: Characters of overlap between consecutive chunks. Default: 64.
        min_chunk_size: Chunks shorter than this are dropped. Default: 50.
        separators:    Ordered list of separators to try. Falls back to character
                       splitting when the list is exhausted.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ChunkError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size}).",
                stage="chunker",
                component="RecursiveChunker",
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._separators = separators if separators is not None else _DEFAULT_SEPARATORS

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split the document content into overlapping chunks.

        Args:
            document: Document to chunk.

        Returns:
            List of Chunks derived from this document.

        Raises:
            ChunkError: If the document produces no chunks above min_chunk_size.
        """
        text = document.content
        if not text or not text.strip():
            raise ChunkError(
                f"Document '{document.doc_id}' has empty content. Cannot chunk.",
                stage="chunker",
                component="RecursiveChunker",
            )

        raw_chunks = self._split(text.strip(), self._separators)
        filtered = [c for c in raw_chunks if len(c.strip()) >= self._min_chunk_size]

        if not filtered:
            raise ChunkError(
                f"Document '{document.doc_id}' produced no chunks above "
                f"min_chunk_size={self._min_chunk_size}. "
                f"Lower min_chunk_size or provide longer content.",
                stage="chunker",
                component="RecursiveChunker",
            )

        total = len(filtered)
        return [
            Chunk(
                text=text.strip(),
                chunk_index=i,
                total_chunks=total,
                parent_doc_id=document.doc_id,
                metadata=document.metadata,
            )
            for i, text in enumerate(filtered)
        ]

    # Private helpers

    def _split(self, text: str, separators: list[str]) -> list[str]:
        """
        Recursively split text into pieces <= chunk_size using the separator cascade.
        Returns a flat list of text pieces ready for merging.
        """
        if not text:
            return []

        # Find the first separator that actually appears in the text (or use "").
        sep = ""
        remaining = separators[1:] if separators else []
        for i, s in enumerate(separators):
            if s == "" or s in text:
                sep = s
                remaining = separators[i + 1 :]
                break

        if sep == "":
            # Character-level — no separator found; slice directly.
            return self._char_split(text)

        raw_splits = [s for s in text.split(sep) if s]

        # Pieces that are small enough to merge; batched until we hit a large one.
        good: list[str] = []
        result: list[str] = []

        for piece in raw_splits:
            if len(piece) <= self._chunk_size:
                good.append(piece)
            else:
                # Flush accumulated small pieces first.
                if good:
                    result.extend(self._merge(good, sep))
                    good = []
                # Recurse into the oversized piece with the remaining separators.
                if remaining:
                    result.extend(self._split(piece, remaining))
                else:
                    result.extend(self._char_split(piece))

        if good:
            result.extend(self._merge(good, sep))

        return result

    def _merge(self, splits: list[str], sep: str) -> list[str]:
        """
        Greedily merge small text pieces into chunks of ~chunk_size with overlap.
        """
        sep_len = len(sep)
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        def _join(parts: list[str]) -> str:
            return sep.join(parts)

        for piece in splits:
            piece_len = len(piece)
            add_len = piece_len + (sep_len if current else 0)

            if current and current_len + add_len > self._chunk_size:
                # Emit the current chunk.
                chunks.append(_join(current))

                # Trim from the left until the retained overlap fits.
                while current and current_len > self._chunk_overlap:
                    removed = current.pop(0)
                    current_len -= len(removed) + sep_len
                    if current_len < 0:
                        current_len = 0

            current.append(piece)
            current_len = len(_join(current))

        if current:
            chunks.append(_join(current))

        return chunks

    def _char_split(self, text: str) -> list[str]:
        """Split at character level when all separators are exhausted."""
        step = max(1, self._chunk_size - self._chunk_overlap)
        return [text[i : i + self._chunk_size] for i in range(0, len(text), step)]
