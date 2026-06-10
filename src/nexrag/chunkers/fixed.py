"""
FixedChunker — splits documents into fixed-size character chunks with optional overlap.

Simpler than RecursiveChunker: no separator cascade, just a sliding window over the
raw character sequence. Use RecursiveChunker when semantic boundary preservation matters.
"""

from __future__ import annotations

from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.models.chunk import Chunk
from nexrag.core.models.document import Document
from nexrag.exceptions import ChunkError


class FixedChunker(BaseChunker):
    """
    Splits text into fixed-size chunks by character count with optional overlap.

    Args:
        chunk_size:     Maximum character length per chunk. Default: 512.
        chunk_overlap:  Characters of overlap between consecutive chunks. Default: 64.
        min_chunk_size: Chunks shorter than this are dropped. Default: 50.
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
                component="FixedChunker",
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split the document content into fixed-size character chunks.

        Args:
            document: Document to chunk.

        Returns:
            List of Chunks derived from this document.

        Raises:
            ChunkError: If the document produces no chunks above min_chunk_size.
        """
        text = document.content.strip()
        if not text:
            raise ChunkError(
                f"Document '{document.doc_id}' has empty content. Cannot chunk.",
                stage="chunker",
                component="FixedChunker",
            )

        step = max(1, self._chunk_size - self._chunk_overlap)
        raw = [text[i : i + self._chunk_size] for i in range(0, len(text), step)]
        filtered = [c for c in raw if len(c.strip()) >= self._min_chunk_size]

        if not filtered:
            raise ChunkError(
                f"Document '{document.doc_id}' produced no chunks above "
                f"min_chunk_size={self._min_chunk_size}. "
                f"Lower min_chunk_size or provide longer content.",
                stage="chunker",
                component="FixedChunker",
            )

        total = len(filtered)
        return [
            Chunk(
                text=c,
                chunk_index=i,
                total_chunks=total,
                parent_doc_id=document.doc_id,
                metadata=document.metadata,
            )
            for i, c in enumerate(filtered)
        ]
