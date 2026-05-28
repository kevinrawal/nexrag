"""
BaseChunker — contract for all text chunkers.

A Chunker receives a Document and returns a list of Chunks.
It is responsible for:
    1. Splitting the document text into appropriately sized pieces.
    2. Populating chunk_index, total_chunks, source, parent_doc_id.
    3. Inheriting the Document's metadata into each Chunk's metadata.

The content_hash on each Chunk is auto-computed by the Chunk dataclass —
chunkers do not set it manually.

Built-in strategies (Phase 1):
    FixedChunker, RecursiveChunker, SentenceChunker, ParagraphChunker

Custom implementation pattern:
    class MySemanticChunker(BaseChunker):
        def chunk(self, document: Document) -> list[Chunk]:
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nexrag.core.models.chunk import Chunk
from nexrag.core.models.document import Document


class BaseChunker(ABC):
    """Abstract base class for all NexRAG chunkers."""

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """
        Split a Document into a list of Chunks.

        Args:
            document: The Document to split.

        Returns:
            Ordered list of Chunks. Never empty — raise ChunkError if the
            document produced no usable chunks.

        Raises:
            ChunkError: If chunking fails or produces no output.
        """
