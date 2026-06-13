"""Chunk Models for RAG pipelines."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """
    A single chunk of text derived from a Document.

    Attributes:
        text:          The chunk text.
        chunk_index:   Position of this chunk within its parent document (0-based).
        total_chunks:  Total number of chunks the parent document was split into.
        parent_doc_id: ID of the Document this chunk came from.
        content_hash:  sha256 of text. Auto-computed. Identifies identical text —
                       never pass this manually.
        row_id:        Document-scoped storage ID, sha256(parent_doc_id:content_hash).
                       Auto-computed. This — not content_hash — is the vector-DB row
                       key, so identical text in two different documents stays in two
                       separate rows. Never pass this manually.
        metadata:      Inherited from parent Document metadata, merged with
                       chunk-level fields by the Chunker.
    """

    text: str
    chunk_index: int
    total_chunks: int
    parent_doc_id: str
    metadata: dict[str, Any] = field(default_factory=dict)

    # Computed automatically — do not pass manually.
    content_hash: str = field(init=False)
    row_id: str = field(init=False)

    def __post_init__(self) -> None:
        # frozen=True means we can't do self.content_hash = ...
        # object.__setattr__ is the correct escape hatch for frozen dataclasses.
        content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()
        object.__setattr__(self, "content_hash", content_hash)
        # Document-scoped row key: two documents containing identical text produce
        # the same content_hash but different row_ids, so neither overwrites the other.
        object.__setattr__(
            self,
            "row_id",
            hashlib.sha256(f"{self.parent_doc_id}:{content_hash}".encode()).hexdigest(),
        )

    def __repr__(self) -> str:
        preview = self.text[:50].replace("\n", " ")
        if len(self.text) > 50:
            preview += "..."
        return (
            f"Chunk(index={self.chunk_index}/{self.total_chunks}, "
            f"hash={self.content_hash[:8]}…, "
            f"text={preview!r})"
        )


@dataclass(frozen=True)
class ScoredChunk:
    """
    A Chunk paired with its retrieval score and rank.
    Produced by a Retriever. Consumed by a PromptBuilder.

    Attributes:
        chunk: The retrieved Chunk.
        score: Similarity score from the vector DB (higher = more relevant).
        rank:  Position in the retrieval result list (1 = most similar).
    """

    chunk: Chunk
    score: float
    rank: int

    def __repr__(self) -> str:
        return (
            f"ScoredChunk(rank={self.rank}, score={self.score:.4f}, "
            f"hash={self.chunk.content_hash[:8]}…)"
        )
