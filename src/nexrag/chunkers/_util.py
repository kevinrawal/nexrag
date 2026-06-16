"""
Shared helpers for chunker implementations.

Every chunker turns a Document into a list of text pieces, then wraps them in
Chunk objects with the standard struct fields and inherited metadata. These
helpers centralise the empty-content guard, the min_chunk_size filter, and the
Chunk assembly so each chunker only has to express its splitting logic.
"""

from __future__ import annotations

from typing import Any

from nexrag.core.models.chunk import Chunk
from nexrag.core.models.document import Document
from nexrag.exceptions import ChunkError


def require_content(document: Document, component: str) -> str:
    """Return the document content, raising ChunkError if it is empty/whitespace."""
    text = document.content
    if not text or not text.strip():
        raise ChunkError(
            f"Document '{document.doc_id}' has empty content. Cannot chunk.",
            stage="chunker",
            component=component,
        )
    return text


def assemble_chunks(
    texts: list[str],
    document: Document,
    *,
    min_chunk_size: int,
    component: str,
    per_chunk_metadata: list[dict[str, Any]] | None = None,
) -> list[Chunk]:
    """
    Filter pieces below min_chunk_size and wrap the survivors in Chunk objects.

    per_chunk_metadata, when given, supplies extra metadata merged into each
    chunk's inherited document metadata (e.g. markdown header paths). It must be
    the same length as texts.

    Raises:
        ChunkError: If no piece survives the min_chunk_size filter.
    """
    metas = per_chunk_metadata if per_chunk_metadata is not None else [{} for _ in texts]
    pairs = [(t, m) for t, m in zip(texts, metas, strict=True) if len(t.strip()) >= min_chunk_size]
    if not pairs:
        raise ChunkError(
            f"Document '{document.doc_id}' produced no chunks above "
            f"min_chunk_size={min_chunk_size}. Lower min_chunk_size or provide longer content.",
            stage="chunker",
            component=component,
        )

    total = len(pairs)
    return [
        Chunk(
            text=text.strip(),
            chunk_index=i,
            total_chunks=total,
            parent_doc_id=document.doc_id,
            metadata={**document.metadata, **extra} if extra else document.metadata,
        )
        for i, (text, extra) in enumerate(pairs)
    ]
