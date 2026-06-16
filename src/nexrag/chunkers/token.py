"""
TokenChunker — fixed-size chunks measured in tokens, with token overlap.

Unlike FixedChunker (character windows) this is tokenizer-aware: chunk_size and
chunk_overlap are counted in tokens, so chunks line up with what the embedder/LLM
actually sees. Deterministic and predictable — the recommended baseline when token
budgets matter.

Requires: pip install "nexrag[tiktoken]"  (tiktoken)
"""

from __future__ import annotations

from typing import Any

from nexrag.chunkers._util import assemble_chunks, require_content
from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.models.chunk import Chunk
from nexrag.core.models.document import Document
from nexrag.exceptions import ChunkError


class TokenChunker(BaseChunker):
    """
    Splits text into fixed-size token windows with token overlap.

    Args:
        chunk_size:     Maximum tokens per chunk. Default 512.
        chunk_overlap:  Tokens of overlap between consecutive chunks. Default 64.
        min_chunk_size: Chunks whose text is shorter than this (chars) are dropped. Default 50.
        encoding:       tiktoken encoding name. Default "cl100k_base".
        model:          Optional model name — if set, its encoding overrides ``encoding``.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        min_chunk_size: int = 50,
        encoding: str = "cl100k_base",
        model: str | None = None,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ChunkError(
                f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size}).",
                stage="chunker",
                component="TokenChunker",
            )
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._encoding = encoding
        self._model = model
        self._encoder: Any = None

    def chunk(self, document: Document) -> list[Chunk]:
        text = require_content(document, "TokenChunker").strip()
        encoder = self._get_encoder()
        tokens = encoder.encode(text)

        step = max(1, self._chunk_size - self._chunk_overlap)
        pieces = [
            encoder.decode(tokens[i : i + self._chunk_size]).strip()
            for i in range(0, len(tokens), step)
        ]
        return assemble_chunks(
            pieces, document, min_chunk_size=self._min_chunk_size, component="TokenChunker"
        )

    def _get_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            import tiktoken
        except ImportError as e:
            raise ChunkError(
                "tiktoken is required for TokenChunker. "
                'Install it: pip install "nexrag[tiktoken]" or pip install tiktoken',
                stage="chunker",
                component="TokenChunker",
                cause=e,
            ) from e
        if self._model:
            self._encoder = tiktoken.encoding_for_model(self._model)
        else:
            self._encoder = tiktoken.get_encoding(self._encoding)
        return self._encoder
