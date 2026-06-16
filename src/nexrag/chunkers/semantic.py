"""
SemanticChunker — splits where adjacent sentences become semantically dissimilar.

Embeds each sentence (optionally buffered with neighbours to reduce noise),
measures the embedding distance between consecutive sentences, and inserts a chunk
boundary wherever that distance exceeds a threshold. The threshold is either given
explicitly or derived from a percentile of the observed distances.

The embedder is supplied as a NESTED component sub-config on the chunker, resolved
independently of the pipeline's main embedder — so you can embed-for-chunking with
a cheap model and generate with a strong one.
"""

from __future__ import annotations

import math

from nexrag.chunkers._util import assemble_chunks, require_content
from nexrag.chunkers.sentence import split_sentences
from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.core.models.chunk import Chunk
from nexrag.core.models.document import Document
from nexrag.exceptions import ChunkError


class SemanticChunker(BaseChunker):
    """
    Groups sentences into chunks at semantic breakpoints.

    Args:
        embedder:       Embedder used to embed sentences (independent of the pipeline embedder).
        threshold:      Explicit distance breakpoint in [0, 2]. If None, derived from ``percentile``.
        percentile:     Percentile of adjacent distances used as the breakpoint when
                        ``threshold`` is None. Default 95 (top 5% of jumps become breaks).
        buffer_size:    Neighbouring sentences combined on each side before embedding. Default 1.
        min_chunk_size: Chunks shorter than this are dropped. Default 1.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        threshold: float | None = None,
        percentile: float = 95.0,
        buffer_size: int = 1,
        min_chunk_size: int = 1,
    ) -> None:
        if embedder is None:
            raise ChunkError(
                "SemanticChunker requires an embedder.",
                stage="chunker",
                component="SemanticChunker",
            )
        if buffer_size < 0:
            raise ChunkError(
                f"buffer_size ({buffer_size}) must be >= 0.",
                stage="chunker",
                component="SemanticChunker",
            )
        self._embedder = embedder
        self._threshold = threshold
        self._percentile = percentile
        self._buffer_size = buffer_size
        self._min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        text = require_content(document, "SemanticChunker")
        sentences = split_sentences(text)

        # Trivial cases: 0/1 sentence → one chunk (no boundaries to compute).
        if len(sentences) <= 1:
            return assemble_chunks(
                [text.strip()],
                document,
                min_chunk_size=self._min_chunk_size,
                component="SemanticChunker",
            )

        combined = [self._window(sentences, i) for i in range(len(sentences))]
        embeddings = self._embedder.embed(combined)

        distances = [
            self._distance(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)
        ]
        breakpoint_distance = self._breakpoint(distances)

        groups: list[str] = []
        current: list[str] = [sentences[0]]
        for i, dist in enumerate(distances):
            if dist > breakpoint_distance:
                groups.append(" ".join(current))
                current = []
            current.append(sentences[i + 1])
        if current:
            groups.append(" ".join(current))

        return assemble_chunks(
            groups, document, min_chunk_size=self._min_chunk_size, component="SemanticChunker"
        )

    def _window(self, sentences: list[str], i: int) -> str:
        start = max(0, i - self._buffer_size)
        end = min(len(sentences), i + self._buffer_size + 1)
        return " ".join(sentences[start:end])

    def _breakpoint(self, distances: list[float]) -> float:
        if self._threshold is not None:
            return self._threshold
        if not distances:
            return float("inf")
        return self._percentile_value(distances, self._percentile)

    @staticmethod
    def _percentile_value(values: list[float], percentile: float) -> float:
        ordered = sorted(values)
        if not ordered:
            return float("inf")
        rank = (percentile / 100.0) * (len(ordered) - 1)
        low = math.floor(rank)
        high = math.ceil(rank)
        if low == high:
            return ordered[int(rank)]
        frac = rank - low
        return ordered[low] * (1 - frac) + ordered[high] * frac

    @staticmethod
    def _distance(a: list[float], b: list[float]) -> float:
        """Cosine distance in [0, 2]; 0 = identical direction."""
        dot = sum(x * y for x, y in zip(a, b, strict=False))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        if na == 0 or nb == 0:
            return 1.0
        return 1.0 - (dot / (na * nb))
