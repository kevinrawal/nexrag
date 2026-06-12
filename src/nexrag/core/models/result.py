"""
PipelineResult is what the user receives from query.
NexRAG never returns a raw string — always this structured object.

Source is a user-facing view of a ScoredChunk — flattened for convenience
so users don't need to navigate nested objects to get what they need.

TokenUsage is defined in metrics.py — import it from there.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from nexrag.core.models.metrics import RunMetrics, TokenUsage

__all__ = ["Source", "PipelineResult", "TokenUsage", "RunMetrics"]


@dataclass(frozen=True)
class Source:
    """
    A single retrieved source as presented to the user.

    Flattened from ScoredChunk for ergonomic access.
    Users do result.sources[0].content and result.sources[0].source
    rather than navigating nested chunk objects.

    source is a convenience field surfaced from chunk.metadata["source"].
    It is whatever the Loader set — a file path, URL, S3 URI, page ID,
    or any other identifier meaningful to the user's application.
    If the loader did not set metadata["source"], this will be an empty string.

    Attributes:
        content:       The chunk text injected into the prompt.
        source:        Origin identifier from chunk metadata. Opaque — set by the Loader.
        metadata:      Full metadata dict from the chunk (includes source and all
                       user-defined fields like vendor, year, department, etc.)
        score:         Similarity score from the vector DB.
        rank:          Position in retrieval results (1 = most relevant).
        chunk_index:   Position of this chunk within its source document (0-based).
        total_chunks:  Total number of chunks produced from the source document.
        parent_doc_id: ID of the Document from which this chunk was created.
    """

    content: str
    source: str
    metadata: dict[str, Any]
    score: float
    rank: int
    chunk_index: int | None = None
    total_chunks: int | None = None
    parent_doc_id: str | None = None


@dataclass(frozen=True)
class PipelineResult:
    """
    Structured result returned by QueryPipeline.run().

    Attributes:
        answer:          The LLM's response text.
        query:           The original user query string.
        sources:         Ranked list of sources that were injected into the prompt.
                         Each Source contains the chunk text, its origin identifier,
                         full metadata, similarity score, and rank.
        scores:          Convenience list of similarity scores (mirrors sources order).
        collection_used: Which vector DB collection was queried.
        latency_ms:      Total wall-clock time for the full query pipeline.
        pipeline_id:     Unique ID for this pipeline run — use for log correlation.
        token_usage:     Token counts from the LLM. None if provider doesn't expose them.
        metadata:        Any extra data the pipeline wants to surface (model name, etc.).
        metrics:         Per-run aggregated metrics (latency, token usage, stage breakdown).
    """

    answer: str
    query: str
    sources: list[Source]
    scores: list[float]
    collection_used: str
    latency_ms: float
    pipeline_id: str
    token_usage: TokenUsage | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    metrics: RunMetrics | None = None

    def __repr__(self) -> str:
        preview = self.answer[:80].replace("\n", " ")
        if len(self.answer) > 80:
            preview += "..."
        return (
            f"PipelineResult(pipeline_id={self.pipeline_id!r}, "
            f"sources={len(self.sources)}, "
            f"latency_ms={self.latency_ms:.1f}, "
            f"answer={preview!r})"
        )
