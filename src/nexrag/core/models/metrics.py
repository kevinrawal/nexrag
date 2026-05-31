"""
TokenUsage and RunMetrics — observability data attached to every pipeline run.

TokenUsage lives here because it is a metrics type, not a result structure.
RunMetrics aggregates all per-run data so callers get one object instead of
having to aggregate individual PipelineEvents themselves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    """
    Token counts from the LLM call.

    Populated when the LLM provider returns usage data.
    None when the provider does not expose token counts (e.g. some Ollama models).
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def __repr__(self) -> str:
        return (
            f"TokenUsage(prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, "
            f"total={self.total_tokens})"
        )


@dataclass(frozen=True)
class RunMetrics:
    """
    Aggregated metrics for a single pipeline run.

    Attached to PipelineResult (query runs) and IngestionResult (ingest runs).
    stage_latencies keys match the stage names emitted as PipelineEvents.

    Attributes:
        pipeline_id:      Ties this metrics object back to the pipeline run logs.
        total_latency_ms: Wall-clock time for the full run (includes overhead).
        stage_latencies:  Per-stage duration in ms. Sum ≤ total_latency_ms.
        token_usage:      LLM token counts. None for ingest runs or when LLM
                          does not expose usage (e.g. Ollama).
        model:            LLM model name used. None for ingest runs.
        chunks_retrieved: Number of chunks retrieved from vector DB (query runs).
        chunks_written:   Number of chunks written to vector DB (ingest runs).
    """

    pipeline_id: str
    total_latency_ms: float
    stage_latencies: dict[str, float] = field(default_factory=dict)
    token_usage: TokenUsage | None = None
    model: str | None = None
    chunks_retrieved: int | None = None
    chunks_written: int | None = None

    def __repr__(self) -> str:
        parts = [
            f"pipeline_id={self.pipeline_id!r}",
            f"total_latency_ms={self.total_latency_ms:.1f}",
        ]
        if self.token_usage is not None:
            parts.append(f"tokens={self.token_usage.total_tokens}")
        if self.model is not None:
            parts.append(f"model={self.model!r}")
        return f"RunMetrics({', '.join(parts)})"

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON logging or dashboards."""
        d: dict[str, Any] = {
            "pipeline_id": self.pipeline_id,
            "total_latency_ms": self.total_latency_ms,
            "stage_latencies": dict(self.stage_latencies),
        }
        if self.token_usage is not None:
            d["token_usage"] = {
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "total_tokens": self.token_usage.total_tokens,
            }
        if self.model is not None:
            d["model"] = self.model
        if self.chunks_retrieved is not None:
            d["chunks_retrieved"] = self.chunks_retrieved
        if self.chunks_written is not None:
            d["chunks_written"] = self.chunks_written
        return d
