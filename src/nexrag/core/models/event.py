"""
PipelineEvent is emitted by every stage in both pipelines.
The Observer receives these and decides what to do with them
(print to console in V1, ship to OpenTelemetry in V2+).

The event system is how NexRAG achieves observability without
coupling pipeline code to any specific logging or tracing library.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# Valid stage names — both pipelines combined.
StageName = Literal[
    "loader",
    "sanitizer",
    "chunker",
    "embedder",
    "fingerprint_check",
    "idempotency_check",
    "index_writer",
    "retriever",
    "reranker",
    "prompt_builder",
    "llm",
    "response_builder",
    "guardrail",
    "pipeline",
]

EventStatus = Literal["started", "completed", "failed"]


@dataclass(frozen=True)
class PipelineEvent:
    """
    Emitted at the start, completion, or failure of every pipeline stage.

    Attributes:
        pipeline_id:  Ties all events from one pipeline run together.
        stage:        Which stage emitted this event.
        status:       "started" | "completed" | "failed"
        duration_ms:  Wall-clock time for this stage. 0.0 on "started" events.
        metadata:     Stage-specific payload — chunk counts, token counts,
                      collection name, model name, error type, etc.
                      Each stage documents what it puts here.
    """

    pipeline_id: str
    stage: StageName
    status: EventStatus
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"PipelineEvent(pipeline_id={self.pipeline_id!r}, "
            f"stage={self.stage!r}, status={self.status!r}, "
            f"duration_ms={self.duration_ms:.1f})"
        )
