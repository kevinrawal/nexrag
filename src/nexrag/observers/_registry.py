"""
Declarative metric registry for NexRAG OpenTelemetry instruments.

Each entry maps a (stage, metadata-key) pair to an OTel instrument
descriptor. The OpenTelemetryObserver reads this table to decide which
instruments to record for a given PipelineEvent.

Extension point: add a new row here when a new metric is needed for an
existing stage. New pipeline stages get the generic per-stage instruments
(duration, invocations, errors) automatically at zero cost.

MetricSpec fields:
  stage        Stage name from PipelineEvent.stage.
  meta_key     Key(s) in event.metadata that carry the value. A tuple means
               "try each key in order; use the first one found".
  name         OTel instrument name (nexrag.<name>).
  kind         "histogram" | "counter" | "gauge".
  unit         UCUM unit string (ms, {chunks}, {tokens}, USD, 1).
  description  Short human-readable description.
  attributes   Extra static attributes merged into every data point.
  status_filter If set, only record when event.status matches this value.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class MetricSpec:
    stage: str
    meta_key: str | tuple[str, ...]
    name: str
    kind: str
    unit: str
    description: str
    attributes: dict[str, str] = field(default_factory=dict)
    status_filter: str | None = "completed"


# ─── Retrieval ────────────────────────────────────────────────────────────────

RETRIEVAL_SPECS: list[MetricSpec] = [
    MetricSpec(
        stage="retriever",
        meta_key="chunks_retrieved",
        name="retrieval.chunks_returned",
        kind="histogram",
        unit="{chunks}",
        description="Number of chunks returned by the retriever",
    ),
    MetricSpec(
        stage="retriever",
        meta_key="top_score",
        name="retrieval.top_score",
        kind="histogram",
        unit="1",
        description="Cosine similarity of the best-matched chunk",
    ),
    MetricSpec(
        stage="retriever",
        meta_key="avg_score",
        name="retrieval.avg_score",
        kind="histogram",
        unit="1",
        description="Average cosine similarity across retrieved chunks",
    ),
    MetricSpec(
        stage="retriever",
        meta_key="bottom_score",
        name="retrieval.bottom_score",
        kind="histogram",
        unit="1",
        description="Cosine similarity of the weakest retrieved chunk",
    ),
    MetricSpec(
        stage="retriever",
        meta_key="score_spread",
        name="retrieval.score_spread",
        kind="histogram",
        unit="1",
        description="Difference between top and bottom retrieval score",
    ),
    MetricSpec(
        stage="retriever",
        meta_key="chunks_retrieved",
        name="retrieval.empty_results",
        kind="counter",
        unit="{queries}",
        description="Queries that returned zero chunks",
        status_filter="completed",
    ),
]

# ─── Reranker ─────────────────────────────────────────────────────────────────

RERANKER_SPECS: list[MetricSpec] = [
    MetricSpec(
        stage="reranker",
        meta_key="chunks_in",
        name="reranker.chunks_in",
        kind="histogram",
        unit="{chunks}",
        description="Chunks sent to the reranker",
    ),
    MetricSpec(
        stage="reranker",
        meta_key="chunks_out",
        name="reranker.chunks_out",
        kind="histogram",
        unit="{chunks}",
        description="Chunks returned after reranking",
    ),
]

# ─── LLM ──────────────────────────────────────────────────────────────────────

LLM_SPECS: list[MetricSpec] = [
    MetricSpec(
        stage="llm",
        meta_key=("token_usage.prompt_tokens",),
        name="llm.tokens_input",
        kind="histogram",
        unit="{tokens}",
        description="LLM prompt tokens per query",
    ),
    MetricSpec(
        stage="llm",
        meta_key=("token_usage.completion_tokens",),
        name="llm.tokens_output",
        kind="histogram",
        unit="{tokens}",
        description="LLM completion tokens per query",
    ),
    MetricSpec(
        stage="llm",
        meta_key=("token_usage.total_tokens",),
        name="llm.tokens_total",
        kind="histogram",
        unit="{tokens}",
        description="Total LLM tokens per query (prompt + completion)",
    ),
    MetricSpec(
        stage="llm",
        meta_key="response_length",
        name="llm.response_length_chars",
        kind="histogram",
        unit="{chars}",
        description="Character length of the LLM response",
    ),
]

# ─── Embedding ────────────────────────────────────────────────────────────────

EMBEDDING_SPECS: list[MetricSpec] = [
    MetricSpec(
        stage="embedder",
        meta_key="chunk_count",
        name="embedding.batch_size",
        kind="histogram",
        unit="{chunks}",
        description="Chunks embedded per batch call",
    ),
    MetricSpec(
        stage="embedder",
        meta_key="dimensions",
        name="embedding.dimensions",
        kind="gauge",
        unit="{dimensions}",
        description="Vector dimensionality of the embedding model",
    ),
]

# ─── Context (prompt builder) ─────────────────────────────────────────────────

CONTEXT_SPECS: list[MetricSpec] = [
    MetricSpec(
        stage="prompt_builder",
        meta_key="prompt_length",
        name="context.prompt_length_chars",
        kind="histogram",
        unit="{chars}",
        description="Total prompt character length sent to the LLM",
    ),
    MetricSpec(
        stage="prompt_builder",
        meta_key="estimated_tokens",
        name="context.token_budget_used",
        kind="histogram",
        unit="{tokens}",
        description="Estimated tokens in the constructed context",
    ),
    MetricSpec(
        stage="prompt_builder",
        meta_key="chunks_sent",
        name="context.chunks_sent",
        kind="histogram",
        unit="{chunks}",
        description="Number of chunks included in the LLM context",
    ),
]

# ─── Ingestion ────────────────────────────────────────────────────────────────

INGESTION_SPECS: list[MetricSpec] = [
    MetricSpec(
        stage="loader",
        meta_key="document_count",
        name="ingest.documents_loaded",
        kind="histogram",
        unit="{documents}",
        description="Documents produced by the loader per ingest call",
    ),
    MetricSpec(
        stage="chunker",
        meta_key="chunk_count",
        name="ingest.chunks_produced",
        kind="histogram",
        unit="{chunks}",
        description="Chunks produced per document",
    ),
    MetricSpec(
        stage="index_writer",
        meta_key="chunks_written",
        name="ingest.chunks_written",
        kind="histogram",
        unit="{chunks}",
        description="Chunks successfully written to the vector DB",
    ),
    MetricSpec(
        stage="idempotency_check",
        meta_key="sources_skipped",
        name="ingest.sources_skipped",
        kind="counter",
        unit="{sources}",
        description="Sources skipped due to idempotency (on_conflict=skip/overwrite)",
    ),
    MetricSpec(
        stage="idempotency_check",
        meta_key="sources_overwritten",
        name="ingest.sources_overwritten",
        kind="counter",
        unit="{sources}",
        description="Sources overwritten in the vector DB",
    ),
]

# ─── Evaluation (LLM-as-judge) ────────────────────────────────────────────────

EVALUATION_SPECS: list[MetricSpec] = [
    MetricSpec(
        stage="evaluation",
        meta_key="value",
        name="eval.metric_value",
        kind="histogram",
        unit="1",
        description="LLM-as-judge evaluation metric value",
    ),
]

# ─── Combined registry ────────────────────────────────────────────────────────

ALL_SPECS: list[MetricSpec] = (
    RETRIEVAL_SPECS
    + RERANKER_SPECS
    + LLM_SPECS
    + EMBEDDING_SPECS
    + CONTEXT_SPECS
    + INGESTION_SPECS
    + EVALUATION_SPECS
)


def specs_for_stage(stage: str) -> list[MetricSpec]:
    return [s for s in ALL_SPECS if s.stage == stage]
