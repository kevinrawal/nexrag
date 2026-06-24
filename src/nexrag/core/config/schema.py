"""
pydantic v2 models that map exactly to nexrag.yaml.

Every field has a default where sensible. Required fields raise a clear
ConfigError with the field path when missing — not a raw Pydantic error.

Structure mirrors nexrag.yaml exactly:
    NexRAGConfig
        ingestion: IngestionConfig
            loader:    LoaderConfig
            sanitizer: SanitizerConfig
            chunker:   ChunkerConfig
            embedder:  EmbedderConfig
            vector_db: VectorDBConfig
        query: QueryConfig
            retriever:      RetrieverConfig
            prompt:         PromptConfig
            llm:            LLMConfig
        observability: ObservabilityConfig
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ChromaDB (>=1.5) collection naming rules: 3–512 chars from [a-zA-Z0-9._-],
# must start and end with an alphanumeric character. Validating here means an
# invalid name fails fast at config load instead of crashing at the first DB call.
_COLLECTION_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._-]{1,510}[a-zA-Z0-9]$")

# Shared component configs
# Defined first because they are reused across ingestion, query, AND as nested
# sub-configs inside a chunker block (semantic/proposition chunkers resolve their
# own embedder/LLM, independent of the pipeline's main models).


class EmbedderConfig(BaseModel):
    provider: Literal["openai", "huggingface", "gemini", "ollama", "custom"]
    model: str
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int = 100
    max_retries: int = 2
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def __repr__(self) -> str:
        return f"EmbedderConfig(provider={self.provider!r}, model={self.model!r}, api_key=***)"


class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "gemini", "ollama", "custom"]
    model: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout: int = 30
    max_retries: int = 2
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def __repr__(self) -> str:
        return f"LLMConfig(provider={self.provider!r}, model={self.model!r}, api_key=***)"


# Ingestion sub-configs


class LoaderConfig(BaseModel):
    # Only types the factory can actually build are accepted. Schema-only stubs
    # (excel/json/code/word/html) are intentionally excluded until their loaders
    # are wired into _factory.py — otherwise they pass validation then fail at wiring.
    type: Literal["auto", "pdf", "txt", "text", "custom"] = "auto"
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)
    # ignore this if Loader doesn't support metadata extraction
    # but if it does, these fields will be extracted and stored in the vector DB
    # Extraction logic will be loader specific
    metadata_fields: list[str] | None = None
    include_metadata: bool = True

    model_config = {"populate_by_name": True}


class SanitizerConfig(BaseModel):
    enabled: bool = False
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def class_required_when_enabled(self) -> SanitizerConfig:
        if self.enabled and not self.class_path:
            raise ValueError(
                "sanitizer.class is required when sanitizer.enabled is true. "
                "Provide a dotted class path: myproject.sanitizers.MySanitizer"
            )
        return self


class ChunkerConfig(BaseModel):
    strategy: Literal[
        "recursive",
        "fixed",
        "token",
        "sentence",
        "sentence_window",
        "markdown",
        "code",
        "semantic",
        "proposition",
        "custom",
    ] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    separator: str = "\n\n"
    # Nested component sub-configs, resolved INDEPENDENTLY of the pipeline's main
    # embedder/LLM. semantic chunking needs an embedder; proposition chunking needs
    # an LLM. This lets a user run a cheap model for chunking and a strong one for
    # generation. Custom chunkers may declare either and receive it as a kwarg.
    embedder: EmbedderConfig | None = None
    llm: LLMConfig | None = None
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def validate_strategy_requirements(self) -> ChunkerConfig:
        if self.strategy == "custom" and not self.class_path:
            raise ValueError(
                "chunker.class is required when chunker.strategy is 'custom'. "
                "Provide a dotted class path: myproject.chunkers.MyChunker"
            )
        if self.strategy == "semantic" and self.embedder is None:
            raise ValueError(
                "chunker.embedder is required when chunker.strategy is 'semantic'. "
                "Provide a nested embedder block (resolved independently of the pipeline embedder)."
            )
        if self.strategy == "proposition" and self.llm is None:
            raise ValueError(
                "chunker.llm is required when chunker.strategy is 'proposition'. "
                "Provide a nested llm block (resolved independently of the query LLM)."
            )
        return self


class CollectionConfig(BaseModel):
    path: str | None = None  # local ChromaDB persist path
    host: str | None = None  # remote ChromaDB HTTP host
    port: int | None = None
    mode: Literal["memory", "persistent", "server"] = "persistent"
    description: str | None = None  # used by V2 router
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorDBConfig(BaseModel):
    provider: Literal["chroma", "pinecone", "custom"] = "chroma"
    default_collection: str
    collections: dict[str, CollectionConfig]
    on_conflict: Literal["overwrite", "skip", "append"] = "overwrite"
    upsert_batch_size: int = 500
    query_batch_size: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @field_validator("default_collection")
    @classmethod
    def validate_default_collection_name(cls, v: str) -> str:
        if not _COLLECTION_NAME_RE.match(v):
            raise ValueError(
                f"Invalid collection name '{v}'. Must be 3–512 characters, "
                "alphanumeric, dots, hyphens, or underscores; must start and end "
                "with an alphanumeric character."
            )
        return v

    @model_validator(mode="after")
    def validate_collections(self) -> VectorDBConfig:
        for name in self.collections:
            if not _COLLECTION_NAME_RE.match(name):
                raise ValueError(
                    f"Invalid collection name '{name}' in vector_db.collections. "
                    "Must be 3–512 characters, alphanumeric, dots, hyphens, or underscores; "
                    "must start and end with an alphanumeric character."
                )
        if self.default_collection not in self.collections:
            raise ValueError(
                f"vector_db.default_collection '{self.default_collection}' is not defined "
                f"in vector_db.collections. "
                f"Add it: collections:\n  {self.default_collection}:\n    path: ./nexrag_store/{self.default_collection}"
            )
        return self


class IngestionConfig(BaseModel):
    loader: LoaderConfig = Field(default_factory=LoaderConfig)
    sanitizer: SanitizerConfig = Field(default_factory=SanitizerConfig)
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedder: EmbedderConfig
    vector_db: VectorDBConfig


# Query sub-configs


class SparseConfig(BaseModel):
    """Sparse retriever component config — used inside RetrieverConfig for hybrid mode."""

    provider: Literal["bm25", "custom"] = "bm25"
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def class_required_for_custom(self) -> SparseConfig:
        if self.provider == "custom" and not self.class_path:
            raise ValueError(
                "retriever.sparse.class is required when retriever.sparse.provider is 'custom'. "
                "Provide a dotted class path: myproject.retrievers.MySparseRetriever"
            )
        return self


class RetrieverConfig(BaseModel):
    provider: Literal["dense", "hybrid", "bm25", "custom"] = "dense"
    top_k: int = 5
    score_threshold: float = 0.0
    metadata_filter: dict[str, Any] | None = None
    # hybrid-specific — ignored when provider != "hybrid"
    alpha: float = 0.7
    sparse_top_k: int | None = None
    sparse: SparseConfig = Field(default_factory=SparseConfig)
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class PromptConfig(BaseModel):
    system: str = (
        "You are a helpful assistant. Answer the user's question using only "
        "the context provided. If the answer is not in the context, say so "
        "clearly. Do not make up information."
    )
    template: Literal["default", "qa", "summarize", "custom"] | None = None
    context_format: Literal["numbered", "labeled", "plain"] = "numbered"
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class RerankerConfig(BaseModel):
    provider: Literal["cohere", "cross_encoder", "custom"]
    model: str
    api_key: str | None = None
    top_n: int = 5
    device: str | None = None
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def class_required_for_custom(self) -> RerankerConfig:
        if self.provider == "custom" and not self.class_path:
            raise ValueError(
                "reranker.class is required when reranker.provider is 'custom'. "
                "Provide a dotted class path: myproject.rerankers.MyReranker"
            )
        return self


class QueryConfig(BaseModel):
    embedder: EmbedderConfig | Literal["inherit"] = "inherit"
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    reranker: RerankerConfig | None = None
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    llm: LLMConfig
    max_query_length: int = 8000


# Observability


class OTelSignalsConfig(BaseModel):
    metrics: bool = True
    traces: bool = True
    logs: bool = True


class PrometheusExporterConfig(BaseModel):
    enabled: bool = False
    host: str = "0.0.0.0"
    port: int = 9464


class OTLPExporterConfig(BaseModel):
    enabled: bool = False
    endpoint: str = "http://localhost:4317"
    protocol: Literal["grpc", "http"] = "grpc"
    headers: dict[str, str] = Field(default_factory=dict)
    insecure: bool = True


class ConsoleExporterConfig(BaseModel):
    enabled: bool = False


class ExportersConfig(BaseModel):
    prometheus: PrometheusExporterConfig = Field(default_factory=PrometheusExporterConfig)
    otlp: OTLPExporterConfig = Field(default_factory=OTLPExporterConfig)
    console: ConsoleExporterConfig = Field(default_factory=ConsoleExporterConfig)


class PricingConfig(BaseModel):
    """Per-model token pricing in USD per 1K tokens. Used to compute llm.cost_per_query_usd."""

    input: float = 0.0
    output: float = 0.0


class MetricsConfig(BaseModel):
    retrieval: bool = True
    llm: bool = True
    cost: bool = True
    embedding: bool = True
    ingestion: bool = True
    pipeline: bool = True
    pricing: dict[str, PricingConfig] = Field(default_factory=dict)


class EvaluatorConfig(BaseModel):
    """Per-metric evaluator config. LLM/embedder are resolved independently per evaluator."""

    enabled: bool = False
    sample_rate: float | None = None
    llm: LLMConfig | None = None
    embedder: EmbedderConfig | None = None
    params: dict[str, Any] = Field(default_factory=dict)


class CustomEvaluatorConfig(BaseModel):
    """
    Config entry for a user-defined evaluator class.

    The class must subclass ``nexrag.core.interfaces.evaluator.BaseEvaluator``.
    It is instantiated with ``**params`` as keyword arguments — inject your own
    LLM or embedder there::

        observability:
          evaluations:
            enabled: true
            custom:
              - class: myapp.evals.DomainAccuracyEvaluator
                sample_rate: 0.5
                params:
                  threshold: 0.8
    """

    enabled: bool = True
    class_path: str = Field(alias="class")
    sample_rate: float | None = None
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class EvaluationsConfig(BaseModel):
    """
    Optional LLM-as-judge evaluation tier.

    Runs asynchronously on a sampled fraction of queries — never on the response
    path. Results are exported as OTel metrics/spans under the 'evaluation' stage.
    """

    enabled: bool = False
    sample_rate: float = 0.1
    max_concurrency: int = 4
    faithfulness: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    answer_relevance: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    answer_completeness: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    answer_coherence: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    context_diversity: EvaluatorConfig = Field(default_factory=EvaluatorConfig)
    custom: list[CustomEvaluatorConfig] = Field(default_factory=list)


class ObservabilityConfig(BaseModel):
    """
    Full OpenTelemetry observability config.

    NexRAG emits metrics, traces, and structured logs for every pipeline stage.
    Users can export to Prometheus (pull /metrics scrape) and/or any OTLP
    endpoint (push to Grafana Alloy, Jaeger, Loki, etc.).

    Data retention is fully managed by the user's backend — NexRAG stores nothing.

    Optional LLM-as-judge evaluations (faithfulness, relevance, etc.) run off the
    response path on a configurable sample of queries.
    """

    enabled: bool = True
    service_name: str = "nexrag"
    resource_attributes: dict[str, str] = Field(default_factory=dict)
    signals: OTelSignalsConfig = Field(default_factory=OTelSignalsConfig)
    exporters: ExportersConfig = Field(default_factory=ExportersConfig)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    evaluations: EvaluationsConfig = Field(default_factory=EvaluationsConfig)

    # Escape hatch: bring your own BaseObserver subclass
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


# Guardrails


class GuardConfig(BaseModel):
    type: Literal[
        "pii",
        "access_control",
        "prompt_injection",
        "groundedness",
        "topic",
        "model",
        "custom",
    ]
    enabled: bool = True
    # Nested LLM sub-config for the 'model' guard (Llama-Guard-style), resolved
    # independently of the pipeline LLM — same nesting pattern as chunker.llm.
    llm: LLMConfig | None = None
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def validate_guard(self) -> GuardConfig:
        if self.type == "custom" and not self.class_path:
            raise ValueError(
                "guard.class is required when guard.type is 'custom'. "
                "Provide a dotted class path: myproject.guards.MyGuard"
            )
        if self.type == "model" and self.llm is None and not self.class_path:
            raise ValueError(
                "guard.llm is required when guard.type is 'model' "
                "(the moderation model is resolved from a nested llm block)."
            )
        return self


class GuardChainConfig(BaseModel):
    enabled: bool = False
    # fail_open: a broken guard is treated as ALLOW. fail_closed: as BLOCK.
    policy: Literal["fail_open", "fail_closed"] = "fail_open"
    guards: list[GuardConfig] = Field(default_factory=list)


class GuardrailsConfig(BaseModel):
    """
    Four ordered guard chains, each bound to a pipeline phase:
        ingestion — runs on document text after the sanitizer, before chunking.
        input     — runs on the user query at the start of the query pipeline.
        retrieved — runs on each retrieved chunk before it enters the prompt.
        output    — runs on the LLM answer before it is returned/streamed.
    """

    ingestion: GuardChainConfig = Field(default_factory=GuardChainConfig)
    input: GuardChainConfig = Field(default_factory=GuardChainConfig)
    retrieved: GuardChainConfig = Field(default_factory=GuardChainConfig)
    output: GuardChainConfig = Field(default_factory=GuardChainConfig)


# Root config


class NexRAGConfig(BaseModel):
    """
    Root configuration object.
    Produced by config/loader.py and passed to the pipeline orchestrators.
    """

    version: str = "1.0"
    mode: Literal["sync", "async"] = "sync"
    ingestion: IngestionConfig
    query: QueryConfig
    guardrails: GuardrailsConfig = Field(default_factory=GuardrailsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
