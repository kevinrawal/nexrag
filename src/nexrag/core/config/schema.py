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

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# Ingestion sub-configs


class LoaderConfig(BaseModel):
    type: Literal["auto", "pdf", "txt", "excel", "json", "code", "word", "html", "custom"] = "auto"
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

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
    strategy: Literal["fixed", "sentence", "paragraph", "recursive", "custom"] = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50
    separator: str = "\n\n"
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def class_required_for_custom(self) -> ChunkerConfig:
        if self.strategy == "custom" and not self.class_path:
            raise ValueError(
                "chunker.class is required when chunker.strategy is 'custom'. "
                "Provide a dotted class path: myproject.chunkers.MyChunker"
            )
        return self


class EmbedderConfig(BaseModel):
    provider: Literal["openai", "huggingface", "ollama", "custom"]
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


class CollectionConfig(BaseModel):
    path: str | None = None  # local ChromaDB persist path
    host: str | None = None  # remote ChromaDB HTTP host
    port: int | None = None
    mode: Literal["memory", "persistent", "server"] = "persistent"
    description: str | None = None  # used by V2 router
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorDBConfig(BaseModel):
    provider: Literal["chroma", "custom"] = "chroma"
    default_collection: str
    collections: dict[str, CollectionConfig]
    on_conflict: Literal["overwrite", "skip", "append"] = "overwrite"
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def default_collection_must_exist(self) -> VectorDBConfig:
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


class RetrieverConfig(BaseModel):
    provider: Literal["dense", "custom"] = "dense"
    top_k: int = 5
    score_threshold: float = 0.0
    metadata_filter: dict[str, Any] | None = None
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


class LLMConfig(BaseModel):
    provider: Literal["openai", "anthropic", "ollama", "custom"]
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


class QueryConfig(BaseModel):
    embedder: EmbedderConfig | Literal["inherit"] = "inherit"
    retriever: RetrieverConfig = Field(default_factory=RetrieverConfig)
    prompt: PromptConfig = Field(default_factory=PromptConfig)
    llm: LLMConfig


# Observability


class ObservabilityConfig(BaseModel):
    enabled: bool = True
    observer: Literal["console", "custom"] = "console"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"
    class_path: str | None = Field(default=None, alias="class")
    params: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


# Root config


class NexRAGConfig(BaseModel):
    """
    Root configuration object.
    Produced by config/loader.py and passed to the pipeline orchestrators.
    """

    version: str = "1.0"
    ingestion: IngestionConfig
    query: QueryConfig
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
