"""
Internal wiring module — builds and connects all pipeline components from config.

Not part of the public API. Import from nexrag, not from here.
"""

from __future__ import annotations

from nexrag.core.config.resolver import resolve_class
from nexrag.core.config.schema import (
    ChunkerConfig,
    EmbedderConfig,
    LLMConfig,
    LoaderConfig,
    NexRAGConfig,
    ObservabilityConfig,
    PromptConfig,
    RetrieverConfig,
    SanitizerConfig,
    VectorDBConfig,
)
from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.core.interfaces.llm import BaseLLM
from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.interfaces.observer import BaseObserver, NoOpObserver
from nexrag.core.interfaces.prompt_builder import BasePromptBuilder
from nexrag.core.interfaces.retriever import BaseRetriever
from nexrag.core.interfaces.sanitizer import BaseSanitizer
from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.pipeline.ingestion import IngestionPipeline, IngestionResult  # noqa: F401
from nexrag.core.pipeline.query import QueryPipeline
from nexrag.exceptions import ConfigError


def wire(config: NexRAGConfig) -> tuple[IngestionPipeline, QueryPipeline]:
    """
    Instantiate all components from config and wire them into both pipelines.

    Returns a (ingestion_pipeline, query_pipeline) tuple so the caller (NexRAG.from_config)
    can construct the NexRAG instance without creating a circular import.
    """
    observer = _build_observer(config.observability)

    # Ingestion components
    embedder = _build_embedder(config.ingestion.embedder)
    chunker = _build_chunker(config.ingestion.chunker)
    loader = _build_loader(config.ingestion.loader)
    sanitizer = _build_sanitizer(config.ingestion.sanitizer)
    vector_db = _build_vector_db(config.ingestion.vector_db)
    collection = config.ingestion.vector_db.default_collection

    ingestion_pipeline = IngestionPipeline(
        chunker=chunker,
        embedder=embedder,
        vector_db=vector_db,
        collection=collection,
        loader=loader,
        sanitizer=sanitizer,
        on_conflict=config.ingestion.vector_db.on_conflict,
        observer=observer,
    )

    # Query components
    query_embedder: BaseEmbedder
    if config.query.embedder == "inherit":
        query_embedder = embedder
    else:
        query_embedder = _build_embedder(config.query.embedder)

    retriever = _build_retriever(config.query.retriever, vector_db)
    prompt_builder = _build_prompt_builder(config.query.prompt)
    llm = _build_llm(config.query.llm)

    query_pipeline = QueryPipeline(
        embedder=query_embedder,
        retriever=retriever,
        prompt_builder=prompt_builder,
        llm=llm,
        collection=collection,
        top_k=config.query.retriever.top_k,
        score_threshold=config.query.retriever.score_threshold,
        observer=observer,
    )

    return ingestion_pipeline, query_pipeline


# Component builders


def _build_embedder(config: EmbedderConfig) -> BaseEmbedder:
    if config.provider == "custom":
        if not config.class_path:
            raise ConfigError(
                "embedder.class is required when embedder.provider is 'custom'.",
                stage="config",
                component="embedder",
            )
        return resolve_class(config.class_path, BaseEmbedder, config.params, stage="embedder")  # type: ignore[type-abstract]

    if config.provider == "openai":
        from nexrag.adapters.embedders.openai import OpenAIEmbedder

        return OpenAIEmbedder(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            batch_size=config.batch_size,
        )

    # TODO: yet to implement in adapters/embedders
    # if config.provider == "ollama":
    #     from nexrag.adapters.embedders.ollama import (
    #         OllamaEmbedder,  # type: ignore[import-not-found]
    #     )
    #     return OllamaEmbedder(model=config.model, base_url=config.base_url, **config.params)  # type: ignore[no-any-return]

    # TODO: yet to implement in adapters/embedders
    # if config.provider == "huggingface":
    #     from nexrag.adapters.embedders.huggingface import (
    #         HuggingFaceEmbedder,  # type: ignore[import-not-found]
    #     )
    #     return HuggingFaceEmbedder(model=config.model, **config.params)  # type: ignore[no-any-return]

    raise ConfigError(
        f"Unknown embedder provider: {config.provider!r}. " f"Supported: openai, custom",
        stage="config",
        component="embedder",
    )


def _build_chunker(config: ChunkerConfig) -> BaseChunker:
    if config.strategy == "custom":
        if not config.class_path:
            raise ConfigError(
                "chunker.class is required when chunker.strategy is 'custom'.",
                stage="config",
                component="chunker",
            )
        return resolve_class(config.class_path, BaseChunker, config.params, stage="chunker")  # type: ignore[type-abstract]

    if config.strategy == "recursive":
        from nexrag.chunkers.recursive import RecursiveChunker

        return RecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_size=config.min_chunk_size,
        )

    # TODO: yet to implement in chunkers/
    # if config.strategy == "fixed":
    #     from nexrag.chunkers.fixed import FixedChunker  # type: ignore[attr-defined]
    #     return FixedChunker(  # type: ignore[no-any-return]
    #         chunk_size=config.chunk_size,
    #         chunk_overlap=config.chunk_overlap,
    #         **config.params,
    #     )

    raise ConfigError(
        f"Unknown chunker strategy: {config.strategy!r}. " f"Supported: recursive, custom",
        stage="config",
        component="chunker",
    )


def _build_loader(config: LoaderConfig) -> BaseLoader | None:
    if config.type == "auto":
        return None

    if config.type == "custom":
        if not config.class_path:
            raise ConfigError(
                "loader.class is required when loader.type is 'custom'.",
                stage="config",
                component="loader",
            )
        return resolve_class(config.class_path, BaseLoader, config.params, stage="loader")  # type: ignore[type-abstract]

    if config.type == "pdf":
        from nexrag.loaders.pdf import PDFLoader

        return PDFLoader(**config.params)

    if config.type in ("txt", "text"):
        from nexrag.loaders.raw import RawTextLoader

        return RawTextLoader(**config.params)

    raise ConfigError(
        f"Unknown loader type: {config.type!r}. " f"Supported: auto, pdf, txt, custom",
        stage="config",
        component="loader",
    )


def _build_sanitizer(config: SanitizerConfig) -> BaseSanitizer | None:
    if not config.enabled:
        return None
    return resolve_class(
        config.class_path,  # type: ignore[arg-type]
        BaseSanitizer,  # type: ignore[type-abstract]
        config.params,
        stage="sanitizer",
    )


def _build_vector_db(config: VectorDBConfig) -> BaseVectorDB:
    if config.provider == "custom":
        if not config.class_path:
            raise ConfigError(
                "vector_db.class is required when vector_db.provider is 'custom'.",
                stage="config",
                component="vector_db",
            )
        return resolve_class(config.class_path, BaseVectorDB, config.params, stage="vector_db")  # type: ignore[type-abstract]

    if config.provider == "chroma":
        from nexrag.adapters.vector_dbs.chroma import ChromaDBAdapter

        coll_cfg = config.collections[config.default_collection]
        return ChromaDBAdapter(
            path=coll_cfg.path,
            mode=coll_cfg.mode,
        )

    raise ConfigError(
        f"Unknown vector_db provider: {config.provider!r}. Supported: chroma, custom",
        stage="config",
        component="vector_db",
    )


def _build_retriever(config: RetrieverConfig, vector_db: BaseVectorDB) -> BaseRetriever:
    if config.provider == "custom":
        if not config.class_path:
            raise ConfigError(
                "retriever.class is required when retriever.provider is 'custom'.",
                stage="config",
                component="retriever",
            )
        return resolve_class(config.class_path, BaseRetriever, config.params, stage="retriever")  # type: ignore[type-abstract]

    if config.provider == "dense":
        from nexrag.retrievers.dense import DenseRetriever

        return DenseRetriever(vector_db=vector_db)

    raise ConfigError(
        f"Unknown retriever provider: {config.provider!r}. Supported: dense, custom",
        stage="config",
        component="retriever",
    )


def _build_prompt_builder(config: PromptConfig) -> BasePromptBuilder:
    if config.class_path:
        return resolve_class(
            config.class_path,
            BasePromptBuilder,
            config.params,
            stage="prompt_builder",  # type: ignore[type-abstract]
        )

    from nexrag.defaults.prompt_builder import DefaultPromptBuilder

    return DefaultPromptBuilder(
        system=config.system,
        context_format=config.context_format,
    )


def _build_llm(config: LLMConfig) -> BaseLLM:
    if config.provider == "custom":
        if not config.class_path:
            raise ConfigError(
                "llm.class is required when llm.provider is 'custom'.",
                stage="config",
                component="llm",
            )
        return resolve_class(config.class_path, BaseLLM, config.params, stage="llm")  # type: ignore[type-abstract]

    if config.provider == "openai":
        from nexrag.adapters.llms.openai import OpenAILLM

        return OpenAILLM(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )

    if config.provider == "ollama":
        from nexrag.adapters.llms.ollama import OllamaLLM

        return OllamaLLM(
            model=config.model,
            base_url=config.base_url or "http://localhost:11434",
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
        )

    # TODO: yet to implement in adapters/llms
    # if config.provider == "anthropic":
    #     from nexrag.adapters.llms.anthropic import AnthropicLLM  # type: ignore[import-not-found]
    #     return AnthropicLLM(  # type: ignore[no-any-return]
    #         model=config.model,
    #         api_key=config.api_key,
    #         temperature=config.temperature,
    #         max_tokens=config.max_tokens,
    #         **config.params,
    #     )

    raise ConfigError(
        f"Unknown LLM provider: {config.provider!r}. " f"Supported: openai, ollama, custom",
        stage="config",
        component="llm",
    )


def _build_observer(config: ObservabilityConfig) -> BaseObserver:
    if not config.enabled:
        return NoOpObserver()

    if config.observer == "console":
        from nexrag.observers.console import ConsoleObserver

        return ConsoleObserver(log_level=config.log_level, format=config.format)

    if config.observer == "custom":
        if not config.class_path:
            raise ConfigError(
                "observability.class is required when observability.observer is 'custom'.",
                stage="config",
                component="observer",
            )
        return resolve_class(
            config.class_path,
            BaseObserver,
            config.params,
            stage="observer",  # type: ignore[type-abstract]
        )

    raise ConfigError(
        f"Unknown observer: {config.observer!r}. Supported: console, custom",
        stage="config",
        component="observer",
    )
