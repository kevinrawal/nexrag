"""
Internal wiring module — builds and connects all pipeline components from config.

Not part of the public API. Import from nexrag, not from here.
"""

from __future__ import annotations

from typing import TypedDict

from nexrag.core.config.resolver import resolve_class
from nexrag.core.config.schema import (
    ChunkerConfig,
    EmbedderConfig,
    LLMConfig,
    LoaderConfig,
    NexRAGConfig,
    ObservabilityConfig,
    PromptConfig,
    RerankerConfig,
    RetrieverConfig,
    SanitizerConfig,
    SparseConfig,
    VectorDBConfig,
)
from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.core.interfaces.llm import BaseLLM
from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.interfaces.observer import BaseObserver, NoOpObserver
from nexrag.core.interfaces.prompt_builder import BasePromptBuilder
from nexrag.core.interfaces.reranker import BaseReranker
from nexrag.core.interfaces.retriever import BaseRetriever
from nexrag.core.interfaces.sanitizer import BaseSanitizer
from nexrag.core.interfaces.sparse_retriever import BaseSparseRetriever
from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.pipeline.async_ingestion import AsyncIngestionPipeline
from nexrag.core.pipeline.async_query import AsyncQueryPipeline
from nexrag.core.pipeline.ingestion import IngestionPipeline
from nexrag.core.pipeline.query import QueryPipeline
from nexrag.exceptions import ConfigError
from nexrag.loaders.auto import AutoLoader


def wire(
    config: NexRAGConfig,
) -> tuple[
    IngestionPipeline | AsyncIngestionPipeline,
    QueryPipeline | AsyncQueryPipeline,
    BaseRetriever,
]:
    """
    Instantiate all components from config and wire them into pipelines.

    When config.mode is "async", returns AsyncIngestionPipeline + AsyncQueryPipeline.
    When config.mode is "sync" (default), returns the standard sync pipelines.

    Returns a (ingestion_pipeline, query_pipeline, retriever) tuple so the caller
    (NexRAG.from_config) can store the retriever for post-ingest cache invalidation.
    """
    observer = _build_observer(config.observability)

    # Ingestion components
    embedder = _build_embedder(config.ingestion.embedder)
    chunker = _build_chunker(config.ingestion.chunker)
    loader = _build_loader(config.ingestion.loader)
    sanitizer = _build_sanitizer(config.ingestion.sanitizer)
    vector_db = _build_vector_db(config.ingestion.vector_db)
    collection = config.ingestion.vector_db.default_collection

    # Query components
    query_embedder: BaseEmbedder
    if config.query.embedder == "inherit":
        query_embedder = embedder
    else:
        query_embedder = _build_embedder(config.query.embedder)

    retriever = _build_retriever(config.query.retriever, vector_db)
    reranker = _build_reranker(config.query.reranker) if config.query.reranker is not None else None
    prompt_builder = _build_prompt_builder(config.query.prompt)
    llm = _build_llm(config.query.llm)
    valid_collections = frozenset(config.ingestion.vector_db.collections.keys())

    ingestion_pipeline: IngestionPipeline | AsyncIngestionPipeline
    query_pipeline: QueryPipeline | AsyncQueryPipeline

    if config.mode == "async":
        ingestion_pipeline = AsyncIngestionPipeline(
            chunker=chunker,
            embedder=embedder,
            vector_db=vector_db,
            collection=collection,
            loader=loader,
            sanitizer=sanitizer,
            on_conflict=config.ingestion.vector_db.on_conflict,
            observer=observer,
            embed_batch_size=config.ingestion.embedder.batch_size,
            valid_collections=valid_collections,
        )
        query_pipeline = AsyncQueryPipeline(
            embedder=query_embedder,
            retriever=retriever,
            prompt_builder=prompt_builder,
            llm=llm,
            collection=collection,
            top_k=config.query.retriever.top_k,
            score_threshold=config.query.retriever.score_threshold,
            observer=observer,
            reranker=reranker,
            max_query_length=config.query.max_query_length,
        )
    else:
        ingestion_pipeline = IngestionPipeline(
            chunker=chunker,
            embedder=embedder,
            vector_db=vector_db,
            collection=collection,
            loader=loader,
            sanitizer=sanitizer,
            on_conflict=config.ingestion.vector_db.on_conflict,
            observer=observer,
            valid_collections=valid_collections,
        )
        query_pipeline = QueryPipeline(
            embedder=query_embedder,
            retriever=retriever,
            prompt_builder=prompt_builder,
            llm=llm,
            collection=collection,
            top_k=config.query.retriever.top_k,
            score_threshold=config.query.retriever.score_threshold,
            observer=observer,
            reranker=reranker,
            max_query_length=config.query.max_query_length,
        )

    return ingestion_pipeline, query_pipeline, retriever


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
            max_retries=config.max_retries,
        )

    if config.provider == "ollama":
        from nexrag.adapters.embedders.ollama import OllamaEmbedder

        return OllamaEmbedder(
            model=config.model,
            base_url=config.base_url or "http://localhost:11434",
            batch_size=config.batch_size,
        )

    if config.provider == "huggingface":
        from nexrag.adapters.embedders.huggingface import HuggingFaceEmbedder

        return HuggingFaceEmbedder(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            batch_size=config.batch_size,
        )

    raise ConfigError(
        f"Unknown embedder provider: {config.provider!r}. Supported: openai, huggingface, ollama, custom",
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
        return AutoLoader()

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

        return PDFLoader(
            metadata_fields=config.metadata_fields,
            include_metadata=config.include_metadata,
            **config.params,
        )

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
        from nexrag.adapters.vector_dbs.chroma import ChromaDBAdapter, _MultiChromaAdapter

        class _ChromaShared(TypedDict):
            upsert_batch_size: int
            query_batch_size: int
            max_retries: int
            retry_delay: float

        shared: _ChromaShared = dict(
            upsert_batch_size=config.upsert_batch_size,
            query_batch_size=config.query_batch_size,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
        )

        def _make(cfg: object) -> ChromaDBAdapter:
            from nexrag.core.config.schema import CollectionConfig as _CC

            assert isinstance(cfg, _CC)
            return ChromaDBAdapter(
                mode=cfg.mode, path=cfg.path, host=cfg.host, port=cfg.port, **shared
            )

        def _key(cfg: object) -> tuple[str, str | None, str | None, int | None]:
            from nexrag.core.config.schema import CollectionConfig as _CC

            assert isinstance(cfg, _CC)
            return (cfg.mode, cfg.path, cfg.host, cfg.port)

        default_cfg = config.collections[config.default_collection]
        default_key = _key(default_cfg)
        default_adapter = _make(default_cfg)

        # Build per-collection adapters for collections whose config differs from the default.
        # Reuse the same adapter instance for collections that share (mode, path, host, port).
        key_to_adapter: dict[tuple[str, str | None, str | None, int | None], ChromaDBAdapter] = {}
        collection_adapters: dict[str, ChromaDBAdapter] = {}
        for name, coll_cfg in config.collections.items():
            k = _key(coll_cfg)
            if k == default_key:
                continue
            if k not in key_to_adapter:
                key_to_adapter[k] = _make(coll_cfg)
            collection_adapters[name] = key_to_adapter[k]

        if not collection_adapters:
            return default_adapter

        return _MultiChromaAdapter(
            default_adapter=default_adapter, collection_adapters=collection_adapters
        )

    raise ConfigError(
        f"Unknown vector_db provider: {config.provider!r}. Supported: chroma, custom",
        stage="config",
        component="vector_db",
    )


def _build_sparse_retriever(config: SparseConfig, vector_db: BaseVectorDB) -> BaseSparseRetriever:
    if config.provider == "custom":
        if not config.class_path:
            raise ConfigError(
                "retriever.sparse.class is required when retriever.sparse.provider is 'custom'.",
                stage="config",
                component="sparse_retriever",
            )
        return resolve_class(
            config.class_path,
            BaseSparseRetriever,  # type: ignore[type-abstract]
            config.params,
            stage="sparse_retriever",
        )

    if config.provider == "bm25":
        from nexrag.retrievers.sparse.bm25 import BM25Retriever

        return BM25Retriever(vector_db=vector_db)

    raise ConfigError(
        f"Unknown sparse retriever provider: {config.provider!r}. Supported: bm25, custom",
        stage="config",
        component="sparse_retriever",
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

    if config.provider == "bm25":
        from nexrag.retrievers.sparse.bm25 import BM25Retriever

        return BM25Retriever(vector_db=vector_db)

    if config.provider == "hybrid":
        from nexrag.retrievers.hybrid import HybridRetriever

        sparse = _build_sparse_retriever(config.sparse, vector_db)
        return HybridRetriever(
            vector_db=vector_db,
            alpha=config.alpha,
            sparse_top_k=config.sparse_top_k,
            sparse=sparse,
        )

    raise ConfigError(
        f"Unknown retriever provider: {config.provider!r}. Supported: dense, hybrid, bm25, custom",
        stage="config",
        component="retriever",
    )


def _build_prompt_builder(config: PromptConfig) -> BasePromptBuilder:
    if config.class_path:
        return resolve_class(
            config.class_path,
            BasePromptBuilder,  # type: ignore[type-abstract]
            config.params,
            stage="prompt_builder",
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
            max_retries=config.max_retries,
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

    if config.provider == "anthropic":
        from nexrag.adapters.llms.anthropic import AnthropicLLM

        return AnthropicLLM(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    raise ConfigError(
        f"Unknown LLM provider: {config.provider!r}. Supported: openai, ollama, anthropic, custom",
        stage="config",
        component="llm",
    )


def _build_reranker(config: RerankerConfig) -> BaseReranker:
    if config.provider == "custom":
        if not config.class_path:
            raise ConfigError(
                "reranker.class is required when reranker.provider is 'custom'.",
                stage="config",
                component="reranker",
            )
        return resolve_class(config.class_path, BaseReranker, config.params, stage="reranker")  # type: ignore[type-abstract]

    if config.provider == "cohere":
        from nexrag.adapters.rerankers.cohere import CohereReranker

        return CohereReranker(
            model=config.model,
            api_key=config.api_key,
            top_n=config.top_n,
        )

    if config.provider == "cross_encoder":
        from nexrag.adapters.rerankers.cross_encoder import CrossEncoderReranker

        return CrossEncoderReranker(
            model=config.model,
            top_n=config.top_n,
            device=config.device,
        )

    raise ConfigError(
        f"Unknown reranker provider: {config.provider!r}. Supported: cohere, cross_encoder, custom",
        stage="config",
        component="reranker",
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
            BaseObserver,  # type: ignore[type-abstract]
            config.params,
            stage="observer",
        )

    raise ConfigError(
        f"Unknown observer: {config.observer!r}. Supported: console, custom",
        stage="config",
        component="observer",
    )
