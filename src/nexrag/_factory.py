"""
Internal wiring module — builds and connects all pipeline components from config.

Not part of the public API. Import from nexrag, not from here.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from typing import Any, TypedDict

from nexrag.core.config.resolver import resolve_class
from nexrag.core.config.schema import (
    CacheConfig,
    ChunkerConfig,
    ContextStrategyConfig,
    EmbedderConfig,
    GuardChainConfig,
    GuardConfig,
    LLMConfig,
    LoaderConfig,
    NexRAGConfig,
    ObservabilityConfig,
    PromptConfig,
    RateLimitConfig,
    RerankerConfig,
    RetrieverConfig,
    SanitizerConfig,
    SessionConfig,
    SparseConfig,
    VectorDBConfig,
)
from nexrag.core.guards.chain import GuardChain
from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.interfaces.context_strategy import BaseContextStrategy
from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.core.interfaces.guard import BaseGuard
from nexrag.core.interfaces.llm import BaseLLM
from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.interfaces.observer import BaseObserver, NoOpObserver
from nexrag.core.interfaces.prompt_builder import BasePromptBuilder
from nexrag.core.interfaces.query_cache import BaseQueryCache
from nexrag.core.interfaces.rate_limiter import BaseRateLimiter
from nexrag.core.interfaces.reranker import BaseReranker
from nexrag.core.interfaces.retriever import BaseRetriever
from nexrag.core.interfaces.sanitizer import BaseSanitizer
from nexrag.core.interfaces.session_store import BaseSessionStore
from nexrag.core.interfaces.sparse_retriever import BaseSparseRetriever
from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.observability.runner import EvaluationRunner, NoOpEvaluationRunner
from nexrag.core.pipeline.async_ingestion import AsyncIngestionPipeline
from nexrag.core.pipeline.async_query import AsyncQueryPipeline
from nexrag.core.pipeline.ingestion import IngestionPipeline
from nexrag.core.pipeline.query import QueryPipeline
from nexrag.core.runtime import QueryRuntime
from nexrag.defaults.rate_limiter import TokenBucketRateLimiter
from nexrag.exceptions import ConfigError
from nexrag.loaders.auto import AutoLoader

_log = logging.getLogger("nexrag")


def wire(
    config: NexRAGConfig,
) -> tuple[
    IngestionPipeline | AsyncIngestionPipeline,
    QueryPipeline | AsyncQueryPipeline,
    BaseRetriever,
    EvaluationRunner | NoOpEvaluationRunner,
    QueryRuntime,
]:
    """
    Instantiate all components from config and wire them into pipelines.

    When config.mode is "async", returns AsyncIngestionPipeline + AsyncQueryPipeline.
    When config.mode is "sync" (default), returns the standard sync pipelines.

    Returns a (ingestion_pipeline, query_pipeline, retriever, eval_runner, runtime)
    tuple so the caller (NexRAG.from_config) can store retriever for cache
    invalidation, eval_runner for post-query sampling, and the QueryRuntime
    (cache / rate limiter / session store / context strategy) for the facade.
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

    # Guardrail chains (None when a chain is disabled or has no enabled guards).
    ingestion_guards = _build_guard_chain(config.guardrails.ingestion, observer, "ingestion")
    input_guards = _build_guard_chain(config.guardrails.input, observer, "input")
    retrieved_guards = _build_guard_chain(config.guardrails.retrieved, observer, "retrieved")
    output_guards = _build_guard_chain(config.guardrails.output, observer, "output")

    # Evaluation runner (off-path, optional).
    eval_runner = _build_evaluation_runner(config.observability, observer)

    # Facade-level query runtime (cache / rate limit / sessions). All optional.
    runtime = _build_query_runtime(
        config.query.cache, config.query.session, config.query.rate_limit
    )

    # Streaming + output guards are incompatible by design: output guards must see
    # the full answer, so streaming buffers it and emits one chunk (no live tokens).
    # Warn once at construction so this is never a silent surprise in production.
    if output_guards is not None:
        _log.warning(
            "Output guards are enabled: stream_query()/astream_query() will buffer "
            "the full response and emit it as a single chunk (output guards cannot "
            "edit an already-sent token stream). Use query()/async_query() for the "
            "same guaranteed guarding, or disable the output guard chain for true "
            "live token streaming."
        )

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
            ingestion_guards=ingestion_guards,
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
            input_guards=input_guards,
            retrieved_guards=retrieved_guards,
            output_guards=output_guards,
            evaluation_runner=eval_runner,
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
            ingestion_guards=ingestion_guards,
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
            input_guards=input_guards,
            retrieved_guards=retrieved_guards,
            output_guards=output_guards,
            evaluation_runner=eval_runner,
        )

    return ingestion_pipeline, query_pipeline, retriever, eval_runner, runtime


# Facade-level query runtime builders


def _build_query_runtime(
    cache_cfg: CacheConfig,
    session_cfg: SessionConfig,
    rate_cfg: RateLimitConfig,
) -> QueryRuntime:
    cache = _build_query_cache(cache_cfg) if cache_cfg.enabled else None
    rate_limiter = _build_rate_limiter(rate_cfg) if rate_cfg.enabled else None
    session_store = _build_session_store(session_cfg) if session_cfg.enabled else None
    context_strategy = (
        _build_context_strategy(session_cfg.context_strategy) if session_cfg.enabled else None
    )
    return QueryRuntime(
        cache=cache,
        rate_limiter=rate_limiter,
        session_store=session_store,
        context_strategy=context_strategy,
    )


def _build_rate_limiter(config: RateLimitConfig) -> BaseRateLimiter:
    if config.backend == "custom":
        return resolve_class(
            config.class_path,  # type: ignore[arg-type]
            BaseRateLimiter,  # type: ignore[type-abstract]
            config.params,
            stage="rate_limiter",
        )

    return TokenBucketRateLimiter(
        requests_per_minute=config.requests_per_minute, burst=config.burst
    )


def _build_query_cache(config: CacheConfig) -> BaseQueryCache:
    if config.backend == "custom":
        # Forward the top-level cache settings so a custom backend (e.g. a semantic
        # cache) receives them without the user duplicating each value into params.
        params = _forward_config_fields(
            config.class_path,
            config.params,
            {
                "similarity_threshold": config.similarity_threshold,
                "max_size": config.max_size,
                "ttl_seconds": config.ttl_seconds,
            },
        )
        return resolve_class(
            config.class_path,  # type: ignore[arg-type]
            BaseQueryCache,  # type: ignore[type-abstract]
            params,
            stage="query_cache",
        )

    from nexrag.defaults.query_cache import InMemoryQueryCache

    return InMemoryQueryCache(max_size=config.max_size, ttl_seconds=config.ttl_seconds)


def _forward_config_fields(
    class_path: str | None,
    base_params: dict[str, Any],
    forwardable: dict[str, Any],
) -> dict[str, Any]:
    """
    Merge top-level config values into ``base_params`` for a resolved custom class,
    but only the fields its ``__init__`` can actually accept.

    A field is forwarded only when the class declares a matching keyword parameter
    (or a ``**kwargs`` catch-all) AND the user did not already set it in ``params``
    (explicit params always win). This lets ``query.cache.similarity_threshold`` reach
    a semantic cache backend without duplication, while never breaking a backend whose
    constructor doesn't take those names. If the class can't be imported/introspected,
    ``base_params`` is returned unchanged so ``resolve_class`` surfaces the real error.
    """
    params = dict(base_params)
    if not class_path:
        return params
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        cls = getattr(importlib.import_module(module_path), class_name)
        signature = inspect.signature(cls)
    except Exception:
        return params
    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
    )
    declared = set(signature.parameters)
    for name, value in forwardable.items():
        if name not in params and (accepts_kwargs or name in declared):
            params[name] = value
    return params


def _build_session_store(config: SessionConfig) -> BaseSessionStore:
    if config.backend == "custom":
        return resolve_class(
            config.class_path,  # type: ignore[arg-type]
            BaseSessionStore,  # type: ignore[type-abstract]
            config.params,
            stage="session_store",
        )

    from nexrag.defaults.session_store import InMemorySessionStore

    return InMemorySessionStore(ttl_seconds=config.session_ttl_seconds, persist=config.persist)


def _build_context_strategy(config: ContextStrategyConfig) -> BaseContextStrategy:
    if config.type == "custom":
        return resolve_class(
            config.class_path,  # type: ignore[arg-type]
            BaseContextStrategy,  # type: ignore[type-abstract]
            config.params,
            stage="context_strategy",
        )

    if config.type == "window":
        from nexrag.defaults.context_strategy import WindowStrategy

        return WindowStrategy(max_turns=config.max_history_turns)

    if config.type == "token_budget":
        from nexrag.defaults.context_strategy import TokenBudgetStrategy

        return TokenBudgetStrategy(max_tokens=config.max_tokens)

    raise ConfigError(
        f"Unknown context strategy type: {config.type!r}. Supported: window, token_budget, custom",
        stage="config",
        component="context_strategy",
    )


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

    if config.provider == "gemini":
        from nexrag.adapters.embedders.gemini import GeminiEmbedder

        return GeminiEmbedder(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            batch_size=config.batch_size,
            max_retries=config.max_retries,
        )

    raise ConfigError(
        f"Unknown embedder provider: {config.provider!r}. "
        "Supported: openai, huggingface, gemini, ollama, custom",
        stage="config",
        component="embedder",
    )


def _build_chunker(config: ChunkerConfig) -> BaseChunker:
    # Resolve nested component sub-configs INDEPENDENTLY of the pipeline's main
    # embedder/LLM. semantic needs an embedder; proposition needs an LLM; custom
    # chunkers may declare either and receive it as a constructor kwarg.
    chunker_embedder = _build_embedder(config.embedder) if config.embedder is not None else None
    chunker_llm = _build_llm(config.llm) if config.llm is not None else None

    if config.strategy == "custom":
        if not config.class_path:
            raise ConfigError(
                "chunker.class is required when chunker.strategy is 'custom'.",
                stage="config",
                component="chunker",
            )
        params = dict(config.params)
        if chunker_embedder is not None:
            params["embedder"] = chunker_embedder
        if chunker_llm is not None:
            params["llm"] = chunker_llm
        return resolve_class(config.class_path, BaseChunker, params, stage="chunker")  # type: ignore[type-abstract]

    if config.strategy == "recursive":
        from nexrag.chunkers.recursive import RecursiveChunker

        return RecursiveChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_size=config.min_chunk_size,
        )

    if config.strategy == "fixed":
        from nexrag.chunkers.fixed import FixedChunker

        return FixedChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_size=config.min_chunk_size,
        )

    if config.strategy == "token":
        from nexrag.chunkers.token import TokenChunker

        return TokenChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_size=config.min_chunk_size,
            **config.params,
        )

    if config.strategy == "sentence":
        from nexrag.chunkers.sentence import SentenceChunker

        return SentenceChunker(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_size=config.min_chunk_size,
            **config.params,
        )

    if config.strategy == "sentence_window":
        from nexrag.chunkers.sentence import SentenceWindowChunker

        return SentenceWindowChunker(min_chunk_size=config.min_chunk_size, **config.params)

    if config.strategy == "markdown":
        from nexrag.chunkers.markdown import MarkdownChunker

        return MarkdownChunker(
            chunk_size=config.chunk_size, min_chunk_size=config.min_chunk_size, **config.params
        )

    if config.strategy == "code":
        from nexrag.chunkers.code import CodeChunker

        return CodeChunker(
            chunk_size=config.chunk_size, min_chunk_size=config.min_chunk_size, **config.params
        )

    if config.strategy == "semantic":
        from nexrag.chunkers.semantic import SemanticChunker

        if chunker_embedder is None:
            raise ConfigError(
                "chunker.embedder is required when chunker.strategy is 'semantic'.",
                stage="config",
                component="chunker",
            )
        return SemanticChunker(
            embedder=chunker_embedder, min_chunk_size=config.min_chunk_size, **config.params
        )

    if config.strategy == "proposition":
        from nexrag.chunkers.proposition import PropositionChunker

        if chunker_llm is None:
            raise ConfigError(
                "chunker.llm is required when chunker.strategy is 'proposition'.",
                stage="config",
                component="chunker",
            )
        return PropositionChunker(
            llm=chunker_llm, min_chunk_size=config.min_chunk_size, **config.params
        )

    raise ConfigError(
        f"Unknown chunker strategy: {config.strategy!r}. Supported: recursive, fixed, token, "
        "sentence, sentence_window, markdown, code, semantic, proposition, custom",
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
        f"Unknown loader type: {config.type!r}. Supported: auto, pdf, txt, custom",
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

    if config.provider == "pinecone":
        from nexrag.adapters.vector_dbs.pinecone import PineconeVectorDB

        params = config.params
        index_name = params.get("index_name") or params.get("index")
        if not index_name:
            raise ConfigError(
                "vector_db.params.index_name is required for the 'pinecone' provider.",
                stage="config",
                component="vector_db",
            )
        return PineconeVectorDB(
            index_name=index_name,
            api_key=params.get("api_key"),
            cloud=params.get("cloud", "aws"),
            region=params.get("region", "us-east-1"),
            metric=params.get("metric", "cosine"),
            upsert_batch_size=config.upsert_batch_size,
            query_batch_size=config.query_batch_size,
            max_retries=config.max_retries,
            retry_delay=config.retry_delay,
        )

    raise ConfigError(
        f"Unknown vector_db provider: {config.provider!r}. Supported: chroma, pinecone, custom",
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

    if config.provider == "gemini":
        from nexrag.adapters.llms.gemini import GeminiLLM

        return GeminiLLM(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    if config.provider == "groq":
        from nexrag.adapters.llms.groq import GroqLLM

        return GroqLLM(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout,
            max_retries=config.max_retries,
        )

    raise ConfigError(
        f"Unknown LLM provider: {config.provider!r}. "
        "Supported: openai, anthropic, gemini, groq, ollama, custom",
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

    # Custom observer escape hatch — takes priority over OTel when class_path is set.
    if config.class_path:
        return resolve_class(
            config.class_path,
            BaseObserver,  # type: ignore[type-abstract]
            config.params,
            stage="observer",
        )

    # Default: OpenTelemetry observer.
    from nexrag.core.observability.pricing import build_pricing_table
    from nexrag.observers.otel import OpenTelemetryObserver

    pricing_table = build_pricing_table(dict(config.metrics.pricing))
    return OpenTelemetryObserver(config=config, pricing_table=pricing_table)


def _build_evaluation_runner(
    config: ObservabilityConfig,
    observer: BaseObserver,
) -> EvaluationRunner | NoOpEvaluationRunner:
    if not config.enabled or not config.evaluations.enabled:
        return NoOpEvaluationRunner()

    ev_cfg = config.evaluations
    evaluators: list[tuple[object, float]] = []

    def _rate(per_metric: float | None) -> float:
        return per_metric if per_metric is not None else ev_cfg.sample_rate

    if ev_cfg.faithfulness.enabled and ev_cfg.faithfulness.llm is not None:
        from nexrag.evaluators.faithfulness import FaithfulnessEvaluator

        llm = _build_llm(ev_cfg.faithfulness.llm)
        evaluators.append(
            (
                FaithfulnessEvaluator(llm=llm, **ev_cfg.faithfulness.params),
                _rate(ev_cfg.faithfulness.sample_rate),
            )
        )

    if ev_cfg.answer_relevance.enabled and ev_cfg.answer_relevance.llm is not None:
        from nexrag.evaluators.answer_relevance import AnswerRelevanceEvaluator

        llm = _build_llm(ev_cfg.answer_relevance.llm)
        embedder = (
            _build_embedder(ev_cfg.answer_relevance.embedder)
            if ev_cfg.answer_relevance.embedder is not None
            else None
        )
        evaluators.append(
            (
                AnswerRelevanceEvaluator(
                    llm=llm, embedder=embedder, **ev_cfg.answer_relevance.params
                ),
                _rate(ev_cfg.answer_relevance.sample_rate),
            )
        )

    if ev_cfg.answer_completeness.enabled and ev_cfg.answer_completeness.llm is not None:
        from nexrag.evaluators.answer_completeness import AnswerCompletenessEvaluator

        llm = _build_llm(ev_cfg.answer_completeness.llm)
        evaluators.append(
            (
                AnswerCompletenessEvaluator(llm=llm, **ev_cfg.answer_completeness.params),
                _rate(ev_cfg.answer_completeness.sample_rate),
            )
        )

    if ev_cfg.answer_coherence.enabled and ev_cfg.answer_coherence.llm is not None:
        from nexrag.evaluators.answer_coherence import AnswerCoherenceEvaluator

        llm = _build_llm(ev_cfg.answer_coherence.llm)
        evaluators.append(
            (
                AnswerCoherenceEvaluator(llm=llm, **ev_cfg.answer_coherence.params),
                _rate(ev_cfg.answer_coherence.sample_rate),
            )
        )

    if ev_cfg.context_diversity.enabled and ev_cfg.context_diversity.embedder is not None:
        from nexrag.evaluators.context_diversity import ContextDiversityEvaluator

        embedder = _build_embedder(ev_cfg.context_diversity.embedder)
        evaluators.append(
            (
                ContextDiversityEvaluator(embedder=embedder, **ev_cfg.context_diversity.params),
                _rate(ev_cfg.context_diversity.sample_rate),
            )
        )

    for custom_cfg in ev_cfg.custom:
        if not custom_cfg.enabled:
            continue
        from nexrag.core.interfaces.evaluator import BaseEvaluator

        ev = resolve_class(
            custom_cfg.class_path,
            BaseEvaluator,  # type: ignore[type-abstract]
            custom_cfg.params,
            stage="evaluator",
        )
        evaluators.append((ev, _rate(custom_cfg.sample_rate)))

    return EvaluationRunner(
        evaluators=evaluators,  # type: ignore[arg-type]
        global_sample_rate=ev_cfg.sample_rate,
        max_concurrency=ev_cfg.max_concurrency,
        observer=observer,
    )


# Guardrails


def _build_guard(config: GuardConfig) -> BaseGuard:
    if config.type == "custom":
        if not config.class_path:
            raise ConfigError(
                "guard.class is required when guard.type is 'custom'.",
                stage="config",
                component="guard",
            )
        params = dict(config.params)
        if config.llm is not None:
            params["llm"] = _build_llm(config.llm)
        return resolve_class(config.class_path, BaseGuard, params, stage="guard")  # type: ignore[type-abstract]

    if config.type == "pii":
        from nexrag.guards.pii import PIIGuard

        return PIIGuard(**config.params)

    if config.type == "access_control":
        from nexrag.guards.access_control import AccessControlGuard

        return AccessControlGuard(**config.params)

    if config.type == "prompt_injection":
        from nexrag.guards.prompt_injection import PromptInjectionGuard

        return PromptInjectionGuard(**config.params)

    if config.type == "groundedness":
        from nexrag.guards.groundedness import CitationGuard

        return CitationGuard(**config.params)

    if config.type == "topic":
        from nexrag.guards.topic import TopicGuard

        return TopicGuard(**config.params)

    if config.type == "model":
        from nexrag.guards.model_guard import ModelGuard

        if config.llm is None:
            raise ConfigError(
                "guard.llm is required when guard.type is 'model'.",
                stage="config",
                component="guard",
            )
        return ModelGuard(llm=_build_llm(config.llm), **config.params)

    raise ConfigError(
        f"Unknown guard type: {config.type!r}. Supported: pii, access_control, "
        "prompt_injection, groundedness, topic, model, custom",
        stage="config",
        component="guard",
    )


def _build_guard_chain(
    config: GuardChainConfig, observer: BaseObserver, name: str
) -> GuardChain | None:
    """Build a GuardChain, or None when the chain is disabled or has no enabled guards."""
    if not config.enabled:
        return None
    guards = [_build_guard(g) for g in config.guards if g.enabled]
    if not guards:
        return None
    return GuardChain(guards, policy=config.policy, observer=observer, name=name)
