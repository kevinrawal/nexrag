# Changelog

All notable changes to NexRAG will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.1] - 2026-06-10

### Added

- `stream_query()` and `astream_query()` now yield `RunMetrics` as the final stream item — callers access per-stage latencies (`embedder`, `retriever`, `reranker`, `prompt_builder`, `llm`) and `chunks_retrieved` without a separate non-streaming call; use `isinstance(item, RunMetrics)` to separate tokens from the metrics object
- `RunMetrics` exported from the top-level `nexrag` package (`from nexrag import NexRAG, RunMetrics`)
- BM25 index caching — corpus is built once per collection on first query and reused across subsequent queries; TTL configurable via `cache_ttl` on `BM25Retriever`; cache invalidated automatically after every `ingest()`, `ingest_documents()`, and `async_ingest()` call
- `BM25Retriever.invalidate_cache(collection=None)` — public API to flush cache for one collection or all
- Per-collection ChromaDB isolation — when collections declare different `(mode, path, host, port)` configs, each collection is routed to its own `ChromaDBAdapter` instance (backed by `_MultiChromaAdapter`); collections sharing the same config share one adapter instance
- `pyproject.toml` extras: `all-sparse` (`bm25`), `all-rerankers` (`cohere` + `cross-encoder`), `all-retrieval` (`all-sparse` + `all-rerankers`) for targeted installs without pulling PyTorch

### Fixed

- `stream_query()` and `astream_query()` return types corrected from `Iterator[str]` / `AsyncIterator[str]` to `Iterator[str | RunMetrics]` / `AsyncIterator[str | RunMetrics]`
- All pre-LLM pipeline stages (embed, retrieve, rerank, prompt_build) now timed in the streaming path, matching the non-streaming `query()` / `async_query()` behaviour

---

## [0.3.0] - 2026-05-31

### Added

**Async support**
- Async ABCs on all pipeline interfaces (`async_embed`, `async_generate`, `async_retrieve`, `async_upsert`, etc.) with `asyncio.to_thread` defaults
- `AsyncIngestionPipeline` and `AsyncQueryPipeline` — opt in via `mode: async` in config
- `NexRAG.async_query()`, `NexRAG.async_ingest()` facade methods

**Streaming**
- `QueryPipeline.stream()` — sync token-by-token LLM streaming
- `NexRAG.stream_query()` (sync) and `NexRAG.astream_query()` (async) on the public facade

**Hybrid & sparse retrieval**
- `BM25Retriever` — corpus-level keyword retrieval
- `HybridRetriever` — fused dense+sparse with configurable `alpha` weight
- `BaseSparseRetriever` interface; pluggable sparse retriever system with `SparseConfig`
- `provider: bm25` and `provider: hybrid` options in `RetrieverConfig`
- `BaseVectorDB.get_all()` / `async_get_all()` — required by BM25 corpus fetch

**Reranking**
- Optional post-retrieval reranking stage in both sync and async query pipelines
- `BaseReranker` ABC; `CohereReranker` and `CrossEncoderReranker` adapters
- `reranker:` block in `QueryConfig` (`provider`, `model`, `api_key`, `top_n`)

**Observability**
- Token usage (`prompt_tokens`, `completion_tokens`, `total_tokens`) in LLM `completed` events and `PipelineResult.token_usage`
- `RunMetrics` frozen dataclass — per-run latency breakdown by stage; attached to `PipelineResult.metrics` and `IngestionResult.metrics`
- `ConsoleObserver` prints per-stage latency summary and token count

**Multi-collection routing**
- `ingest()`, `ingest_batch()`, `ingest_documents()`, `async_ingest()` accept `collection: str | None`
- `IngestionResult.collection_used` field

**PDF metadata extraction**
- `PDFLoader` extracts up to 8 standard metadata fields (`author`, `title`, `subject`, `creator`, `producer`, `created_at`, `modified_at`, `page_count`)
- `LoaderConfig` gains `include_metadata` and `metadata_fields` fields

**Config**
- `VectorDBConfig` gains `upsert_batch_size`, `query_batch_size`, `max_retries`, `retry_delay`
- `NexRAGConfig` gains `mode: sync | async`
- `per-call metadata` param on all `ingest*()` methods

### Changed

- Loaders are now converter-only: `PDFLoader.load()` accepts `bytes`; `RawTextLoader.load()` accepts `str`. Passing a file path raises `LoaderError`.
- `AutoLoader` redesigned from file-extension dispatch to content-type dispatch
- `IngestionPipeline.ingest()` uses `doc.with_metadata()` (non-mutating) instead of `doc.metadata.update()`
- `ChunkerConfig.strategy` reduced from 5 literals to `recursive | custom` (unimplemented values removed)
- `bm25_top_k` config key renamed to `sparse_top_k`
- `StageName` gains `"reranker"` and `"pipeline"` entries

### Fixed

- `ChromaDBAdapter`: six bugs fixed — `None` metadata crash, missing struct fields (`chunk_index`, `total_chunks`, `parent_doc_id`) on round-trip, filter operator always `$eq`, missing batch upsert, connection retry, unimplemented `list_collections()`
- `Document.doc_id` stabilised across re-ingests via `sha256(source)[:32]` when `metadata["source"]` is set
- `AutoLoader` no longer injects `source="raw_text"` fallback on text documents
- `loader source` metadata made optional — absent source skips idempotency check
- `QueryPipeline.stream()` pre-LLM stages wrapped in outer `try/except`

---

## [0.2.0] - 2026-05-28

First public release. Full naive RAG pipeline — ingest a document, query it, get a structured result.

### Added

**Core pipeline**
- `IngestionPipeline`: hash-based idempotent ingestion (Loader → Sanitizer → Chunker → Embedder → VectorDB)
- `QueryPipeline`: embed → retrieve → prompt → LLM with structured `PipelineResult`
- `NexRAG` facade: `from_config()`, `ingest()`, `ingest_batch()`, `query()`
- `_factory.py`: SRP wiring module — builds and connects all components from config

**Config system**
- Pydantic v2 schema (`NexRAGConfig`) mapping 1:1 to `nexrag.yaml`
- YAML loader with `${ENV_VAR}` substitution — secrets never hardcoded
- Dynamic dotted class path resolver with secret-key redaction in error messages
- `nexrag.example.yaml` — copy-paste starting template

**Interfaces (ABCs)**
- `BaseLoader`, `BaseSanitizer`, `BaseChunker`, `BaseEmbedder`, `BaseVectorDB`
- `BaseRetriever`, `BasePromptBuilder`, `BaseLLM`, `BaseObserver`
- All 9 stages are swappable via `class:` in YAML

**Data models**
- `Document`, `Chunk` (with SHA-256 `content_hash`), `ScoredChunk`
- `PipelineResult`, `PipelineEvent`, typed exception hierarchy (`NexRAGError` subclasses)

**Loaders**
- `PDFLoader` — text extraction via `pypdf`
- `RawTextLoader` — plain `.txt` / `.md` files
- `AutoLoader` — auto-detects loader by file extension (`.pdf`, `.txt`, `.md`)

**Chunkers**
- `RecursiveChunker` — paragraph-aware splitting with configurable `chunk_size`, `chunk_overlap`, `min_chunk_size`

**Embedders**
- `OpenAIEmbedder` — batched embeddings via OpenAI API, configurable `batch_size` and `max_retries` with exponential backoff
- `OllamaEmbedder` — local embeddings via Ollama; one call per text (no batch endpoint)
- `HuggingFaceEmbedder` — batched embeddings via HuggingFace Inference API (`InferenceClient.feature_extraction`); supports Dedicated Endpoints via `base_url`

**LLMs**
- `OpenAILLM` — chat completions with streaming support and configurable `max_retries`
- `OllamaLLM` — local inference via Ollama with streaming
- `AnthropicLLM` — Claude models via Anthropic Messages API; handles `system=` separately; retry/backoff on rate limits

**Vector databases**
- `ChromaDBAdapter` — three modes: `memory` (ephemeral), `persistent` (disk), `server` (remote `HttpClient`)

**Retrievers**
- `DenseRetriever` — cosine similarity search with `top_k` and `score_threshold` filtering

**Prompt builder**
- `DefaultPromptBuilder` — numbered context blocks, configurable system prompt and context format

**Observability**
- `ConsoleObserver` — structured JSON or text event logging at configurable log level
- `NoOpObserver` — zero-overhead no-op for when observability is disabled

**Retry / resilience**
- Exponential backoff with jitter on `OpenAIEmbedder`, `OpenAILLM`, `AnthropicLLM`
- `max_retries` configurable per component in YAML (default 2; set 0 to disable)

**Tests**
- 213 unit tests — 100% pass, mypy strict clean, ruff clean
- All adapter tests use `sys.modules` injection — no optional packages required in CI

### Changed
- `LLMConfig`: removed dead `streaming: bool` field (never read by `QueryPipeline`)

---

## [Unreleased]

Nothing yet.

---

## Version History

| Version | Date | Description |
|---|---|---|
| 0.3.1 | 2026-06-10 | Streaming metrics, BM25 caching, per-collection ChromaDB isolation, targeted extras |
| 0.3.0 | 2026-05-31 | Async pipelines, streaming, hybrid/sparse retrieval, reranking, RunMetrics, multi-collection routing |
| 0.2.0 | 2026-05-28 | Public release — full naive RAG pipeline |
