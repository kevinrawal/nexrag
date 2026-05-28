# Changelog

All notable changes to NexRAG will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
| 0.2.0 | 2026-05-28 | Public release — full naive RAG pipeline |
