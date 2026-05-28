# NexRAG V1 — Context & Task Tracker

> Internal development reference. Tracks what is built, what remains for V1, and what is deferred to V2.

---

## What NexRAG Is

A framework-agnostic RAG pipeline SDK. Core philosophy: **NexRAG owns the pipeline shape. Users own the components.**

Every stage is a clean ABC interface. NexRAG ships default implementations. Users swap any stage by implementing the interface and declaring it in YAML. No LangChain. No LlamaIndex. No hidden magic.

**Two pipelines:**
```
INGESTION  →  Loader → Sanitizer → Chunker → Embedder → VectorDB
QUERY      →  Embedder → Retriever → PromptBuilder → LLM → PipelineResult
```

**V1 target:** Working naive RAG — ingest a PDF, query it, get a structured result. Sync-only.

---

## Current State (what is built and tested)

| Component | Location | Status |
|---|---|---|
| All 9 pipeline stage interfaces (ABCs) | `core/interfaces/` | Done |
| Config schema (Pydantic v2) | `core/config/schema.py` | Done |
| YAML loader + `${ENV_VAR}` substitution | `core/config/loader.py` | Done |
| Dotted class path resolver + secret redaction | `core/config/resolver.py` | Done |
| `IngestionPipeline` (hash-based, idempotent) | `core/pipeline/ingestion.py` | Done |
| `QueryPipeline` (embed → retrieve → prompt → LLM) | `core/pipeline/query.py` | Done |
| `NexRAG` facade (`from_config`, `ingest`, `query`) | `__init__.py` | Done |
| `_factory.py` wiring module (SRP refactor) | `_factory.py` | Done |
| `RecursiveChunker` | `chunkers/recursive.py` | Done |
| `PDFLoader` | `loaders/pdf.py` | Done |
| `RawTextLoader` | `loaders/raw.py` | Done |
| `OpenAIEmbedder` | `adapters/embedders/openai.py` | Done |
| `ChromaDBAdapter` (memory + persistent) | `adapters/vector_dbs/chroma.py` | Done |
| `DenseRetriever` | `retrievers/dense.py` | Done |
| `DefaultPromptBuilder` | `defaults/prompt_builder.py` | Done |
| `OpenAILLM` | `adapters/llms/openai.py` | Done |
| `OllamaLLM` | `adapters/llms/ollama.py` | Done |
| `ConsoleObserver` | `observers/console.py` | Done |
| Unit tests (125) + integration tests (15) | `tests/` | Done |
| mypy strict + ruff clean | — | Done |

---

## V1 Remaining Tasks

### High priority (must ship with V1)

| Task | File(s) | Notes |
|---|---|---|
| `__repr__` masking for `EmbedderConfig` / `LLMConfig` | `core/config/schema.py` | Security: `api_key` must show `***` in repr, never the real value |
| `nexrag.yaml` example file in the repo | `nexrag.example.yaml` | Users need a working template to copy |
| Update README "Supported Providers" table | `README.md` | Currently lists HuggingFace, Word/Excel/JSON/HTML loaders that don't exist yet |

### Medium priority (rounds out V1)

| Task | File(s) | Notes |
|---|---|---|
| `OllamaEmbedder` | `adapters/embedders/ollama.py` | LLM side is done; embedder side missing |
| `HuggingFaceEmbedder` | `adapters/embedders/huggingface.py` | README promises this |
| `AnthropicLLM` | `adapters/llms/anthropic.py` | README promises this; wiring hook is already commented in `_factory.py` |
| `FixedChunker` | `chunkers/fixed.py` | Wiring hook is already commented in `_factory.py` |
| ChromaDB `HttpClient` mode | `adapters/vector_dbs/chroma.py` | Adds `mode: server` with `host` + `port` config; needed for self-hosted ChromaDB server |

**ChromaDB server mode** — the config shape when implemented:
```yaml
vector_db:
  provider: chroma
  collections:
    documents:
      mode: server
      host: chroma.internal
      port: 8000
```
`ChromaDBAdapter.__init__` would branch on `mode == "server"` and call `chromadb.HttpClient(host=host, port=port)`.

### Low priority (before v1.0 tag)

| Task | Notes |
|---|---|
| `CHANGELOG.md` | Required before PyPI publish |
| `docs/` architecture diagrams | README links to `docs/` — currently empty |
| PyPI publishing setup (twine / uv publish) | Needs classifiers, license, long_description in `pyproject.toml` |
| GitHub Actions CI (lint + test on push) | `.github/workflows/ci.yml` |
| Test coverage report | `uv run pytest --cov=nexrag tests/` |

---

## V2 Deferred (out of V1 scope)

These are explicitly not V1. The interfaces are already designed for them (no breaking changes needed).

| Feature | Why deferred |
|---|---|
| **Streaming** (`stream_query()`) | `BaseLLM.stream()` already exists; `QueryPipeline` just needs a streaming path |
| **Async** (`aembed`, `agenerate`) | Needs async pipeline runners; V1 is sync-only |
| **Hybrid retrieval** (BM25 + dense) | `BaseRetriever.retrieve()` already accepts both `query` and `query_embedding`; just needs a `HybridRetriever` impl |
| **Multi-collection routing** | Query two indexes, merge results |
| **Secrets backends** | AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager — V1 uses env vars only |
| **Observability backends** | OpenTelemetry, Langfuse, Prometheus — V1 ships console only |
| **More vector DBs** | Pinecone, Weaviate, Qdrant, pgvector |
| **More loaders** | Word (.docx), Excel, JSON, HTML, code files |
| **Modular pipeline** (configurable stage order) | V1 pipeline sequence is fixed |
| **Multi-modal** (text + image embedders) | Requires separate embedder instances per modality |

---

## Architecture Constraints (never violate)

1. **`core/` never imports from `adapters/`, `loaders/`, `chunkers/`** — only the other direction
2. **YAML is the only config mechanism** — no programmatic config dicts or env var magic
3. **Typed exceptions only** — all errors extend `NexRAGError`; no bare `raise Exception`
4. **Secrets never in logs** — resolver redacts sensitive param keys; observer never emits config

---

## How to run checks

```bash
uv run pytest tests/ -v                  # 140 tests, all must pass
uv run mypy src/nexrag                   # zero errors (strict mode)
uv run ruff check src/ tests/            # zero warnings
```
