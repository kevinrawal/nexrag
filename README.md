# NexRAG

```text
███╗   ██╗███████╗██╗  ██╗██████╗  █████╗  ██████╗
████╗  ██║██╔════╝╚██╗██╔╝██╔══██╗██╔══██╗██╔════╝
██╔██╗ ██║█████╗   ╚███╔╝ ██████╔╝███████║██║  ███╗
██║╚██╗██║██╔══╝   ██╔██╗ ██╔══██╗██╔══██║██║   ██║
██║ ╚████║███████╗██╔╝ ██╗██║  ██║██║  ██║╚██████╔╝
╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝

●plug ⇄swap ▶scale
```

> Framework-agnostic RAG pipeline SDK. Plug in any component, swap any stage, configure everything in YAML.

[![PyPI version](https://img.shields.io/pypi/v/nexrag.svg)](https://pypi.org/project/nexrag/)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

---

## What is NexRAG?

NexRAG is a production-grade RAG (Retrieval-Augmented Generation) pipeline SDK for Python.

**NexRAG owns the pipeline shape. You own the components.**

Every stage — loading, chunking, embedding, retrieval, generation — is a clean interface. NexRAG ships default implementations for each. You can swap any of them by implementing the interface and declaring it in YAML. No framework lock-in. No magic. No hidden behavior.

---

## Quickstart

```bash
pip install "nexrag[openai,chromadb,pdf]"
export OPENAI_API_KEY=sk-...
cp nexrag.example.yaml nexrag.yaml   # edit to taste
```

```python
from pathlib import Path

from nexrag import NexRAG, RunMetrics

pipeline = NexRAG.from_config("nexrag.yaml")

# Ingest a PDF. Loaders are converter-only: pass the file *content* (bytes for
# PDF, str for text), not a path — NexRAG never opens files for you.
pdf_bytes = Path("contracts/agreement.pdf").read_bytes()
result = pipeline.ingest(pdf_bytes, metadata={"source": "agreement.pdf"})
print(f"Ingested {result.documents_loaded} doc, {result.chunks_produced} chunks")

# Query (blocking)
result = pipeline.query("What are the termination clauses?")
print(result.answer)
for source in result.sources:
    print(f"  [{source.rank}] score={source.score:.3f}  {source.source}")

# Streaming — tokens arrive live; RunMetrics is the final item
metrics = None
for item in pipeline.stream_query("Summarise the key obligations."):
    if isinstance(item, RunMetrics):
        metrics = item
    else:
        print(item, end="", flush=True)
print(f"\n\n{metrics.total_latency_ms:.0f}ms — {metrics.chunks_retrieved} chunks retrieved")
```

```yaml
# nexrag.yaml (minimal)
ingestion:
  loader:
    type: pdf
  embedder:
    provider: openai
    model: text-embedding-3-small
    api_key: ${OPENAI_API_KEY}
  vector_db:
    provider: chroma
    default_collection: documents
    collections:
      documents:
        mode: persistent
        path: ./.nexrag/chroma

query:
  embedder: inherit
  llm:
    provider: openai
    model: gpt-4o
    api_key: ${OPENAI_API_KEY}
```

See [docs/](docs/) for the full documentation site.

---

## Installation

```bash
# Core only — pydantic + pyyaml. No provider SDKs; add the extras you use below.
pip install nexrag

# Default getting-started stack (OpenAI embedder/LLM + ChromaDB vector store)
pip install "nexrag[openai,chromadb]"

# Provider extras — install only what you use
pip install "nexrag[openai]"         # OpenAI embedder + LLM
pip install "nexrag[chromadb]"       # ChromaDB vector store
pip install "nexrag[anthropic]"      # Anthropic (Claude) LLM
pip install "nexrag[ollama]"         # Ollama local LLM + embedder
pip install "nexrag[huggingface]"    # HuggingFace embedder

# Document loaders
pip install "nexrag[pdf]"            # PDFLoader (pypdf)

# Retrieval extras
pip install "nexrag[bm25]"           # BM25Retriever keyword search (rank-bm25)
pip install "nexrag[cohere]"         # CohereReranker
pip install "nexrag[cross-encoder]"  # CrossEncoderReranker (sentence-transformers)

# Observability
pip install "nexrag[observability]"  # OpenTelemetry metrics + traces + logs (Prometheus / OTLP)

# Convenience bundles
pip install "nexrag[all-sparse]"     # all sparse retrievers (bm25)
pip install "nexrag[all-rerankers]"  # all rerankers (cohere + cross-encoder)
pip install "nexrag[all-retrieval]"  # all-sparse + all-rerankers
pip install "nexrag[all-loaders]"    # all document loaders
pip install "nexrag[all-providers]"  # all LLM + embedder providers

# Full bundle — dev/CI; pulls PyTorch via sentence-transformers
pip install "nexrag[all]"
```

---

## Design Principles

| Principle | What it means |
|---|---|
| Interface-first | Every stage is a contract. Implementation is secondary. |
| Config-driven | YAML configures the pipeline. Code defines the logic. |
| Zero lock-in | Core has no dependency on LangChain, LlamaIndex, or any AI SDK. |
| Explicit over implicit | No hidden defaults. Every behavior is declared or documented. |
| Extensible by design | New components plug in without touching core. |

---

## Architecture

NexRAG has two independent pipelines:

```
INGESTION  →  Loader → Sanitizer → Chunker → Embedder → VectorDB
QUERY      →  Embedder → Retriever → Reranker → PromptBuilder → LLM → PipelineResult
```

Every stage runs through a guard chain (ingestion / input / retrieved / output) and emits
a `PipelineEvent` to the observer.

See [Architecture Documentation](docs/) for full pipeline diagrams.

---

## Supported Providers

| Category | Providers |
|---|---|
| Embedders | OpenAI, Gemini, Ollama, HuggingFace |
| Vector DBs | ChromaDB (in-memory, persistent, server), Pinecone |
| LLMs | OpenAI, Anthropic, Gemini, Ollama |
| Loaders | PDF, plain text, auto-detect (Word / HTML / Excel are planned, not yet wired) |
| Chunkers | Recursive, fixed, token, sentence, sentence-window, markdown, code, semantic, proposition |
| Retrievers | Dense (cosine similarity), BM25 (keyword), Hybrid (dense + BM25) |
| Rerankers | Cohere, CrossEncoder (sentence-transformers) |
| Guardrails | PII, access control, prompt-injection, groundedness, topic, model (LLM-as-judge) |
| Observability | OpenTelemetry — Prometheus pull, OTLP push (Grafana Alloy, Jaeger, etc.) |

Every provider is selected in YAML and swappable with a `custom` class path. See
[docs/](docs/) for the full configuration reference.

---

## Production features

Beyond the core pipeline, the following are configured entirely in YAML and off by default:

| Feature | Config block | What it does |
|---|---|---|
| Async & streaming | `mode: async`, `stream_query()` / `astream_query()` | Native-async stages and token-by-token streaming. |
| Guardrails | `guardrails.{ingestion,input,retrieved,output}` | Ordered guard chains with `fail_open` / `fail_closed` policy; verdicts ALLOW / BLOCK / REDACT. |
| Query cache | `query.cache` | Exact-match in-memory cache (LRU + TTL) with per-collection invalidation; pluggable backend. |
| Multi-turn sessions | `query.session` + `query_session()` / `clear_session()` / `delete_turns()` | Conversation history injected into the prompt; window / token-budget context strategies; `persist: false` privacy mode. |
| Rate limiting | `query.rate_limit` | Client-side token-bucket throttle; raises `LLMRateLimitError(retry_after_seconds=...)`. |
| Evaluations | `observability.evaluations` | Off-path LLM-as-judge: faithfulness, answer relevance, completeness, coherence, context diversity, at a sample rate. |

---

## Observability

Connect Prometheus + Grafana with two YAML lines:

```bash
pip install "nexrag[observability]"
```

```yaml
# nexrag.yaml
observability:
  enabled: true
  service_name: my-rag-app
  exporters:
    prometheus:
      enabled: true    # exposes GET http://host:9464/metrics
      port: 9464
    otlp:
      enabled: false   # push to OTel Collector / Grafana Alloy
      endpoint: ${OTEL_EXPORTER_OTLP_ENDPOINT:-http://localhost:4317}
      protocol: grpc
  signals:
    metrics: true
    traces: true
    logs: true
```

Always-on metrics cover every pipeline stage — latency, token counts, retrieval scores, ingestion stats, and cost. Optional LLM-as-judge evaluators (faithfulness, answer relevance, completeness, coherence, context diversity) run off the response path at a configurable sample rate, each with its own model and embedder.


---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
