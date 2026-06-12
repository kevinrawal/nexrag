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
pip install "nexrag[openai,pdf]"
export OPENAI_API_KEY=sk-...
cp nexrag.example.yaml nexrag.yaml   # edit to taste
```

```python
from nexrag import NexRAG, RunMetrics

pipeline = NexRAG.from_config("nexrag.yaml")

# Ingest a PDF
result = pipeline.ingest("contracts/agreement.pdf")
print(f"Ingested {result.documents_loaded} doc, {result.chunks_produced} chunks")

# Query (blocking)
result = pipeline.query("What are the termination clauses?")
print(result.answer)
for source in result.sources:
    print(f"  [{source.rank}] score={source.score:.3f}  {source.chunk.metadata.get('source')}")

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
# Core only
pip install nexrag

# Provider extras — install only what you use
pip install "nexrag[openai]"         # OpenAI embedder + LLM
pip install "nexrag[anthropic]"      # Anthropic (Claude) LLM
pip install "nexrag[ollama]"         # Ollama local LLM + embedder
pip install "nexrag[huggingface]"    # HuggingFace embedder

# Document loaders
pip install "nexrag[pdf]"            # PDFLoader (pypdf)
pip install "nexrag[word]"           # Word documents (python-docx)
pip install "nexrag[html]"           # HTML pages (beautifulsoup4)

# Retrieval extras
pip install "nexrag[bm25]"           # BM25Retriever keyword search (rank-bm25)
pip install "nexrag[cohere]"         # CohereReranker
pip install "nexrag[cross-encoder]"  # CrossEncoderReranker (sentence-transformers)

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
QUERY      →  Embedder → Retriever → PromptBuilder → LLM → PipelineResult
```

See [Architecture Documentation](docs/) for full pipeline diagrams.

---

## Supported Providers

| Category | Providers |
|---|---|
| Embedders | OpenAI, Ollama, HuggingFace |
| Vector DBs | ChromaDB (in-memory, persistent, remote server) |
| LLMs | OpenAI, Ollama, Anthropic |
| Loaders | PDF, plain text, Word, HTML, Excel |
| Chunkers | Recursive (separator-aware) |
| Retrievers | Dense (cosine similarity), BM25 (keyword), Hybrid (dense + BM25) |
| Rerankers | Cohere, CrossEncoder (sentence-transformers) |

---

## Contributing

NexRAG is in early development. Contribution guidelines will be published with v1.0.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
