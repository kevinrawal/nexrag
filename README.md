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
[![License](https://img.shields.io/badge/license-TBD-lightgrey.svg)]()

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
from nexrag import NexRAG

pipeline = NexRAG.from_config("nexrag.yaml")

# Ingest a PDF
result = pipeline.ingest("contracts/agreement.pdf")
print(f"Ingested {result.documents_loaded} doc, {result.chunks_written} chunks")

# Query
result = pipeline.query("What are the termination clauses?")
print(result.answer)
for source in result.sources:
    print(f"  [{source.rank}] score={source.score:.3f}  {source.chunk.metadata.get('source')}")
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

See [docs/user-guide.md](docs/user-guide.md) for the full guide.

---

## Installation

```bash
# Core only
pip install nexrag

# With OpenAI support
pip install "nexrag[openai]"

# With everything
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

**Available now**

| Category | Providers |
|---|---|
| Embedders | OpenAI |
| Vector DBs | ChromaDB (local persistent, in-memory) |
| LLMs | OpenAI, Ollama |
| Loaders | PDF, plain text |
| Chunkers | Recursive (separator-aware) |

**Coming in V1**

| Category | Providers |
|---|---|
| Embedders | Ollama, HuggingFace |
| Vector DBs | ChromaDB (remote server via HttpClient) |
| LLMs | Anthropic |

---

## Contributing

NexRAG is in early development. Contribution guidelines will be published with v1.0.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
