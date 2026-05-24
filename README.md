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

> **Note:** NexRAG v1.0 is under active development. This section will be updated on first release.

```python
from nexrag import NexRAG

pipeline = NexRAG.from_config("nexrag.yaml")

# Ingest documents
pipeline.ingest("docs/contracts/")

# Query
result = pipeline.query("What are the termination clauses?")
print(result.answer)
print(result.source_chunks)
```

```yaml
# nexrag.yaml
nexrag:
  version: "1.0"

ingestion:
  chunker:
    strategy: recursive
    chunk_size: 512
  embedder:
    provider: openai
    model: text-embedding-3-small
  vector_db:
    provider: chroma
    default_collection: contracts

query:
  llm:
    provider: openai
    model: gpt-4o
```

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

## Supported Providers (V1)

| Category | Providers |
|---|---|
| Embedders | OpenAI, HuggingFace, Ollama |
| Vector DBs | ChromaDB (local + remote) |
| LLMs | OpenAI, Anthropic, Ollama |
| Loaders | PDF, TXT/MD, Word, Excel, JSON, HTML, Code |

---

## Contributing

NexRAG is in early development. Contribution guidelines will be published with v1.0.

---

## Changelog

See [CHANGELOG.md](CHANGELOG.md).
