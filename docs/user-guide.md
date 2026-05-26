# NexRAG User Guide

> This guide is for developers integrating NexRAG into an application.
> For high-level overview and design principles, see the [README](../README.md).

---

## Table of Contents

1. [Installation](#installation)
2. [Minimal configuration](#minimal-configuration)
3. [Ingest your first document](#ingest-your-first-document)
4. [Query your data](#query-your-data)
5. [Secret management](#secret-management)
6. [Ingesting without a file](#ingesting-without-a-file)
7. [Overriding query parameters at runtime](#overriding-query-parameters-at-runtime)
8. [Swapping a component](#swapping-a-component)
9. [Testing without API calls](#testing-without-api-calls)
10. [Result reference](#result-reference)

---

## Installation

Install the core package plus the extras you need:

```bash
# Core + OpenAI + PDF support (most common)
pip install "nexrag[openai,pdf]"

# Core + OpenAI only (no PDF)
pip install "nexrag[openai]"

# Everything
pip install "nexrag[all]"

# Core only (bring your own adapters)
pip install nexrag
```

**Available extras**

| Extra | Installs | When you need it |
|---|---|---|
| `openai` | `openai` | OpenAI embeddings or LLM |
| `pdf` | `pypdf` | PDF ingestion |
| `ollama` | `ollama` | Local Ollama LLM or embeddings |
| `chromadb` | `chromadb` | Default vector store |
| `all` | All of the above | Full installation |

---

## Minimal configuration

NexRAG is configured entirely through a YAML file (default: `nexrag.yaml` in your working directory).

```yaml
# nexrag.yaml

ingestion:
  loader:
    type: pdf

  chunker:
    strategy: recursive
    chunk_size: 512
    chunk_overlap: 64

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
  embedder: inherit       # reuse the ingestion embedder

  retriever:
    provider: dense
    top_k: 5
    score_threshold: 0.0

  llm:
    provider: openai
    model: gpt-4o
    api_key: ${OPENAI_API_KEY}

  prompt:
    system: |
      You are a helpful assistant. Answer the question using only the provided context.
      If the context does not contain the answer, say "I don't know."
```

> **Rule**: Never hardcode API keys in YAML. Use `${ENV_VAR}` — see [Secret management](#secret-management).

---

## Ingest your first document

```python
import os
from nexrag import NexRAG

os.environ["OPENAI_API_KEY"] = "sk-..."   # or set it in your shell / .env

pipeline = NexRAG.from_config("nexrag.yaml")

result = pipeline.ingest("contracts/agreement.pdf")

print(f"Ingested {result.docs} document(s)")
print(f"Wrote {result.chunks_written} chunks in {result.latency_ms:.1f} ms")
```

`ingest()` accepts:
- A file path as `str` or `pathlib.Path`
- Raw `bytes` (e.g. from S3, HTTP response)
- A plain string (if your loader type is `txt`)

---

## Query your data

```python
result = pipeline.query("What are the termination clauses?")

print(result.answer)

for source in result.sources:
    print(f"  [{source.rank}] score={source.score:.3f}  {source.chunk.metadata.get('source')}")
```

The full `PipelineResult` fields are documented in [Result reference](#result-reference).

---

## Secret management

NexRAG supports `${ENV_VAR}` substitution in YAML before any parsing occurs.

**Required variable** — fails at startup if not set:
```yaml
embedder:
  api_key: ${OPENAI_API_KEY}
```

**Optional with fallback** — uses empty string if not set:
```yaml
embedder:
  api_key: ${OPENAI_API_KEY:-}
```

**Default value** — uses `"gpt-4o"` if `LLM_MODEL` is not set:
```yaml
llm:
  model: ${LLM_MODEL:-gpt-4o}
```

**How to set env vars**

```bash
# Shell
export OPENAI_API_KEY=sk-...

# .env file (loaded before NexRAG.from_config)
from dotenv import load_dotenv
load_dotenv()
pipeline = NexRAG.from_config("nexrag.yaml")

# Docker
docker run --env-file .env myapp

# GitHub Actions
env:
  OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

> **Never** put a real API key in a YAML file. YAML files are often committed to version control.

---

## Ingesting without a file

### Raw string

```python
text = "Alice joined Acme Corp in 2019 as a senior engineer..."

result = pipeline.ingest(("alice_bio", text))
# The first element becomes metadata["source"]
```

Or configure `loader.type: txt` and call:

```python
result = pipeline.ingest(text)
```

### Pre-built Document objects

Use `ingest_documents()` when your data comes from a database, API, or S3 — anywhere a file path
doesn't make sense. You build the `Document` objects; NexRAG handles chunking, embedding, and storage.

```python
from nexrag.core.models.document import Document

documents = [
    Document(
        content="The quick brown fox jumped over the lazy dog.",
        metadata={"source": "s3://my-bucket/fox.txt", "year": 2024},
    ),
    Document(
        content="NexRAG is a framework-agnostic RAG pipeline SDK.",
        metadata={"source": "internal-wiki", "team": "platform"},
    ),
]

result = pipeline.ingest_documents(documents)
```

### Raw bytes (e.g. from S3)

```python
import boto3

s3 = boto3.client("s3")
pdf_bytes = s3.get_object(Bucket="my-bucket", Key="report.pdf")["Body"].read()

result = pipeline.ingest(pdf_bytes)
# metadata["source"] defaults to "pdf_bytes" — override with PDFLoader(source_override="s3://...")
```

---

## Overriding query parameters at runtime

All query parameters can be overridden per-call without changing the YAML:

```python
# Query a different collection
result = pipeline.query("What is the refund policy?", collection="terms_of_service")

# Retrieve more chunks
result = pipeline.query("Summarize the contract", top_k=10)

# Raise the similarity threshold
result = pipeline.query("Exact clause text", score_threshold=0.75)

# Filter by metadata
result = pipeline.query(
    "What changed in 2024?",
    metadata_filter={"year": 2024},
)
```

---

## Swapping a component

Any component can be replaced by providing `provider: custom` and a dotted `class:` path in YAML.
Your class must implement the corresponding interface from `nexrag.core.interfaces`.

### Example: custom embedder

```python
# myapp/embedders.py
from nexrag.core.interfaces.embedder import BaseEmbedder

class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, model: str = "all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model)
        self._model_name = model

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode([text])[0].tolist()
```

```yaml
# nexrag.yaml
ingestion:
  embedder:
    provider: custom
    class: myapp.embedders.SentenceTransformerEmbedder
    params:
      model: all-MiniLM-L6-v2
```

The same pattern works for every stage: `loader`, `chunker`, `vector_db`, `retriever`,
`prompt_builder`, `llm`, `observability`.

---

## Testing without API calls

### Use memory mode for ChromaDB

No disk I/O, no persistent state between test runs:

```yaml
# nexrag.test.yaml
vector_db:
  provider: chroma
  default_collection: test
  collections:
    test:
      mode: memory
```

```python
pipeline = NexRAG.from_config("nexrag.test.yaml")
```

### Mock the embedder and LLM in tests

```python
import pytest
from unittest.mock import MagicMock
from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.core.interfaces.llm import BaseLLM

class FixedEmbedder(BaseEmbedder):
    DIM = 8

    @property
    def model_name(self) -> str:
        return "fixed-test"

    @property
    def dimensions(self) -> int:
        return self.DIM

    def embed(self, texts):
        return [[float(i % self.DIM) for i in range(self.DIM)] for _ in texts]

    def embed_query(self, text):
        return [float(i % self.DIM) for i in range(self.DIM)]


class EchoLLM(BaseLLM):
    def generate(self, prompt: str) -> str:
        return "test answer"

    def stream(self, prompt):
        yield "test answer"
```

Wire them directly without YAML:

```python
from nexrag import NexRAG
from nexrag.adapters.vector_dbs.chroma import ChromaDBAdapter
from nexrag.chunkers.recursive import RecursiveChunker
from nexrag.core.pipeline.ingestion import IngestionPipeline
from nexrag.core.pipeline.query import QueryPipeline
from nexrag.defaults.prompt_builder import DefaultPromptBuilder
from nexrag.retrievers.dense import DenseRetriever

vector_db = ChromaDBAdapter(mode="memory")
embedder = FixedEmbedder()
llm = EchoLLM()

ingestion = IngestionPipeline(
    chunker=RecursiveChunker(),
    embedder=embedder,
    vector_db=vector_db,
    collection="test",
)
query = QueryPipeline(
    embedder=embedder,
    retriever=DenseRetriever(vector_db=vector_db),
    prompt_builder=DefaultPromptBuilder(system="You are helpful."),
    llm=llm,
    collection="test",
)
pipeline = NexRAG(ingestion=ingestion, query=query)
```

---

## Result reference

### `IngestionResult`

Returned by `pipeline.ingest()` and `pipeline.ingest_documents()`.

| Field | Type | Description |
|---|---|---|
| `pipeline_id` | `str` | UUID for this ingestion run |
| `docs` | `int` | Number of documents loaded |
| `chunks` | `int` | Total chunks produced by the chunker |
| `chunks_written` | `int` | Chunks actually written (skips duplicates on `skip` conflict strategy) |
| `latency_ms` | `float` | Wall-clock time for the full ingestion pipeline |

### `PipelineResult`

Returned by `pipeline.query()`.

| Field | Type | Description |
|---|---|---|
| `pipeline_id` | `str` | UUID for this query run |
| `answer` | `str` | LLM-generated answer |
| `sources` | `list[ScoredChunk]` | Retrieved chunks, ordered by similarity score |
| `latency_ms` | `float` | Wall-clock time for the full query pipeline |

### `ScoredChunk`

Each element in `result.sources`.

| Field | Type | Description |
|---|---|---|
| `chunk.text` | `str` | The chunk's text content |
| `chunk.metadata` | `dict` | Metadata from the original document (e.g. `source`, `page_count`) |
| `chunk.chunk_index` | `int` | Position of this chunk within its parent document |
| `chunk.total_chunks` | `int` | Total chunks from the same parent document |
| `score` | `float` | Cosine similarity score (0–1, higher = more similar) |
| `rank` | `int` | 1-based rank in the result list |
