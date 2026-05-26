"""
NexRAG — Framework-agnostic RAG pipeline SDK.

Public API:
    NexRAG.from_config("nexrag.yaml")  →  NexRAG
    pipeline.ingest("resume.pdf")      →  IngestionResult
    pipeline.query("What skills?")     →  PipelineResult

All other symbols are implementation detail. Import from nexrag, not from internals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.models.result import PipelineResult
from nexrag.core.pipeline.ingestion import IngestionPipeline, IngestionResult
from nexrag.core.pipeline.query import QueryPipeline
from nexrag.exceptions import NexRAGError

__version__ = "0.1.0"
__all__ = [
    "NexRAG",
    "PipelineResult",
    "IngestionResult",
    "NexRAGError",
    "__version__",
]


class NexRAG:
    """
    The NexRAG pipeline facade.

    Instantiate via from_config() — do not call __init__ directly.

        pipeline = NexRAG.from_config("nexrag.yaml")
        pipeline.ingest("my_document.pdf")
        result = pipeline.query("What does the document say about X?")
    """

    def __init__(
        self,
        ingestion: IngestionPipeline,
        query: QueryPipeline,
    ) -> None:
        self._ingestion = ingestion
        self._query = query

    @classmethod
    def from_config(cls, path: str | Path = "nexrag.yaml") -> NexRAG:
        """
        Load nexrag.yaml, resolve all components, wire both pipelines.

        Args:
            path: Path to the YAML config file. Resolved relative to CWD.

        Returns:
            A fully wired NexRAG instance ready to ingest and query.

        Raises:
            ConfigError:          If the YAML is missing, invalid, or fails validation.
            ClassResolutionError: If a custom class_path cannot be imported.
        """
        from nexrag._factory import wire
        from nexrag.core.config.loader import load_config

        config = load_config(path)
        ingestion, query = wire(config)
        return cls(ingestion=ingestion, query=query)

    # Public pipeline methods

    def ingest(self, data: Any, loader: BaseLoader | None = None) -> IngestionResult:
        """
        Ingest data through the configured loader, chunker, embedder, and vector DB.

        Args:
            data:   Anything the loader accepts — file path, bytes, string, etc.
            loader: Optional loader override for this call only.
                    If not given, uses the loader from nexrag.yaml.

        Returns:
            IngestionResult with document count, chunk count, and latency.
        """
        return self._ingestion.ingest(data, loader)

    def ingest_documents(self, documents: list[Any]) -> IngestionResult:
        """
        Ingest pre-built Document objects, skipping the loader stage.

        Use this when your data comes from S3, an API, a database, or anywhere
        that doesn't fit a file path. You produce the Documents; NexRAG handles the rest.

        Args:
            documents: List of nexrag.core.models.document.Document objects.

        Returns:
            IngestionResult with counts and pipeline_id.
        """
        return self._ingestion.ingest_documents(documents)

    def query(
        self,
        text: str,
        *,
        collection: str | None = None,
        top_k: int | None = None,
        score_threshold: float | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> PipelineResult:
        """
        Run the full query pipeline: embed → retrieve → prompt → LLM → result.

        Args:
            text:             The user's question as a plain string.
            collection:       Override the default collection for this query.
            top_k:            Override the top_k retrieval count.
            score_threshold:  Override the minimum similarity score.
            metadata_filter:  Optional key-value metadata filters, e.g. {"year": 2024}.

        Returns:
            PipelineResult with answer, sources, scores, and latency.
        """
        return self._query.run(
            text,
            collection=collection,
            top_k=top_k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter,
        )
