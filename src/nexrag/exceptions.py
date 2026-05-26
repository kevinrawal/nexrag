"""
NexRAG exception hierarchy.

Every exception carries: stage name, component name, pipeline_id, and the
original exception.
"""

from __future__ import annotations


class NexRAGError(Exception):
    """Base exception for all NexRAG errors."""

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        component: str | None = None,
        pipeline_id: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.stage = stage
        self.component = component
        self.pipeline_id = pipeline_id
        self.cause = cause
        super().__init__(self._format(message))

    def _format(self, message: str) -> str:
        parts = [message]
        if self.stage:
            parts.append(f"stage={self.stage}")
        if self.component:
            parts.append(f"component={self.component}")
        if self.pipeline_id:
            parts.append(f"pipeline_id={self.pipeline_id}")
        if self.cause:
            parts.append(f"cause={type(self.cause).__name__}: {self.cause}")
        return " | ".join(parts)


# Configuration


class ConfigError(NexRAGError):
    """Bad or missing nexrag.yaml values."""


class ClassResolutionError(ConfigError):
    """Dotted class path not found, not importable, or wrong interface."""


# Ingestion stages


class LoaderError(NexRAGError):
    """Failed to read or parse a source file."""


class SanitizerError(NexRAGError):
    """User-provided sanitizer raised an exception."""


class ChunkError(NexRAGError):
    """Chunking failed — empty output, invalid config, or runtime error."""


class EmbedderError(NexRAGError):
    """Embedding API failed or returned an unexpected shape."""


class EmbedderMismatchError(EmbedderError):
    """
    Embedding model changed since the collection was created.

    Vectors produced by different models are incompatible.
    Resolution: run with --rebuild to wipe and re-ingest the collection.
    """

    def __init__(
        self,
        stored_model: str,
        configured_model: str,
        collection: str,
        **kwargs: object,
    ) -> None:
        self.stored_model = stored_model
        self.configured_model = configured_model
        self.collection = collection
        message = (
            f"Embedding model mismatch in collection '{collection}'. "
            f"Stored: '{stored_model}', configured: '{configured_model}'. "
            f"Vectors are incompatible. Run: nexrag rebuild --config nexrag.yaml"
        )
        super().__init__(message, **kwargs)  # type: ignore[arg-type]


class VectorDBError(NexRAGError):
    """Vector database operation failed."""


class VectorDBConnectionError(VectorDBError):
    """Could not connect to the vector database."""


class VectorDBUpsertError(VectorDBError):
    """Failed to write chunks to the vector database."""


# Query stages


class RetrieverError(NexRAGError):
    """Retrieval failed or returned no results."""


class PromptError(NexRAGError):
    """Prompt template rendering failed."""


class LLMError(NexRAGError):
    """LLM API call failed."""


class LLMTimeoutError(LLMError):
    """LLM call exceeded the configured timeout."""


class LLMRateLimitError(LLMError):
    """LLM provider rate limit hit."""


# Pipeline orchestration


class PipelineError(NexRAGError):
    """
    Orchestration-level error.

    Wraps a stage-level exception with pipeline context.
    Inspect .cause for the original stage error.
    """
