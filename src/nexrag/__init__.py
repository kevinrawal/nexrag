"""
NexRAG — Framework-agnostic RAG pipeline SDK.

Public API surface. Import from here, not from internal modules.

    from nexrag import NexRAG, PipelineResult
    from nexrag.exceptions import NexRAGError

Everything under nexrag.core, nexrag.adapters, nexrag.loaders, etc.
is internal. Internal APIs may change between minor versions.
The public surface below is stable across minor versions.
"""

from nexrag.exceptions import NexRAGError

__version__ = "0.1.0"
__all__ = [
    "NexRAG",
    "PipelineResult",
    "NexRAGError",
    "__version__",
]


class NexRAG:
    """Placeholder — real implementation comes in Phase 3."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "NexRAG entrypoint is not yet implemented. "
            "Core interfaces and config system are next."
        )


class PipelineResult:
    """Placeholder — real implementation comes in Phase 0 data models."""
    def __init__(self) -> None:
        raise NotImplementedError("PipelineResult is not yet implemented.")
