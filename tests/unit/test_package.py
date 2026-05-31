"""Basic smoke tests — verify the package is importable and structured correctly."""

import nexrag
from nexrag import NexRAG, PipelineResult
from nexrag.exceptions import (
    ChunkError,
    ClassResolutionError,
    ConfigError,
    EmbedderError,
    EmbedderMismatchError,
    LLMError,
    LLMRateLimitError,
    LLMTimeoutError,
    LoaderError,
    NexRAGError,
    PipelineError,
    PromptError,
    RetrieverError,
    SanitizerError,
    VectorDBConnectionError,
    VectorDBError,
    VectorDBUpsertError,
)


def test_version_exists():
    assert hasattr(nexrag, "__version__")
    assert isinstance(nexrag.__version__, str)
    assert nexrag.__version__ == "0.3.0"


def test_public_api_importable():
    assert NexRAG is not None
    assert PipelineResult is not None
    assert NexRAGError is not None


def test_nexrag_requires_pipelines():
    import pytest

    with pytest.raises(TypeError):
        NexRAG()  # type: ignore[call-arg]


def test_pipeline_result_requires_fields():
    import pytest

    with pytest.raises(TypeError):
        PipelineResult()  # type: ignore[call-arg]


def test_exception_hierarchy():
    # Every stage exception must be catchable as NexRAGError
    assert issubclass(ConfigError, NexRAGError)
    assert issubclass(ClassResolutionError, ConfigError)
    assert issubclass(LoaderError, NexRAGError)
    assert issubclass(SanitizerError, NexRAGError)
    assert issubclass(ChunkError, NexRAGError)
    assert issubclass(EmbedderError, NexRAGError)
    assert issubclass(EmbedderMismatchError, EmbedderError)
    assert issubclass(VectorDBError, NexRAGError)
    assert issubclass(VectorDBConnectionError, VectorDBError)
    assert issubclass(VectorDBUpsertError, VectorDBError)
    assert issubclass(RetrieverError, NexRAGError)
    assert issubclass(PromptError, NexRAGError)
    assert issubclass(LLMError, NexRAGError)
    assert issubclass(LLMTimeoutError, LLMError)
    assert issubclass(LLMRateLimitError, LLMError)
    assert issubclass(PipelineError, NexRAGError)


def test_embedder_mismatch_error_message():
    err = EmbedderMismatchError(
        stored_model="text-embedding-3-small",
        configured_model="text-embedding-3-large",
        collection="contracts",
    )
    assert "contracts" in str(err)
    assert "text-embedding-3-small" in str(err)
    assert "text-embedding-3-large" in str(err)
    assert "rebuild" in str(err)
