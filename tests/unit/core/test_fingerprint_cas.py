"""
Fingerprint check best-effort compare-and-set (v0.3.3 issue #7).

When two first-ingests with different embedders race into an empty collection,
the loser must raise EmbedderMismatchError instead of silently writing into a
collection that already carries another model's fingerprint.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexrag.core.models.document import Document
from nexrag.core.pipeline.ingestion import IngestionPipeline, _compute_fingerprint
from nexrag.exceptions import EmbedderMismatchError, PipelineError


def _embedder(model: str, dims: int):
    e = MagicMock()
    e.model_name = model
    e.dimensions = dims
    e.embed.side_effect = lambda texts: [[0.1] * dims for _ in texts]
    return e


def _chunker():
    c = MagicMock()
    chunk = MagicMock(text="text", metadata={"source": "s"}, row_id="r1")
    c.chunk.return_value = [chunk]
    return c


def test_losing_writer_detects_competitors_fingerprint_via_cas():
    """
    Our get_collection_metadata reads empty first (so we write our fingerprint),
    but the re-read after set returns a *different* model's fingerprint — as if a
    racing process won the first write. The CAS must surface EmbedderMismatchError.
    """
    competitor = {
        "embedding_model": "competitor-model",
        "embedding_dimensions": 8,
        "fingerprint": _compute_fingerprint("competitor-model", 8),
    }

    vdb = MagicMock()
    # 1st read: empty (we proceed to set). 2nd read (CAS re-read): competitor won.
    vdb.get_collection_metadata.side_effect = [{}, competitor]

    sanitizer = MagicMock()
    sanitizer.sanitize.side_effect = lambda d: d

    pipe = IngestionPipeline(
        chunker=_chunker(),
        embedder=_embedder("our-model", 4),
        vector_db=vdb,
        collection="c",
        sanitizer=sanitizer,
    )

    # The mismatch is raised by the fingerprint stage and surfaces wrapped in a
    # PipelineError at the facade boundary; its cause is the EmbedderMismatchError.
    with pytest.raises(PipelineError) as excinfo:
        pipe.ingest_documents([Document(content="hello", metadata={"source": "s"})])
    assert isinstance(excinfo.value.__cause__, EmbedderMismatchError)


def test_uncontested_first_ingest_succeeds():
    """Re-read returns our own fingerprint — no mismatch, ingest proceeds."""
    ours = {
        "embedding_model": "our-model",
        "embedding_dimensions": 4,
        "fingerprint": _compute_fingerprint("our-model", 4),
    }
    vdb = MagicMock()
    vdb.get_collection_metadata.side_effect = [{}, ours]
    vdb.get_ids_by_metadata.return_value = []

    sanitizer = MagicMock()
    sanitizer.sanitize.side_effect = lambda d: d

    pipe = IngestionPipeline(
        chunker=_chunker(),
        embedder=_embedder("our-model", 4),
        vector_db=vdb,
        collection="c",
        sanitizer=sanitizer,
    )

    result = pipe.ingest_documents([Document(content="hello", metadata={"source": "s"})])
    assert result.chunks_written == 1
