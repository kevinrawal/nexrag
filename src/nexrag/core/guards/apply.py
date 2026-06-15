"""
Guard-chain application helpers shared by the sync and async query pipelines.

These are intentionally synchronous: guards are regex/CPU bound (the model guard
makes a blocking LLM call, which the async pipeline accepts for simplicity). The
helpers translate a GuardChain's ChainOutcome into pipeline actions — raising on a
BLOCK, threading REDACT text forward, dropping blocked chunks, and merging
access-control filters.
"""

from __future__ import annotations

from typing import Any

from nexrag.core.guards.chain import GuardChain
from nexrag.core.interfaces.guard import GuardContext
from nexrag.core.models.chunk import Chunk, ScoredChunk
from nexrag.core.models.document import Document
from nexrag.exceptions import GuardrailBlockedError


def apply_ingestion_guards(
    chain: GuardChain | None, document: Document, *, pipeline_id: str
) -> Document:
    """Run the ingestion chain on a document's content. REDACT rewrites it; BLOCK rejects it."""
    if chain is None:
        return document
    context = GuardContext(pipeline_id=pipeline_id, stage="ingestion")
    outcome = chain.run(document.content, context)
    if outcome.blocked:
        raise GuardrailBlockedError(
            outcome.reason or f"Document '{document.doc_id}' blocked by guardrails.",
            guard=outcome.blocking_guard,
            stage="guardrail",
            pipeline_id=pipeline_id,
        )
    if outcome.text != document.content:
        return document.with_content(outcome.text)
    return document


def merge_filters(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
    """Combine two retrieval filters conjunctively (vector DBs support $and)."""
    if not a:
        return b
    if not b:
        return a
    return {"$and": [a, b]}


def apply_input_guards(
    chain: GuardChain | None,
    query: str,
    metadata_filter: dict[str, Any] | None,
    *,
    pipeline_id: str,
    auth_context: dict[str, Any] | None,
) -> tuple[str, dict[str, Any] | None]:
    """Run the input chain on the query. Returns (possibly redacted query, merged filter)."""
    if chain is None:
        return query, metadata_filter
    context = GuardContext(
        pipeline_id=pipeline_id,
        stage="query_input",
        query=query,
        auth_context=auth_context,
    )
    outcome = chain.run(query, context)
    if outcome.blocked:
        raise GuardrailBlockedError(
            outcome.reason or "Query blocked by guardrails.",
            guard=outcome.blocking_guard,
            stage="guardrail",
            pipeline_id=pipeline_id,
        )
    return outcome.text, merge_filters(metadata_filter, outcome.metadata_filter)


def apply_retrieved_guards(
    chain: GuardChain | None,
    chunks: list[ScoredChunk],
    *,
    pipeline_id: str,
    query: str,
) -> list[ScoredChunk]:
    """Run the retrieved chain over each chunk. BLOCK drops the chunk; REDACT rewrites it."""
    if chain is None:
        return chunks
    context = GuardContext(pipeline_id=pipeline_id, stage="retrieved", query=query)
    kept: list[tuple[Chunk, float]] = []
    for sc in chunks:
        outcome = chain.run(sc.chunk.text, context)
        if outcome.blocked:
            continue
        chunk = sc.chunk
        if outcome.text != chunk.text:
            chunk = Chunk(
                text=outcome.text,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                parent_doc_id=chunk.parent_doc_id,
                metadata=chunk.metadata,
            )
        kept.append((chunk, sc.score))
    return [ScoredChunk(chunk=c, score=s, rank=i) for i, (c, s) in enumerate(kept, start=1)]


def apply_output_guards(
    chain: GuardChain | None,
    answer: str,
    *,
    pipeline_id: str,
    query: str,
    sources: list[str],
) -> str:
    """Run the output chain on the answer. Returns the (possibly redacted) answer; raises on BLOCK."""
    if chain is None:
        return answer
    context = GuardContext(pipeline_id=pipeline_id, stage="output", query=query, sources=sources)
    outcome = chain.run(answer, context)
    if outcome.blocked:
        raise GuardrailBlockedError(
            outcome.reason or "Answer blocked by guardrails.",
            guard=outcome.blocking_guard,
            stage="guardrail",
            pipeline_id=pipeline_id,
        )
    return outcome.text
