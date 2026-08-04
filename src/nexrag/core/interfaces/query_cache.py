"""
BaseQueryCache — contract for caching whole query results.

A query cache short-circuits the pipeline: when a key hits, the stored
PipelineResult is returned without embedding, retrieval, or an LLM call. This is
the single biggest cost lever for read-heavy deployments (FAQ bots, support KBs)
where the same questions recur.

Two design rules this interface enforces:

1. **Correctness over hit rate.** The default key strategy is *exact-match* — a
   SHA-256 of the normalised query plus every parameter that changes the answer
   (collection, top_k, score_threshold, metadata_filter, auth_context). Two
   different questions never collide. Semantic caching (returning a cached answer
   for a "similar" query) is opt-in and risky — "refund policy" and "return
   policy" are embedding-near but have different answers — so it is never the
   default and must be requested explicitly.

2. **Invalidation on write.** Cached answers go stale the moment new documents
   are ingested into a collection. ``invalidate(collection)`` is called by the
   facade after every ingest so a backend can drop (or version past) that
   collection's entries. In-memory and Redis backends differ in *how* they do
   this, hence it is part of the contract.

Backends ship separately: the default ``InMemoryQueryCache`` is single-process;
production multi-replica deploys plug a shared backend (e.g. Redis) via
``query.cache.class``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any

from nexrag.core.models.result import PipelineResult


def make_cache_key(
    query: str,
    *,
    collection: str,
    top_k: int | None,
    score_threshold: float | None,
    metadata_filter: dict[str, Any] | None,
    auth_context: dict[str, Any] | None,
) -> str:
    """
    Build a stable exact-match cache key.

    Every argument that can change the answer is folded into the key. auth_context
    is included so a tenant never sees another tenant's cached answer — the
    access-control guard turns auth_context into a retrieval filter, so two
    principals with different auth must key differently.

    The query is stripped and lower-cased so trivial whitespace/case differences
    share an entry; semantically different queries still differ.
    """
    payload = {
        "q": query.strip().lower(),
        "collection": collection,
        "top_k": top_k,
        "score_threshold": score_threshold,
        "metadata_filter": metadata_filter,
        "auth_context": auth_context,
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()


class BaseQueryCache(ABC):
    """Abstract base class for all NexRAG query-result caches."""

    @abstractmethod
    def get(self, key: str, *, collection: str) -> PipelineResult | None:
        """
        Return the cached PipelineResult for ``key``, or None on miss/expiry.

        ``collection`` is passed so backends that namespace by collection version
        (for invalidation) can check staleness.
        """

    @abstractmethod
    def set(self, key: str, result: PipelineResult, *, collection: str) -> None:
        """Store ``result`` under ``key``, tagged with ``collection`` for invalidation."""

    @abstractmethod
    def invalidate(self, collection: str) -> None:
        """
        Drop (or version past) all cached entries for ``collection``.

        Called by the facade after every ingest into ``collection`` so stale
        answers are never served. Must be cheap — it runs on the ingest path.
        """

    # Async variants — the async facade methods (async_query) call these so a
    # network-backed cache (e.g. Redis) never blocks the event loop. The defaults
    # run the sync methods in a thread pool, so existing sync-only backends keep
    # working unmodified; override with a native async client for true async I/O.

    async def aget(self, key: str, *, collection: str) -> PipelineResult | None:
        """Async variant of :meth:`get`. Default: runs ``get`` in a thread pool."""
        return await asyncio.to_thread(self.get, key, collection=collection)

    async def aset(self, key: str, result: PipelineResult, *, collection: str) -> None:
        """Async variant of :meth:`set`. Default: runs ``set`` in a thread pool."""
        await asyncio.to_thread(self.set, key, result, collection=collection)
