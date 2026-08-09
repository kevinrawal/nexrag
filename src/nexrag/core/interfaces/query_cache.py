"""
BaseQueryCache — contract for caching whole query results.

A query cache short-circuits the pipeline: when a lookup hits, the stored
PipelineResult is returned without embedding, retrieval, or an LLM call. This is
the single biggest cost lever for read-heavy deployments (FAQ bots, support KBs)
where the same (or near-same) questions recur.

Two design rules this interface enforces:

1. **The raw query travels with the lookup, not just a digest.** ``get``/``set``
   take a :class:`CacheLookup` carrying the *raw* query text alongside every
   parameter that changes the answer (collection, top_k, score_threshold,
   metadata_filter, auth_context). A pre-hashed key would make exact-match the
   only implementable strategy; semantic caching needs the actual text to embed
   and compare, so it must reach the backend. Most real deployments plug a
   semantic backend here — "refund policy" and "how do I get a refund" should
   share an entry — so the interface is designed around that as the common case,
   not the exception.

2. **Invalidation on write.** Cached answers go stale the moment new documents
   are ingested into a collection. ``invalidate(collection)`` (or the async
   ``ainvalidate``) is called by the facade after every ingest so a backend can
   drop (or version past) that collection's entries. In-memory and Redis backends
   differ in *how* they do this, hence it is part of the contract.

Backends ship separately under ``nexrag/caches/`` (one module per backend, same
convention as ``nexrag/retrievers/`` and ``nexrag/guards/``) and are resolved by
``_factory.py`` from ``query.cache.backend``. The default ``InMemoryQueryCache``
(``nexrag/caches/memory.py``) is single-process and exact-match; production
multi-replica deploys — and anyone wanting semantic matching — plug a backend via
``query.cache.backend: custom`` + ``class``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from nexrag.core.models.result import PipelineResult


@dataclass(frozen=True)
class CacheLookup:
    """
    Everything a cache backend needs to check or store one query's result.

    ``query`` is the raw, un-normalised text — a semantic backend embeds it and
    searches for a near neighbour. The remaining fields must still match exactly
    regardless of strategy: each one changes the answer or the access boundary
    (``auth_context`` becomes a retrieval filter), so a similarity match must
    never cross them. Exact-match backends fold every field, including the
    query, into a single key via :meth:`BaseQueryCache.make_key`.
    """

    query: str
    collection: str
    top_k: int | None
    score_threshold: float | None
    metadata_filter: dict[str, Any] | None
    auth_context: dict[str, Any] | None


class BaseQueryCache(ABC):
    """Abstract base class for all NexRAG query-result caches."""

    def make_key(self, lookup: CacheLookup) -> str:
        """
        Build a stable exact-match key for ``lookup``.

        This is the default key-derivation strategy, not part of the storage
        contract itself — it exists as an overridable seam. ``InMemoryQueryCache``
        (and any backend that wants exact matching) calls this from ``get``/
        ``set``; subclass it and override just this method to change *what
        counts as the same request* — e.g. drop ``auth_context`` for a
        single-tenant deploy — without reimplementing LRU/TTL/invalidation.

        A semantic backend typically ignores this method entirely: it matches
        ``lookup.query`` by embedding similarity instead of exact key equality,
        while still using the non-query fields to scope the search (see
        :class:`CacheLookup`).

        The query is stripped and lower-cased so trivial whitespace/case
        differences share an entry; semantically different queries still differ.
        """
        payload = {
            "q": lookup.query.strip().lower(),
            "collection": lookup.collection,
            "top_k": lookup.top_k,
            "score_threshold": lookup.score_threshold,
            "metadata_filter": lookup.metadata_filter,
            "auth_context": lookup.auth_context,
        }
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    @abstractmethod
    def get(self, lookup: CacheLookup) -> PipelineResult | None:
        """
        Return the cached PipelineResult for ``lookup``, or None on miss/expiry.

        ``lookup.collection`` is available so backends that namespace by
        collection version (for invalidation) can check staleness.
        """

    @abstractmethod
    def set(self, lookup: CacheLookup, result: PipelineResult) -> None:
        """Store ``result`` for ``lookup``, tagged with its collection for invalidation."""

    @abstractmethod
    def invalidate(self, collection: str) -> None:
        """
        Drop (or version past) all cached entries for ``collection``.

        Called by the facade after every ingest into ``collection`` so stale
        answers are never served. Must be cheap — it runs on the ingest path.
        """

    # Async variants — the async facade methods (async_query, async_ingest, ...)
    # call these so a network-backed cache (e.g. Redis) never blocks the event
    # loop. The defaults run the sync methods in a thread pool, so existing
    # sync-only backends keep working unmodified; override with a native async
    # client for true async I/O.

    async def aget(self, lookup: CacheLookup) -> PipelineResult | None:
        """Async variant of :meth:`get`. Default: runs ``get`` in a thread pool."""
        return await asyncio.to_thread(self.get, lookup)

    async def aset(self, lookup: CacheLookup, result: PipelineResult) -> None:
        """Async variant of :meth:`set`. Default: runs ``set`` in a thread pool."""
        await asyncio.to_thread(self.set, lookup, result)

    async def ainvalidate(self, collection: str) -> None:
        """Async variant of :meth:`invalidate`. Default: runs ``invalidate`` in a thread pool."""
        await asyncio.to_thread(self.invalidate, collection)
