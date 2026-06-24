"""
InMemoryQueryCache — the default, process-local query-result cache.

Exact-match only (see BaseQueryCache for why semantic caching is not the default).
Backed by an ``OrderedDict`` for O(1) LRU eviction, guarded by a lock so concurrent
queries are safe.

Invalidation uses a per-collection **version counter** rather than scanning and
deleting keys: ``invalidate(collection)`` bumps the counter, and a cached entry is
only a hit if it was written at the collection's current version. Stale entries are
left to fall out via LRU/TTL. This keeps ``invalidate`` O(1) on the ingest path.

This backend is single-process: each replica has its own cache. For a shared cache
across replicas (higher hit rate, survives restart) plug a backend such as Redis
via ``query.cache.class``.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from nexrag.core.interfaces.query_cache import BaseQueryCache
from nexrag.core.models.result import PipelineResult


@dataclass
class _Entry:
    result: PipelineResult
    expires_at: float  # monotonic seconds; float("inf") when ttl disabled
    version: int  # collection version this entry was written at


class InMemoryQueryCache(BaseQueryCache):
    """
    LRU + TTL in-memory query cache.

    Args:
        max_size:    Maximum number of cached results. Oldest (least-recently-used)
                     entries are evicted past this. Default 1000.
        ttl_seconds: Time-to-live per entry in seconds. 0 or None disables expiry.
                     Default 300 (5 minutes).
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: float | None = 300) -> None:
        self._max_size = max(1, int(max_size))
        self._ttl = float(ttl_seconds) if ttl_seconds else 0.0
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, _Entry] = OrderedDict()
        self._versions: dict[str, int] = {}

    def get(self, key: str, *, collection: str) -> PipelineResult | None:
        now = time.monotonic()
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            if entry.version != self._versions.get(collection, 0):
                # Collection was re-ingested since this entry was written — stale.
                del self._entries[key]
                return None
            if entry.expires_at <= now:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)  # mark most-recently-used
            return entry.result

    def set(self, key: str, result: PipelineResult, *, collection: str) -> None:
        expires_at = (time.monotonic() + self._ttl) if self._ttl else float("inf")
        with self._lock:
            self._entries[key] = _Entry(
                result=result,
                expires_at=expires_at,
                version=self._versions.get(collection, 0),
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_size:
                self._entries.popitem(last=False)  # evict LRU

    def invalidate(self, collection: str) -> None:
        with self._lock:
            self._versions[collection] = self._versions.get(collection, 0) + 1

    # Introspection (used by tests and observability).

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
