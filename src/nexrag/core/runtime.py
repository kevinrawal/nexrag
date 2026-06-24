"""
QueryRuntime — facade-level query components wired from config.

Caching, rate limiting, and conversation sessions sit *around* the query pipeline
rather than inside it: they short-circuit, throttle, or contextualise a request
before/after the pipeline runs, and they apply uniformly to the sync, async, and
streaming entry points. Bundling them in one object keeps the ``wire()`` return
small and lets the NexRAG facade hold a single handle.

All fields are optional. When a feature is disabled in config its field is None and
the facade skips it — there is no behavioural cost for the common "off" case.
"""

from __future__ import annotations

from dataclasses import dataclass

from nexrag.core.interfaces.context_strategy import BaseContextStrategy
from nexrag.core.interfaces.query_cache import BaseQueryCache
from nexrag.core.interfaces.session_store import BaseSessionStore
from nexrag.defaults.rate_limiter import TokenBucketRateLimiter


@dataclass(frozen=True)
class QueryRuntime:
    """Facade-level components that wrap the query pipeline."""

    cache: BaseQueryCache | None = None
    rate_limiter: TokenBucketRateLimiter | None = None
    session_store: BaseSessionStore | None = None
    context_strategy: BaseContextStrategy | None = None
