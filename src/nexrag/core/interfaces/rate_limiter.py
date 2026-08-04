"""
BaseRateLimiter — contract for client-side query-path throttling.

The rate limiter sits at the very front of every query entry point on the facade:
it throttles (or rejects) a request *before* any pipeline stage runs, so a retry
storm or runaway loop is rejected cheaply and locally with an
``LLMRateLimitError`` instead of fanning out into provider 429s.

Like the query cache and session store, the limiter is a thin, swappable interface:

    - ``TokenBucketRateLimiter`` (default) — process-local token bucket. Correct
      for a single process; in a multi-worker/multi-replica deployment each
      process holds its own bucket, so the effective limit scales with worker
      count. Fine for development and single-process services.
    - A shared/distributed backend (e.g. Redis-backed) — plugged via
      ``query.rate_limit.class`` for multi-process correctness.

Implementations MUST be safe to call concurrently from both threaded (sync facade
methods, worker pools) and asyncio (async facade methods) callers — the token
maths must never block the event loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseRateLimiter(ABC):
    """Abstract base class for all NexRAG client-side rate limiters."""

    @abstractmethod
    def acquire(self) -> None:
        """
        Consume one unit of quota, or raise ``LLMRateLimitError`` if none is available.

        Called synchronously at the start of every sync query entry point.

        Raises:
            LLMRateLimitError: With ``retry_after_seconds`` set, when over the limit.
        """

    async def aacquire(self) -> None:
        """
        Async variant of :meth:`acquire`.

        Default implementation calls the synchronous :meth:`acquire` — correct for
        limiters whose quota maths never blocks (the built-in token bucket).
        Override with a native async implementation when the backend performs I/O
        (e.g. a Redis round-trip) so the event loop is never blocked.

        Raises:
            LLMRateLimitError: With ``retry_after_seconds`` set, when over the limit.
        """
        self.acquire()
