"""
TokenBucketRateLimiter — client-side request throttle for the query path.

A single misbehaving caller (a retry storm, a load test, a runaway loop) can
exhaust an upstream LLM/embedder quota for everyone in seconds. NexRAG's facade
applies this limiter *before* any pipeline stage so excess requests are rejected
cheaply, locally, with a clear ``LLMRateLimitError`` carrying ``retry_after_seconds``
— rather than fanning out into provider 429s.

Algorithm: a classic token bucket. The bucket holds up to ``burst`` tokens and
refills continuously at ``requests_per_minute / 60`` tokens per second. Each request
spends one token; when the bucket is empty the request is rejected and the caller is
told how long until one token is available. Pure stdlib, no external dependency.

Thread-safe (``threading.Lock``) and usable from async code: the bucket maths is
non-blocking, so the async path guards the same bucket without an event-loop hop.
"""

from __future__ import annotations

import threading
import time

from nexrag.exceptions import LLMRateLimitError


class TokenBucketRateLimiter:
    """
    Token-bucket rate limiter.

    Args:
        requests_per_minute: Sustained refill rate. Default 60 (one per second).
        burst:               Maximum tokens the bucket can hold — how many requests
                             may fire back-to-back before throttling kicks in.
                             Default 10.
    """

    def __init__(self, requests_per_minute: int = 60, burst: int = 10) -> None:
        if requests_per_minute <= 0:
            raise ValueError("requests_per_minute must be positive")
        self._rate_per_sec = requests_per_minute / 60.0
        self._capacity = float(max(1, burst))
        self._tokens = self._capacity
        self._updated = time.monotonic()
        self._lock = threading.Lock()

    def _take_token(self) -> float:
        """
        Try to spend one token. Returns 0.0 on success, else seconds until the next
        token is available. Must be called under the lock.
        """
        now = time.monotonic()
        elapsed = now - self._updated
        self._updated = now
        self._tokens = min(self._capacity, self._tokens + elapsed * self._rate_per_sec)
        if self._tokens >= 1.0:
            self._tokens -= 1.0
            return 0.0
        return (1.0 - self._tokens) / self._rate_per_sec

    def acquire(self) -> None:
        """
        Consume one token or raise ``LLMRateLimitError`` if none is available.

        Raises:
            LLMRateLimitError: With ``retry_after_seconds`` set, when over the limit.
        """
        with self._lock:
            wait = self._take_token()
        if wait > 0.0:
            raise LLMRateLimitError(
                f"Rate limit exceeded. Retry in {wait:.2f}s.",
                retry_after_seconds=round(wait, 3),
                stage="rate_limit",
                component="TokenBucketRateLimiter",
            )

    async def aacquire(self) -> None:
        """Async variant — same bucket, same semantics (the maths never blocks)."""
        self.acquire()
