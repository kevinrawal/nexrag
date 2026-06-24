"""Unit tests for TokenBucketRateLimiter."""

import asyncio

import pytest

from nexrag.defaults.rate_limiter import TokenBucketRateLimiter
from nexrag.exceptions import LLMRateLimitError


class TestTokenBucketRateLimiter:
    def test_allows_up_to_burst_immediately(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60, burst=5)
        for _ in range(5):
            limiter.acquire()  # should not raise

    def test_rejects_after_burst_exhausted(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60, burst=3)
        for _ in range(3):
            limiter.acquire()
        with pytest.raises(LLMRateLimitError):
            limiter.acquire()

    def test_rejection_sets_retry_after_seconds(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60, burst=1)
        limiter.acquire()
        with pytest.raises(LLMRateLimitError) as exc:
            limiter.acquire()
        assert exc.value.retry_after_seconds is not None
        assert exc.value.retry_after_seconds > 0

    def test_refills_over_time(self, monkeypatch):
        clock = {"t": 1000.0}
        monkeypatch.setattr("nexrag.defaults.rate_limiter.time.monotonic", lambda: clock["t"])
        limiter = TokenBucketRateLimiter(requests_per_minute=60, burst=1)  # 1 token/sec
        limiter.acquire()
        with pytest.raises(LLMRateLimitError):
            limiter.acquire()
        clock["t"] += 1.0  # one second passes → one token refilled
        limiter.acquire()  # should not raise

    def test_rate_is_respected(self, monkeypatch):
        clock = {"t": 0.0}
        monkeypatch.setattr("nexrag.defaults.rate_limiter.time.monotonic", lambda: clock["t"])
        limiter = TokenBucketRateLimiter(requests_per_minute=120, burst=1)  # 2 tokens/sec
        limiter.acquire()
        clock["t"] += 0.5  # exactly one token at 2/sec
        limiter.acquire()

    def test_invalid_rate_raises(self):
        with pytest.raises(ValueError):
            TokenBucketRateLimiter(requests_per_minute=0, burst=5)

    def test_aacquire_matches_acquire(self):
        limiter = TokenBucketRateLimiter(requests_per_minute=60, burst=1)

        async def run() -> None:
            await limiter.aacquire()
            with pytest.raises(LLMRateLimitError):
                await limiter.aacquire()

        asyncio.run(run())
