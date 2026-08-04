"""
Tests for the rate-limit foundation (issue #56):
  - RateLimitConfig load-time validation, matching CacheConfig/SessionConfig.
  - Pluggable backend resolution in the factory (backend: memory | custom).

The heavier per-user/scope redesign is deferred to a follow-up; these cover the
validation fixes and the swappable-backend foundation only.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nexrag._factory import _build_rate_limiter
from nexrag.core.config.schema import RateLimitConfig
from nexrag.core.interfaces.rate_limiter import BaseRateLimiter
from nexrag.defaults.rate_limiter import TokenBucketRateLimiter


class FakeRateLimiter(BaseRateLimiter):
    """A resolvable custom limiter used to prove backend: custom wiring."""

    def __init__(self, tag: str = "custom") -> None:
        self.tag = tag

    def acquire(self) -> None:  # never throttles — just a wiring probe
        pass


class TestRateLimitConfigValidation:
    def test_defaults_preserve_current_behavior(self):
        cfg = RateLimitConfig()
        assert cfg.backend == "memory"
        assert cfg.requests_per_minute == 60
        assert cfg.burst == 10

    def test_zero_requests_per_minute_rejected_at_load(self):
        with pytest.raises(ValidationError, match="requests_per_minute"):
            RateLimitConfig(requests_per_minute=0)

    def test_non_positive_burst_rejected_at_load(self):
        # Previously only caught (and silently clamped) at limiter construction.
        with pytest.raises(ValidationError, match="burst"):
            RateLimitConfig(burst=0)
        with pytest.raises(ValidationError, match="burst"):
            RateLimitConfig(burst=-5)

    def test_custom_backend_requires_class(self):
        with pytest.raises(ValidationError, match="class"):
            RateLimitConfig(backend="custom")

    def test_custom_backend_with_class_is_valid(self):
        cfg = RateLimitConfig.model_validate(
            {"backend": "custom", "class": "myproject.limits.MyLimiter"}
        )
        assert cfg.class_path == "myproject.limits.MyLimiter"

    def test_custom_backend_skips_numeric_bounds(self):
        # A custom backend owns its own validation; numeric bounds are the
        # built-in limiter's contract and must not block a custom config.
        cfg = RateLimitConfig.model_validate(
            {"backend": "custom", "class": "x.Y", "requests_per_minute": 0}
        )
        assert cfg.backend == "custom"


class TestBuildRateLimiter:
    def test_memory_backend_builds_token_bucket(self):
        limiter = _build_rate_limiter(
            RateLimitConfig(enabled=True, requests_per_minute=120, burst=7)
        )
        assert isinstance(limiter, TokenBucketRateLimiter)

    def test_custom_backend_resolves_class(self):
        cfg = RateLimitConfig.model_validate(
            {
                "enabled": True,
                "backend": "custom",
                "class": f"{__name__}.FakeRateLimiter",
                "params": {"tag": "redis"},
            }
        )
        limiter = _build_rate_limiter(cfg)
        assert isinstance(limiter, FakeRateLimiter)
        assert limiter.tag == "redis"
