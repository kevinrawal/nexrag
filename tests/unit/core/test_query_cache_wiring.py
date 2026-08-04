"""
Tests for the query-cache wiring foundation (issue #57):
  - Top-level CacheConfig fields (similarity_threshold/max_size/ttl_seconds) reach a
    custom backend without the user duplicating them into `params`.
  - Forwarding is signature-aware: a backend whose __init__ doesn't accept those
    names is never broken by the forwarding.

The heavier interface change (passing the raw query to get/set so semantic caching
is possible) is deferred to a follow-up; this covers the async-safety + config
forwarding foundation only.
"""

from __future__ import annotations

from typing import Any

from nexrag._factory import _build_query_cache
from nexrag.core.config.schema import CacheConfig
from nexrag.core.interfaces.query_cache import BaseQueryCache
from nexrag.core.models.result import PipelineResult


class ForwardingCache(BaseQueryCache):
    """Custom backend that opts into the forwarded config fields."""

    def __init__(
        self,
        similarity_threshold: float | None = None,
        max_size: int | None = None,
        ttl_seconds: int | None = None,
        tag: str = "",
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.tag = tag

    def get(self, key: str, *, collection: str) -> PipelineResult | None:
        return None

    def set(self, key: str, result: PipelineResult, *, collection: str) -> None:
        pass

    def invalidate(self, collection: str) -> None:
        pass


class MinimalCache(BaseQueryCache):
    """Custom backend that accepts ONLY its own params — must not break on forwarding."""

    def __init__(self, url: str) -> None:
        self.url = url

    def get(self, key: str, *, collection: str) -> PipelineResult | None:
        return None

    def set(self, key: str, result: PipelineResult, *, collection: str) -> None:
        pass

    def invalidate(self, collection: str) -> None:
        pass


class KwargsCache(BaseQueryCache):
    """Custom backend with **kwargs — should receive all forwarded fields."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def get(self, key: str, *, collection: str) -> PipelineResult | None:
        return None

    def set(self, key: str, result: PipelineResult, *, collection: str) -> None:
        pass

    def invalidate(self, collection: str) -> None:
        pass


class TestCustomCacheConfigForwarding:
    def test_declared_fields_are_forwarded(self):
        cfg = CacheConfig.model_validate(
            {
                "enabled": True,
                "backend": "custom",
                "class": f"{__name__}.ForwardingCache",
                "similarity_threshold": 0.9,
                "max_size": 42,
                "ttl_seconds": 77,
                "params": {"tag": "semantic"},
            }
        )
        cache = _build_query_cache(cfg)
        assert isinstance(cache, ForwardingCache)
        assert cache.similarity_threshold == 0.9
        assert cache.max_size == 42
        assert cache.ttl_seconds == 77
        assert cache.tag == "semantic"

    def test_explicit_params_win_over_forwarded(self):
        cfg = CacheConfig.model_validate(
            {
                "enabled": True,
                "backend": "custom",
                "class": f"{__name__}.ForwardingCache",
                "max_size": 42,
                "params": {"max_size": 999},  # explicit params take precedence
            }
        )
        cache = _build_query_cache(cfg)
        assert isinstance(cache, ForwardingCache)
        assert cache.max_size == 999

    def test_backend_without_those_kwargs_is_not_broken(self):
        # MinimalCache.__init__ takes only `url`; forwarding must not pass it
        # similarity_threshold/max_size/ttl_seconds and blow up with a TypeError.
        cfg = CacheConfig.model_validate(
            {
                "enabled": True,
                "backend": "custom",
                "class": f"{__name__}.MinimalCache",
                "params": {"url": "redis://localhost"},
            }
        )
        cache = _build_query_cache(cfg)
        assert isinstance(cache, MinimalCache)
        assert cache.url == "redis://localhost"

    def test_var_kwargs_backend_receives_all_fields(self):
        cfg = CacheConfig.model_validate(
            {
                "enabled": True,
                "backend": "custom",
                "class": f"{__name__}.KwargsCache",
                "similarity_threshold": 0.8,
                "max_size": 10,
                "ttl_seconds": 5,
            }
        )
        cache = _build_query_cache(cfg)
        assert isinstance(cache, KwargsCache)
        assert cache.kwargs["similarity_threshold"] == 0.8
        assert cache.kwargs["max_size"] == 10
        assert cache.kwargs["ttl_seconds"] == 5
