"""Unit tests for InMemoryQueryCache and make_cache_key."""

import asyncio

from nexrag.core.interfaces.query_cache import make_cache_key
from nexrag.core.models.result import PipelineResult
from nexrag.defaults.query_cache import InMemoryQueryCache


def _result(answer: str = "a") -> PipelineResult:
    return PipelineResult(
        answer=answer,
        query="q",
        sources=[],
        scores=[],
        collection_used="default",
        latency_ms=1.0,
        pipeline_id="pid",
    )


class TestMakeCacheKey:
    def test_deterministic(self):
        kwargs = dict(
            collection="c", top_k=5, score_threshold=0.0, metadata_filter=None, auth_context=None
        )
        assert make_cache_key("hello", **kwargs) == make_cache_key("hello", **kwargs)

    def test_case_and_whitespace_insensitive(self):
        kwargs = dict(
            collection="c", top_k=5, score_threshold=0.0, metadata_filter=None, auth_context=None
        )
        assert make_cache_key("  Hello ", **kwargs) == make_cache_key("hello", **kwargs)

    def test_different_query_differs(self):
        kwargs = dict(
            collection="c", top_k=5, score_threshold=0.0, metadata_filter=None, auth_context=None
        )
        assert make_cache_key("a", **kwargs) != make_cache_key("b", **kwargs)

    def test_auth_context_changes_key(self):
        base = dict(collection="c", top_k=5, score_threshold=0.0, metadata_filter=None)
        k1 = make_cache_key("q", auth_context={"tenant": "acme"}, **base)
        k2 = make_cache_key("q", auth_context={"tenant": "globex"}, **base)
        assert k1 != k2

    def test_collection_changes_key(self):
        base = dict(top_k=5, score_threshold=0.0, metadata_filter=None, auth_context=None)
        assert make_cache_key("q", collection="a", **base) != make_cache_key(
            "q", collection="b", **base
        )


class TestInMemoryQueryCache:
    def test_set_then_get_hit(self):
        cache = InMemoryQueryCache()
        r = _result()
        cache.set("k", r, collection="c")
        assert cache.get("k", collection="c") is r

    def test_miss_returns_none(self):
        cache = InMemoryQueryCache()
        assert cache.get("nope", collection="c") is None

    def test_invalidate_drops_entry(self):
        cache = InMemoryQueryCache()
        cache.set("k", _result(), collection="c")
        cache.invalidate("c")
        assert cache.get("k", collection="c") is None

    def test_invalidate_is_per_collection(self):
        cache = InMemoryQueryCache()
        cache.set("k1", _result("x"), collection="c1")
        cache.set("k2", _result("y"), collection="c2")
        cache.invalidate("c1")
        assert cache.get("k1", collection="c1") is None
        assert cache.get("k2", collection="c2") is not None

    def test_ttl_expiry(self, monkeypatch):
        clock = {"t": 100.0}
        monkeypatch.setattr("nexrag.defaults.query_cache.time.monotonic", lambda: clock["t"])
        cache = InMemoryQueryCache(ttl_seconds=10)
        cache.set("k", _result(), collection="c")
        clock["t"] += 5
        assert cache.get("k", collection="c") is not None
        clock["t"] += 6  # now 11s elapsed > 10s ttl
        assert cache.get("k", collection="c") is None

    def test_ttl_zero_disables_expiry(self, monkeypatch):
        clock = {"t": 0.0}
        monkeypatch.setattr("nexrag.defaults.query_cache.time.monotonic", lambda: clock["t"])
        cache = InMemoryQueryCache(ttl_seconds=0)
        cache.set("k", _result(), collection="c")
        clock["t"] += 10_000
        assert cache.get("k", collection="c") is not None

    def test_lru_eviction(self):
        cache = InMemoryQueryCache(max_size=2)
        cache.set("a", _result("a"), collection="c")
        cache.set("b", _result("b"), collection="c")
        cache.get("a", collection="c")  # touch a → b is now LRU
        cache.set("c", _result("c"), collection="c")  # evicts b
        assert cache.get("a", collection="c") is not None
        assert cache.get("b", collection="c") is None
        assert cache.get("c", collection="c") is not None

    def test_len_reflects_entries(self):
        cache = InMemoryQueryCache()
        assert len(cache) == 0
        cache.set("k", _result(), collection="c")
        assert len(cache) == 1


class TestAsyncCacheInterface:
    """aget/aset default to the sync methods so sync-only backends work on async paths (#57)."""

    def test_aset_then_aget_hit(self):
        cache = InMemoryQueryCache()
        r = _result()

        async def run() -> None:
            await cache.aset("k", r, collection="c")
            got = await cache.aget("k", collection="c")
            assert got is r

        asyncio.run(run())

    def test_aget_miss_returns_none(self):
        cache = InMemoryQueryCache()

        async def run() -> None:
            assert await cache.aget("nope", collection="c") is None

        asyncio.run(run())
