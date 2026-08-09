"""Unit tests for InMemoryQueryCache and BaseQueryCache.make_key."""

import asyncio

from nexrag.caches.memory import InMemoryQueryCache
from nexrag.core.interfaces.query_cache import CacheLookup
from nexrag.core.models.result import PipelineResult


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


def _lookup(query: str = "q", **overrides) -> CacheLookup:
    fields = dict(
        query=query,
        collection="c",
        top_k=5,
        score_threshold=0.0,
        metadata_filter=None,
        auth_context=None,
    )
    fields.update(overrides)
    return CacheLookup(**fields)


class TestMakeKey:
    def test_deterministic(self):
        cache = InMemoryQueryCache()
        assert cache.make_key(_lookup("hello")) == cache.make_key(_lookup("hello"))

    def test_case_and_whitespace_insensitive(self):
        cache = InMemoryQueryCache()
        assert cache.make_key(_lookup("  Hello ")) == cache.make_key(_lookup("hello"))

    def test_different_query_differs(self):
        cache = InMemoryQueryCache()
        assert cache.make_key(_lookup("a")) != cache.make_key(_lookup("b"))

    def test_auth_context_changes_key(self):
        cache = InMemoryQueryCache()
        k1 = cache.make_key(_lookup(auth_context={"tenant": "acme"}))
        k2 = cache.make_key(_lookup(auth_context={"tenant": "globex"}))
        assert k1 != k2

    def test_collection_changes_key(self):
        cache = InMemoryQueryCache()
        assert cache.make_key(_lookup(collection="a")) != cache.make_key(_lookup(collection="b"))


class TestInMemoryQueryCache:
    def test_set_then_get_hit(self):
        cache = InMemoryQueryCache()
        r = _result()
        cache.set(_lookup(), r)
        assert cache.get(_lookup()) is r

    def test_miss_returns_none(self):
        cache = InMemoryQueryCache()
        assert cache.get(_lookup("nope")) is None

    def test_invalidate_drops_entry(self):
        cache = InMemoryQueryCache()
        cache.set(_lookup(), _result())
        cache.invalidate("c")
        assert cache.get(_lookup()) is None

    def test_invalidate_is_per_collection(self):
        cache = InMemoryQueryCache()
        cache.set(_lookup("k1", collection="c1"), _result("x"))
        cache.set(_lookup("k2", collection="c2"), _result("y"))
        cache.invalidate("c1")
        assert cache.get(_lookup("k1", collection="c1")) is None
        assert cache.get(_lookup("k2", collection="c2")) is not None

    def test_ttl_expiry(self, monkeypatch):
        clock = {"t": 100.0}
        monkeypatch.setattr("nexrag.caches.memory.time.monotonic", lambda: clock["t"])
        cache = InMemoryQueryCache(ttl_seconds=10)
        cache.set(_lookup(), _result())
        clock["t"] += 5
        assert cache.get(_lookup()) is not None
        clock["t"] += 6  # now 11s elapsed > 10s ttl
        assert cache.get(_lookup()) is None

    def test_ttl_zero_disables_expiry(self, monkeypatch):
        clock = {"t": 0.0}
        monkeypatch.setattr("nexrag.caches.memory.time.monotonic", lambda: clock["t"])
        cache = InMemoryQueryCache(ttl_seconds=0)
        cache.set(_lookup(), _result())
        clock["t"] += 10_000
        assert cache.get(_lookup()) is not None

    def test_lru_eviction(self):
        cache = InMemoryQueryCache(max_size=2)
        cache.set(_lookup("a"), _result("a"))
        cache.set(_lookup("b"), _result("b"))
        cache.get(_lookup("a"))  # touch a → b is now LRU
        cache.set(_lookup("c"), _result("c"))  # evicts b
        assert cache.get(_lookup("a")) is not None
        assert cache.get(_lookup("b")) is None
        assert cache.get(_lookup("c")) is not None

    def test_len_reflects_entries(self):
        cache = InMemoryQueryCache()
        assert len(cache) == 0
        cache.set(_lookup(), _result())
        assert len(cache) == 1


class TestAsyncCacheInterface:
    """aget/aset/ainvalidate default to the sync methods so sync-only backends
    work on async paths (#57)."""

    def test_aset_then_aget_hit(self):
        cache = InMemoryQueryCache()
        r = _result()

        async def run() -> None:
            await cache.aset(_lookup(), r)
            got = await cache.aget(_lookup())
            assert got is r

        asyncio.run(run())

    def test_aget_miss_returns_none(self):
        cache = InMemoryQueryCache()

        async def run() -> None:
            assert await cache.aget(_lookup("nope")) is None

        asyncio.run(run())

    def test_ainvalidate_drops_entry(self):
        cache = InMemoryQueryCache()

        async def run() -> None:
            await cache.aset(_lookup(), _result())
            await cache.ainvalidate("c")
            assert await cache.aget(_lookup()) is None

        asyncio.run(run())
