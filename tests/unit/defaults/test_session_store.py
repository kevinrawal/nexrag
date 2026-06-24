"""Unit tests for InMemorySessionStore."""

import time

from nexrag.defaults.session_store import InMemorySessionStore


class TestInMemorySessionStore:
    def test_append_then_get(self):
        store = InMemorySessionStore()
        store.append("s1", "user", "hello")
        store.append("s1", "assistant", "hi there")
        history = store.get_history("s1")
        assert [t.content for t in history] == ["hello", "hi there"]
        assert [t.role for t in history] == ["user", "assistant"]

    def test_unknown_session_is_empty(self):
        store = InMemorySessionStore()
        assert store.get_history("nope") == []

    def test_sessions_are_isolated(self):
        store = InMemorySessionStore()
        store.append("s1", "user", "a")
        store.append("s2", "user", "b")
        assert [t.content for t in store.get_history("s1")] == ["a"]
        assert [t.content for t in store.get_history("s2")] == ["b"]

    def test_clear(self):
        store = InMemorySessionStore()
        store.append("s1", "user", "a")
        store.clear("s1")
        assert store.get_history("s1") == []

    def test_clear_unknown_is_noop(self):
        store = InMemorySessionStore()
        store.clear("nope")  # should not raise

    def test_ttl_expiry(self):
        # Real time + short TTL: the "old" turn ages past the TTL before "new" is added.
        store = InMemorySessionStore(ttl_seconds=0.05)
        store.append("s1", "user", "old")
        time.sleep(0.1)  # old is now older than the 0.05s TTL
        store.append("s1", "user", "new")
        history = store.get_history("s1")
        assert [t.content for t in history] == ["new"]

    def test_persist_false_never_stores(self):
        store = InMemorySessionStore(persist=False)
        store.append("s1", "user", "secret")
        assert store.get_history("s1") == []

    def test_delete_before(self):
        store = InMemorySessionStore()
        store.append("s1", "user", "old")
        cutoff = time.time() + 0.001
        time.sleep(0.005)
        store.append("s1", "user", "new")
        removed = store.delete_before("s1", cutoff)
        assert removed == 1
        assert [t.content for t in store.get_history("s1")] == ["new"]

    def test_delete_before_unknown_session(self):
        store = InMemorySessionStore()
        assert store.delete_before("nope", time.time()) == 0
