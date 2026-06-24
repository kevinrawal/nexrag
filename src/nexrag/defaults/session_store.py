"""
InMemorySessionStore — the default, process-local conversation store.

Keeps each session's turns in a list, guarded by a lock for concurrent access.
TTL expiry is evaluated lazily on read: ``get_history`` drops turns older than
``ttl_seconds`` so callers never see stale context and memory is reclaimed as
sessions are touched. A ``persist=False`` mode makes the store a no-op writer for
privacy-sensitive deployments that must never retain history beyond a request.

Single-process: each replica has its own sessions. For shared/long-lived history
across replicas, plug a persistent backend (e.g. Redis) via ``query.session.class``.
"""

from __future__ import annotations

import threading
import time

from nexrag.core.interfaces.session_store import BaseSessionStore
from nexrag.core.models.conversation import ConversationTurn, Role


class InMemorySessionStore(BaseSessionStore):
    """
    In-memory session store with lazy TTL expiry.

    Args:
        ttl_seconds: Turns older than this (by ``created_at``) are dropped on read.
                     0 or None disables expiry. Default 1800 (30 minutes).
        persist:     When False, ``append`` is a no-op and history is always empty —
                     the request never writes history (privacy mode). Default True.
    """

    def __init__(self, ttl_seconds: float | None = 1800, persist: bool = True) -> None:
        self._ttl = float(ttl_seconds) if ttl_seconds else 0.0
        self._persist = persist
        self._lock = threading.Lock()
        self._sessions: dict[str, list[ConversationTurn]] = {}

    def get_history(self, session_id: str) -> list[ConversationTurn]:
        if not self._persist:
            return []
        cutoff = (time.time() - self._ttl) if self._ttl else None
        with self._lock:
            turns = self._sessions.get(session_id)
            if not turns:
                return []
            if cutoff is not None:
                turns = [t for t in turns if t.created_at >= cutoff]
                if turns:
                    self._sessions[session_id] = turns
                else:
                    self._sessions.pop(session_id, None)
            return list(turns)

    def append(self, session_id: str, role: Role, content: str) -> ConversationTurn:
        turn = ConversationTurn(role=role, content=content)
        if not self._persist:
            return turn
        with self._lock:
            self._sessions.setdefault(session_id, []).append(turn)
        return turn

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)

    def delete_before(self, session_id: str, timestamp: float) -> int:
        with self._lock:
            turns = self._sessions.get(session_id)
            if not turns:
                return 0
            kept = [t for t in turns if t.created_at >= timestamp]
            removed = len(turns) - len(kept)
            if kept:
                self._sessions[session_id] = kept
            else:
                self._sessions.pop(session_id, None)
            return removed
