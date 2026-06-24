"""
BaseSessionStore — contract for persisting multi-turn conversation history.

A session is an ordered list of ConversationTurns keyed by an opaque session_id
the caller owns (a chat thread id, a websocket id, etc.). The store is the only
stateful part of the query path, so it is deliberately a thin, swappable interface:

    - ``InMemorySessionStore`` (default) — process-local, TTL-expiring. Fine for a
      single replica or development; lost on restart.
    - A persistent/shared backend (e.g. Redis) — plugged via ``query.session.class``
      for multi-replica deploys or long-lived history.

Retention is a first-class concern: some deployments want history kept for days,
others must forget it immediately for privacy/compliance. The store exposes both
TTL-based expiry (``get_history`` skips expired turns) and explicit deletion
(``clear`` / ``delete_before``) so callers can satisfy "forget my data" requests
that automated trimming never covers.

Context-window management (how much history to actually *send* to the LLM) is a
separate concern handled by BaseContextStrategy — the store keeps the record;
the strategy decides what fits.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nexrag.core.models.conversation import ConversationTurn, Role


class BaseSessionStore(ABC):
    """Abstract base class for all NexRAG session stores."""

    @abstractmethod
    def get_history(self, session_id: str) -> list[ConversationTurn]:
        """
        Return the full (non-expired) history for ``session_id`` in turn order.

        Returns an empty list for an unknown or fully expired session. Implementations
        must drop TTL-expired turns here so callers never see stale context.
        """

    @abstractmethod
    def append(self, session_id: str, role: Role, content: str) -> ConversationTurn:
        """Append a turn to ``session_id`` and return the stored ConversationTurn."""

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """Delete all turns for ``session_id``. Idempotent — unknown ids are a no-op."""

    @abstractmethod
    def delete_before(self, session_id: str, timestamp: float) -> int:
        """
        Delete turns in ``session_id`` whose ``created_at`` is older than ``timestamp``.

        Targeted removal of outdated context (compliance / pruning) without
        clearing the whole session. Returns the number of turns deleted.
        """
