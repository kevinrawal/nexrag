"""
BaseContextStrategy — contract for deciding how much conversation history to send.

Conversation history cannot grow without bound. Three independent pressures force
trimming on every turn:

    - the LLM **context window** is finite;
    - **cost** grows with every turn re-sent as input tokens;
    - **relevance decays** — stale turns dilute the prompt and degrade answers.

A strategy receives the full stored history and returns the subset to actually
include in the prompt for the current query. It never mutates the store — the
record is preserved; only what is *sent* is trimmed. Shipped strategies:

    - ``WindowStrategy`` — keep the last N turns. Simplest, zero extra cost.
    - ``TokenBudgetStrategy`` — keep the most recent turns that fit a token budget.

A custom strategy (e.g. LLM summarisation of older turns) plugs in via
``query.session.context_strategy.class``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nexrag.core.models.conversation import ConversationTurn


class BaseContextStrategy(ABC):
    """Abstract base class for conversation context-management strategies."""

    @abstractmethod
    def apply(self, history: list[ConversationTurn], current_query: str) -> list[ConversationTurn]:
        """
        Return the subset of ``history`` to include in the prompt for ``current_query``.

        Args:
            history:       Full stored history in turn order (oldest first).
            current_query: The user's current question (so strategies can budget
                           for it; the current query is never part of the returned
                           history).

        Returns:
            Turns to send, in turn order (oldest first). May be empty.
        """
