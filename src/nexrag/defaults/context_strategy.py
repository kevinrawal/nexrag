"""
Built-in context-management strategies.

Both are zero-extra-cost (no LLM calls): they only choose which already-stored
turns to send. An LLM summarisation strategy (which compresses old turns into a
rolling summary) is a heavier, opt-in follow-up that plugs in via
``query.session.context_strategy.class``.
"""

from __future__ import annotations

from nexrag.core.interfaces.context_strategy import BaseContextStrategy
from nexrag.core.models.conversation import ConversationTurn


class WindowStrategy(BaseContextStrategy):
    """
    Keep the most recent ``max_turns`` turns.

    "Turn" here means a single ConversationTurn (one user or one assistant message),
    so a max of 6 keeps roughly the last three exchanges. Simplest possible policy;
    constant cost; no token accounting.

    Args:
        max_turns: Maximum number of trailing turns to include. Default 6.
    """

    def __init__(self, max_turns: int = 6) -> None:
        self._max_turns = max(0, int(max_turns))

    def apply(self, history: list[ConversationTurn], current_query: str) -> list[ConversationTurn]:
        if self._max_turns == 0:
            return []
        return history[-self._max_turns :]


class TokenBudgetStrategy(BaseContextStrategy):
    """
    Keep the most recent turns that fit within a token budget.

    Walks history newest-first, accumulating an estimated token count, and stops
    before exceeding ``max_tokens``. Respects the LLM context window directly,
    which a fixed turn count cannot (turns vary wildly in length).

    Token counts are estimated as ``len(text) / chars_per_token`` — a deliberately
    cheap heuristic (no tokenizer dependency on the hot path). Set ``chars_per_token``
    to match your model's language/tokenizer if needed; the default of 4 is a good
    English approximation for GPT-style BPE tokenizers.

    Args:
        max_tokens:      Approximate token budget for included history. Default 2000.
        chars_per_token: Characters per token for the estimate. Default 4.0.
    """

    def __init__(self, max_tokens: int = 2000, chars_per_token: float = 4.0) -> None:
        self._max_tokens = max(0, int(max_tokens))
        self._chars_per_token = max(1.0, float(chars_per_token))

    def _estimate_tokens(self, text: str) -> int:
        return int(len(text) / self._chars_per_token) + 1

    def apply(self, history: list[ConversationTurn], current_query: str) -> list[ConversationTurn]:
        if self._max_tokens == 0:
            return []
        # Reserve budget for the current query so we never crowd it out.
        budget = self._max_tokens - self._estimate_tokens(current_query)
        if budget <= 0:
            return []
        kept: list[ConversationTurn] = []
        used = 0
        for turn in reversed(history):
            cost = self._estimate_tokens(turn.content)
            if used + cost > budget:
                break
            kept.append(turn)
            used += cost
        kept.reverse()  # restore oldest-first order
        return kept
