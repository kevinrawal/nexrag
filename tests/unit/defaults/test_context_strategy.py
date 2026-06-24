"""Unit tests for WindowStrategy and TokenBudgetStrategy."""

from nexrag.core.models.conversation import ConversationTurn
from nexrag.defaults.context_strategy import TokenBudgetStrategy, WindowStrategy


def _turns(n: int) -> list[ConversationTurn]:
    return [
        ConversationTurn(role="user" if i % 2 == 0 else "assistant", content=f"turn{i}")
        for i in range(n)
    ]


class TestWindowStrategy:
    def test_keeps_last_n_turns(self):
        strat = WindowStrategy(max_turns=3)
        kept = strat.apply(_turns(10), "q")
        assert [t.content for t in kept] == ["turn7", "turn8", "turn9"]

    def test_shorter_than_window_returns_all(self):
        strat = WindowStrategy(max_turns=5)
        kept = strat.apply(_turns(2), "q")
        assert len(kept) == 2

    def test_zero_returns_empty(self):
        strat = WindowStrategy(max_turns=0)
        assert strat.apply(_turns(5), "q") == []

    def test_preserves_order(self):
        strat = WindowStrategy(max_turns=4)
        kept = strat.apply(_turns(6), "q")
        assert [t.content for t in kept] == ["turn2", "turn3", "turn4", "turn5"]


class TestTokenBudgetStrategy:
    def test_keeps_recent_within_budget(self):
        # chars_per_token=4 default; each "turnN" ~ 6 chars ~ 2 tokens (+1) ≈ 2-3 tokens.
        strat = TokenBudgetStrategy(max_tokens=100, chars_per_token=4)
        kept = strat.apply(_turns(5), "q")
        # all 5 short turns fit comfortably
        assert len(kept) == 5

    def test_drops_oldest_when_over_budget(self):
        long_turns = [ConversationTurn(role="user", content="x" * 400) for _ in range(5)]
        # each ~ 100 tokens; budget 250 fits ~2 newest
        strat = TokenBudgetStrategy(max_tokens=250, chars_per_token=4)
        kept = strat.apply(long_turns, "q")
        assert len(kept) <= 2
        assert kept == long_turns[-len(kept) :]  # newest retained, in order

    def test_preserves_oldest_first_order(self):
        strat = TokenBudgetStrategy(max_tokens=1000)
        kept = strat.apply(_turns(4), "q")
        assert [t.content for t in kept] == ["turn0", "turn1", "turn2", "turn3"]

    def test_current_query_reserves_budget(self):
        # A huge current query leaves no room for history.
        strat = TokenBudgetStrategy(max_tokens=10, chars_per_token=4)
        kept = strat.apply(_turns(5), "x" * 400)
        assert kept == []

    def test_zero_budget_returns_empty(self):
        strat = TokenBudgetStrategy(max_tokens=0)
        assert strat.apply(_turns(3), "q") == []
