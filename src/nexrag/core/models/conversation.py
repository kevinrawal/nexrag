"""
ConversationTurn — one message in a multi-turn session.

Sessions are an ordered list of ConversationTurns. The query pipeline injects a
(trimmed) history into the prompt so the LLM can answer follow-up questions that
depend on earlier turns ("what about the second one?").

Turns are plain value objects: no behaviour, just (role, content, created_at).
Retrieval never uses history — only the current query is embedded — so history
shape has no effect on what is retrieved, only on what the LLM sees.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass(frozen=True)
class ConversationTurn:
    """
    A single turn in a conversation.

    Attributes:
        role:       "user" or "assistant".
        content:    The message text.
        created_at: Unix timestamp (seconds) when the turn was recorded. Used by
                    TTL expiry and delete_turns(before=...).
    """

    role: Role
    content: str
    created_at: float = field(default_factory=time.time)

    def __repr__(self) -> str:
        preview = self.content if len(self.content) <= 40 else self.content[:37] + "..."
        return f"ConversationTurn(role={self.role!r}, content={preview!r})"
