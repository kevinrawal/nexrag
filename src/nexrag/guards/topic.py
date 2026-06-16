"""
TopicGuard — config-driven allow/deny topic filtering.

A lightweight gate on the user query: block queries that mention any denied
keyword, and (optionally) block queries that mention none of an allowed set.
Keyword matching is whole-word and case-insensitive. Optional only.
"""

from __future__ import annotations

import re

from nexrag.core.interfaces.guard import BaseGuard, GuardContext, GuardResult


class TopicGuard(BaseGuard):
    """
    Args:
        deny:  Keywords that, if present, BLOCK the query.
        allow: If non-empty, the query is blocked unless it contains at least one of these.
    """

    name = "topic"

    def __init__(self, deny: list[str] | None = None, allow: list[str] | None = None) -> None:
        self._deny = [self._compile(k) for k in (deny or [])]
        self._allow = [self._compile(k) for k in (allow or [])]

    @staticmethod
    def _compile(keyword: str) -> re.Pattern[str]:
        return re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)

    def check(self, text: str, context: GuardContext) -> GuardResult:
        for rx in self._deny:
            if rx.search(text):
                return GuardResult.block(reason=f"Query matches a denied topic: {rx.pattern!r}.")

        if self._allow and not any(rx.search(text) for rx in self._allow):
            return GuardResult.block(reason="Query does not match any allowed topic.")

        return GuardResult.allow()
