"""
PromptInjectionGuard — heuristic prompt-injection / jailbreak detection.

Scans text against a denylist of common injection phrasings. Critically, it is
meant to run on BOTH the user query AND retrieved content — retrieved documents are
an injection vector (a poisoned chunk can carry "ignore previous instructions").

Honesty (per SECURITY.md): this catches common, well-known patterns. It is NOT a
complete defence against a determined adversary — pair it with the access-control
guard and, for higher assurance, a model-based guard.
"""

from __future__ import annotations

import re

from nexrag.core.interfaces.guard import BaseGuard, GuardContext, GuardResult

_DEFAULT_PATTERNS: list[str] = [
    r"ignore\s+(?:all\s+|the\s+)?(?:previous|prior|above)\s+instructions",
    r"disregard\s+(?:all\s+|the\s+)?(?:previous|prior|above)",
    r"forget\s+(?:everything|all\s+previous|your\s+instructions)",
    r"reveal\s+(?:your\s+)?(?:system\s+)?prompt",
    r"(?:print|show|repeat)\s+(?:your\s+)?(?:system\s+)?(?:prompt|instructions)",
    r"you\s+are\s+now\s+(?:a|an|in)\b",
    r"\bdeveloper\s+mode\b",
    r"\bjailbreak\b",
    r"\bDAN\b",
    r"pretend\s+(?:to\s+be|you\s+are)",
    r"act\s+as\s+(?:if|a|an|though)",
    r"override\s+(?:your\s+)?(?:safety|guidelines|instructions)",
]


class PromptInjectionGuard(BaseGuard):
    """
    Args:
        mode:     "block" (default) rejects the text; "redact" strips the offending spans.
        patterns: Extra regex patterns to add to the built-in denylist.
    """

    name = "prompt_injection"

    def __init__(self, mode: str = "block", patterns: list[str] | None = None) -> None:
        self._mode = mode
        all_patterns = list(_DEFAULT_PATTERNS) + list(patterns or [])
        self._regexes = [re.compile(p, re.IGNORECASE) for p in all_patterns]

    def check(self, text: str, context: GuardContext) -> GuardResult:
        matched = [rx for rx in self._regexes if rx.search(text)]
        if not matched:
            return GuardResult.allow()

        reason = "Possible prompt injection detected."
        if self._mode == "redact":
            redacted = text
            for rx in matched:
                redacted = rx.sub("[REDACTED]", redacted)
            return GuardResult.redact(redacted, reason=reason)
        return GuardResult.block(reason=reason)
