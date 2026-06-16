"""
CitationGuard — cheap output groundedness check.

Verifies the LLM answer is lexically supported by the retrieved context (the
"cheap version" of groundedness). It measures the fraction of the answer's
significant words that appear in the retrieved sources; below a threshold the
answer is treated as ungrounded.

This is a heuristic, not a faithfulness judge. LLM-judged faithfulness (the
expensive version) is intentionally out of scope here — route that to the eval
harness, not the inline request path. See SECURITY.md.

Runs on the output chain (GuardContext.sources carries the retrieved chunk texts).
"""

from __future__ import annotations

import re

from nexrag.core.interfaces.guard import BaseGuard, GuardContext, GuardResult

_WORD_RE = re.compile(r"[A-Za-z0-9]+")


class CitationGuard(BaseGuard):
    """
    Args:
        min_overlap:  Minimum fraction of the answer's significant words that must
                      appear in the sources. Default 0.2.
        min_word_len: Words shorter than this are ignored (drops stopword noise). Default 4.
        mode:         "block" (default) rejects an ungrounded answer; "allow" only emits an event.
    """

    name = "groundedness"

    def __init__(
        self, min_overlap: float = 0.2, min_word_len: int = 4, mode: str = "block"
    ) -> None:
        self._min_overlap = min_overlap
        self._min_word_len = min_word_len
        self._mode = mode

    def check(self, text: str, context: GuardContext) -> GuardResult:
        sources = context.sources or []
        if not sources or not text.strip():
            return GuardResult.allow()  # nothing to ground against

        answer_words = self._significant_words(text)
        if not answer_words:
            return GuardResult.allow()

        source_words = set()
        for source in sources:
            source_words |= self._significant_words(source)

        overlap = sum(1 for w in answer_words if w in source_words) / len(answer_words)
        if overlap >= self._min_overlap:
            return GuardResult.allow()

        reason = (
            f"Answer appears ungrounded: only {overlap:.0%} of its terms are supported by the "
            f"retrieved context (threshold {self._min_overlap:.0%})."
        )
        if self._mode == "block":
            return GuardResult.block(reason=reason)
        return GuardResult.allow()

    def _significant_words(self, text: str) -> set[str]:
        return {w.lower() for w in _WORD_RE.findall(text) if len(w) >= self._min_word_len}
