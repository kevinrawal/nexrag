"""
ModelGuard — optional model-based moderation tier (Llama-Guard / NeMo style).

Calls a moderation LLM (supplied as a nested ``llm`` sub-config, resolved
independently of the pipeline LLM) to classify the text as safe/unsafe. This is the
highest-assurance guard and the most expensive — it adds a full LLM round-trip per
check. Not free; opt-in. See SECURITY.md.

The default prompt asks the model to answer SAFE / UNSAFE; ``unsafe_pattern``
controls how the response is interpreted. Swap the prompt/model for a dedicated
Llama-Guard deployment when you have one.
"""

from __future__ import annotations

import re

from nexrag.core.interfaces.guard import BaseGuard, GuardContext, GuardResult
from nexrag.core.interfaces.llm import BaseLLM

_DEFAULT_PROMPT = (
    "You are a content-safety classifier. Decide whether the CONTENT below is safe "
    "to process in a retrieval-augmented assistant. Respond with exactly one word: "
    "SAFE or UNSAFE.\n\n---\n\nCONTENT:\n{text}"
)


class ModelGuard(BaseGuard):
    """
    Args:
        llm:            Moderation LLM (independent of the pipeline LLM).
        prompt:         Prompt template containing a ``{text}`` placeholder.
        unsafe_pattern: Regex; if it matches the model's response, the text is BLOCKED.
                        Default matches a leading "unsafe".
    """

    name = "model"

    def __init__(
        self,
        llm: BaseLLM,
        prompt: str = _DEFAULT_PROMPT,
        unsafe_pattern: str = r"^\s*unsafe\b",
    ) -> None:
        if "{text}" not in prompt:
            raise ValueError("ModelGuard prompt must contain a '{text}' placeholder.")
        self._llm = llm
        self._prompt = prompt
        self._unsafe = re.compile(unsafe_pattern, re.IGNORECASE)

    def check(self, text: str, context: GuardContext) -> GuardResult:
        if not text.strip():
            return GuardResult.allow()
        answer, _usage = self._llm.generate(self._prompt.format(text=text))
        if self._unsafe.search(answer.strip()):
            return GuardResult.block(
                reason=f"Model guard flagged content as unsafe: {answer.strip()[:80]!r}."
            )
        return GuardResult.allow()
