"""
DefaultPromptBuilder — assembles prompts from a system message + numbered context.

The prompt structure sent to the LLM:

    [system]  You are a helpful assistant...
    [user]    Context:
              [1] <chunk text>
              [2] <chunk text>
              ...
              Question: <user query>

This format works with OpenAI chat completions (gpt-4o) and most instruction-tuned
models. The system message and context_format are configurable.
"""

from __future__ import annotations

from nexrag.core.interfaces.prompt_builder import BasePromptBuilder
from nexrag.core.models.chunk import ScoredChunk
from nexrag.exceptions import PromptError

_DEFAULT_SYSTEM = (
    "You are a helpful assistant. Answer the user's question using only "
    "the context provided below. If the answer is not in the context, say so "
    "clearly. Do not make up information."
)


class DefaultPromptBuilder(BasePromptBuilder):
    """
    Builds a system + user message prompt for chat-style LLMs.

    The returned string is formatted as a two-part prompt that LLM adapters
    can split into system/user messages or use as-is for completion endpoints.

    Args:
        system:         The system prompt text. Defines the LLM's role and rules.
        context_format: How to label context chunks.
                        "numbered" → [1], [2], ...
                        "labeled"  → Source 1:, Source 2:, ...
                        "plain"    → no labels, chunks separated by blank lines.
    """

    _SEPARATOR = "\n\n---\n\n"

    def __init__(
        self,
        system: str = _DEFAULT_SYSTEM,
        context_format: str = "numbered",
    ) -> None:
        self._system = system.strip()
        self._context_format = context_format

    def build(self, query: str, chunks: list[ScoredChunk]) -> str:
        """
        Assemble the prompt.

        Args:
            query:  User's question.
            chunks: Retrieved chunks in relevance order.

        Returns:
            A string with system prompt, context block, and question.
            Format: "<SYSTEM>\\n\\n<CONTEXT>\\n\\nQuestion: <query>"

        Raises:
            PromptError: If query is empty.
        """
        if not query or not query.strip():
            raise PromptError(
                "Query is empty. Cannot build a prompt without a question.",
                stage="prompt_builder",
                component="DefaultPromptBuilder",
            )

        context_block = self._build_context(chunks)
        user_part = f"{context_block}\n\nQuestion: {query.strip()}"

        return f"{self._system}{self._SEPARATOR}{user_part}"

    @property
    def system_prompt(self) -> str:
        return self._system

    # Private helpers

    def _build_context(self, chunks: list[ScoredChunk]) -> str:
        if not chunks:
            return "Context: (no relevant documents found)"

        lines: list[str] = ["Context:"]

        for i, sc in enumerate(chunks, start=1):
            text = sc.chunk.text.strip()
            if self._context_format == "numbered":
                lines.append(f"[{i}] {text}")
            elif self._context_format == "labeled":
                lines.append(f"Source {i}: {text}")
            else:
                lines.append(text)
                if i < len(chunks):
                    lines.append("")

        return "\n".join(lines)
