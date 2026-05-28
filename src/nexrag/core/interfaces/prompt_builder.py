"""
BasePromptBuilder — contract for prompt assembly.

The PromptBuilder takes the user query and retrieved chunks and assembles
the final prompt string that gets sent to the LLM.

Three slots, always in this order:
    [system_prompt]   Who the LLM is. Defined in nexrag.yaml.
    [context_block]   Retrieved chunks injected here.
    [user_query]      The original user question.

Why is this an interface?
    Different use cases need different prompt shapes:
    - Q&A: "Answer using only the context below."
    - Summarization: "Summarize the following documents."
    - Extraction: "Extract all dates from the context."
    Users declare a custom class in YAML to get full control.

V1 built-in templates: default, qa, summarize.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nexrag.core.models.chunk import ScoredChunk


class BasePromptBuilder(ABC):
    """Abstract base class for all NexRAG prompt builders."""

    @abstractmethod
    def build(self, query: str, chunks: list[ScoredChunk]) -> str:
        """
        Assemble the final prompt string from a query and retrieved chunks.

        Args:
            query:  The original user query string.
            chunks: Retrieved chunks ordered by relevance (rank 1 = most relevant).

        Returns:
            The complete prompt string to pass to the LLM.
            The format (chat messages vs raw string) depends on what
            the paired LLM adapter expects — document this in your implementation.

        Raises:
            PromptError: If template rendering fails.
        """
