"""
PropositionChunker — LLM-based decomposition into atomic propositions.

An LLM rewrites the source text into a list of atomic, self-contained statements
("propositions"), each of which becomes one chunk. This produces the cleanest
retrieval units but is the most expensive strategy — one LLM call per text window.
Opt-in only.

The LLM is supplied as a NESTED component sub-config on the chunker, resolved
independently of the pipeline's query LLM — so you can decompose with a cheap model
and answer with a strong one.

Cost note: expect roughly (document tokens) of input + the propositions of output
PER document, per ingest. Budget accordingly before enabling on large corpora.
"""

from __future__ import annotations

import re

from nexrag.chunkers._util import assemble_chunks, require_content
from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.interfaces.llm import BaseLLM
from nexrag.core.models.chunk import Chunk
from nexrag.core.models.document import Document
from nexrag.exceptions import ChunkError

_DEFAULT_PROMPT = (
    "Decompose the following text into a list of clear, atomic, self-contained "
    "propositions. Each proposition must:\n"
    "- express a single fact or idea,\n"
    "- be understandable without the surrounding text (resolve pronouns and references),\n"
    "- preserve the original meaning.\n"
    "Return ONLY the propositions, one per line, with no numbering or bullets.\n\n"
    "---\n\n"
    "TEXT:\n{text}"
)

# Strip leading list markers like "1.", "-", "*", "•".
_LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


class PropositionChunker(BaseChunker):
    """
    Rewrites text into atomic propositions via an LLM; one proposition per chunk.

    Args:
        llm:            LLM used to generate propositions (independent of the query LLM).
        window_size:    Characters of source text per LLM call. Larger windows mean fewer,
                        cheaper calls but risk truncation. Default 2000.
        prompt:         Prompt template containing a ``{text}`` placeholder.
        min_chunk_size: Propositions shorter than this are dropped. Default 1.
    """

    def __init__(
        self,
        llm: BaseLLM,
        window_size: int = 2000,
        prompt: str = _DEFAULT_PROMPT,
        min_chunk_size: int = 1,
    ) -> None:
        if llm is None:
            raise ChunkError(
                "PropositionChunker requires an llm.",
                stage="chunker",
                component="PropositionChunker",
            )
        if "{text}" not in prompt:
            raise ChunkError(
                "PropositionChunker prompt must contain a '{text}' placeholder.",
                stage="chunker",
                component="PropositionChunker",
            )
        self._llm = llm
        self._window_size = window_size
        self._prompt = prompt
        self._min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        text = require_content(document, "PropositionChunker")

        propositions: list[str] = []
        for window in self._windows(text):
            propositions.extend(self._propositions_for(window))

        return assemble_chunks(
            propositions,
            document,
            min_chunk_size=self._min_chunk_size,
            component="PropositionChunker",
        )

    def _windows(self, text: str) -> list[str]:
        """Split into ~window_size character windows on paragraph boundaries."""
        text = text.strip()
        if len(text) <= self._window_size:
            return [text]
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
        windows: list[str] = []
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            add = len(para) + (2 if current else 0)
            if current and current_len + add > self._window_size:
                windows.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(para)
            current_len += add
        if current:
            windows.append("\n\n".join(current))
        return windows

    def _propositions_for(self, window: str) -> list[str]:
        prompt = self._prompt.format(text=window)
        answer, _usage = self._llm.generate(prompt)
        props: list[str] = []
        for line in answer.splitlines():
            cleaned = _LIST_MARKER_RE.sub("", line).strip()
            if cleaned:
                props.append(cleaned)
        return props
