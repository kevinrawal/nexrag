"""
MarkdownChunker — structure-aware splitting on the heading hierarchy.

Splits a Markdown document at ATX headings (``#`` … ``######``), emitting one
chunk per section and recording the full heading path (e.g. "Guide > Setup >
Install") in metadata["header_path"]. Oversized sections are split further on
blank lines up to chunk_size. High value for docs/wikis where the heading path is
strong retrieval context.

No external dependencies.
"""

from __future__ import annotations

import re
from typing import Any

from nexrag.chunkers._util import assemble_chunks, require_content
from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.models.chunk import Chunk
from nexrag.core.models.document import Document

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


class MarkdownChunker(BaseChunker):
    """
    Splits Markdown on heading hierarchy, preserving the header path in metadata.

    Args:
        chunk_size:     Sections longer than this are further split on blank lines. Default 512.
        min_chunk_size: Chunks shorter than this are dropped. Default 1.
    """

    def __init__(self, chunk_size: int = 512, min_chunk_size: int = 1) -> None:
        self._chunk_size = chunk_size
        self._min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        text = require_content(document, "MarkdownChunker")

        texts: list[str] = []
        metas: list[dict[str, Any]] = []
        for header_path, section in self._sections(text):
            for piece in self._split_oversized(section):
                texts.append(piece)
                metas.append({"header_path": header_path} if header_path else {})

        return assemble_chunks(
            texts,
            document,
            min_chunk_size=self._min_chunk_size,
            component="MarkdownChunker",
            per_chunk_metadata=metas,
        )

    def _sections(self, text: str) -> list[tuple[str, str]]:
        """Yield (header_path, section_text) pairs. Section text includes its heading line."""
        sections: list[tuple[str, str]] = []
        stack: list[tuple[int, str]] = []  # (level, title)
        current_lines: list[str] = []
        current_path = ""

        def flush() -> None:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_path, body))

        for line in text.splitlines():
            match = _HEADING_RE.match(line)
            if match:
                flush()
                current_lines = []
                level = len(match.group(1))
                title = match.group(2).strip()
                # Pop deeper-or-equal headings, then push this one.
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                current_path = " > ".join(t for _, t in stack)
                current_lines.append(line)
            else:
                current_lines.append(line)
        flush()
        return sections

    def _split_oversized(self, section: str) -> list[str]:
        """Split a section that exceeds chunk_size on blank lines, packing to chunk_size."""
        if len(section) <= self._chunk_size:
            return [section]
        paragraphs = [p for p in re.split(r"\n\s*\n", section) if p.strip()]
        pieces: list[str] = []
        current: list[str] = []
        current_len = 0
        for para in paragraphs:
            add = len(para) + (2 if current else 0)
            if current and current_len + add > self._chunk_size:
                pieces.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(para)
            current_len += add
        if current:
            pieces.append("\n\n".join(current))
        return pieces
