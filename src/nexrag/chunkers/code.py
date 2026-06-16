"""
CodeChunker — language-aware splitting on function/class boundaries.

Uses tree-sitter (via tree-sitter-language-pack) to split source code at top-level
definitions so functions and classes stay intact. When tree-sitter is not
installed it falls back to a regex/blank-line splitter, so the chunker always
works — just less precisely for non-Python languages.

Each chunk records metadata["language"].

Requires (for AST splitting): pip install "nexrag[code]"  (tree-sitter, tree-sitter-language-pack)
"""

from __future__ import annotations

import re
from typing import Any

from nexrag.chunkers._util import assemble_chunks, require_content
from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.models.chunk import Chunk
from nexrag.core.models.document import Document

# Tree-sitter node types that represent a top-level definition worth keeping whole.
_DEFINITION_NODE_TYPES: frozenset[str] = frozenset(
    {
        "function_definition",
        "function_declaration",
        "method_definition",
        "class_definition",
        "class_declaration",
        "decorated_definition",
        "impl_item",
        "struct_item",
        "enum_item",
    }
)

# Fallback: top-level Python definition starts (column 0).
_PY_DEF_RE = re.compile(r"^(?:@|def |async def |class )", re.MULTILINE)


class CodeChunker(BaseChunker):
    """
    Splits source code at function/class boundaries.

    Args:
        language:       Source language (e.g. "python", "javascript", "go"). Default "python".
        chunk_size:     Definitions larger than this are split on blank lines. Default 1500.
        min_chunk_size: Chunks shorter than this are dropped. Default 1.
    """

    def __init__(
        self,
        language: str = "python",
        chunk_size: int = 1500,
        min_chunk_size: int = 1,
    ) -> None:
        self._language = language
        self._chunk_size = chunk_size
        self._min_chunk_size = min_chunk_size

    def chunk(self, document: Document) -> list[Chunk]:
        text = require_content(document, "CodeChunker")
        pieces = self._tree_sitter_split(text)
        if pieces is None:
            pieces = self._fallback_split(text)

        # Split any oversized piece (e.g. a very long function) on blank lines.
        final: list[str] = []
        for piece in pieces:
            final.extend(self._split_oversized(piece))

        return assemble_chunks(
            final,
            document,
            min_chunk_size=self._min_chunk_size,
            component="CodeChunker",
            per_chunk_metadata=[{"language": self._language} for _ in final],
        )

    def _tree_sitter_split(self, text: str) -> list[str] | None:
        """Split on top-level definitions via tree-sitter. Returns None if unavailable."""
        try:
            from tree_sitter_language_pack import get_parser
        except ImportError:
            return None
        try:
            parser: Any = get_parser(self._language)
            tree = parser.parse(text.encode("utf-8"))
        except Exception:
            return None

        data = text.encode("utf-8")
        pieces: list[str] = []
        buffer_start: int | None = None
        buffer_end: int | None = None

        def flush_buffer() -> None:
            nonlocal buffer_start, buffer_end
            if buffer_start is not None and buffer_end is not None:
                segment = data[buffer_start:buffer_end].decode("utf-8", errors="ignore").strip()
                if segment:
                    pieces.append(segment)
            buffer_start = buffer_end = None

        for node in tree.root_node.children:
            segment = data[node.start_byte : node.end_byte].decode("utf-8", errors="ignore")
            if node.type in _DEFINITION_NODE_TYPES:
                flush_buffer()
                if segment.strip():
                    pieces.append(segment.strip())
            else:
                # Accumulate non-definition top-level nodes (imports, constants) together.
                if buffer_start is None:
                    buffer_start = node.start_byte
                buffer_end = node.end_byte
        flush_buffer()

        return pieces or None

    def _fallback_split(self, text: str) -> list[str]:
        """Regex/blank-line fallback when tree-sitter is unavailable."""
        # Split at top-level Python definition boundaries.
        boundaries = [m.start() for m in _PY_DEF_RE.finditer(text)]
        if boundaries:
            pieces: list[str] = []
            if boundaries[0] > 0:
                head = text[: boundaries[0]].strip()
                if head:
                    pieces.append(head)
            for i, start in enumerate(boundaries):
                end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
                segment = text[start:end].strip()
                if segment:
                    pieces.append(segment)
            return pieces
        # No definitions found — fall back to blank-line paragraphs.
        return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    def _split_oversized(self, piece: str) -> list[str]:
        if len(piece) <= self._chunk_size:
            return [piece]
        blocks = [b for b in re.split(r"\n\s*\n", piece) if b.strip()]
        out: list[str] = []
        current: list[str] = []
        current_len = 0
        for block in blocks:
            add = len(block) + (2 if current else 0)
            if current and current_len + add > self._chunk_size:
                out.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(block)
            current_len += add
        if current:
            out.append("\n\n".join(current))
        return out
