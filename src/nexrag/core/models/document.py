"""
First object in the pipeline. Created by a Loader, optionally transformed
by a Sanitizer, consumed by a Chunker.
"""

from __future__ import annotations

import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """
    Raw document as produced by a Loader.

    Attributes:
        content:  Full extracted text.
        metadata: Open dict for user-defined fields (vendor, year, dept…).
        doc_id:   Stable ID. Auto-generated if not supplied. Chunks carry
                  this as parent_doc_id so every chunk is traceable to its
                  source document.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    doc_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def with_content(self, new_content: str) -> Document:
        """
        Return a new Document with replaced content, all other fields intact.
        Sanitizers use this — they never mutate, they return a new instance.
        """
        return Document(
            content=new_content,
            metadata=self.metadata,
            doc_id=self.doc_id,
        )

    def with_metadata(self, extra: dict[str, Any]) -> Document:
        """
        Return a new Document with merged metadata. Top-level dict values are
        merged one level deep — nested keys from the original are preserved
        unless extra explicitly overwrites them.
        """
        merged = deepcopy(self.metadata)
        for key, value in extra.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        return Document(content=self.content, metadata=merged, doc_id=self.doc_id)

    def __repr__(self) -> str:
        preview = self.content[:60].replace("\n", " ")
        if len(self.content) > 60:
            preview += "..."
        return f"Document(doc_id={self.doc_id!r}, content={preview!r})"
