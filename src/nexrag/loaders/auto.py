"""
AutoLoader — selects the appropriate loader based on file extension.

Dispatches to PDFLoader (.pdf) or RawTextLoader (.txt, .md).
Raises LoaderError for unknown extensions with a clear message.

This is used when loader.type is set to 'auto' in nexrag.yaml.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.models.document import Document
from nexrag.exceptions import LoaderError

_EXTENSION_MAP = {
    ".pdf": "nexrag.loaders.pdf.PDFLoader",
    ".txt": "nexrag.loaders.raw.RawTextLoader",
    ".md": "nexrag.loaders.raw.RawTextLoader",
}


class AutoLoader(BaseLoader):
    """
    Extension-based loader dispatcher.

    Supported extensions: .pdf, .txt, .md
    """

    def load(self, source: Any) -> list[Document]:
        ext = Path(str(source)).suffix.lower()

        if ext == ".pdf":
            from nexrag.loaders.pdf import PDFLoader

            return PDFLoader().load(source)

        if ext in (".txt", ".md"):
            from nexrag.loaders.raw import RawTextLoader

            return RawTextLoader().load(source)

        supported = ", ".join(sorted(_EXTENSION_MAP))
        raise LoaderError(
            f"Cannot auto-detect loader for extension {ext!r}. "
            f"Supported extensions: {supported}. "
            "Use an explicit loader type (pdf, txt, custom) in nexrag.yaml.",
            stage="loader",
            component="AutoLoader",
        )
