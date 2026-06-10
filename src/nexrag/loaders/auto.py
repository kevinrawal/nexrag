"""
AutoLoader — detects data format from content and dispatches to the right loader.

Accepts bytes or str. No file paths — fetch the data yourself before passing it in.

Currently supported formats:
    bytes starting with %PDF  →  PDFLoader
    str                       →  RawTextLoader

Unsupported formats (images, JSON, Excel, etc.) raise LoaderError with a
message directing the caller to use a specific loader. Support grows as new loaders
are added to NexRAG.
"""

from __future__ import annotations

from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.models.document import Document
from nexrag.exceptions import LoaderError

_PDF_MAGIC = b"%PDF"


class AutoLoader(BaseLoader):
    """
    Content-type-aware loader dispatcher.

    Detects the data format from the content itself (magic bytes for binary
    formats, Python type for text), then delegates to the appropriate loader.

    Supports multi-format upload use cases: pass any supported bytes or str
    through the same pipeline without pre-selecting a loader.

    Args:
        source_override: Propagated to the delegated loader as source identifier.
    """

    def __init__(self, source_override: str | None = None) -> None:
        self._source_override = source_override

    def load(self, data: bytes | str) -> list[Document]:
        """
        Args:
            data: Raw content as bytes or str. File paths are not accepted.

        Returns:
            Documents produced by the detected loader.

        Raises:
            LoaderError: If data is not bytes or str, or if the bytes format is unsupported.
        """
        if isinstance(data, bytes):
            if data[:4] == _PDF_MAGIC:
                from nexrag.loaders.pdf import PDFLoader

                return PDFLoader(source_override=self._source_override).load(data)

            raise LoaderError(
                "AutoLoader could not identify the data format from its content. "
                f"Got bytes starting with {data[:8]!r}. "
                "Supported: PDF (bytes starting with %PDF). "
                "Use a specific loader: PDFLoader(bytes), RawTextLoader(str).",
                stage="loader",
                component="AutoLoader",
            )

        if isinstance(data, str):
            from nexrag.loaders.raw import RawTextLoader

            return RawTextLoader(source=self._source_override).load(data)

        raise LoaderError(
            f"AutoLoader expects bytes or str. Got: {type(data).__name__}. "
            "File paths are not accepted — fetch the data first.",
            stage="loader",
            component="AutoLoader",
        )
