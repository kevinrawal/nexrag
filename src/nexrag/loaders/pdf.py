"""
PDFLoader — extracts text from PDF files using pypdf.

Accepts:
  - str / Path  → reads from filesystem, sets metadata["source"] = resolved path
  - bytes       → reads from memory (e.g. downloaded from S3, received over HTTP)

All pages are merged into a single Document. Use RecursiveChunker to split it.

Requires: pip install "nexrag[pdf]"  (pypdf)
"""

from __future__ import annotations

import io
from pathlib import Path

from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.models.document import Document
from nexrag.exceptions import LoaderError


class PDFLoader(BaseLoader):
    """
    Loads a PDF file into a single Document containing all page text.

    Args:
        source_override: If set, overrides the auto-detected source identifier.
                         Useful when loading bytes from a known URI (e.g. an S3 key).
    """

    def __init__(self, source_override: str | None = None) -> None:
        self._source_override = source_override

    def load(self, data: str | Path | bytes) -> list[Document]:
        """
        Args:
            data: A file path (str or Path) or raw PDF bytes.

        Returns:
            A list containing one Document with all page text joined by double newlines.
            metadata["source"] is the resolved file path or "pdf_bytes".
            metadata["page_count"] is the number of pages.

        Raises:
            LoaderError: If pypdf is not installed, the file cannot be read,
                         the PDF is encrypted, or no text could be extracted.
        """
        try:
            from pypdf import PdfReader  # type: ignore[import-not-found]
        except ImportError as e:
            raise LoaderError(
                "pypdf is required for PDFLoader. "
                'Install it: pip install "nexrag[pdf]" or pip install pypdf',
                stage="loader",
                component="PDFLoader",
                cause=e,
            ) from e

        reader, source = self._open(data, PdfReader)
        return self._extract(reader, source)

    # TODO: rename the PdfReader type to avoid the type: ignore[import-not-found] in the signature
    def _open(self, data: str | Path | bytes, PdfReader: type) -> tuple:  # type: ignore[type-arg]
        if isinstance(data, bytes):
            source = self._source_override or "pdf_bytes"
            try:
                reader = PdfReader(io.BytesIO(data))
            except Exception as e:
                raise LoaderError(
                    f"Failed to parse PDF from bytes: {e}",
                    stage="loader",
                    component="PDFLoader",
                    cause=e,
                ) from e
        elif isinstance(data, (str, Path)):
            path = Path(data)
            source = self._source_override or str(path.resolve())
            if not path.exists():
                raise LoaderError(
                    f"PDF file not found: {path.resolve()}",
                    stage="loader",
                    component="PDFLoader",
                )
            try:
                reader = PdfReader(str(path))
            except Exception as e:
                raise LoaderError(
                    f"Failed to open PDF '{path}': {e}",
                    stage="loader",
                    component="PDFLoader",
                    cause=e,
                ) from e
        else:
            raise LoaderError(
                f"PDFLoader expects str, Path, or bytes. Got: {type(data).__name__}",
                stage="loader",
                component="PDFLoader",
            )

        if reader.is_encrypted:
            raise LoaderError(
                f"PDF is encrypted. Decrypt it before passing to PDFLoader. " f"source={source!r}",
                stage="loader",
                component="PDFLoader",
            )

        return reader, source

    def _extract(self, reader: object, source: str) -> list[Document]:
        page_texts: list[str] = []
        for page in reader.pages:  # type: ignore[attr-defined]
            text = page.extract_text()
            if text and text.strip():
                page_texts.append(text.strip())

        if not page_texts:
            raise LoaderError(
                f"PDF contains no extractable text. It may be a scanned image PDF "
                f"or have no text layer. source={source!r}",
                stage="loader",
                component="PDFLoader",
            )

        return [
            Document(
                content="\n\n".join(page_texts),
                metadata={
                    "source": source,
                    "page_count": len(reader.pages),  # type: ignore[attr-defined]
                },
            )
        ]
