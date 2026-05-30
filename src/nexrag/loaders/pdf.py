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
from typing import Any

from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.models.document import Document
from nexrag.exceptions import LoaderError


def _parse_pdf_date(raw: str) -> str | None:
    """
    Convert a PDF date string (D:YYYYMMDDHHmmSS[OHH'mm']) to ISO 8601.
    Returns None if the string cannot be parsed rather than raising.
    """
    if not raw:
        return None
    s = raw.strip()
    if s.startswith("D:"):
        s = s[2:]
    digits = ""
    for ch in s:
        if ch.isdigit():
            digits += ch
        else:
            break
    if len(digits) < 8:
        return None
    try:
        year = digits[0:4]
        month = digits[4:6] if len(digits) >= 6 else "01"
        day = digits[6:8]
        hour = digits[8:10] if len(digits) >= 10 else "00"
        minute = digits[10:12] if len(digits) >= 12 else "00"
        second = digits[12:14] if len(digits) >= 14 else "00"
        return f"{year}-{month}-{day}T{hour}:{minute}:{second}"
    except Exception:
        return None


class PDFLoader(BaseLoader):
    """
    Loads a PDF file into a single Document containing all page text.

    By default extracts all available PDF metadata fields alongside page text.
    Fields are silently omitted when not present in the source PDF.

    Args:
        source_override:  If set, overrides the auto-detected source identifier.
                          Useful when loading bytes from a known URI (e.g. an S3 key).
        metadata_fields:  Whitelist of metadata field names to include. Default (None)
                          includes all available fields. Supported names:
                          author, title, subject, creator, producer,
                          created_at, modified_at, page_count.
        include_metadata: Set to False to suppress all metadata extraction.
                          Only metadata["source"] will be set on the Document.
    """

    def __init__(
        self,
        source_override: str | None = None,
        metadata_fields: list[str] | None = None,
        include_metadata: bool = True,
    ) -> None:
        self._source_override = source_override
        self._metadata_fields = metadata_fields
        self._include_metadata = include_metadata

    def load(self, data: str | Path | bytes) -> list[Document]:
        """
        Args:
            data: A file path (str or Path) or raw PDF bytes.

        Returns:
            A list containing one Document with all page text joined by double newlines.
            metadata["source"] is always set (resolved file path or "pdf_bytes").
            Additional metadata fields are included per include_metadata / metadata_fields config.

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
        elif isinstance(data, str | Path):
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
                f"PDF is encrypted. Decrypt it before passing to PDFLoader. source={source!r}",
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

        metadata: dict[str, Any] = {"source": source}
        metadata.update(self._extract_pdf_metadata(reader))

        return [Document(content="\n\n".join(page_texts), metadata=metadata)]

    def _extract_pdf_metadata(self, reader: object) -> dict[str, Any]:
        """
        Extract available metadata fields from the PDF reader.

        Returns an empty dict when include_metadata=False.
        Silently omits fields that are not present in the source PDF.
        Applies metadata_fields whitelist if configured.
        """
        if not self._include_metadata:
            return {}

        pages = getattr(reader, "pages", [])
        result: dict[str, Any] = {"page_count": len(pages)}

        meta = getattr(reader, "metadata", None)
        if meta is not None:
            # String fields — try pypdf attribute access first, fall back to dict key
            for attr, key in [
                ("author", "author"),
                ("title", "title"),
                ("subject", "subject"),
                ("creator", "creator"),
                ("producer", "producer"),
            ]:
                val = getattr(meta, attr, None)
                if not val:
                    val = meta.get(f"/{attr.capitalize()}")
                if val:
                    result[key] = str(val)

            # Date fields — pypdf ≥ 3.x returns datetime objects via .creation_date /
            # .modification_date; older versions return raw D:... strings via dict access.
            for attr, dict_key, output_key in [
                ("creation_date", "/CreationDate", "created_at"),
                ("modification_date", "/ModDate", "modified_at"),
            ]:
                val = getattr(meta, attr, None)
                if val is not None:
                    if hasattr(val, "isoformat"):
                        result[output_key] = val.isoformat()
                    else:
                        parsed = _parse_pdf_date(str(val))
                        if parsed:
                            result[output_key] = parsed
                else:
                    raw = meta.get(dict_key)
                    if raw:
                        parsed = _parse_pdf_date(str(raw))
                        if parsed:
                            result[output_key] = parsed

        if self._metadata_fields is not None:
            result = {k: v for k, v in result.items() if k in self._metadata_fields}

        return result
