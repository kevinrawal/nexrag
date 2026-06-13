"""
PDFLoader — converts PDF bytes into a Document.

Accepts bytes only. Fetch the bytes yourself before calling load():
    data = Path("file.pdf").read_bytes()          # local file
    data = s3_client.get_object(...)["Body"].read()  # S3
    data = response.content                         # HTTP

Requires: pip install "nexrag[pdf]"  (pypdf)
"""

from __future__ import annotations

import io
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
    except (ValueError, IndexError):
        return None


class PDFLoader(BaseLoader):
    """
    Converts PDF bytes into a single Document containing all page text.

    Accepts bytes only. File reading and path resolution are the caller's
    responsibility.

    Args:
        source_override:  Stable identifier for the content (used by idempotency
                          check). Set this to the origin URI, S3 key, or filename.
                          Defaults to "pdf_bytes" when not provided.
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

    def load(self, data: bytes) -> list[Document]:
        """
        Args:
            data: Raw PDF bytes. Must be bytes — passing a file path raises LoaderError.
                  To load from a file: loader.load(Path("file.pdf").read_bytes())

        Returns:
            A list containing one Document with all page text joined by double newlines.
            metadata["source"] is always set (source_override or "pdf_bytes").
            Additional metadata fields are extracted per include_metadata / metadata_fields.

        Raises:
            LoaderError: If data is not bytes, pypdf is not installed,
                         the PDF is encrypted, or no text could be extracted.
        """
        if not isinstance(data, bytes):
            raise LoaderError(
                f"PDFLoader expects bytes. "
                f"Read the file first: data = Path('file.pdf').read_bytes(). "
                f"Got: {type(data).__name__}",
                stage="loader",
                component="PDFLoader",
            )

        try:
            from pypdf import PdfReader
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

    def _open(self, data: bytes, pdf_reader_cls: type) -> tuple:  # type: ignore[type-arg]
        source = (
            self._source_override
        )  # None when unset; idempotency disabled until caller provides one
        try:
            reader = pdf_reader_cls(io.BytesIO(data))
        except Exception as e:
            raise LoaderError(
                f"Failed to parse PDF from bytes: {e}",
                stage="loader",
                component="PDFLoader",
                cause=e,
            ) from e

        if reader.is_encrypted:
            raise LoaderError(
                f"PDF is encrypted. Decrypt it before passing to PDFLoader. source={source!r}",
                stage="loader",
                component="PDFLoader",
            )

        return reader, source

    def _extract(self, reader: object, source: str | None) -> list[Document]:
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

        metadata: dict[str, Any] = {}
        if source is not None:
            metadata["source"] = source
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
