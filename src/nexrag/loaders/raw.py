"""
RawTextLoader — accepts a plain string and returns a single Document.

Useful for:
  - In-memory text (programmatic ingestion without a file)
  - Testing and development (no file I/O needed)
  - API responses, database records, or any text already in memory

Usage:
    loader = RawTextLoader()
    docs = loader.load("Hello world. This is my text.")
    docs = loader.load("My text", source="my-doc-id")
"""

from __future__ import annotations

from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.models.document import Document
from nexrag.exceptions import LoaderError


class RawTextLoader(BaseLoader):
    """
    Loads a plain text string as a single Document.

    Args:
        source: Optional stable identifier for the document.
                Used by the idempotency check. Defaults to "raw_text".
                Override per-call via load(text, source="my-id").
    """

    def __init__(self, source: str = "raw_text") -> None:
        self._default_source = source

    def load(self, data: str | tuple[str, str]) -> list[Document]:
        """
        Args:
            data: Either a plain string, or a (text, source) tuple to override
                  the default source identifier.

        Returns:
            A list containing a single Document.

        Raises:
            LoaderError: If data is not a string, is empty, or is whitespace-only.
        """
        if isinstance(data, tuple):
            if len(data) != 2:  # noqa: PLR2004
                raise LoaderError(
                    "RawTextLoader tuple input must be (text, source). "
                    f"Got a tuple of length {len(data)}.",
                    stage="loader",
                    component="RawTextLoader",
                )
            text, source = data
        elif isinstance(data, str):
            text = data
            source = self._default_source
        else:
            raise LoaderError(
                f"RawTextLoader expects a str or (str, str) tuple. Got: {type(data).__name__}",
                stage="loader",
                component="RawTextLoader",
            )

        if not text or not text.strip():
            raise LoaderError(
                "RawTextLoader received empty or whitespace-only text.",
                stage="loader",
                component="RawTextLoader",
            )

        return [
            Document(
                content=text,
                metadata={"source": source},
            )
        ]
