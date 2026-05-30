"""
RawTextLoader — converts a plain string into a single Document.

Accepts str only. Fetch the text yourself before calling load():
    text = Path("file.txt").read_text()    # local file
    text = response.text                    # HTTP
    text = record["body"]                   # database

Usage:
    loader = RawTextLoader()
    docs = loader.load("Hello world. This is my text.")
    docs = loader.load(("My text", "my-doc-id"))   # with source override
"""

from __future__ import annotations

from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.models.document import Document
from nexrag.exceptions import LoaderError


class RawTextLoader(BaseLoader):
    """
    Loads a plain text string as a single Document.

    Args:
        source: Optional stable identifier for the document used by the idempotency
                check. When not set, no source is stored and every ingest always writes
                (safe default — two different texts never silently overwrite each other).
                Override per-call via load((text, "my-doc-id")).
    """

    def __init__(self, source: str | None = None) -> None:
        self._default_source = source

    def load(self, data: str | tuple[str, str]) -> list[Document]:
        """
        Args:
            data: Either a plain str, or a (text, source) tuple to override
                  the default source identifier. Passing a file path or bytes
                  raises LoaderError — read the text first.

        Returns:
            A list containing a single Document.

        Raises:
            LoaderError: If data is not a str or tuple, is empty, or is whitespace-only.
        """
        from pathlib import Path

        if isinstance(data, (bytes, Path)):
            raise LoaderError(
                f"RawTextLoader expects str. "
                f"Read the file first: text = Path('file.txt').read_text(). "
                f"Got: {type(data).__name__}",
                stage="loader",
                component="RawTextLoader",
            )

        text: str
        source: str | None
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

        metadata: dict[str, object] = {}
        if source is not None:
            metadata["source"] = source

        return [Document(content=text, metadata=metadata)]
