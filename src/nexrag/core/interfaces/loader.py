"""
BaseLoader — contract for all document loaders.

Loaders are converters, not fetchers. They accept already-fetched raw data
(bytes from S3/HTTP, str from a database, etc.) and return Document objects.

File I/O, HTTP calls, and path resolution are the caller's responsibility.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nexrag.core.models.document import Document


class BaseLoader(ABC):
    """
    Abstract base class for all NexRAG document loaders.

    Contract:
        - load() accepts raw data (bytes or str), never a file path
        - load() performs no file I/O or network calls
        - metadata["source"] is optional. When set it must be a stable identifier
          used by the idempotency check to detect re-ingestion. When absent the
          pipeline always writes (no dedup). Set it via source_override at
          construction or via the loader's per-call mechanism (e.g. tuple input).
    """

    @abstractmethod
    def load(self, data: Any) -> list[Document]:
        """
        Convert raw data into a list of Documents.

        Args:
            data: Already-fetched content. Type depends on the concrete loader:
                  PDFLoader expects bytes; RawTextLoader expects str.
                  Passing a file path raises LoaderError — read bytes first.

        Returns:
            One or more Document objects with metadata["source"] set.

        Raises:
            LoaderError: If data is the wrong type, malformed, or unparseable.
        """
