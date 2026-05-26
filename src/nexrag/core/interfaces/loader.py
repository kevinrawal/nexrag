"""
BaseLoader — contract for all document loaders.
A Loader receives raw data in some format and returns a list of Documents.
"""

# TODO: required critical changes.
# Loader should be something, which load and convert the data into document.
# default behaviour of loader is having raw data, always.
# User can write a wrapper where they can fetch the data from source and then pass it to loader.
# loader can accept row data and the type of data.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nexrag.core.models.document import Document


class BaseLoader(ABC):
    """
    Abstract base class for all NexRAG document loaders.

    Subclass this to parse any data format into Documents.
    """

    @abstractmethod
    def load(self, data: Any) -> list[Document]:
        """
        Parse data into a list of Documents.

        data is whatever you fetched — its type depends on this loader's
        implementation.

        Every returned Document MUST have metadata["source"] set to a stable
        identifier for this content. The idempotency check depends on it.

        Args:
            data: Already-fetched content in whatever format this loader accepts.
                  Could be str, bytes, dict, list[dict], Path, or any other type.

        Returns:
            One or more Document objects.

        Raises:
            LoaderError: If data is the wrong type, malformed, or unparseable.
        """
