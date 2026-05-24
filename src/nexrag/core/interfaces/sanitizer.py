"""
BaseSanitizer — contract for document sanitizers.

A Sanitizer receives a Document and returns a cleaned Document.
The logic is 100% user-defined. NexRAG provides a PassthroughSanitizer
(no-op) as the default when sanitizer.enabled: false in config.

Why sanitizers exist:
    Raw extracted text from PDFs/Word/HTML is often noisy — headers,
    footers, watermarks, boilerplate legal text, encoding artifacts.
    Sanitizers let users clean this before chunking without modifying
    their source files or their loaders.

Contract:
    - Always return a Document (never None, never raise on clean input).
    - Use document.with_content() to return a modified copy.
    - Never mutate the input Document (it's frozen).

Custom implementation pattern:
    class RemoveBoilerplateSanitizer(BaseSanitizer):
        def sanitize(self, document: Document) -> Document:
            cleaned = document.content.replace("CONFIDENTIAL", "")
            return document.with_content(cleaned.strip())
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from nexrag.core.models.document import Document


class BaseSanitizer(ABC):
    """Abstract base class for all NexRAG sanitizers."""

    @abstractmethod
    def sanitize(self, document: Document) -> Document:
        """
        Clean a Document and return a new Document with cleaned content.

        Args:
            document: The Document to sanitize.

        Returns:
            A new Document instance. Use document.with_content() or
            document.with_metadata() — never mutate in place.

        Raises:
            SanitizerError: If sanitization fails unexpectedly.
        """


class PassthroughSanitizer(BaseSanitizer):
    """
    Default no-op sanitizer.

    Used when sanitizer.enabled: false in nexrag.yaml.
    Returns the Document unchanged.
    """

    def sanitize(self, document: Document) -> Document:
        return document
