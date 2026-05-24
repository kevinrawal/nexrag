"""
BaseEmbedder — contract for all embedding model adapters.

The Embedder is the only component shared between both pipelines:
    - Ingestion pipeline: embed(texts) to convert chunks to vectors.
    - Query pipeline:     embed_query(text) to convert the user query to a vector.

Both pipelines must use the same Embedder instance configured in nexrag.yaml.
A different model in query vs ingestion = vectors in different spaces = broken retrieval.
This is enforced at startup by the fingerprint check (Phase 1).

Built-in adapters (Phase 1):
    OpenAIEmbedder, HuggingFaceEmbedder, OllamaEmbedder

The model_name and dimensions properties are used by the fingerprint check
to detect embedding model changes between ingestion runs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseEmbedder(ABC):
    """Abstract base class for all NexRAG embedding adapters."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Identifier for the embedding model being used.
        e.g. "text-embedding-3-small", "nomic-embed-text"
        Used in the fingerprint stored alongside the vector collection.
        """

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """
        Output vector dimensionality.
        e.g. 1536 for text-embedding-3-small, 768 for nomic-embed-text.
        Used to validate vectors before writing to the DB.
        """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of texts.

        Args:
            texts: List of strings to embed. May be empty — return [] if so.

        Returns:
            List of vectors, one per input text, in the same order.
            Each vector has length == self.dimensions.

        Raises:
            EmbedderError: If the API call fails or returns unexpected output.
        """

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """
        Embed a single query string.

        Separate from embed() because some providers apply different
        processing for queries vs documents (e.g. Cohere input_type param).
        For providers that don't distinguish, implement as: return self.embed([text])[0]

        Args:
            text: The user query string.

        Returns:
            A single vector of length == self.dimensions.

        Raises:
            EmbedderError: If the API call fails.
        """
