"""core.interfaces — abstract contracts for every pipeline component."""

from nexrag.core.interfaces.chunker import BaseChunker
from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.core.interfaces.llm import BaseLLM
from nexrag.core.interfaces.loader import BaseLoader
from nexrag.core.interfaces.observer import BaseObserver, NoOpObserver
from nexrag.core.interfaces.prompt_builder import BasePromptBuilder
from nexrag.core.interfaces.retriever import BaseRetriever
from nexrag.core.interfaces.sanitizer import BaseSanitizer, PassthroughSanitizer
from nexrag.core.interfaces.vector_db import BaseVectorDB

__all__ = [
    "BaseChunker",
    "BaseEmbedder",
    "BaseLLM",
    "BaseLoader",
    "BaseObserver",
    "BasePromptBuilder",
    "BaseRetriever",
    "BaseSanitizer",
    "BaseVectorDB",
    "NoOpObserver",
    "PassthroughSanitizer",
]
