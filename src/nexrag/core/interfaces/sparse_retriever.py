"""
BaseSparseRetriever — abstract base for all sparse (keyword-based) retrieval strategies.

Subclasses BaseRetriever without adding new abstract methods. The separate class exists so
resolve_class() can validate that a user-supplied class is specifically a sparse retriever,
and so HybridRetriever can type-annotate its sparse slot clearly.

IS-A BaseRetriever — usable standalone or injected into HybridRetriever.

Built-in: BM25Retriever (nexrag.retrievers.sparse.bm25)
Custom:   extend this class, point retriever.sparse.class at your dotted path.
"""

from __future__ import annotations

from nexrag.core.interfaces.retriever import BaseRetriever


class BaseSparseRetriever(BaseRetriever):
    """
    Abstract base for sparse (keyword-based) retrieval strategies.

    Inherits the full BaseRetriever interface:
        retrieve(query, query_embedding, top_k, collection, score_threshold, filters)
        async_retrieve(...)  — default: wraps retrieve() in asyncio.to_thread

    Custom sparse classes do not need to accept vector_db unless their corpus source
    requires it — pass it via SparseConfig.params if needed.

    Example custom implementation::

        class MyTFIDFRetriever(BaseSparseRetriever):
            def __init__(self, index_path: str) -> None:
                ...
            def retrieve(self, query, query_embedding, top_k, collection,
                         score_threshold=0.0, filters=None):
                ...
    """

    # retrieve() remains abstract (inherited from BaseRetriever).
    # No new abstract methods — the interface contract is identical.
