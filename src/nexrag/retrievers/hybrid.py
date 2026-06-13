"""
HybridRetriever — fuses dense vector scores with sparse keyword scores.

Combines the complementary strengths of semantic (dense) and keyword (sparse)
retrieval using a configurable alpha weight:

    fused_score = alpha * dense_score + (1 - alpha) * sparse_score

Both score lists are min-max normalized to [0, 1] before fusion.

  alpha = 1.0  →  pure dense  (identical to DenseRetriever)
  alpha = 0.0  →  pure sparse (identical to the injected sparse retriever)
  alpha = 0.7  →  recommended default: 70% semantic, 30% keyword

The sparse component is pluggable — any BaseSparseRetriever can be injected via
the factory (retriever.sparse.provider in YAML). Defaults to BM25Retriever.

Async: dense and sparse sub-retrievers run concurrently via asyncio.gather.
"""

from __future__ import annotations

import asyncio
from typing import Any

from nexrag.core.interfaces.retriever import BaseRetriever
from nexrag.core.interfaces.sparse_retriever import BaseSparseRetriever
from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.models.chunk import ScoredChunk
from nexrag.retrievers.dense import DenseRetriever
from nexrag.retrievers.sparse.bm25 import _normalize_scores


class HybridRetriever(BaseRetriever):
    """
    Retrieves chunks by fusing dense vector search with sparse keyword scores.

    Args:
        vector_db:    Vector DB used by the DenseRetriever (and by the default BM25
                      sparse retriever when no sparse retriever is injected).
        alpha:        Weight for the dense score. 0.0–1.0. Default 0.7.
        sparse_top_k: Candidate set size for the sparse sub-retriever. If None,
                      uses the same top_k as the dense path. Set higher (e.g. 3x)
                      to improve recall before fusion.
        sparse:       Pre-built BaseSparseRetriever to use. Defaults to
                      BM25Retriever(vector_db=vector_db) when not supplied.
    """

    def __init__(
        self,
        vector_db: BaseVectorDB,
        alpha: float = 0.7,
        sparse_top_k: int | None = None,
        sparse: BaseSparseRetriever | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha!r}")
        self._dense = DenseRetriever(vector_db=vector_db)
        if sparse is None:
            from nexrag.retrievers.sparse.bm25 import BM25Retriever

            self._sparse: BaseSparseRetriever = BM25Retriever(vector_db=vector_db)
        else:
            self._sparse = sparse
        self._alpha = alpha
        self._sparse_top_k = sparse_top_k

    def retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        collection: str,
        score_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """
        Retrieve chunks by fusing dense and sparse scores.

        Metadata filters are applied to BOTH the dense and sparse paths so the
        fused result set can never include a chunk that violates the filter —
        critical for multi-tenant isolation, where a sparse-only keyword match
        must not leak a chunk from another tenant into the results.

        Args:
            query:           Raw query string — passed to the sparse retriever.
            query_embedding: Pre-computed query vector — passed to DenseRetriever.
            top_k:           Final number of chunks to return after fusion.
            collection:      Vector collection to search.
            score_threshold: Applied to fused scores. 0.0 returns all results.
            filters:         Applied to both the dense and sparse paths.

        Returns:
            List of ScoredChunk ordered by fused score descending, trimmed to top_k.
        """
        sparse_k = self._sparse_top_k if self._sparse_top_k is not None else top_k

        dense_results = self._dense.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            collection=collection,
            score_threshold=0.0,
            filters=filters,
        )
        sparse_results = self._sparse.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=sparse_k,
            collection=collection,
            score_threshold=0.0,
            filters=filters,
        )

        return self._fuse(dense_results, sparse_results, top_k, score_threshold)

    async def async_retrieve(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int,
        collection: str,
        score_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[ScoredChunk]:
        """
        Concurrent async retrieval: dense and sparse run simultaneously via asyncio.gather.

        Both sub-retrievers' async_retrieve() are awaited concurrently. Custom sparse
        retrievers that override async_retrieve() with a native async implementation
        (e.g. an Elasticsearch async client) get true async parallelism here.
        """
        sparse_k = self._sparse_top_k if self._sparse_top_k is not None else top_k

        dense_results, sparse_results = await asyncio.gather(
            self._dense.async_retrieve(
                query=query,
                query_embedding=query_embedding,
                top_k=top_k,
                collection=collection,
                score_threshold=0.0,
                filters=filters,
            ),
            self._sparse.async_retrieve(
                query=query,
                query_embedding=query_embedding,
                top_k=sparse_k,
                collection=collection,
                score_threshold=0.0,
                filters=filters,
            ),
        )

        return self._fuse(dense_results, sparse_results, top_k, score_threshold)

    def _fuse(
        self,
        dense: list[ScoredChunk],
        sparse: list[ScoredChunk],
        top_k: int,
        score_threshold: float,
    ) -> list[ScoredChunk]:
        """
        Fuse two scored lists by normalizing then applying alpha weighting.

        Both lists are keyed by content_hash. Chunks that only appear in one
        list get score 0.0 from the absent list.
        """
        dense_map: dict[str, float] = {sc.chunk.content_hash: sc.score for sc in dense}
        sparse_map: dict[str, float] = {sc.chunk.content_hash: sc.score for sc in sparse}
        chunk_map: dict[str, ScoredChunk] = {sc.chunk.content_hash: sc for sc in dense + sparse}

        all_hashes = list(chunk_map.keys())

        dense_scores = _normalize_scores([dense_map.get(h, 0.0) for h in all_hashes])
        sparse_scores = _normalize_scores([sparse_map.get(h, 0.0) for h in all_hashes])

        fused: list[tuple[str, float]] = []
        for h, d_score, s_score in zip(all_hashes, dense_scores, sparse_scores, strict=True):
            fused_score = self._alpha * d_score + (1.0 - self._alpha) * s_score
            fused.append((h, fused_score))

        fused.sort(key=lambda x: x[1], reverse=True)

        results: list[ScoredChunk] = []
        for rank, (content_hash, score) in enumerate(fused[:top_k], start=1):
            if score_threshold > 0.0 and score < score_threshold:
                break
            original = chunk_map[content_hash]
            results.append(ScoredChunk(chunk=original.chunk, score=score, rank=rank))

        return results
