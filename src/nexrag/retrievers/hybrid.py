"""
HybridRetriever — fuses dense vector scores with BM25 keyword scores.

Combines the complementary strengths of semantic (dense) and keyword (BM25)
retrieval using a configurable alpha weight:

    fused_score = alpha * dense_score + (1 - alpha) * bm25_score

Both score lists are min-max normalized to [0, 1] before fusion.

  alpha = 1.0  →  pure dense  (identical to DenseRetriever)
  alpha = 0.0  →  pure BM25   (identical to BM25Retriever)
  alpha = 0.7  →  recommended default: 70% semantic, 30% keyword

Typical use: retrieve a larger BM25 candidate set, fuse with dense scores, and
return the top-k by fused score. This handles both semantic and exact-match queries.

Requires: pip install "nexrag[bm25]"  (rank-bm25, via BM25Retriever)
"""

from __future__ import annotations

from typing import Any

from nexrag.core.interfaces.retriever import BaseRetriever
from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.models.chunk import ScoredChunk
from nexrag.retrievers.bm25 import BM25Retriever, _normalize_scores
from nexrag.retrievers.dense import DenseRetriever


class HybridRetriever(BaseRetriever):
    """
    Retrieves chunks by fusing dense vector search with BM25 keyword scores.

    Args:
        vector_db:   Vector DB used by both sub-retrievers.
        alpha:       Weight for the dense score. 0.0–1.0. Default 0.7.
        bm25_top_k:  Candidate set size for BM25 sub-retriever. If None, uses
                     the same top_k as the dense path. Set higher than top_k
                     (e.g. 3x) to improve recall before fusion.
    """

    def __init__(
        self,
        vector_db: BaseVectorDB,
        alpha: float = 0.7,
        bm25_top_k: int | None = None,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be between 0.0 and 1.0, got {alpha!r}")
        self._dense = DenseRetriever(vector_db=vector_db)
        self._bm25 = BM25Retriever(vector_db=vector_db)
        self._alpha = alpha
        self._bm25_top_k = bm25_top_k

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
        Retrieve chunks by fusing dense and BM25 scores.

        Dense retrieval respects metadata filters. BM25 retrieval does NOT
        apply metadata filters (V1 limitation).

        Args:
            query:           Raw query string — passed to BM25.
            query_embedding: Pre-computed query vector — passed to DenseRetriever.
            top_k:           Final number of chunks to return after fusion.
            collection:      Vector collection to search.
            score_threshold: Applied to fused scores. 0.0 returns all results.
            filters:         Applied to dense path only.

        Returns:
            List of ScoredChunk ordered by fused score descending, trimmed to top_k.
        """
        bm25_k = self._bm25_top_k if self._bm25_top_k is not None else top_k

        dense_results = self._dense.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=top_k,
            collection=collection,
            score_threshold=0.0,
            filters=filters,
        )
        bm25_results = self._bm25.retrieve(
            query=query,
            query_embedding=query_embedding,
            top_k=bm25_k,
            collection=collection,
            score_threshold=0.0,
            filters=None,
        )

        return self._fuse(dense_results, bm25_results, top_k, score_threshold)

    def _fuse(
        self,
        dense: list[ScoredChunk],
        bm25: list[ScoredChunk],
        top_k: int,
        score_threshold: float,
    ) -> list[ScoredChunk]:
        """
        Fuse two scored lists by normalizing then applying alpha weighting.

        Both lists are keyed by content_hash. Chunks that only appear in one
        list get score 0.0 from the absent list.
        """
        dense_map: dict[str, float] = {sc.chunk.content_hash: sc.score for sc in dense}
        bm25_map: dict[str, float] = {sc.chunk.content_hash: sc.score for sc in bm25}
        chunk_map: dict[str, ScoredChunk] = {sc.chunk.content_hash: sc for sc in dense + bm25}

        all_hashes = list(chunk_map.keys())

        dense_scores = _normalize_scores([dense_map.get(h, 0.0) for h in all_hashes])
        bm25_scores = _normalize_scores([bm25_map.get(h, 0.0) for h in all_hashes])

        fused: list[tuple[str, float]] = []
        for h, d_score, b_score in zip(all_hashes, dense_scores, bm25_scores, strict=True):
            fused_score = self._alpha * d_score + (1.0 - self._alpha) * b_score
            fused.append((h, fused_score))

        fused.sort(key=lambda x: x[1], reverse=True)

        results: list[ScoredChunk] = []
        for rank, (content_hash, score) in enumerate(fused[:top_k], start=1):
            if score_threshold > 0.0 and score < score_threshold:
                break
            original = chunk_map[content_hash]
            results.append(ScoredChunk(chunk=original.chunk, score=score, rank=rank))

        return results
