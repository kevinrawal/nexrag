"""
BM25Retriever — keyword-based retrieval using BM25Okapi from rank_bm25.

Fetches the full chunk corpus from the vector DB at query time, builds a BM25
index, and scores the query against it. Returns results as ScoredChunk objects
with normalized BM25 scores.

V1 limitations (documented):
  - Corpus is fetched fresh on every retrieve() call. For large collections this
    can be slow. Future versions will cache/pre-build the index.
  - Metadata filters are not applied to the BM25 path. Pass filters to
    DenseRetriever via HybridRetriever for filter-aware hybrid retrieval.
  - Tokenization is whitespace + lowercase — English-tuned. Other languages
    work but may produce suboptimal ranking.

Requires: pip install "nexrag[bm25]"  (rank-bm25)
"""

from __future__ import annotations

from typing import Any

from nexrag.core.interfaces.retriever import BaseRetriever
from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.models.chunk import ScoredChunk
from nexrag.exceptions import RetrieverError


class BM25Retriever(BaseRetriever):
    """
    Retrieves chunks using BM25 keyword scoring over the full collection corpus.

    BM25 is complementary to dense retrieval — it excels at exact keyword matches
    where dense vectors may assign low similarity due to vocabulary mismatch.

    Args:
        vector_db: Any BaseVectorDB that implements get_all(). Used to fetch the
                   corpus at query time.

    V1 Note: corpus is fetched on every call. Not suitable for collections with
    millions of chunks without caching. See issue roadmap for pre-built index support.
    """

    def __init__(self, vector_db: BaseVectorDB) -> None:
        self._vector_db = vector_db

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
        Score all chunks in the collection by BM25 keyword relevance to the query.

        Args:
            query:           The raw query string used for BM25 scoring.
            query_embedding: Ignored — BM25 is keyword-only.
            top_k:           Maximum chunks to return (before score_threshold).
            collection:      Which vector collection to fetch the corpus from.
            score_threshold: Minimum BM25 score (0.0–1.0 after normalization).
            filters:         Ignored in V1 — BM25 operates on full corpus.

        Returns:
            List of ScoredChunk ordered by BM25 score descending, trimmed to top_k.

        Raises:
            RetrieverError: If rank_bm25 is not installed or corpus fetch fails.
        """
        try:
            from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
        except ImportError as e:
            raise RetrieverError(
                "rank_bm25 package is required for BM25Retriever. "
                'Install it: pip install "nexrag[bm25]" or pip install rank-bm25',
                stage="retriever",
                component="BM25Retriever",
                cause=e,
            ) from e

        try:
            corpus = self._vector_db.get_all(collection)
        except Exception as e:
            raise RetrieverError(
                f"BM25Retriever failed to fetch corpus from collection '{collection}': {e}",
                stage="retriever",
                component="BM25Retriever",
                cause=e,
            ) from e

        if not corpus:
            return []

        tokenized_corpus = [doc.text.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        query_tokens = query.lower().split()
        raw_scores = bm25.get_scores(query_tokens)

        normalized = _normalize_scores(raw_scores.tolist())

        scored: list[tuple[int, float]] = sorted(
            enumerate(normalized), key=lambda x: x[1], reverse=True
        )

        results: list[ScoredChunk] = []
        for rank, (idx, score) in enumerate(scored[:top_k], start=1):
            if score_threshold > 0.0 and score < score_threshold:
                break
            results.append(ScoredChunk(chunk=corpus[idx], score=score, rank=rank))

        return results


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0 if s > 0 else 0.0 for s in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]
