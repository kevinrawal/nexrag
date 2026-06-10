"""
BM25Retriever — keyword-based retrieval using BM25Okapi from rank_bm25.

Fetches the chunk corpus from the vector DB, builds a BM25 index, and scores
the query against it. The index is cached per-collection and invalidated when
new documents are ingested into that collection.

Requires: pip install "nexrag[bm25]"  (rank-bm25)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from nexrag.core.interfaces.sparse_retriever import BaseSparseRetriever
from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.models.chunk import Chunk, ScoredChunk
from nexrag.exceptions import RetrieverError


@dataclass
class _BM25CacheEntry:
    bm25: Any
    corpus: list[Chunk]
    built_at: float = field(default_factory=time.monotonic)


class BM25Retriever(BaseSparseRetriever):
    """
    Retrieves chunks using BM25 keyword scoring over the collection corpus.

    BM25 is complementary to dense retrieval — it excels at exact keyword matches
    where dense vectors may assign low similarity due to vocabulary mismatch.

    The BM25 index is built once per collection and cached in memory. It is
    invalidated automatically after ingestion and optionally after a TTL expires.

    Args:
        vector_db:  Any BaseVectorDB that implements get_all(). Used to fetch the
                    corpus when building or refreshing the index.
        cache_ttl:  Optional cache TTL in seconds. If set, the index is rebuilt
                    after this many seconds regardless of ingestion activity.
                    Default None — rely solely on explicit invalidation.
    """

    def __init__(self, vector_db: BaseVectorDB, cache_ttl: float | None = None) -> None:
        self._vector_db = vector_db
        self._cache_ttl = cache_ttl
        self._cache: dict[str, _BM25CacheEntry] = {}
        self._lock = threading.Lock()

    def invalidate_cache(self, collection: str | None = None) -> None:
        """
        Evict the BM25 index cache.

        Args:
            collection: Collection name to invalidate. Pass None to clear all.
        """
        with self._lock:
            if collection is None:
                self._cache.clear()
            else:
                self._cache.pop(collection, None)

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
        Score chunks in the collection by BM25 keyword relevance to the query.

        The BM25 index is served from the in-memory cache when available. On a
        cache miss (first call, post-invalidation, or TTL expiry) the full corpus
        is fetched from the vector DB and the index is rebuilt.

        Metadata filters are applied post-scoring: the full corpus is scored and
        sorted, then chunks that do not match every filter key-value pair are
        dropped before the top-k window is applied.

        Args:
            query:           The raw query string used for BM25 scoring.
            query_embedding: Ignored — BM25 is keyword-only.
            top_k:           Maximum chunks to return.
            collection:      Which vector collection to score against.
            score_threshold: Minimum normalized BM25 score (0.0–1.0). Chunks
                             below this threshold are excluded.
            filters:         Metadata key-value pairs. Only chunks whose metadata
                             contains every pair are returned. Supports scalar
                             equality only (same semantics as DenseRetriever's
                             single-level filter).

        Returns:
            List of ScoredChunk ordered by BM25 score descending, trimmed to top_k.

        Raises:
            RetrieverError: If rank_bm25 is not installed or corpus fetch fails.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise RetrieverError(
                "rank_bm25 package is required for BM25Retriever. "
                'Install it: pip install "nexrag[bm25]" or pip install rank-bm25',
                stage="retriever",
                component="BM25Retriever",
                cause=e,
            ) from e

        entry = self._get_or_build(collection, BM25Okapi)
        if entry is None:
            return []

        raw_scores = entry.bm25.get_scores(query.lower().split())
        normalized = _normalize_scores(raw_scores.tolist())

        scored = sorted(enumerate(normalized), key=lambda x: x[1], reverse=True)

        results: list[ScoredChunk] = []
        rank = 0
        for idx, score in scored:
            if score_threshold > 0.0 and score < score_threshold:
                break
            chunk = entry.corpus[idx]
            if filters and not all(chunk.metadata.get(k) == v for k, v in filters.items()):
                continue
            rank += 1
            results.append(ScoredChunk(chunk=chunk, score=score, rank=rank))
            if rank >= top_k:
                break

        return results

    # Private helpers

    def _get_or_build(self, collection: str, BM25Okapi: type) -> _BM25CacheEntry | None:  # noqa: N803
        with self._lock:
            entry = self._cache.get(collection)
            if entry is not None:
                if self._cache_ttl is None or (time.monotonic() - entry.built_at) < self._cache_ttl:
                    return entry

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
            return None

        tokenized = [doc.text.lower().split() for doc in corpus]
        bm25 = BM25Okapi(tokenized)
        new_entry = _BM25CacheEntry(bm25=bm25, corpus=corpus, built_at=time.monotonic())

        with self._lock:
            self._cache[collection] = new_entry

        return new_entry


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0 if s > 0 else 0.0 for s in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]
