"""
BM25Retriever — keyword-based retrieval using BM25Okapi from rank_bm25.

Fetches the chunk corpus from the vector DB, builds a BM25 index, and scores
the query against it. The index is cached in two tiers:

  - L1: in-process memory (fast, lost on restart). Always on.
  - L2: on disk (opt-in via ``cache_dir``, e.g. ``.nexrag/bm25``). When enabled, a
    cold process — or the first query after an L1 invalidation — reloads the
    pickled index instead of re-tokenising and rebuilding over the whole corpus.
    The on-disk index is trusted only while the collection's document count is
    unchanged and the TTL has not expired; otherwise it is rebuilt.

Scaling note (read before deploying at scale):
    The index is held entirely in memory and rebuilt from the full corpus on a
    cache miss. This is fine into the ~10^5-chunk range; beyond that, rebuild
    latency and resident memory grow linearly. Cache invalidation is also
    **single-process** — an ``ingest()`` in worker A (or a separate ETL process)
    does not invalidate worker B's index. For multi-worker deployments rely on
    ``cache_ttl`` (finite by default) to bound staleness. A server-side sparse
    backend (SQLite FTS5 / Elasticsearch) is the scalable path — tracked for a
    future release.

Requires: pip install "nexrag[bm25]"  (rank-bm25)
"""

from __future__ import annotations

import hashlib
import pickle
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nexrag.core.interfaces.sparse_retriever import BaseSparseRetriever
from nexrag.core.interfaces.vector_db import BaseVectorDB
from nexrag.core.models.chunk import Chunk, ScoredChunk
from nexrag.exceptions import RetrieverError

_DEFAULT_CACHE_TTL = 300.0
_FETCH_PAGE_SIZE = 1000


@dataclass
class _BM25CacheEntry:
    bm25: Any
    corpus: list[Chunk]
    doc_count: int
    built_at: float = field(default_factory=time.time)


class BM25Retriever(BaseSparseRetriever):
    """
    Retrieves chunks using BM25 keyword scoring over the collection corpus.

    BM25 is complementary to dense retrieval — it excels at exact keyword matches
    where dense vectors may assign low similarity due to vocabulary mismatch.

    The BM25 index is built once per collection and cached in memory and on disk.
    The in-memory cache is invalidated automatically after ingestion (single
    process only — see the module docstring) and after ``cache_ttl`` expires.

    Args:
        vector_db:  Any BaseVectorDB that implements get_all(). Used to fetch the
                    corpus when building or refreshing the index.
        cache_ttl:  Cache TTL in seconds. After this many seconds the index is
                    rebuilt regardless of ingestion activity. Defaults to 300s so
                    multi-worker deployments do not serve indefinitely-stale
                    results; pass None to disable TTL expiry (single-process only).
        cache_dir:  Directory for the optional on-disk (L2) index, e.g.
                    ``.nexrag/bm25``. Defaults to None (memory-only). Set a path to
                    persist the built index across restarts/processes — useful to
                    skip the cold-start rebuild on large corpora.
    """

    def __init__(
        self,
        vector_db: BaseVectorDB,
        cache_ttl: float | None = _DEFAULT_CACHE_TTL,
        cache_dir: str | None = None,
    ) -> None:
        self._vector_db = vector_db
        self._cache_ttl = cache_ttl
        self._cache_dir = Path(cache_dir) if cache_dir else None
        self._cache: dict[str, _BM25CacheEntry] = {}
        self._lock = threading.Lock()

    def invalidate_cache(self, collection: str | None = None) -> None:
        """
        Evict the BM25 index cache (both in-memory and on-disk tiers).

        Args:
            collection: Collection name to invalidate. Pass None to clear all.
        """
        with self._lock:
            if collection is None:
                self._cache.clear()
                self._clear_disk()
            else:
                self._cache.pop(collection, None)
                self._clear_disk(collection)

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

        The BM25 index is served from the in-memory cache when available, then
        from the on-disk cache, and only rebuilt from the corpus on a full miss.

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
            # L1: in-memory.
            entry = self._cache.get(collection)
            if entry is not None and self._is_fresh(entry, collection):
                return entry

            # L2: on disk — reload the prebuilt index instead of rebuilding.
            entry = self._load_disk(collection)
            if entry is not None and self._is_fresh(entry, collection):
                self._cache[collection] = entry
                return entry

            # Miss: rebuild from the (paginated) corpus.
            corpus = self._fetch_corpus(collection)
            if not corpus:
                return None

            tokenized = [doc.text.lower().split() for doc in corpus]
            bm25 = BM25Okapi(tokenized)
            new_entry = _BM25CacheEntry(bm25=bm25, corpus=corpus, doc_count=len(corpus))
            self._cache[collection] = new_entry
            self._save_disk(collection, new_entry)
            return new_entry

    def _is_fresh(self, entry: _BM25CacheEntry, collection: str) -> bool:
        """
        Fresh = within TTL and the collection's document count is unchanged.

        The count probe is only allowed to *invalidate* on a confirmed integer
        mismatch — a failed or non-numeric probe falls back to TTL-only freshness
        rather than forcing an expensive rebuild on uncertainty.
        """
        if self._cache_ttl is not None and (time.time() - entry.built_at) >= self._cache_ttl:
            return False
        try:
            count = self._vector_db.count(collection)
        except Exception:
            return True
        if isinstance(count, int) and count != entry.doc_count:
            return False
        return True

    def _fetch_corpus(self, collection: str) -> list[Chunk]:
        """Fetch the whole corpus in pages so one call never spikes memory."""
        out: list[Chunk] = []
        offset = 0
        try:
            while True:
                batch = self._vector_db.get_all(collection, limit=_FETCH_PAGE_SIZE, offset=offset)
                if not batch:
                    break
                out.extend(batch)
                if len(batch) < _FETCH_PAGE_SIZE:
                    break
                offset += _FETCH_PAGE_SIZE
        except Exception as e:
            raise RetrieverError(
                f"BM25Retriever failed to fetch corpus from collection '{collection}': {e}",
                stage="retriever",
                component="BM25Retriever",
                cause=e,
            ) from e
        return out

    # On-disk cache (best-effort: failures degrade to an in-memory rebuild).

    def _disk_path(self, collection: str) -> Path | None:
        if self._cache_dir is None:
            return None
        digest = hashlib.sha256(collection.encode("utf-8")).hexdigest()[:16]
        return self._cache_dir / f"{digest}.pkl"

    def _load_disk(self, collection: str) -> _BM25CacheEntry | None:
        path = self._disk_path(collection)
        if path is None or not path.exists():
            return None
        try:
            with path.open("rb") as fh:
                entry = pickle.load(fh)
            return entry if isinstance(entry, _BM25CacheEntry) else None
        except Exception:
            return None

    def _save_disk(self, collection: str, entry: _BM25CacheEntry) -> None:
        path = self._disk_path(collection)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("wb") as fh:
                pickle.dump(entry, fh)
        except Exception:
            pass

    def _clear_disk(self, collection: str | None = None) -> None:
        if self._cache_dir is None:
            return
        try:
            if collection is None:
                for f in self._cache_dir.glob("*.pkl"):
                    f.unlink(missing_ok=True)
            else:
                path = self._disk_path(collection)
                if path is not None:
                    path.unlink(missing_ok=True)
        except Exception:
            pass


def _normalize_scores(scores: list[float]) -> list[float]:
    """Min-max normalize a list of scores to [0, 1]."""
    if not scores:
        return scores
    min_s = min(scores)
    max_s = max(scores)
    if max_s == min_s:
        return [1.0 if s > 0 else 0.0 for s in scores]
    return [(s - min_s) / (max_s - min_s) for s in scores]
