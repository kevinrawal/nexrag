"""
ContextDiversityEvaluator — average pairwise semantic distance between retrieved chunks.

Low diversity = all chunks are semantically near-identical (over-retrieval of the same
information). High diversity = retrieval set covers different facets of the query topic.

Cost: N*(N-1)/2 cosine comparisons over existing chunk vectors — no LLM call needed.
If embedder is provided, re-embeds chunks to get fresh vectors. If chunks are very short
and the embedder is expensive, consider caching at the retriever level instead.
"""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.core.interfaces.evaluator import BaseEvaluator, EvalSample, MetricValue


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine_similarity (so 0 = identical, 1 = orthogonal, 2 = opposite)."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return 1.0 - (dot / (norm_a * norm_b))


class ContextDiversityEvaluator(BaseEvaluator):
    """
    Embedding-based context diversity scorer.

    Args:
        embedder: Used to embed the retrieved chunks. Use a cheap embedding model.
    """

    def __init__(self, embedder: BaseEmbedder, **kwargs: Any) -> None:
        self._embedder = embedder

    @property
    def metric_names(self) -> list[str]:
        return ["context.diversity_score"]

    def evaluate(self, sample: EvalSample) -> list[MetricValue]:
        attrs = {"model": self._embedder.model_name}
        chunks = sample.context
        if len(chunks) < 2:
            return [MetricValue(name="context.diversity_score", value=1.0, attributes=attrs)]

        try:
            embeddings = self._embedder.embed(chunks)
        except Exception as exc:
            return [
                MetricValue(
                    name="context.diversity_score",
                    value=0.0,
                    success=False,
                    attributes={"error": str(exc)[:200]},
                )
            ]

        pairs = list(combinations(range(len(embeddings)), 2))
        if not pairs:
            return [MetricValue(name="context.diversity_score", value=1.0, attributes=attrs)]

        distances = [_cosine_distance(embeddings[i], embeddings[j]) for i, j in pairs]
        diversity = sum(distances) / len(distances)
        return [MetricValue(name="context.diversity_score", value=diversity, attributes=attrs)]
