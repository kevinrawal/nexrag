"""
AnswerRelevanceEvaluator — measures whether the answer addresses the question.

Approach: generate hypothetical questions the answer would respond to, then
compute cosine similarity between those and the original query embedding.
High similarity = the answer is on-topic. Low similarity = the answer drifted.

Requires: llm (to generate reverse questions) + embedder (cosine similarity).
If no embedder is configured, falls back to 0.0 (not scored).
"""

from __future__ import annotations

import json
import math
from typing import Any

from nexrag.core.interfaces.embedder import BaseEmbedder
from nexrag.core.interfaces.evaluator import BaseEvaluator, EvalSample, MetricValue
from nexrag.core.interfaces.llm import BaseLLM

_REVERSE_PROMPT = """\
Given the following ANSWER, generate exactly 3 questions that this answer
directly and fully addresses. Focus on specificity — each question should
be one that the answer clearly resolves.

ANSWER:
{answer}

Return ONLY valid JSON:
{{"questions": ["question 1", "question 2", "question 3"]}}
"""


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class AnswerRelevanceEvaluator(BaseEvaluator):
    """
    Reverse-question relevance evaluator.

    Args:
        llm:     Generates reverse questions from the answer.
        embedder: Embeds questions for cosine comparison with the original query.
                  If None, the score is always 0.0 (not very useful).
    """

    def __init__(self, llm: BaseLLM, embedder: BaseEmbedder | None = None, **kwargs: Any) -> None:
        self._llm = llm
        self._embedder = embedder

    @property
    def metric_names(self) -> list[str]:
        return ["answer.relevance_score"]

    def evaluate(self, sample: EvalSample) -> list[MetricValue]:
        attrs: dict[str, str] = {"model": self._llm.model_name or ""}
        try:
            raw, _ = self._llm.generate(_REVERSE_PROMPT.format(answer=sample.answer[:3000]))
            data = _parse_json(raw)
            questions = data.get("questions", [])[:3]
        except Exception as exc:
            return [
                MetricValue(
                    name="answer.relevance_score",
                    value=0.0,
                    success=False,
                    attributes={"error": str(exc)[:200]},
                )
            ]

        if not questions or self._embedder is None:
            return [MetricValue(name="answer.relevance_score", value=0.0, attributes=attrs)]

        try:
            query_emb = self._embedder.embed_query(sample.query)
            q_embs = self._embedder.embed(questions)
            similarities = [_cosine(query_emb, q) for q in q_embs]
            score = sum(similarities) / len(similarities)
        except Exception as exc:
            return [
                MetricValue(
                    name="answer.relevance_score",
                    value=0.0,
                    success=False,
                    attributes={"error": str(exc)[:200]},
                )
            ]

        return [MetricValue(name="answer.relevance_score", value=score, attributes=attrs)]


def _parse_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    result: dict[str, Any] = json.loads(text[start:end])
    return result
