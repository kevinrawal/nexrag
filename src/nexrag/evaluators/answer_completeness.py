"""
AnswerCompletenessEvaluator — did the answer cover all aspects of the query?

The judge is asked to identify the information needs implied by the query,
then check which needs were addressed in the answer.
"""

from __future__ import annotations

import json
from typing import Any

from nexrag.core.interfaces.evaluator import BaseEvaluator, EvalSample, MetricValue
from nexrag.core.interfaces.llm import BaseLLM

_COMPLETENESS_PROMPT = """\
You are an answer completeness judge.

QUERY:
{query}

ANSWER:
{answer}

Instructions:
1. List all distinct information needs or sub-questions implied by the query.
2. For each need, determine if the answer fully, partially, or did not address it.
3. Score: fully=1.0, partially=0.5, not addressed=0.0.
4. Return ONLY valid JSON:
{{
  "needs": [
    {{"need": "...", "addressed": "fully"}},
    {{"need": "...", "addressed": "partially"}},
    {{"need": "...", "addressed": "none"}}
  ],
  "completeness_score": 0.75
}}
"""

_SCORE_MAP = {"fully": 1.0, "partially": 0.5, "none": 0.0}


class AnswerCompletenessEvaluator(BaseEvaluator):
    """
    Completeness judge using an LLM to assess coverage of all query intents.

    Args:
        llm: Any BaseLLM adapter. Use a small/fast model.
    """

    def __init__(self, llm: BaseLLM, **kwargs: Any) -> None:
        self._llm = llm

    @property
    def metric_names(self) -> list[str]:
        return ["answer.completeness_score"]

    def evaluate(self, sample: EvalSample) -> list[MetricValue]:
        attrs: dict[str, str] = {"model": self._llm.model_name or ""}
        try:
            prompt = _COMPLETENESS_PROMPT.format(
                query=sample.query[:2000],
                answer=sample.answer[:4000],
            )
            raw, _ = self._llm.generate(prompt)
            data = _parse_json(raw)
        except Exception as exc:
            return [
                MetricValue(
                    name="answer.completeness_score",
                    value=0.0,
                    success=False,
                    attributes={"error": str(exc)[:200]},
                )
            ]

        # Prefer explicit score if the LLM calculated it.
        if "completeness_score" in data:
            score = float(data["completeness_score"])
            return [MetricValue(name="answer.completeness_score", value=score, attributes=attrs)]

        needs = data.get("needs", [])
        if not needs:
            return [MetricValue(name="answer.completeness_score", value=0.0, attributes=attrs)]

        total = sum(_SCORE_MAP.get(n.get("addressed", "none"), 0.0) for n in needs)
        score = total / len(needs)
        return [MetricValue(name="answer.completeness_score", value=score, attributes=attrs)]


def _parse_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    result: dict[str, Any] = json.loads(text[start:end])
    return result
