"""
AnswerCoherenceEvaluator — logical flow, grammar, and readability of the answer.

Low coherence with high faithfulness usually means the LLM is retrieving correct
information but struggling to express it — noisy or conflicting context chunks.
"""

from __future__ import annotations

import json
from typing import Any

from nexrag.core.interfaces.evaluator import BaseEvaluator, EvalSample, MetricValue
from nexrag.core.interfaces.llm import BaseLLM

_COHERENCE_PROMPT = """\
You are a text quality judge. Evaluate the ANSWER on three dimensions:
  - Logical flow: does the answer progress logically from one idea to the next?
  - Grammar: is the answer grammatically correct?
  - Readability: is the answer clear and easy to understand?

ANSWER:
{answer}

Score each dimension from 0.0 (very poor) to 1.0 (excellent).
Return ONLY valid JSON:
{{
  "logical_flow": 0.9,
  "grammar": 1.0,
  "readability": 0.85,
  "coherence_score": 0.92
}}
"""


class AnswerCoherenceEvaluator(BaseEvaluator):
    """
    Coherence judge using an LLM to score logical flow, grammar, and readability.

    Args:
        llm: Any BaseLLM adapter. Use a small/fast model.
    """

    def __init__(self, llm: BaseLLM, **kwargs: Any) -> None:
        self._llm = llm

    @property
    def metric_names(self) -> list[str]:
        return ["answer.coherence_score"]

    def evaluate(self, sample: EvalSample) -> list[MetricValue]:
        attrs: dict[str, str] = {"model": self._llm.model_name or ""}
        try:
            prompt = _COHERENCE_PROMPT.format(answer=sample.answer[:4000])
            raw, _ = self._llm.generate(prompt)
            data = _parse_json(raw)
        except Exception as exc:
            return [
                MetricValue(
                    name="answer.coherence_score",
                    value=0.0,
                    success=False,
                    attributes={"error": str(exc)[:200]},
                )
            ]

        if "coherence_score" in data:
            score = float(data["coherence_score"])
        else:
            sub_scores = [
                float(data.get("logical_flow", 0.0)),
                float(data.get("grammar", 0.0)),
                float(data.get("readability", 0.0)),
            ]
            score = sum(sub_scores) / len(sub_scores) if sub_scores else 0.0

        return [MetricValue(name="answer.coherence_score", value=score, attributes=attrs)]


def _parse_json(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    result: dict[str, Any] = json.loads(text[start:end])
    return result
