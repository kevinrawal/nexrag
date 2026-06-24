"""
FaithfulnessEvaluator — LLM-as-judge for hallucination detection.

Decomposes the LLM answer into individual factual claims and checks each
against the retrieved context. Returns:
  - faithfulness.score            (0.0-1.0) fraction of claims grounded in context
  - faithfulness.hallucinated_count  raw count of unsupported claims
  - faithfulness.critical_hallucination_count  claims involving numbers/IPs/dates/names
"""

from __future__ import annotations

import json
import re
from typing import Any

from nexrag.core.interfaces.evaluator import BaseEvaluator, EvalSample, MetricValue
from nexrag.core.interfaces.llm import BaseLLM

_JUDGE_PROMPT = """\
You are a faithfulness judge. Your task is to check whether each factual claim in
the ANSWER is directly supported by the CONTEXT.

CONTEXT:
{context}

ANSWER:
{answer}

Instructions:
1. Decompose the answer into individual factual claims (ignore opinions/hedges).
2. For each claim, check if it is explicitly or implicitly supported by the context.
3. Mark any claim that contains numbers, IP addresses, dates, timestamps, usernames,
   or named entities as "critical" if unsupported.
4. Return ONLY valid JSON in this exact format:
{{
  "claims": [
    {{"claim": "...", "supported": true, "critical": false}},
    {{"claim": "...", "supported": false, "critical": true}}
  ]
}}
"""

_CRITICAL_PATTERN = re.compile(
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b"  # IP address
    r"|\b\d{4}-\d{2}-\d{2}\b"  # date
    r"|\b\d+(?:\.\d+)?%\b"  # percentage
    r"|\b\d{5,}\b"  # large number
)


class FaithfulnessEvaluator(BaseEvaluator):
    """
    Faithfulness judge using an LLM to decompose claims and verify against context.

    Args:
        llm: Any BaseLLM adapter. Use a small/fast model (Haiku, gpt-4o-mini).
    """

    def __init__(self, llm: BaseLLM, **kwargs: Any) -> None:
        self._llm = llm

    @property
    def metric_names(self) -> list[str]:
        return [
            "faithfulness.score",
            "faithfulness.hallucinated_count",
            "faithfulness.critical_hallucination_count",
        ]

    def evaluate(self, sample: EvalSample) -> list[MetricValue]:
        try:
            context_str = "\n\n".join(f"[{i+1}] {chunk}" for i, chunk in enumerate(sample.context))
            prompt = _JUDGE_PROMPT.format(
                context=context_str[:8000],
                answer=sample.answer[:4000],
            )
            raw, _ = self._llm.generate(prompt)
            data = _parse_json(raw)
            claims = data.get("claims", [])
        except Exception as exc:
            return [
                MetricValue(
                    name="faithfulness.score",
                    value=0.0,
                    success=False,
                    attributes={"error": str(exc)[:200]},
                )
            ]

        total = len(claims)
        if total == 0:
            return [
                MetricValue(
                    name="faithfulness.score", value=1.0, attributes={"model": self._llm.model_name}
                )
            ]

        supported = sum(1 for c in claims if c.get("supported", False))
        hallucinated = total - supported

        # Critical: unsupported claims that contain named entities / numbers
        critical = sum(
            1
            for c in claims
            if not c.get("supported", False)
            and (c.get("critical", False) or bool(_CRITICAL_PATTERN.search(c.get("claim", ""))))
        )

        attrs = {"model": self._llm.model_name}
        return [
            MetricValue(name="faithfulness.score", value=supported / total, attributes=attrs),
            MetricValue(
                name="faithfulness.hallucinated_count", value=float(hallucinated), attributes=attrs
            ),
            MetricValue(
                name="faithfulness.critical_hallucination_count",
                value=float(critical),
                attributes=attrs,
            ),
        ]


def _parse_json(text: str) -> dict[str, Any]:
    """Extract the first JSON object from the LLM response."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        return {}
    return json.loads(text[start:end])
