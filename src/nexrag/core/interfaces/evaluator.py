"""
BaseEvaluator — contract for optional LLM-as-judge metrics.

Evaluators run off the query response path on a sampled fraction of queries.
Results are emitted as PipelineEvent(stage="evaluation") so they flow through
the same OTel export path as every other metric.

Rules:
  - evaluate() must never raise — catch all errors internally and return a
    MetricValue with success=False and the error message in attributes.
  - Each evaluator works with injected BaseLLM / BaseEmbedder adapters so the
    user can pick any model/embedding per metric independently of the pipeline.
  - async_evaluate() defaults to running evaluate() in a thread pool. Override
    when the evaluator uses native async I/O.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvalSample:
    """
    Input to an evaluator — the minimal view of one query run.

    Attributes:
        query:          The original user query string.
        answer:         The LLM answer from that run.
        context:        Retrieved chunk texts that were included in the prompt.
        pipeline_id:    Ties the evaluation back to the pipeline run's logs.
        metadata:       Stage-level metadata (model, token counts, scores, etc.)
                        pulled directly from the PipelineEvent payloads.
    """

    query: str
    answer: str
    context: list[str]
    pipeline_id: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricValue:
    """
    Result produced by one evaluator for one sample.

    Attributes:
        name:       Metric key (e.g. "faithfulness.score"). Used as the OTel
                    instrument name.
        value:      Numeric result. Float for scores; int for counts.
        success:    False when the evaluation itself failed (error in judge call).
        attributes: Extra key-value pairs attached to the OTel metric point
                    (e.g. {"model": "claude-haiku-4-5-20251001"}).
    """

    name: str
    value: float
    success: bool = True
    attributes: dict[str, str] = field(default_factory=dict)


class BaseEvaluator(ABC):
    """Abstract base for all NexRAG LLM-as-judge evaluators."""

    @property
    @abstractmethod
    def metric_names(self) -> list[str]:
        """Names of all metrics this evaluator produces."""

    @abstractmethod
    def evaluate(self, sample: EvalSample) -> list[MetricValue]:
        """
        Run the evaluation and return metric values.

        Must never raise. Catch all errors and return MetricValue(success=False).
        """

    async def async_evaluate(self, sample: EvalSample) -> list[MetricValue]:
        """Async variant — defaults to thread-pool. Override for native async."""
        return await asyncio.to_thread(self.evaluate, sample)
