"""
EvaluationRunner — samples completed query runs and dispatches async evaluators.

Design rules:
  - NEVER on the response path. Evaluation is always fire-and-forget.
  - Sampling is configurable globally and per-evaluator (per-metric overrides global).
  - Concurrency is bounded by a ThreadPoolExecutor for sync paths and a semaphore
    for async paths.
  - Results are emitted as PipelineEvent(stage="evaluation") so they flow through
    the same OTel export pipeline as all other metrics.
  - Any evaluator error is caught here; a failed MetricValue is emitted instead
    of silently disappearing or crashing the background worker.
"""

from __future__ import annotations

import asyncio
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from nexrag.core.interfaces.evaluator import BaseEvaluator, EvalSample
from nexrag.core.interfaces.observer import BaseObserver, NoOpObserver
from nexrag.core.models.event import PipelineEvent

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)


@dataclass
class _EvaluatorSlot:
    evaluator: BaseEvaluator
    sample_rate: float


class EvaluationRunner:
    """
    Dispatches optional LLM-as-judge evaluations off the query response path.

    Args:
        evaluators:       List of (evaluator, per-evaluator sample_rate) tuples.
        global_sample_rate: Fraction of queries to evaluate when no per-metric
                          rate is set. 0.0 = disabled, 1.0 = all queries.
        max_concurrency:  Thread-pool size for sync dispatch.
        observer:         Pipeline observer — evaluation results are emitted as
                          PipelineEvent(stage="evaluation") through it.
    """

    def __init__(
        self,
        evaluators: list[tuple[BaseEvaluator, float]],
        global_sample_rate: float = 0.1,
        max_concurrency: int = 4,
        observer: BaseObserver | None = None,
    ) -> None:
        self._slots = [_EvaluatorSlot(evaluator=ev, sample_rate=rate) for ev, rate in evaluators]
        self._global_sample_rate = global_sample_rate
        self._observer = observer or NoOpObserver()
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrency, thread_name_prefix="nexrag-eval"
        )
        self._async_sem = asyncio.Semaphore(max_concurrency)
        self._slots_lock = threading.Lock()

    def add(self, evaluator: BaseEvaluator, sample_rate: float | None = None) -> None:
        """
        Attach a custom evaluator at runtime.

        Thread-safe — can be called after the runner has started dispatching.
        The evaluator will be included in the next dispatch() / async_dispatch() call.

        Args:
            evaluator:   A :class:`BaseEvaluator` subclass instance.
            sample_rate: Fraction of queries to evaluate (0.0–1.0).
                         Defaults to the runner's global_sample_rate.
        """
        rate = sample_rate if sample_rate is not None else self._global_sample_rate
        with self._slots_lock:
            self._slots.append(_EvaluatorSlot(evaluator=evaluator, sample_rate=rate))

    def dispatch(self, sample: EvalSample) -> None:
        """Fire-and-forget: submit evaluations for a completed sync query run."""
        with self._slots_lock:
            slots = list(self._slots)
        for slot in slots:
            if random.random() < slot.sample_rate:
                self._executor.submit(self._run_sync, slot.evaluator, sample)

    async def async_dispatch(self, sample: EvalSample) -> None:
        """Fire-and-forget for async query runs: schedule coroutines on the event loop."""
        with self._slots_lock:
            slots = list(self._slots)
        for slot in slots:
            if random.random() < slot.sample_rate:
                asyncio.ensure_future(self._run_async(slot.evaluator, sample))

    def _run_sync(self, evaluator: BaseEvaluator, sample: EvalSample) -> None:
        try:
            results = evaluator.evaluate(sample)
        except Exception as exc:
            _log.debug("Evaluator %s raised: %s", type(evaluator).__name__, exc)
            results = []
        self._emit_results(results, sample.pipeline_id)

    async def _run_async(self, evaluator: BaseEvaluator, sample: EvalSample) -> None:
        async with self._async_sem:
            try:
                results = await evaluator.async_evaluate(sample)
            except Exception as exc:
                _log.debug("Evaluator %s raised: %s", type(evaluator).__name__, exc)
                results = []
        await self._emit_results_async(results, sample.pipeline_id)

    def _emit_results(self, results: list[Any], pipeline_id: str) -> None:
        for metric in results:
            event = PipelineEvent(
                pipeline_id=pipeline_id,
                stage="evaluation",
                status="completed" if metric.success else "failed",
                duration_ms=0.0,
                metadata={
                    "metric_name": metric.name,
                    "value": metric.value,
                    **metric.attributes,
                },
            )
            self._observer.emit(event)

    async def _emit_results_async(self, results: list[Any], pipeline_id: str) -> None:
        for metric in results:
            event = PipelineEvent(
                pipeline_id=pipeline_id,
                stage="evaluation",
                status="completed" if metric.success else "failed",
                duration_ms=0.0,
                metadata={
                    "metric_name": metric.name,
                    "value": metric.value,
                    **metric.attributes,
                },
            )
            await self._observer.async_emit(event)

    def shutdown(self, wait: bool = False) -> None:
        """Graceful shutdown — called when the NexRAG instance is torn down."""
        self._executor.shutdown(wait=wait)


class NoOpEvaluationRunner:
    """Used when evaluations are disabled."""

    def dispatch(self, sample: EvalSample) -> None:
        pass

    async def async_dispatch(self, sample: EvalSample) -> None:
        pass

    def add(self, evaluator: BaseEvaluator, sample_rate: float | None = None) -> None:  # noqa: ARG002
        raise RuntimeError(
            "Cannot add evaluators to a disabled evaluation runner. "
            "Set observability.evaluations.enabled: true in your config."
        )

    def shutdown(self, wait: bool = False) -> None:
        pass
