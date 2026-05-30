"""
BaseObserver — contract for the observability hook.

Every pipeline stage emits a PipelineEvent at start, completion, and failure.
The Observer receives these events and decides what to do with them.

This design decouples pipeline code from any specific logging or tracing library.
The pipeline just calls observer.emit(event). What happens next is the observer's
problem — not the pipeline's.

V1:  ConsoleObserver — prints structured output to stdout.
V2+: OpenTelemetryObserver, LangfuseObserver, DatadogObserver, or any custom class
     declared in nexrag.yaml under observability.class.

Why emit() never raises:
    A failed observer must never crash the pipeline. If your Datadog observer
    has a network issue, you still want the RAG response. Observers swallow
    their own errors and log them internally.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod

from nexrag.core.models.event import PipelineEvent


class BaseObserver(ABC):
    """Abstract base class for all NexRAG observability observers."""

    @abstractmethod
    def emit(self, event: PipelineEvent) -> None:
        """
        Receive and handle a pipeline event.

        This method must never raise. Catch all exceptions internally.

        Args:
            event: The PipelineEvent emitted by a pipeline stage.
        """

    async def async_emit(self, event: PipelineEvent) -> None:
        """
        Async variant of emit(). Default: runs sync emit() in a thread pool.
        Override for observers that use native async I/O (HTTP, async queues).
        Must never raise, same contract as emit().
        """
        await asyncio.to_thread(self.emit, event)


class NoOpObserver(BaseObserver):
    """
    Silent observer. Discards all events.
    Used when observability.enabled: false in nexrag.yaml.
    """

    def emit(self, event: PipelineEvent) -> None:
        pass

    async def async_emit(self, event: PipelineEvent) -> None:
        pass
