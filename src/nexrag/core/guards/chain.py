"""
GuardChain — runs an ordered list of guards with a fail-open / fail-closed policy.

The chain threads REDACT transformations forward, short-circuits on the first
BLOCK, collects any metadata filters contributed by access-control guards, and
emits one observability event per guard firing (guard name, verdict, latency).

Policy on a guard *exception*:
    fail_open   — treat the error as ALLOW (logged via a 'failed' event). Default.
    fail_closed — treat the error as BLOCK. Production teams that must not leak on
                  a broken guard choose this explicitly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Literal

from nexrag.core.interfaces.guard import BaseGuard, GuardContext
from nexrag.core.interfaces.observer import BaseObserver, NoOpObserver
from nexrag.core.models.event import PipelineEvent

GuardPolicy = Literal["fail_open", "fail_closed"]


@dataclass(frozen=True)
class ChainOutcome:
    """Result of running a full guard chain over one piece of text."""

    text: str  # possibly redacted
    blocked: bool = False
    reason: str | None = None
    blocking_guard: str | None = None
    metadata_filter: dict[str, Any] | None = None


class GuardChain:
    """An ordered, policy-governed sequence of guards bound to one pipeline phase."""

    def __init__(
        self,
        guards: list[BaseGuard],
        *,
        policy: GuardPolicy = "fail_open",
        observer: BaseObserver | None = None,
        name: str = "",
    ) -> None:
        self._guards = guards
        self._policy = policy
        self._observer = observer or NoOpObserver()
        self._name = name

    @property
    def is_empty(self) -> bool:
        return not self._guards

    def run(self, text: str, context: GuardContext) -> ChainOutcome:
        """Run every guard in order; return the (possibly redacted) text and verdict."""
        current = text
        merged_filter: dict[str, Any] | None = None

        for guard in self._guards:
            started = time.monotonic()
            try:
                result = guard.check(current, context)
            except Exception as exc:  # noqa: BLE001 — policy decides; never crash the pipeline here
                self._emit_failed(guard, context, started, exc)
                if self._policy == "fail_closed":
                    return ChainOutcome(
                        text=current,
                        blocked=True,
                        reason=f"{self._guard_name(guard)} errored under fail_closed policy: {exc}",
                        blocking_guard=self._guard_name(guard),
                        metadata_filter=merged_filter,
                    )
                continue  # fail_open — treat as ALLOW

            self._emit(guard, context, result.action, started)

            if result.action == "block":
                return ChainOutcome(
                    text=current,
                    blocked=True,
                    reason=result.reason,
                    blocking_guard=self._guard_name(guard),
                    metadata_filter=merged_filter,
                )
            if result.action == "redact" and result.text is not None:
                current = result.text
            if result.metadata_filter:
                merged_filter = self._merge_filters(merged_filter, result.metadata_filter)

        return ChainOutcome(text=current, blocked=False, metadata_filter=merged_filter)

    # Helpers

    @staticmethod
    def _guard_name(guard: BaseGuard) -> str:
        return getattr(guard, "name", type(guard).__name__)

    @staticmethod
    def _merge_filters(a: dict[str, Any] | None, b: dict[str, Any] | None) -> dict[str, Any] | None:
        if not a:
            return b
        if not b:
            return a
        # Combine independent guard filters conjunctively; vector DBs support $and.
        return {"$and": [a, b]}

    def _emit(self, guard: BaseGuard, context: GuardContext, verdict: str, started: float) -> None:
        self._observer.emit(
            PipelineEvent(
                pipeline_id=context.pipeline_id,
                stage="guardrail",
                status="completed",
                duration_ms=(time.monotonic() - started) * 1000,
                metadata={
                    "chain": self._name or context.stage,
                    "phase": context.stage,
                    "guard": self._guard_name(guard),
                    "verdict": verdict,
                },
            )
        )

    def _emit_failed(
        self, guard: BaseGuard, context: GuardContext, started: float, exc: Exception
    ) -> None:
        self._observer.emit(
            PipelineEvent(
                pipeline_id=context.pipeline_id,
                stage="guardrail",
                status="failed",
                duration_ms=(time.monotonic() - started) * 1000,
                metadata={
                    "chain": self._name or context.stage,
                    "phase": context.stage,
                    "guard": self._guard_name(guard),
                    "policy": self._policy,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                },
            )
        )
