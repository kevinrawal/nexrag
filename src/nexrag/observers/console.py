"""
ConsoleObserver — prints PipelineEvents to stdout.

Supports two output formats:
  - json (default): one JSON object per line, machine-parseable.
  - text: human-readable single line per event.

Respects log_level:
  - DEBUG:   emit all events (started, completed, failed)
  - INFO:    emit only completed and failed events
  - WARNING: emit only failed events
  - ERROR:   emit only failed events

The observer contract requires that emit() never raises. All exceptions are
caught internally and silently discarded to prevent observer bugs from
crashing the pipeline.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any

from nexrag.core.interfaces.observer import BaseObserver
from nexrag.core.models.event import PipelineEvent

_LEVEL_RANK = {"DEBUG": 0, "INFO": 1, "WARNING": 2, "ERROR": 3}
_FAILED_LEVEL = 2  # WARNING and above always show "failed"


class ConsoleObserver(BaseObserver):
    """
    Writes PipelineEvents to stdout.

    Args:
        log_level: Minimum level to emit. "DEBUG" | "INFO" | "WARNING" | "ERROR".
        format:    "json" (one JSON line) or "text" (human-readable).
    """

    def __init__(
        self,
        log_level: str = "INFO",
        format: str = "json",
    ) -> None:
        self._level_rank = _LEVEL_RANK.get(log_level.upper(), 1)
        self._format = format

    def emit(self, event: PipelineEvent) -> None:
        """
        Write the event to stdout.

        Always returns without raising — any internal error is silently discarded.
        """
        try:
            if not self._should_emit(event):
                return

            line = self._format_event(event)
            # TODO: consider using a logging framework instead of print() for better performance and features
            print(line, file=sys.stdout, flush=True)
        except Exception:
            pass

    # Private helpers

    def _should_emit(self, event: PipelineEvent) -> bool:
        if event.status == "failed":
            return self._level_rank <= _FAILED_LEVEL
        if event.status == "completed":
            return self._level_rank <= _LEVEL_RANK["INFO"]
        # "started" events are DEBUG only
        return self._level_rank <= _LEVEL_RANK["DEBUG"]

    def _format_event(self, event: PipelineEvent) -> str:
        if self._format == "json":
            return self._to_json(event)
        return self._to_text(event)

    @staticmethod
    def _to_json(event: PipelineEvent) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(tz=UTC).isoformat(),
            "pipeline_id": event.pipeline_id,
            "stage": event.stage,
            "status": event.status,
            "duration_ms": round(event.duration_ms, 2),
        }
        if event.metadata:
            payload["meta"] = event.metadata
        return json.dumps(payload, default=str)

    @staticmethod
    def _to_text(event: PipelineEvent) -> str:
        ts = datetime.now(tz=UTC).strftime("%H:%M:%S")
        parts = [
            f"[{ts}]",
            f"[{event.status.upper():9s}]",
            f"{event.stage:<20s}",
            f"pid={event.pipeline_id[:8]}",
        ]
        if event.duration_ms > 0:
            parts.append(f"{event.duration_ms:.1f}ms")
        if event.metadata:
            meta_str = "  ".join(f"{k}={v}" for k, v in event.metadata.items())
            parts.append(meta_str)
        return "  ".join(parts)
