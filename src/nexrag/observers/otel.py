"""
OpenTelemetryObserver — translates PipelineEvents into OTel metrics, traces, and logs.

Every pipeline stage already emits PipelineEvent(stage, status, duration_ms, metadata).
This observer is the single translation layer from that event model into:
  - OTel Metrics  (histograms, counters, gauges via the registry)
  - OTel Traces   (per-stage spans nested under a per-run root span)
  - OTel Logs     (one structured LogRecord per event)

Design rules (same contract as all BaseObserver implementations):
  - emit() / async_emit() must NEVER raise. All errors are caught and logged
    via Python's standard logging module (not printed).
  - Instruments are created lazily on first use and cached in _instruments.
  - Open spans are tracked in _spans: dict[(pipeline_id, stage), Span], protected
    by a threading.Lock. Spans are pruned when the pipeline-level terminal event
    arrives and capped at _MAX_OPEN_SPANS to prevent unbounded memory growth
    in pathological cases.
  - The LLM cost_usd metric is computed here from token counts in the LLM event
    using the pricing table built by _factory.py.

The observer does NOT import any AI adapter packages — it only imports
opentelemetry (already guarded behind nexrag[observability]).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from nexrag.core.interfaces.observer import BaseObserver
from nexrag.core.models.event import PipelineEvent
from nexrag.observers._registry import specs_for_stage

if TYPE_CHECKING:
    from nexrag.core.config.schema import ObservabilityConfig

_log = logging.getLogger(__name__)

_MAX_OPEN_SPANS = 4096


def _deep_get(d: dict[str, Any], dotted_key: str) -> Any:
    """Resolve a dot-separated key path into a nested dict."""
    parts = dotted_key.split(".")
    current: Any = d
    for part in parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


class OpenTelemetryObserver(BaseObserver):
    """
    Production observer that exports metrics, traces, and logs via OTel.

    Args:
        config:          The resolved ObservabilityConfig.
        pricing_table:   {model_name: (input_usd_per_1k, output_usd_per_1k)}.
                         Pass an empty dict to disable cost metrics.
    """

    def __init__(
        self,
        config: ObservabilityConfig,
        pricing_table: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._config = config
        self._pricing: dict[str, tuple[float, float]] = pricing_table or {}
        self._lock = threading.Lock()
        self._spans: dict[tuple[str, str], Any] = {}  # (pipeline_id, stage) → Span
        self._instruments: dict[str, Any] = {}

        # Lazy OTel imports — only materialise when the observer is actually used.
        self._meter: Any = None
        self._tracer: Any = None
        self._logger: Any = None
        self._providers: tuple[Any, Any, Any] | None = None

        self._setup_done = False

    # ── Setup (called lazily on first emit) ───────────────────────────────────

    def _ensure_setup(self) -> None:
        if self._setup_done:
            return
        with self._lock:
            if self._setup_done:
                return
            try:
                from nexrag.observers._otel_setup import build_providers

                meter_provider, tracer_provider, logger_provider = build_providers(self._config)
                self._providers = (meter_provider, tracer_provider, logger_provider)

                self._meter = meter_provider.get_meter(
                    "nexrag", schema_url="https://opentelemetry.io/schemas/1.26.0"
                )
                self._tracer = tracer_provider.get_tracer("nexrag")
                if logger_provider is not None:
                    from opentelemetry._logs import get_logger

                    self._logger = get_logger("nexrag", logger_provider=logger_provider)
            except Exception as exc:
                _log.error("OpenTelemetryObserver setup failed: %s", exc)
                self._meter = None
                self._tracer = None
                self._logger = None
            finally:
                self._setup_done = True

    # ── Public contract ───────────────────────────────────────────────────────

    def emit(self, event: PipelineEvent) -> None:
        try:
            self._ensure_setup()
            self._record_generic_metrics(event)
            self._record_spec_metrics(event)
            self._record_cost_metric(event)
            self._handle_span(event)
            self._emit_log(event)
            if event.status in ("completed", "failed") and event.stage == "pipeline":
                self._prune_spans(event.pipeline_id)
        except Exception as exc:
            _log.debug("OTel emit error (swallowed): %s", exc)

    async def async_emit(self, event: PipelineEvent) -> None:
        try:
            await asyncio.to_thread(self.emit, event)
        except Exception as exc:
            _log.debug("OTel async_emit error (swallowed): %s", exc)

    def shutdown(self) -> None:
        """Flush all exporters. Call when the application exits."""
        if self._providers is None:
            return
        meter_provider, tracer_provider, _ = self._providers
        try:
            meter_provider.shutdown()
        except Exception:
            pass
        try:
            tracer_provider.shutdown()
        except Exception:
            pass

    # ── Generic per-stage instruments ─────────────────────────────────────────

    def _record_generic_metrics(self, event: PipelineEvent) -> None:
        if self._meter is None:
            return

        attrs = {"stage": event.stage, "status": event.status}

        # stage.duration — histogram (ms) on every completed/failed event
        if event.status in ("completed", "failed") and event.duration_ms > 0:
            self._histogram("nexrag.stage.duration", "ms", "Stage wall-clock duration").record(
                event.duration_ms, attributes=attrs
            )

        # stage.invocations — counter on every "started" event (= one pipeline call)
        if event.status == "started":
            self._counter(
                "nexrag.stage.invocations", "{invocations}", "Pipeline stage invocations"
            ).add(1, attributes={"stage": event.stage})

        # stage.errors — counter on "failed"
        if event.status == "failed":
            error_attrs = {
                "stage": event.stage,
                "error_type": str(event.metadata.get("error_type", "unknown")),
            }
            self._counter("nexrag.stage.errors", "{errors}", "Pipeline stage errors").add(
                1, attributes=error_attrs
            )

    # ── Registry-driven spec metrics ──────────────────────────────────────────

    def _record_spec_metrics(self, event: PipelineEvent) -> None:
        if self._meter is None:
            return

        for spec in specs_for_stage(event.stage):
            if spec.status_filter and event.status != spec.status_filter:
                continue

            value = self._resolve_meta_value(event.metadata, spec.meta_key)
            if value is None:
                continue

            # Special case: retrieval.empty_results — only increment when 0 chunks
            if spec.name == "retrieval.empty_results":
                if int(value) > 0:
                    continue
                self._counter(f"nexrag.{spec.name}", spec.unit, spec.description).add(
                    1, attributes={"stage": event.stage}
                )
                continue

            # Special case: evaluation metrics — add metric_name attribute
            extra_attrs: dict[str, str] = {}
            if event.stage == "evaluation":
                extra_attrs["metric_name"] = str(event.metadata.get("metric_name", "unknown"))

            attrs = {"stage": event.stage, **spec.attributes, **extra_attrs}

            numeric = float(value)
            if spec.kind == "histogram":
                self._histogram(f"nexrag.{spec.name}", spec.unit, spec.description).record(
                    numeric, attributes=attrs
                )
            elif spec.kind == "counter":
                self._counter(f"nexrag.{spec.name}", spec.unit, spec.description).add(
                    int(numeric), attributes=attrs
                )
            elif spec.kind == "gauge":
                self._gauge(f"nexrag.{spec.name}", spec.unit, spec.description).set(
                    numeric, attributes=attrs
                )

    # ── Cost metric ───────────────────────────────────────────────────────────

    def _record_cost_metric(self, event: PipelineEvent) -> None:
        if self._meter is None or not self._pricing:
            return
        if event.stage != "llm" or event.status != "completed":
            return

        model = str(event.metadata.get("model", ""))
        token_usage = event.metadata.get("token_usage")
        if not model or not isinstance(token_usage, dict):
            return

        from nexrag.core.observability.pricing import cost_usd

        usd = cost_usd(
            model=model,
            input_tokens=int(token_usage.get("prompt_tokens", 0)),
            output_tokens=int(token_usage.get("completion_tokens", 0)),
            pricing=self._pricing,
        )
        if usd is not None:
            self._histogram(
                "nexrag.llm.cost_per_query_usd", "USD", "Estimated LLM cost per query"
            ).record(usd, attributes={"stage": "llm", "model": model})

    # ── Traces ────────────────────────────────────────────────────────────────

    def _handle_span(self, event: PipelineEvent) -> None:
        if self._tracer is None or not self._config.signals.traces:
            return

        key = (event.pipeline_id, event.stage)

        if event.status == "started":
            self._start_span(event, key)
        elif event.status in ("completed", "failed"):
            self._end_span(event, key)

    def _start_span(self, event: PipelineEvent, key: tuple[str, str]) -> None:
        try:
            from opentelemetry import trace

            span_name = f"nexrag.{event.stage}"
            if event.stage == "pipeline":
                span_name = f"nexrag.pipeline.{event.metadata.get('kind', 'query')}"

            parent_key = (event.pipeline_id, "pipeline")
            parent_span = self._spans.get(parent_key)
            if parent_span is not None and event.stage != "pipeline":
                ctx = trace.set_span_in_context(parent_span)
            else:
                ctx = None

            span = self._tracer.start_span(
                span_name,
                context=ctx,
                attributes={"pipeline_id": event.pipeline_id, "stage": event.stage},
            )
            with self._lock:
                if len(self._spans) < _MAX_OPEN_SPANS:
                    self._spans[key] = span
        except Exception as exc:
            _log.debug("OTel span start failed: %s", exc)

    def _end_span(self, event: PipelineEvent, key: tuple[str, str]) -> None:
        try:
            from opentelemetry.trace import StatusCode

            with self._lock:
                span = self._spans.pop(key, None)
            if span is None:
                return

            for k, v in event.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    span.set_attribute(k, v)

            span.set_attribute("duration_ms", event.duration_ms)

            if event.status == "failed":
                span.set_status(StatusCode.ERROR, str(event.metadata.get("message", "")))
                span.add_event(
                    "exception",
                    {
                        "exception.type": str(event.metadata.get("error_type", "unknown")),
                        "exception.message": str(event.metadata.get("message", "")),
                    },
                )
            else:
                span.set_status(StatusCode.OK)

            span.end()
        except Exception as exc:
            _log.debug("OTel span end failed: %s", exc)

    def _prune_spans(self, pipeline_id: str) -> None:
        """Close any orphaned child spans when the pipeline terminates."""
        try:
            from opentelemetry.trace import StatusCode

            orphaned = []
            with self._lock:
                orphaned = [(k, v) for k, v in self._spans.items() if k[0] == pipeline_id]
                for k, _ in orphaned:
                    del self._spans[k]

            for _, span in orphaned:
                try:
                    span.set_status(StatusCode.ERROR, "orphaned span — pipeline terminated")
                    span.end()
                except Exception:
                    pass
        except Exception as exc:
            _log.debug("OTel prune_spans failed: %s", exc)

    # ── Logs ──────────────────────────────────────────────────────────────────

    def _emit_log(self, event: PipelineEvent) -> None:
        if self._logger is None or not self._config.signals.logs:
            return

        try:
            from opentelemetry._logs import SeverityNumber
            from opentelemetry.sdk._logs import LogRecord

            severity_map = {
                "started": SeverityNumber.DEBUG,
                "completed": SeverityNumber.INFO,
                "failed": SeverityNumber.ERROR,
            }
            severity = severity_map.get(event.status, SeverityNumber.INFO)

            body = {
                "pipeline_id": event.pipeline_id,
                "stage": event.stage,
                "status": event.status,
                "duration_ms": round(event.duration_ms, 2),
                **event.metadata,
            }

            record = LogRecord(
                timestamp=int(time.time_ns()),
                severity_number=severity,
                severity_text=event.status.upper(),
                body=str(body),
                attributes={
                    "pipeline_id": event.pipeline_id,
                    "stage": event.stage,
                    "status": event.status,
                },
            )
            self._logger.emit(record)
        except Exception as exc:
            _log.debug("OTel log emit failed: %s", exc)

    # ── Instrument factory / cache ────────────────────────────────────────────

    def _histogram(self, name: str, unit: str, description: str) -> Any:
        key = f"hist:{name}"
        if key not in self._instruments:
            self._instruments[key] = self._meter.create_histogram(
                name=name, unit=unit, description=description
            )
        return self._instruments[key]

    def _counter(self, name: str, unit: str, description: str) -> Any:
        key = f"cnt:{name}"
        if key not in self._instruments:
            self._instruments[key] = self._meter.create_counter(
                name=name, unit=unit, description=description
            )
        return self._instruments[key]

    def _gauge(self, name: str, unit: str, description: str) -> Any:
        key = f"gauge:{name}"
        if key not in self._instruments:
            self._instruments[key] = self._meter.create_gauge(
                name=name, unit=unit, description=description
            )
        return self._instruments[key]

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_meta_value(metadata: dict[str, Any], key: str | tuple[str, ...]) -> Any:
        """Try each key variant; support dot-notation for nested dicts."""
        keys = (key,) if isinstance(key, str) else key
        for k in keys:
            val = _deep_get(metadata, k)
            if val is not None:
                return val
        return None
