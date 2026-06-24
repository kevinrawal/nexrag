"""
OTel provider/exporter setup — called once during NexRAG initialization.

Builds:
  - MeterProvider  (metrics) with Prometheus pull and/or OTLP push and/or console
  - TracerProvider (traces)  with OTLP push and/or console
  - LoggerProvider (logs)    with OTLP push and/or console

All three providers are returned as a tuple. The caller (OpenTelemetryObserver)
stores them and uses them to create instruments.

Lazy imports: opentelemetry packages are imported here at call time, not at
module import time. This means `import nexrag` works without nexrag[observability]
installed — the error only surfaces when `enabled: true` and a provider is built.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nexrag.core.config.schema import ObservabilityConfig


def build_providers(config: ObservabilityConfig) -> tuple[Any, Any, Any]:
    """
    Build and return (MeterProvider, TracerProvider, LoggerProvider).

    Raises ConfigError if nexrag[observability] is not installed.
    """
    try:
        from opentelemetry import metrics as otel_metrics
        from opentelemetry import trace as otel_trace
        from opentelemetry.sdk.resources import Resource
    except ImportError as e:
        from nexrag.exceptions import ConfigError

        raise ConfigError(
            "OpenTelemetry SDK is not installed. " "Run: pip install nexrag[observability]",
            stage="config",
            component="observer",
            cause=e,
        ) from e

    resource_attrs: dict[str, Any] = {
        "service.name": config.service_name,
        **config.resource_attributes,
    }
    resource = Resource.create(resource_attrs)

    meter_provider = _build_meter_provider(config, resource)
    tracer_provider = _build_tracer_provider(config, resource)
    logger_provider = _build_logger_provider(config, resource)

    otel_metrics.set_meter_provider(meter_provider)
    otel_trace.set_tracer_provider(tracer_provider)

    return meter_provider, tracer_provider, logger_provider


def _build_meter_provider(config: ObservabilityConfig, resource: Any) -> Any:
    from opentelemetry.sdk.metrics import MeterProvider

    readers = []

    if config.exporters.prometheus.enabled and config.signals.metrics:
        readers.append(_prometheus_reader(config))

    if config.exporters.otlp.enabled and config.signals.metrics:
        readers.append(_otlp_metric_reader(config))

    if config.exporters.console.enabled and config.signals.metrics:
        readers.append(_console_metric_reader())

    return MeterProvider(resource=resource, metric_readers=readers)


def _prometheus_reader(config: ObservabilityConfig) -> Any:
    try:
        from opentelemetry.exporter.prometheus import PrometheusMetricReader
        from prometheus_client import start_http_server
    except ImportError as e:
        from nexrag.exceptions import ConfigError

        raise ConfigError(
            "prometheus_client is required for Prometheus export. "
            "Run: pip install nexrag[observability]",
            stage="config",
            component="observer",
            cause=e,
        ) from e

    host = config.exporters.prometheus.host
    port = config.exporters.prometheus.port
    start_http_server(port=port, addr=host)
    return PrometheusMetricReader()


def _otlp_metric_reader(config: ObservabilityConfig) -> Any:
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

    otlp_cfg = config.exporters.otlp
    if otlp_cfg.protocol == "grpc":
        try:
            from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import (
                OTLPMetricExporter,
            )
        except ImportError as e:
            from nexrag.exceptions import ConfigError

            raise ConfigError(
                "opentelemetry-exporter-otlp-proto-grpc is required for OTLP gRPC export. "
                "Run: pip install nexrag[observability]",
                stage="config",
                component="observer",
                cause=e,
            ) from e
        exporter = OTLPMetricExporter(
            endpoint=otlp_cfg.endpoint,
            headers=dict(otlp_cfg.headers),
            insecure=otlp_cfg.insecure,
        )
    else:
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (  # type: ignore[no-redef]
                OTLPMetricExporter,
            )
        except ImportError as e:
            from nexrag.exceptions import ConfigError

            raise ConfigError(
                "opentelemetry-exporter-otlp-proto-http is required for OTLP HTTP export. "
                "Run: pip install nexrag[observability]",
                stage="config",
                component="observer",
                cause=e,
            ) from e
        exporter = OTLPMetricExporter(
            endpoint=otlp_cfg.endpoint,
            headers=dict(otlp_cfg.headers),
        )
    return PeriodicExportingMetricReader(exporter)


def _console_metric_reader() -> Any:
    from opentelemetry.sdk.metrics.export import (
        ConsoleMetricExporter,
        PeriodicExportingMetricReader,
    )

    return PeriodicExportingMetricReader(ConsoleMetricExporter())


def _build_tracer_provider(config: ObservabilityConfig, resource: Any) -> Any:
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

    provider = TracerProvider(resource=resource)

    if not config.signals.traces:
        return provider

    if config.exporters.otlp.enabled:
        otlp_cfg = config.exporters.otlp
        if otlp_cfg.protocol == "grpc":
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                    OTLPSpanExporter,
                )
            except ImportError as e:
                from nexrag.exceptions import ConfigError

                raise ConfigError(
                    "opentelemetry-exporter-otlp-proto-grpc required for trace export.",
                    stage="config",
                    component="observer",
                    cause=e,
                ) from e
            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=otlp_cfg.endpoint,
                        headers=dict(otlp_cfg.headers),
                        insecure=otlp_cfg.insecure,
                    )
                )
            )
        else:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore[no-redef]
                    OTLPSpanExporter,
                )
            except ImportError as e:
                from nexrag.exceptions import ConfigError

                raise ConfigError(
                    "opentelemetry-exporter-otlp-proto-http required for trace export.",
                    stage="config",
                    component="observer",
                    cause=e,
                ) from e
            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(
                        endpoint=otlp_cfg.endpoint,
                        headers=dict(otlp_cfg.headers),
                    )
                )
            )

    if config.exporters.console.enabled:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    return provider


def _build_logger_provider(config: ObservabilityConfig, resource: Any) -> Any:
    try:
        from opentelemetry._logs import set_logger_provider
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk._logs.export import (
            BatchLogRecordProcessor,
            ConsoleLogExporter,
        )
    except ImportError:
        return None

    provider = LoggerProvider(resource=resource)

    if not config.signals.logs:
        set_logger_provider(provider)
        return provider

    if config.exporters.otlp.enabled:
        otlp_cfg = config.exporters.otlp
        try:
            if otlp_cfg.protocol == "grpc":
                from opentelemetry.exporter.otlp.proto.grpc._log_exporter import (
                    OTLPLogExporter,
                )

                provider.add_log_record_processor(
                    BatchLogRecordProcessor(
                        OTLPLogExporter(
                            endpoint=otlp_cfg.endpoint,
                            headers=dict(otlp_cfg.headers),
                            insecure=otlp_cfg.insecure,
                        )
                    )
                )
            else:
                from opentelemetry.exporter.otlp.proto.http._log_exporter import (  # type: ignore[no-redef]
                    OTLPLogExporter,
                )

                provider.add_log_record_processor(
                    BatchLogRecordProcessor(
                        OTLPLogExporter(
                            endpoint=otlp_cfg.endpoint,
                            headers=dict(otlp_cfg.headers),
                        )
                    )
                )
        except ImportError:
            pass

    if config.exporters.console.enabled:
        provider.add_log_record_processor(BatchLogRecordProcessor(ConsoleLogExporter()))

    set_logger_provider(provider)
    return provider
