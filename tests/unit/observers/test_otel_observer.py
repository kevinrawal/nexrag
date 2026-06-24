"""
Tests for OpenTelemetryObserver.

Uses OTel's in-memory exporters/readers so we don't need a real collector.
No network calls, no ports opened.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nexrag.core.models.event import PipelineEvent

# ── Helpers ───────────────────────────────────────────────────────────────────


def _event(
    stage: str = "retriever",
    status: str = "completed",
    duration_ms: float = 42.0,
    metadata: dict | None = None,
    pipeline_id: str = "test-pipe-001",
) -> PipelineEvent:
    return PipelineEvent(
        pipeline_id=pipeline_id,
        stage=stage,  # type: ignore[arg-type]
        status=status,
        duration_ms=duration_ms,
        metadata=metadata or {},
    )


def _minimal_config(
    prometheus_enabled: bool = False,
    otlp_enabled: bool = False,
    console_enabled: bool = False,
    metrics_enabled: bool = True,
    traces_enabled: bool = True,
    logs_enabled: bool = True,
) -> object:
    """Build a minimal ObservabilityConfig-like object via the real schema."""
    from nexrag.core.config.schema import ObservabilityConfig

    raw: dict = {
        "enabled": True,
        "service_name": "nexrag-test",
        "signals": {
            "metrics": metrics_enabled,
            "traces": traces_enabled,
            "logs": logs_enabled,
        },
        "exporters": {
            "prometheus": {"enabled": prometheus_enabled},
            "otlp": {"enabled": otlp_enabled},
            "console": {"enabled": console_enabled},
        },
    }
    return ObservabilityConfig.model_validate(raw)


# ── Schema tests ──────────────────────────────────────────────────────────────


class TestObservabilityConfigSchema:
    def test_default_config_is_valid(self):
        from nexrag.core.config.schema import ObservabilityConfig

        cfg = ObservabilityConfig()
        assert cfg.enabled is True
        assert cfg.service_name == "nexrag"
        assert cfg.signals.metrics is True
        assert cfg.signals.traces is True
        assert cfg.signals.logs is True
        assert cfg.exporters.prometheus.enabled is False
        assert cfg.exporters.otlp.enabled is False

    def test_pricing_table_is_parsed(self):
        from nexrag.core.config.schema import ObservabilityConfig

        cfg = ObservabilityConfig.model_validate(
            {
                "metrics": {
                    "pricing": {
                        "gpt-4o": {"input": 0.0025, "output": 0.01},
                    }
                }
            }
        )
        assert cfg.metrics.pricing["gpt-4o"].input == 0.0025

    def test_evaluator_config_parsed_with_nested_llm(self):
        from nexrag.core.config.schema import ObservabilityConfig

        cfg = ObservabilityConfig.model_validate(
            {
                "evaluations": {
                    "enabled": True,
                    "faithfulness": {
                        "enabled": True,
                        "sample_rate": 0.5,
                        "llm": {
                            "provider": "openai",
                            "model": "gpt-4o-mini",
                            "api_key": "sk-test",
                        },
                    },
                }
            }
        )
        assert cfg.evaluations.faithfulness.enabled is True
        assert cfg.evaluations.faithfulness.sample_rate == 0.5
        assert cfg.evaluations.faithfulness.llm is not None
        assert cfg.evaluations.faithfulness.llm.model == "gpt-4o-mini"

    def test_disabled_observability_valid(self):
        from nexrag.core.config.schema import ObservabilityConfig

        cfg = ObservabilityConfig.model_validate({"enabled": False})
        assert cfg.enabled is False


# ── Pricing tests ─────────────────────────────────────────────────────────────


class TestPricing:
    def test_cost_usd_known_model(self):
        from nexrag.core.observability.pricing import cost_usd

        pricing = {"gpt-4o": (0.0025, 0.01)}
        cost = cost_usd("gpt-4o", 1000, 500, pricing)
        assert cost is not None
        assert abs(cost - (0.0025 + 0.005)) < 1e-9

    def test_cost_usd_unknown_model_returns_none(self):
        from nexrag.core.observability.pricing import cost_usd

        cost = cost_usd("unknown-model", 1000, 500, {})
        assert cost is None

    def test_build_pricing_table(self):
        from nexrag.core.config.schema import ObservabilityConfig
        from nexrag.core.observability.pricing import build_pricing_table

        cfg = ObservabilityConfig.model_validate(
            {"metrics": {"pricing": {"gpt-4o": {"input": 0.0025, "output": 0.01}}}}
        )
        table = build_pricing_table(dict(cfg.metrics.pricing))
        assert "gpt-4o" in table
        assert table["gpt-4o"] == (0.0025, 0.01)


# ── Observer tests using in-memory OTel ───────────────────────────────────────


class TestOpenTelemetryObserverMetrics:
    """
    These tests use the real OTel SDK with an InMemoryMetricReader so we can
    assert that specific metric data points are produced without opening ports.
    """

    @pytest.fixture()
    def observer_with_reader(self):
        """Return (observer, in_memory_reader) pair. Sets up a fresh MeterProvider."""
        try:
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import InMemoryMetricReader
        except ImportError:
            pytest.skip("opentelemetry-sdk not installed")

        reader = InMemoryMetricReader()
        meter_provider = MeterProvider(metric_readers=[reader])
        meter = meter_provider.get_meter("nexrag-test")

        from nexrag.observers.otel import OpenTelemetryObserver

        obs = OpenTelemetryObserver.__new__(OpenTelemetryObserver)
        obs._config = _minimal_config()
        obs._pricing = {"gpt-4o": (0.0025, 0.01)}
        obs._lock = __import__("threading").Lock()
        obs._spans = {}
        obs._instruments = {}
        obs._meter = meter
        obs._tracer = None
        obs._logger = None
        obs._providers = None
        obs._setup_done = True

        return obs, reader, meter_provider

    def test_stage_duration_recorded_on_completed(self, observer_with_reader):
        obs, reader, mp = observer_with_reader
        obs.emit(_event("embedder", "completed", duration_ms=55.0))

        data = reader.get_metrics_data()
        names = {
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        }
        assert "nexrag.stage.duration" in names

    def test_stage_errors_counted_on_failed(self, observer_with_reader):
        obs, reader, mp = observer_with_reader
        obs.emit(
            _event("llm", "failed", metadata={"error_type": "TimeoutError", "message": "timed out"})
        )

        data = reader.get_metrics_data()
        names = {
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        }
        assert "nexrag.stage.errors" in names

    def test_retrieval_top_score_recorded(self, observer_with_reader):
        obs, reader, mp = observer_with_reader
        obs.emit(
            _event(
                "retriever",
                "completed",
                metadata={
                    "chunks_retrieved": 5,
                    "top_score": 0.92,
                    "avg_score": 0.75,
                    "bottom_score": 0.61,
                    "score_spread": 0.31,
                },
            )
        )

        data = reader.get_metrics_data()
        names = {
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        }
        assert "nexrag.retrieval.top_score" in names
        assert "nexrag.retrieval.avg_score" in names

    def test_retrieval_empty_results_counted(self, observer_with_reader):
        obs, reader, mp = observer_with_reader
        obs.emit(
            _event(
                "retriever",
                "completed",
                metadata={"chunks_retrieved": 0, "collection": "docs"},
            )
        )

        data = reader.get_metrics_data()
        names = {
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        }
        assert "nexrag.retrieval.empty_results" in names

    def test_llm_token_metrics_recorded(self, observer_with_reader):
        obs, reader, mp = observer_with_reader
        obs.emit(
            _event(
                "llm",
                "completed",
                metadata={
                    "model": "gpt-4o",
                    "response_length": 200,
                    "token_usage": {
                        "prompt_tokens": 500,
                        "completion_tokens": 200,
                        "total_tokens": 700,
                    },
                },
            )
        )

        data = reader.get_metrics_data()
        names = {
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        }
        assert "nexrag.llm.tokens_input" in names
        assert "nexrag.llm.tokens_output" in names
        assert "nexrag.llm.cost_per_query_usd" in names

    def test_emit_never_raises_on_malformed_event(self, observer_with_reader):
        obs, reader, mp = observer_with_reader
        obs.emit(object())  # type: ignore[arg-type] — must not raise

    def test_invocation_counter_on_started(self, observer_with_reader):
        obs, reader, mp = observer_with_reader
        obs.emit(_event("embedder", "started", duration_ms=0.0))

        data = reader.get_metrics_data()
        names = {
            m.name for rm in data.resource_metrics for sm in rm.scope_metrics for m in sm.metrics
        }
        assert "nexrag.stage.invocations" in names


# ── Registry tests ────────────────────────────────────────────────────────────


class TestMetricRegistry:
    def test_retriever_specs_exist(self):
        from nexrag.observers._registry import specs_for_stage

        specs = specs_for_stage("retriever")
        names = [s.name for s in specs]
        assert "retrieval.top_score" in names
        assert "retrieval.avg_score" in names
        assert "retrieval.empty_results" in names

    def test_llm_specs_exist(self):
        from nexrag.observers._registry import specs_for_stage

        specs = specs_for_stage("llm")
        names = [s.name for s in specs]
        assert "llm.tokens_input" in names
        assert "llm.tokens_output" in names

    def test_unknown_stage_returns_empty(self):
        from nexrag.observers._registry import specs_for_stage

        assert specs_for_stage("nonexistent_stage") == []

    def test_evaluation_spec_exists(self):
        from nexrag.observers._registry import specs_for_stage

        specs = specs_for_stage("evaluation")
        assert any(s.name == "eval.metric_value" for s in specs)


# ── EvaluationRunner tests ────────────────────────────────────────────────────


class TestEvaluationRunner:
    def test_noop_runner_does_nothing(self):
        from nexrag.core.interfaces.evaluator import EvalSample
        from nexrag.core.observability.runner import NoOpEvaluationRunner

        runner = NoOpEvaluationRunner()
        sample = EvalSample(query="q", answer="a", context=[], pipeline_id="p1")
        runner.dispatch(sample)  # must not raise

    def test_runner_dispatches_to_evaluator(self):
        from nexrag.core.interfaces.evaluator import BaseEvaluator, EvalSample, MetricValue
        from nexrag.core.interfaces.observer import NoOpObserver
        from nexrag.core.observability.runner import EvaluationRunner

        class FakeEval(BaseEvaluator):
            @property
            def metric_names(self) -> list[str]:
                return ["fake.score"]

            def evaluate(self, sample: EvalSample) -> list[MetricValue]:
                return [MetricValue(name="fake.score", value=0.9)]

        runner = EvaluationRunner(
            evaluators=[(FakeEval(), 1.0)],
            global_sample_rate=1.0,
            max_concurrency=1,
            observer=NoOpObserver(),
        )

        sample = EvalSample(query="hello", answer="world", context=["chunk"], pipeline_id="p1")
        runner.dispatch(sample)
        runner.shutdown(wait=True)

    def test_runner_sampling_at_zero_rate(self):
        """Sample rate 0.0 — evaluator must never be called."""
        import threading

        from nexrag.core.interfaces.evaluator import BaseEvaluator, EvalSample, MetricValue
        from nexrag.core.interfaces.observer import NoOpObserver
        from nexrag.core.observability.runner import EvaluationRunner

        called = threading.Event()

        class TrackedEval(BaseEvaluator):
            @property
            def metric_names(self) -> list[str]:
                return ["t.score"]

            def evaluate(self, sample: EvalSample) -> list[MetricValue]:
                called.set()
                return [MetricValue(name="t.score", value=1.0)]

        runner = EvaluationRunner(
            evaluators=[(TrackedEval(), 0.0)],
            global_sample_rate=0.0,
            observer=NoOpObserver(),
        )
        sample = EvalSample(query="q", answer="a", context=[], pipeline_id="p")
        runner.dispatch(sample)
        runner.shutdown(wait=True)
        assert not called.is_set()

    def test_runner_evaluator_exception_does_not_propagate(self):
        from nexrag.core.interfaces.evaluator import BaseEvaluator, EvalSample, MetricValue
        from nexrag.core.interfaces.observer import NoOpObserver
        from nexrag.core.observability.runner import EvaluationRunner

        class BrokenEval(BaseEvaluator):
            @property
            def metric_names(self) -> list[str]:
                return ["x.score"]

            def evaluate(self, sample: EvalSample) -> list[MetricValue]:
                raise RuntimeError("judge exploded")

        runner = EvaluationRunner(
            evaluators=[(BrokenEval(), 1.0)],
            observer=NoOpObserver(),
        )
        sample = EvalSample(query="q", answer="a", context=[], pipeline_id="p")
        runner.dispatch(sample)  # must not raise
        runner.shutdown(wait=True)


# ── Evaluator unit tests ──────────────────────────────────────────────────────


class TestEvaluatorsFaithfulness:
    def test_faithfulness_evaluator_returns_metrics(self):
        from nexrag.core.interfaces.evaluator import EvalSample
        from nexrag.evaluators.faithfulness import FaithfulnessEvaluator

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4o-mini"
        mock_llm.generate.return_value = (
            '{"claims": [{"claim": "sky is blue", "supported": true, "critical": false},'
            '{"claim": "IP is 1.2.3.4", "supported": false, "critical": true}]}',
            None,
        )

        ev = FaithfulnessEvaluator(llm=mock_llm)
        sample = EvalSample(
            query="What color is the sky?",
            answer="The sky is blue and the IP 1.2.3.4 is suspicious.",
            context=["The sky appears blue due to Rayleigh scattering."],
            pipeline_id="p1",
        )
        results = ev.evaluate(sample)
        names = {r.name for r in results}
        assert "faithfulness.score" in names
        assert "faithfulness.hallucinated_count" in names
        assert "faithfulness.critical_hallucination_count" in names

        score = next(r for r in results if r.name == "faithfulness.score")
        assert score.value == 0.5  # 1 of 2 supported
        critical = next(r for r in results if r.name == "faithfulness.critical_hallucination_count")
        assert critical.value == 1.0

    def test_faithfulness_evaluator_llm_error_returns_failure_metric(self):
        from nexrag.core.interfaces.evaluator import EvalSample
        from nexrag.evaluators.faithfulness import FaithfulnessEvaluator

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4o-mini"
        mock_llm.generate.side_effect = RuntimeError("network error")

        ev = FaithfulnessEvaluator(llm=mock_llm)
        sample = EvalSample(query="q", answer="a", context=["c"], pipeline_id="p")
        results = ev.evaluate(sample)
        assert len(results) == 1
        assert results[0].success is False


class TestEvaluatorsRelevance:
    def test_answer_relevance_with_embedder(self):
        from nexrag.core.interfaces.evaluator import EvalSample
        from nexrag.evaluators.answer_relevance import AnswerRelevanceEvaluator

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4o-mini"
        mock_llm.generate.return_value = (
            '{"questions": ["What is RAG?", "How does RAG work?", "What is retrieval?"]}',
            None,
        )

        mock_embedder = MagicMock()
        mock_embedder.model_name = "text-embedding-3-small"
        # Return perfect similarity vectors
        mock_embedder.embed_query.return_value = [1.0, 0.0, 0.0]
        mock_embedder.embed.return_value = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

        ev = AnswerRelevanceEvaluator(llm=mock_llm, embedder=mock_embedder)
        sample = EvalSample(
            query="What is RAG?",
            answer="RAG is retrieval-augmented generation.",
            context=[],
            pipeline_id="p1",
        )
        results = ev.evaluate(sample)
        assert len(results) == 1
        assert results[0].name == "answer.relevance_score"
        assert abs(results[0].value - 1.0) < 0.01

    def test_answer_relevance_without_embedder_returns_zero(self):
        from nexrag.core.interfaces.evaluator import EvalSample
        from nexrag.evaluators.answer_relevance import AnswerRelevanceEvaluator

        mock_llm = MagicMock()
        mock_llm.model_name = "gpt-4o-mini"
        mock_llm.generate.return_value = ('{"questions": ["q1"]}', None)

        ev = AnswerRelevanceEvaluator(llm=mock_llm, embedder=None)
        sample = EvalSample(query="q", answer="a", context=[], pipeline_id="p")
        results = ev.evaluate(sample)
        assert results[0].value == 0.0


class TestContextDiversityEvaluator:
    def test_high_diversity_with_orthogonal_vectors(self):
        from nexrag.core.interfaces.evaluator import EvalSample
        from nexrag.evaluators.context_diversity import ContextDiversityEvaluator

        mock_embedder = MagicMock()
        mock_embedder.model_name = "text-embedding-3-small"
        # Two perfectly orthogonal vectors → cosine distance = 1.0
        mock_embedder.embed.return_value = [[1.0, 0.0], [0.0, 1.0]]

        ev = ContextDiversityEvaluator(embedder=mock_embedder)
        sample = EvalSample(query="q", answer="a", context=["chunk A", "chunk B"], pipeline_id="p")
        results = ev.evaluate(sample)
        assert results[0].name == "context.diversity_score"
        assert abs(results[0].value - 1.0) < 0.01

    def test_single_chunk_returns_one(self):
        from nexrag.core.interfaces.evaluator import EvalSample
        from nexrag.evaluators.context_diversity import ContextDiversityEvaluator

        mock_embedder = MagicMock()
        mock_embedder.model_name = "text-embedding-3-small"

        ev = ContextDiversityEvaluator(embedder=mock_embedder)
        sample = EvalSample(query="q", answer="a", context=["only chunk"], pipeline_id="p")
        results = ev.evaluate(sample)
        assert results[0].value == 1.0


# ── Integration: pipeline emits evaluation events ─────────────────────────────


class TestQueryPipelineWithEvaluationRunner:
    def test_evaluation_runner_dispatched_after_run(self):
        import threading

        from nexrag.core.interfaces.evaluator import BaseEvaluator, EvalSample, MetricValue
        from nexrag.core.interfaces.observer import NoOpObserver
        from nexrag.core.models.chunk import Chunk, ScoredChunk
        from nexrag.core.observability.runner import EvaluationRunner
        from nexrag.core.pipeline.query import QueryPipeline

        dispatched = threading.Event()

        class FakeEval(BaseEvaluator):
            @property
            def metric_names(self) -> list[str]:
                return ["fake.score"]

            def evaluate(self, sample: EvalSample) -> list[MetricValue]:
                dispatched.set()
                return [MetricValue(name="fake.score", value=1.0)]

        runner = EvaluationRunner([(FakeEval(), 1.0)], observer=NoOpObserver())

        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2]
        mock_embedder.model_name = "test-model"

        mock_retriever = MagicMock()
        chunk = Chunk(text="context text", chunk_index=0, total_chunks=1, parent_doc_id="d1")
        mock_retriever.retrieve.return_value = [ScoredChunk(chunk=chunk, score=0.85, rank=1)]

        mock_prompt_builder = MagicMock()
        mock_prompt_builder.build.return_value = "assembled prompt"

        mock_llm = MagicMock()
        mock_llm.generate.return_value = ("The answer.", None)
        mock_llm.model_name = "gpt-4o"

        pipeline = QueryPipeline(
            embedder=mock_embedder,
            retriever=mock_retriever,
            prompt_builder=mock_prompt_builder,
            llm=mock_llm,
            collection="docs",
            evaluation_runner=runner,
        )
        pipeline.run("What is the answer?")
        runner.shutdown(wait=True)

        assert dispatched.wait(timeout=5), "Evaluator was never dispatched"
