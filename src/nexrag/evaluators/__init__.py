"""
NexRAG LLM-as-judge evaluators.

Each evaluator implements BaseEvaluator and uses injected BaseLLM / BaseEmbedder
adapters — resolved independently per metric by the factory from the per-metric
LLM/embedder config blocks under observability.evaluations.*.

Evaluators run off the query response path (sampled, async/thread-pool) and
emit MetricValue objects consumed by the EvaluationRunner.
"""
