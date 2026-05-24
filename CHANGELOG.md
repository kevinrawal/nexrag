# Changelog

All notable changes to NexRAG will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added
- Project skeleton: folder structure, pyproject.toml, Makefile
- Core interface contracts: BaseLoader, BaseSanitizer, BaseChunker, BaseEmbedder, BaseVectorDB, BaseRetriever, BasePromptBuilder, BaseLLM, BaseObserver
- Core data models: Document, Chunk, ScoredChunk, PipelineResult, PipelineEvent
- Exception hierarchy: NexRAGError and all typed stage exceptions
- YAML config system with Pydantic schema validation
- Dynamic class resolver (dotted path → validated instance)

---

## Version History

| Version | Status | Description |
|---|---|---|
| 0.1.0 | 🔧 In Progress | Core interfaces + models + config system |
| 0.2.0 | Planned | All loaders + all chunkers + batch ingestion |
| 0.3.0 | Planned | ChromaDB adapter + fingerprinting + idempotency |
| 0.4.0 | Planned | All LLM and embedder adapters + streaming |
| 0.5.0 | Planned | Prompt builder + full YAML config + observability hook |
| 0.6.0 | Planned | Error handling hardening + integration tests |
| 1.0.0 | Planned | Public release — docs, examples, pip publish |