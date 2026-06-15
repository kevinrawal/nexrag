# NexRAG Security & Threat Model

NexRAG ships a **pluggable guard-chain** for RAG security. This document states,
honestly, what it defends against, what it does **not**, and what each guard
costs. There is no "blocks everything, zero overhead" claim — guards are
defense-in-depth controls you compose and tune, not a silver bullet.

To report a vulnerability, open a private security advisory on the GitHub repo.

## Architecture

A guard returns one of three verdicts:

- **ALLOW** — pass the text through (an input guard may also attach a retrieval
  metadata filter, e.g. access control).
- **BLOCK** — stop the request. The chain short-circuits; the pipeline raises
  `GuardrailBlockedError`.
- **REDACT(text)** — replace the text with a transformed (e.g. masked) version and
  continue.

Guards are composed into four ordered **chains**, each bound to a pipeline phase:

| Chain        | Runs on                                   | Typical guards                          |
|--------------|-------------------------------------------|-----------------------------------------|
| `ingestion`  | document text (after sanitizer)           | PII redaction                           |
| `input`      | the user query                            | access control, prompt-injection, topic |
| `retrieved`  | each retrieved chunk (before the prompt)  | prompt-injection, PII                   |
| `output`     | the LLM answer (before return/stream)     | PII, groundedness                       |

Each chain has a **policy** for guard *errors*:

- `fail_open` (default) — a broken guard is treated as ALLOW (and logged).
- `fail_closed` — a broken guard is treated as BLOCK. Choose this when a guard
  failure must not leak.

Every guard firing emits a `PipelineEvent(stage="guardrail", …)` carrying the
guard name, verdict, and latency — wire it into your observer to measure overhead
and audit decisions.

## Why retrieved content is guarded

Retrieved chunks are an **injection vector**: a poisoned document in your corpus
can carry "ignore previous instructions" into the prompt. The `retrieved` chain
exists specifically to scan retrieved content, not just user input.

## Guards shipped (value-to-cost order)

| Guard            | `type`             | Defends against                              | Overhead |
|------------------|--------------------|----------------------------------------------|----------|
| PII              | `pii`              | leaking emails/SSNs/cards/keys in/out        | regex: ~µs; Presidio: ~10s of ms + model load |
| Access control   | `access_control`   | cross-tenant data exposure                   | negligible (a filter on the query) |
| Prompt injection | `prompt_injection` | common jailbreak/injection phrasings         | ~µs (regex) |
| Groundedness     | `groundedness`     | ungrounded/hallucinated answers (cheap proxy)| ~µs (lexical overlap) |
| Topic            | `topic`            | off-policy / banned topics                   | ~µs (regex) |
| Model            | `model`            | nuanced unsafe content                       | **a full LLM round-trip** — not free |

**Access control is the highest-impact, lowest-cost guard** for multi-tenant
deployments. Pass `auth_context` to `query(...)`; the guard turns it into a
retrieval metadata filter so a request can only ever retrieve documents it is
authorised to see. Proven by test: unauthorized documents are excluded from
results.

## What NexRAG does NOT defend against (honest limits)

- **Prompt-injection heuristics catch common patterns, not a determined
  adversary.** They are regex/denylist based. For higher assurance add the
  `model` guard and rely on access control to bound blast radius.
- **Groundedness here is a cheap lexical-overlap proxy**, not a faithfulness
  judge. True LLM-judged faithfulness is intentionally **out of scope for the
  inline request path** — route it to an offline eval harness.
- **The regex PII path is best-effort.** Install `nexrag[pii]` (Microsoft
  Presidio) for a real PII engine; the regex fallback covers only common,
  high-value entities.
- **NexRAG does not manage** authentication, secret storage, network policy,
  rate limiting, or your vector DB's access controls. It manages data-flow guards
  only — `auth_context` is trusted input; you must authenticate the principal
  upstream.
- **Output guards disable true token streaming.** When an output chain is
  configured, the stream is buffered, guarded, then emitted as one chunk — you
  cannot un-send tokens already streamed.

## Performance honesty

Every guard above lists its measured overhead class. Model-based guards add a full
LLM call per check and are **opt-in**. Measure with the `guardrail` observability
events before enabling heavy guards on hot paths.

## Roadmap

- Streaming-aware incremental output guarding.
- Additional model-guard adapters (Llama-Guard, NeMo Guardrails, Guardrails-AI).
- LLM-judged faithfulness via the eval harness (offline).
