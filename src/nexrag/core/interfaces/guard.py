"""
BaseGuard — contract for a single guardrail.

A guard inspects a piece of text (the user query, a retrieved chunk, or the LLM
answer) and returns a verdict:

    ALLOW                  — let the text through unchanged.
    BLOCK(reason)          — stop the request; the chain short-circuits.
    REDACT(text, reason)   — replace the text with a transformed (e.g. masked) version.

A guard may additionally attach a ``metadata_filter`` to an ALLOW verdict — this is
how the retrieval access-control guard restricts a request to documents the caller
is authorised to see, without transforming the query text.

Guards are composed into ordered GuardChains (input / retrieved / output / ingestion)
by the factory. Threat model and per-guard overhead live in SECURITY.md — guards are
honest about what they catch and what they cost.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Literal

GuardAction = Literal["allow", "block", "redact"]
GuardStage = Literal["ingestion", "query_input", "retrieved", "output"]


@dataclass(frozen=True)
class GuardResult:
    """
    The verdict a guard returns.

    Attributes:
        action:          "allow" | "block" | "redact".
        text:            The transformed text — set only when action == "redact".
        reason:          Human-readable explanation for block/redact (surfaced in errors/events).
        metadata_filter: Optional retrieval filter contributed by the guard (access control).
                         Merged into the query's metadata filter by the input chain.
    """

    action: GuardAction = "allow"
    text: str | None = None
    reason: str | None = None
    metadata_filter: dict[str, Any] | None = None

    @classmethod
    def allow(cls, *, metadata_filter: dict[str, Any] | None = None) -> GuardResult:
        return cls(action="allow", metadata_filter=metadata_filter)

    @classmethod
    def block(cls, reason: str) -> GuardResult:
        return cls(action="block", reason=reason)

    @classmethod
    def redact(cls, text: str, reason: str | None = None) -> GuardResult:
        return cls(action="redact", text=text, reason=reason)


@dataclass
class GuardContext:
    """
    Per-request context handed to every guard.

    Attributes:
        pipeline_id:  Ties guard events back to the pipeline run.
        stage:        Which chain is running this guard.
        query:        The original user query (available to retrieved/output guards too).
        auth_context: Per-request principal info (e.g. {"tenant": "acme", "roles": [...]}),
                      used by the access-control guard to build a retrieval filter.
        sources:      Retrieved chunk texts, populated for the output chain so a
                      groundedness guard can check the answer against its context.
    """

    pipeline_id: str
    stage: GuardStage
    query: str | None = None
    auth_context: dict[str, Any] | None = None
    sources: list[str] | None = None


class BaseGuard(ABC):
    """Abstract base class for all NexRAG guards."""

    #: Short stable identifier, surfaced in observability events. Override per guard.
    name: str = "guard"

    @abstractmethod
    def check(self, text: str, context: GuardContext) -> GuardResult:
        """
        Inspect ``text`` and return a verdict.

        Must not mutate ``text`` in place. Raising is allowed — the GuardChain
        applies the chain's fail_open / fail_closed policy to any exception.
        """
