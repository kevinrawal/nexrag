"""
PIIGuard — detects and redacts (or blocks) personal data.

Prefers Microsoft Presidio (a real PII engine) when installed; otherwise falls
back to a deterministic regex detector covering the common high-value entities
(emails, phone numbers, SSNs, credit cards, IP addresses, API keys). Usable in any
chain — redact document text at ingestion, or scrub the query/answer.

Overhead (honest): the regex path is microseconds; the Presidio path adds tens of
milliseconds plus a one-time model load. Presidio is more accurate. See SECURITY.md.

Requires (for the Presidio path): pip install "nexrag[pii]"
"""

from __future__ import annotations

import re
from typing import Any

from nexrag.core.interfaces.guard import BaseGuard, GuardContext, GuardResult

# (label, compiled pattern) — order matters: more specific patterns first.
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ -]?){13,16}\b")),
    ("API_KEY", re.compile(r"\b(?:sk|pk|api|key|token|secret)[-_][A-Za-z0-9]{16,}\b", re.I)),
    ("PHONE", re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")),
    ("IP_ADDRESS", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


class PIIGuard(BaseGuard):
    """
    Args:
        mode:         "redact" (default) replaces PII with a label; "block" rejects the text.
        use_presidio: Try Presidio first when available. Default True.
        entities:     Optional Presidio entity allowlist (e.g. ["EMAIL_ADDRESS", "PHONE_NUMBER"]).
        language:     Presidio language. Default "en".
        mask:         Redaction template; "{label}" is substituted (default "[{label}]").
    """

    name = "pii"

    def __init__(
        self,
        mode: str = "redact",
        use_presidio: bool = True,
        entities: list[str] | None = None,
        language: str = "en",
        mask: str = "[{label}]",
    ) -> None:
        self._mode = mode
        self._use_presidio = use_presidio
        self._entities = entities
        self._language = language
        self._mask = mask
        self._analyzer: Any = None
        self._anonymizer: Any = None

    def check(self, text: str, context: GuardContext) -> GuardResult:
        if not text.strip():
            return GuardResult.allow()

        found, redacted = self._scan(text)
        if not found:
            return GuardResult.allow()
        if self._mode == "block":
            return GuardResult.block(reason=f"PII detected ({', '.join(sorted(found))}).")
        return GuardResult.redact(redacted, reason=f"Redacted PII ({', '.join(sorted(found))}).")

    def _scan(self, text: str) -> tuple[set[str], str]:
        if self._use_presidio:
            result = self._scan_presidio(text)
            if result is not None:
                return result
        return self._scan_regex(text)

    def _scan_presidio(self, text: str) -> tuple[set[str], str] | None:
        try:
            engines = self._presidio_engines()
        except Exception:
            return None  # not installed or model missing — caller falls back to regex
        analyzer, anonymizer = engines
        try:
            results = analyzer.analyze(text=text, language=self._language, entities=self._entities)
            if not results:
                return set(), text
            labels = {r.entity_type for r in results}
            anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
            return labels, anonymized.text
        except Exception:
            return None

    def _presidio_engines(self) -> tuple[Any, Any]:
        if self._analyzer is None or self._anonymizer is None:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine

            self._analyzer = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
        return self._analyzer, self._anonymizer

    def _scan_regex(self, text: str) -> tuple[set[str], str]:
        found: set[str] = set()
        redacted = text
        for label, pattern in _PATTERNS:
            if pattern.search(redacted):
                found.add(label)
                redacted = pattern.sub(self._mask.format(label=label), redacted)
        return found, redacted
