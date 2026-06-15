"""
AccessControlGuard — per-request retrieval access control.

The single most impactful "security" feature for multi-tenant RAG, and usually
missing: it turns the caller's ``auth_context`` (passed to NexRAG.query(...)) into a
metadata filter so a request can only ever retrieve documents it is authorised to
see. It is cheap — a filter on the query — and depends only on the vector DB's
metadata filtering (Chroma and Pinecone both support it).

Runs on the input chain. Deny-by-default: if no auth context is supplied and
``require_auth`` is true, the request is BLOCKED rather than retrieving everything.
"""

from __future__ import annotations

from typing import Any

from nexrag.core.interfaces.guard import BaseGuard, GuardContext, GuardResult


class AccessControlGuard(BaseGuard):
    """
    Args:
        mapping:      auth_context key -> document metadata field. e.g. {"tenant": "tenant_id"}.
                      If empty, ``fields`` is used; if that is empty too, every auth_context
                      key is matched against the same-named metadata field (identity mapping).
        fields:       Convenience: a list of fields used as an identity mapping.
        require_auth: Block when no usable auth context is present. Default True (secure default).
    """

    name = "access_control"

    def __init__(
        self,
        mapping: dict[str, str] | None = None,
        fields: list[str] | None = None,
        require_auth: bool = True,
    ) -> None:
        if not mapping and fields:
            mapping = {f: f for f in fields}
        self._mapping = mapping or {}
        self._require_auth = require_auth

    def check(self, text: str, context: GuardContext) -> GuardResult:
        auth = context.auth_context or {}
        if not auth:
            if self._require_auth:
                return GuardResult.block(
                    reason="Access denied: no auth_context supplied for access-control guard."
                )
            return GuardResult.allow()

        mapping = self._mapping or {key: key for key in auth}
        metadata_filter: dict[str, Any] = {}
        for auth_key, meta_field in mapping.items():
            if auth_key not in auth:
                continue
            value = auth[auth_key]
            metadata_filter[meta_field] = (
                {"$in": value} if isinstance(value, list) else {"$eq": value}
            )

        if not metadata_filter:
            if self._require_auth:
                return GuardResult.block(
                    reason="Access denied: auth_context has no fields matching the access-control mapping."
                )
            return GuardResult.allow()

        return GuardResult.allow(metadata_filter=metadata_filter)
