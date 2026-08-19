"""Bound and redact tool input before it enters the approval store."""

from __future__ import annotations

import re
from typing import Any

from .state import JsonObject

_SECRET_TEXT_PATTERNS = (
    re.compile(r"(?i)\b(Bearer\s+)[A-Za-z0-9._~+/=-]+"),
    re.compile(r"(?i)\b([A-Z][A-Z0-9_]*(?:KEY|SECRET|TOKEN|PASSWORD)\s*=\s*)([^\s'\"]+|['\"][^'\"]*['\"])"),
    re.compile(
        r"\b(?:sk-[A-Za-z0-9_-]{12,}|ghp_[A-Za-z0-9_-]{12,}|github_pat_[A-Za-z0-9_-]{12,}|xox[baprs]-[A-Za-z0-9_-]{12,})\b",
        re.IGNORECASE,
    ),
    re.compile(r"-----BEGIN [^-]*PRIVATE KEY-----.*?-----END [^-]*PRIVATE KEY-----", re.DOTALL),
)
_MAX_STRING_CHARS = 2_000
_MAX_COLLECTION_ITEMS = 50
_MAX_DEPTH = 6


def redact_approval_payload(value: Any, *, _depth: int = 0) -> Any:
    """Return a size-bounded JSON-safe copy with likely secrets removed."""
    if _depth >= _MAX_DEPTH:
        return "[truncated]"
    if isinstance(value, dict):
        result: JsonObject = {}
        for index, (raw_key, item) in enumerate(value.items()):
            if index >= _MAX_COLLECTION_ITEMS:
                result["[truncated]"] = f"{len(value) - _MAX_COLLECTION_ITEMS} more fields"
                break
            key = str(raw_key)
            result[key] = "[redacted]" if _is_sensitive_key(key) else redact_approval_payload(item, _depth=_depth + 1)
        return result
    if isinstance(value, list | tuple):
        items = [redact_approval_payload(item, _depth=_depth + 1) for item in value[:_MAX_COLLECTION_ITEMS]]
        if len(value) > _MAX_COLLECTION_ITEMS:
            items.append(f"[truncated: {len(value) - _MAX_COLLECTION_ITEMS} more items]")
        return items
    if isinstance(value, str):
        redacted = value
        for pattern in _SECRET_TEXT_PATTERNS:
            if pattern.groups:
                redacted = pattern.sub(r"\1[redacted]", redacted)
            else:
                redacted = pattern.sub("[redacted]", redacted)
        if len(redacted) > _MAX_STRING_CHARS:
            return f"{redacted[:_MAX_STRING_CHARS]}… [truncated {len(redacted) - _MAX_STRING_CHARS} chars]"
        return redacted
    if value is None or isinstance(value, bool | int | float):
        return value
    return f"<{type(value).__name__}>"


def _is_sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.lower())
    return any(
        marker in normalized
        for marker in (
            "apikey",
            "auth",
            "cookie",
            "credential",
            "password",
            "passphrase",
            "privatekey",
            "secret",
            "token",
        )
    )
