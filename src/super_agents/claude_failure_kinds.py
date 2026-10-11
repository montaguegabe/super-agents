"""Structured reasons for a failed Claude Code turn (``Turn.error_kind``).

Callers that retry or speak a failure need a stable kind, not error prose.
The SDK already labels a signed-out CLI (``authentication_failed``) and the
consent-dialog wait labels itself (``blocked_permission_dialog``); this module
names the two remaining operational cases and classifies an exception into
them. Anything else keeps ``error_kind`` empty and is reported by message.
"""

from __future__ import annotations

#: The CLI spawned but never answered the SDK's initialize request (a slow or
#: wedged start). Retrying once is reasonable.
START_TIMEOUT = "start_timeout"
#: The CLI or its transport could not be reached at all (binary missing,
#: connection refused, process gone before the first message). Retry once the
#: backend is back.
BACKEND_UNAVAILABLE = "backend_unavailable"

_START_TIMEOUT_MARKERS = ("control request timeout: initialize",)
_UNAVAILABLE_MARKERS = (
    "not running or not reachable",
    "connection refused",
    "claude code not found",
    "no such file or directory",
    "transport is not ready",
    "cannot write to terminated process",
    "process exited",
)
_UNAVAILABLE_TYPES = ("CLIConnectionError", "CLINotFoundError", "ConnectionRefusedError", "FileNotFoundError")


def classify_turn_failure(exc: BaseException) -> str | None:
    """Map an exception from a Claude Code turn to a known ``error_kind``."""
    text = str(exc).lower()
    if any(marker in text for marker in _START_TIMEOUT_MARKERS):
        return START_TIMEOUT
    for candidate in (exc, exc.__cause__, exc.__context__):
        if candidate is None:
            continue
        if type(candidate).__name__ in _UNAVAILABLE_TYPES:
            return BACKEND_UNAVAILABLE
    if any(marker in text for marker in _UNAVAILABLE_MARKERS):
        return BACKEND_UNAVAILABLE
    return None
