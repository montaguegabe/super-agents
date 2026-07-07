"""Shared approval-queue helpers, provided by the openapprovals package.

This module is the compatibility surface for the pre-openapprovals names;
new code should import from ``openapprovals`` directly. Store-path
resolution (including the legacy ``SUPER_AGENTS_APPROVAL_REQUESTS_FILE``
environment variable and the legacy ``~/.super-agents`` store location)
lives in ``openapprovals.store``.

Response-shape normalization stays here: elicitation requests
(``mcpServer/elicitation/request``) answer with an ``action`` payload while
approval requests answer with a ``decision`` payload, and the shared store
only records decisions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from openapprovals import DEFAULT_REQUESTS_FILE as DEFAULT_APPROVAL_REQUESTS_FILE
from openapprovals import clear_request as clear_shared_permission_request
from openapprovals import is_approval_request as is_permission_request
from openapprovals import pending_requests as shared_permission_requests
from openapprovals import pop_decision as _pop_shared_decision
from openapprovals import read_store as read_permission_store
from openapprovals import record_request as record_shared_permission_request
from openapprovals import write_decision as write_shared_permission_decision
from openapprovals import write_store as write_permission_store

from .app_formatting import as_object
from .app_models import PendingServerRequest
from .state import JsonObject

__all__ = [
    "DEFAULT_APPROVAL_REQUESTS_FILE",
    "clear_shared_permission_request",
    "is_permission_request",
    "normalize_permission_response",
    "permission_response_for_request",
    "pop_shared_permission_decision",
    "read_permission_store",
    "record_shared_permission_request",
    "shared_permission_requests",
    "write_permission_store",
    "write_shared_permission_decision",
]


def permission_response_for_request(
    request: JsonObject | PendingServerRequest | Any,
    decision: Literal["accept", "decline", "cancel"],
) -> JsonObject:
    method = _permission_request_method(request)
    if method == "mcpServer/elicitation/request":
        return {"action": decision, "content": None, "_meta": None}
    return {"decision": decision}


def normalize_permission_response(
    request: JsonObject | PendingServerRequest | Any,
    result: JsonObject,
) -> JsonObject:
    method = _permission_request_method(request)
    decision = result.get("decision")
    if (
        method == "mcpServer/elicitation/request"
        and "action" not in result
        and decision in {"accept", "decline", "cancel"}
    ):
        return permission_response_for_request(
            request,
            cast(Literal["accept", "decline", "cancel"], decision),
        )
    return result


def pop_shared_permission_decision(request_id: str | int, path: str | Path | None = None) -> JsonObject | None:
    """Pop a recorded decision, shaped for the request's method."""
    store = read_permission_store(path)
    requests = as_object(store.get("requests"))
    request = requests.get(str(request_id))
    raw = _pop_shared_decision(request_id, path)
    if raw is None:
        return None
    decision = cast(Literal["accept", "decline", "cancel"], raw["decision"])
    return permission_response_for_request(
        request if isinstance(request, dict) else {"method": ""},
        decision,
    )


def _permission_request_method(request: JsonObject | PendingServerRequest | Any) -> str:
    if isinstance(request, PendingServerRequest):
        return request.method
    if isinstance(request, dict):
        return str(request.get("method") or "")
    return str(getattr(request, "method", "") or "")
