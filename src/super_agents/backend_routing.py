"""Stateless helpers for deterministic configured-backend routing."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .app_models import LabelQueryInput
from .backend_config import CLAUDE_CODE_BACKEND, execution_backend

JsonObject = dict[str, Any]


class BackendResolutionError(ValueError):
    """Base error for deterministic backend resolution failures."""


class BackendNotFoundError(BackendResolutionError):
    """No configured backend owns the requested identifier or name."""


class AmbiguousBackendError(BackendResolutionError):
    """More than one configured backend owns the requested name."""


class BackendEndpointConflictError(BackendResolutionError):
    """Two configured identities would target the same process-global endpoint."""


def backend_has_local_state(identity: str) -> bool:
    if execution_backend(identity) == CLAUDE_CODE_BACKEND:
        from .agent_store import database_path

        return database_path().exists()
    from .app_server_client import DEFAULT_STATE_FILE

    configured = os.environ.get("SUPER_AGENTS_STATE_FILE")
    return (Path(configured).expanduser() if configured else DEFAULT_STATE_FILE).exists()


def require_one_backend(matches: list[str], *, subject: str) -> str:
    unique = list(dict.fromkeys(matches))
    if not unique:
        raise BackendNotFoundError(f"No configured backend owns {subject}.")
    if len(unique) > 1:
        choices = ", ".join(unique)
        raise AmbiguousBackendError(
            f"Ambiguous Super Agents {subject}; matching backends: {choices}. "
            "Provide backend or an authoritative threadId."
        )
    return unique[0]


def session_matches(session: JsonObject, query: LabelQueryInput) -> bool:
    name = session.get("name") or session.get("label")
    cwd = session.get("cwd")
    return name == query.label and (not query.cwd or cwd == query.cwd)


def session_thread_id(session: JsonObject) -> str | None:
    value = session.get("threadId") or session.get("sessionId") or session.get("id")
    return value if isinstance(value, str) and value else None


def client_has_pending_request(client: Any, request_id: str | int) -> bool:
    pending = client.pending_permission_requests()
    return any(str(_request_id(request)) == str(request_id) for request in pending)


def _request_id(request: Any) -> str | int | None:
    if isinstance(request, dict):
        return request.get("id")
    return getattr(request, "id", None)


def annotated_items(value: Any, identity: str) -> list[JsonObject]:
    if not isinstance(value, list):
        return []
    return [{**item, "backend": identity} for item in value if isinstance(item, dict)]


def collect_result_ids(
    value: Any,
    *,
    thread_ids: set[str],
    turn_ids: set[str],
    request_ids: set[str],
    parent_key: str | None = None,
) -> None:
    if isinstance(value, list):
        for item in value:
            collect_result_ids(
                item,
                thread_ids=thread_ids,
                turn_ids=turn_ids,
                request_ids=request_ids,
                parent_key=parent_key,
            )
        return
    if not isinstance(value, dict):
        return
    for key in ("threadId", "sessionId"):
        if isinstance(value.get(key), str) and value[key]:
            thread_ids.add(value[key])
    for key in ("turnId", "queueItemId"):
        if isinstance(value.get(key), str) and value[key]:
            turn_ids.add(value[key])
    if parent_key in {"pendingRequests", "pendingPermissionRequests"}:
        request_id = value.get("id")
        if isinstance(request_id, str | int):
            request_ids.add(str(request_id))
    if value.get("status") == "queued" and isinstance(value.get("id"), str):
        turn_ids.add(value["id"])
    for key, item in value.items():
        collect_result_ids(
            item,
            thread_ids=thread_ids,
            turn_ids=turn_ids,
            request_ids=request_ids,
            parent_key=key,
        )


def optional_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
