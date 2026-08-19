from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Literal, cast

from .app_formatting import as_object
from .app_models import PendingServerRequest
from .app_time import iso_now
from .state import JsonObject

DEFAULT_APPROVAL_REQUESTS_FILE = Path.home() / ".super-agents" / "approval-requests.json"


def is_permission_request(method: str) -> bool:
    return "requestApproval" in method or method == "mcpServer/elicitation/request"


def shared_permission_requests(path: str | Path | None = None) -> list[JsonObject]:
    store = read_permission_store(path)
    raw_requests = as_object(store.get("requests"))
    raw_decisions = as_object(store.get("decisions"))
    return [
        item
        for request_id, item in raw_requests.items()
        if isinstance(item, dict)
        and is_permission_request(str(item.get("method") or ""))
        and request_id not in raw_decisions
    ]


def record_shared_permission_request(
    request: PendingServerRequest,
    path: str | Path | None = None,
) -> None:
    store = read_permission_store(path)
    requests = as_object(store.get("requests"))
    requests[str(request.id)] = request.to_json()
    store["requests"] = requests
    store["decisions"] = as_object(store.get("decisions"))
    write_permission_store(path, store)


def clear_shared_permission_request(request_id: str | int, path: str | Path | None = None) -> None:
    store = read_permission_store(path)
    requests = as_object(store.get("requests"))
    decisions = as_object(store.get("decisions"))
    requests.pop(str(request_id), None)
    decisions.pop(str(request_id), None)
    store["requests"] = requests
    store["decisions"] = decisions
    write_permission_store(path, store)


def write_shared_permission_decision(
    request_id: str | int,
    decision: Literal["accept", "decline", "cancel"],
    path: str | Path | None = None,
) -> bool:
    store = read_permission_store(path)
    requests = as_object(store.get("requests"))
    if str(request_id) not in requests:
        return False
    decisions = as_object(store.get("decisions"))
    decisions[str(request_id)] = {"decision": decision, "decidedAt": iso_now()}
    store["requests"] = requests
    store["decisions"] = decisions
    write_permission_store(path, store)
    return True


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


def _permission_request_method(request: JsonObject | PendingServerRequest | Any) -> str:
    if isinstance(request, PendingServerRequest):
        return request.method
    if isinstance(request, dict):
        return str(request.get("method") or "")
    return str(getattr(request, "method", "") or "")


def pop_shared_permission_decision(request_id: str | int, path: str | Path | None = None) -> JsonObject | None:
    store = read_permission_store(path)
    requests = as_object(store.get("requests"))
    request = requests.get(str(request_id))
    method = str(request.get("method") or "") if isinstance(request, dict) else ""
    decisions = as_object(store.get("decisions"))
    raw_decision = decisions.pop(str(request_id), None)
    if not isinstance(raw_decision, dict):
        return None
    decision = raw_decision.get("decision")
    if decision not in {"accept", "decline", "cancel"}:
        return None
    store["requests"] = requests
    store["decisions"] = decisions
    write_permission_store(path, store)
    return permission_response_for_request(
        request if isinstance(request, dict) else {"method": method},
        decision,
    )


def read_permission_store(path: str | Path | None = None) -> JsonObject:
    store_path = Path(path or os.environ.get("SUPER_AGENTS_APPROVAL_REQUESTS_FILE") or DEFAULT_APPROVAL_REQUESTS_FILE)
    try:
        raw = json.loads(store_path.read_text(encoding="utf-8"))
    except Exception:
        return {"requests": {}, "decisions": {}}
    if not isinstance(raw, dict):
        return {"requests": {}, "decisions": {}}
    return {"requests": as_object(raw.get("requests")), "decisions": as_object(raw.get("decisions"))}


def write_permission_store(path: str | Path | None, store: JsonObject) -> None:
    store_path = Path(path or os.environ.get("SUPER_AGENTS_APPROVAL_REQUESTS_FILE") or DEFAULT_APPROVAL_REQUESTS_FILE)
    store_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Approval payloads can contain commands, paths, and redacted input
    # context. Do not depend on the host's umask to keep that metadata private.
    os.chmod(store_path.parent, 0o700)
    payload = json.dumps(
        {"requests": as_object(store.get("requests")), "decisions": as_object(store.get("decisions"))},
        indent=2,
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=store_path.parent, delete=False) as tmp:
        os.chmod(tmp.name, 0o600)
        tmp.write(payload + "\n")
        tmp_name = tmp.name
    os.replace(tmp_name, store_path)
    os.chmod(store_path, 0o600)
