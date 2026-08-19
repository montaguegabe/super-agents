"""Backend-neutral tool approval gating backed by the native approval store."""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from .app_formatting import as_object
from .app_models import PendingServerRequest
from .app_permissions import (
    clear_shared_permission_request,
    pop_shared_permission_decision,
    read_permission_store,
    record_shared_permission_request,
    write_permission_store,
)
from .app_time import iso_now, parse_iso_ms
from .approval_redaction import redact_approval_payload
from .execution_control import approval_action, canonical_action_digest
from .state import JsonObject

ApprovalDecision = Literal["accept", "decline", "cancel"]

DEFAULT_APPROVAL_TIMEOUT_SECONDS = 300.0
MAX_APPROVAL_TIMEOUT_SECONDS = 3600.0
DECISION_POLL_SECONDS = 0.25
MANAGED_GATE_MARKER = "super-agents-tool-gate-v1"

_DECISION_ALIASES: dict[str, ApprovalDecision] = {
    "accept": "accept",
    "approve": "accept",
    "approved": "accept",
    "decline": "decline",
    "deny": "decline",
    "denied": "decline",
    "reject": "decline",
    "cancel": "cancel",
    "cancelled": "cancel",
    "canceled": "cancel",
}


@dataclass(frozen=True, slots=True)
class ApprovalOutcome:
    decision: ApprovalDecision
    reason: Literal["answered", "timeout", "cancelled"]
    request: PendingServerRequest


def decision_from_answer(result: JsonObject) -> ApprovalDecision | None:
    """Normalize the approval and elicitation answer shapes accepted by MCP."""
    raw = result.get("decision") or result.get("action")
    if not isinstance(raw, str):
        return None
    return _DECISION_ALIASES.get(raw.strip().lower())


class ToolApprovalGate:
    """Track scoped approval requests and wait a bounded time for decisions."""

    def __init__(
        self,
        *,
        backend: str,
        request_method: str,
        requests_file: str | Path | None = None,
        timeout_seconds: float = DEFAULT_APPROVAL_TIMEOUT_SECONDS,
    ) -> None:
        if not backend.strip() or not request_method.strip():
            raise ValueError("backend and request_method are required for approval gating.")
        if timeout_seconds <= 0:
            raise ValueError("Approval timeout must be greater than zero.")
        self.backend = backend
        self.request_method = request_method
        self.requests_file = requests_file
        self.timeout_seconds = min(float(timeout_seconds), MAX_APPROVAL_TIMEOUT_SECONDS)
        self.owner_id = f"gate-{uuid.uuid4().hex}"
        self.owner_pid = os.getpid()
        self._pending: dict[str, PendingServerRequest] = {}
        self._decisions: dict[str, asyncio.Future[ApprovalDecision]] = {}
        self.clear_orphaned_requests()

    def pending_requests(self) -> list[PendingServerRequest]:
        return list(self._pending.values())

    def resolve(
        self,
        request_id: str | int,
        decision: ApprovalDecision,
        *,
        thread_id: str | None = None,
        turn_id: str | None = None,
    ) -> PendingServerRequest | None:
        """Resolve only the exact request and, when supplied, exact scope."""
        key = str(request_id)
        request = self._pending.get(key)
        future = self._decisions.get(key)
        if request is None or future is None or future.done():
            return None
        if thread_id is not None and request.params.get("threadId") != thread_id:
            return None
        if turn_id is not None and request.params.get("turnId") != turn_id:
            return None
        if decision not in {"accept", "decline", "cancel"}:
            return None
        future.set_result(decision)
        return request

    def cancel_scope(self, *, thread_id: str | None = None, turn_id: str | None = None) -> int:
        """Fail closed any pending requests owned by a cancelled scope."""
        cancelled = 0
        for request in list(self._pending.values()):
            if thread_id is not None and request.params.get("threadId") != thread_id:
                continue
            if turn_id is not None and request.params.get("turnId") != turn_id:
                continue
            if self.resolve(request.id, "cancel", thread_id=thread_id, turn_id=turn_id):
                cancelled += 1
        return cancelled

    async def request_permission(
        self,
        *,
        tool_name: str,
        tool_input: JsonObject,
        thread_id: str,
        turn_id: str | None,
        tool_call_id: str | None = None,
        description: str | None = None,
    ) -> ApprovalOutcome:
        """Persist one redacted request and wait for its scoped decision."""
        request = self._record_request(
            tool_name=tool_name,
            tool_input=tool_input,
            thread_id=thread_id,
            turn_id=turn_id,
            tool_call_id=tool_call_id,
            description=description,
        )
        key = str(request.id)
        try:
            return await self._await_decision(request)
        except asyncio.CancelledError:
            self.resolve(request.id, "cancel", thread_id=thread_id, turn_id=turn_id)
            raise
        finally:
            self._pending.pop(key, None)
            self._decisions.pop(key, None)
            clear_shared_permission_request(request.id, self.requests_file)

    def clear_orphaned_requests(self) -> int:
        """Remove expired requests or requests whose owning process exited."""
        store = read_permission_store(self.requests_file)
        requests = as_object(store.get("requests"))
        decisions = as_object(store.get("decisions"))
        removed = 0
        now_ms = int(time.time() * 1000)
        for request_id, raw_request in list(requests.items()):
            if not isinstance(raw_request, dict) or raw_request.get("method") != self.request_method:
                continue
            params = as_object(raw_request.get("params"))
            if params.get("approvalManagedBy") != MANAGED_GATE_MARKER or params.get("backend") != self.backend:
                continue
            expires_at = params.get("approvalExpiresAt")
            expires_ms = parse_iso_ms(expires_at if isinstance(expires_at, str) else None)
            owner_pid = params.get("approvalOwnerPid")
            if (
                (expires_ms and expires_ms <= now_ms)
                or not isinstance(owner_pid, int)
                or not _process_is_alive(owner_pid)
            ):
                requests.pop(request_id, None)
                decisions.pop(request_id, None)
                removed += 1
        if removed:
            store["requests"] = requests
            store["decisions"] = decisions
            write_permission_store(self.requests_file, store)
        return removed

    def _record_request(
        self,
        *,
        tool_name: str,
        tool_input: JsonObject,
        thread_id: str,
        turn_id: str | None,
        tool_call_id: str | None,
        description: str | None,
    ) -> PendingServerRequest:
        expires_at = datetime.now(timezone.utc) + timedelta(seconds=self.timeout_seconds)
        params: JsonObject = {
            "name": f"{self.backend}: {tool_name}",
            "backend": self.backend,
            "toolName": tool_name,
            "input": redact_approval_payload(tool_input),
            "threadId": thread_id,
            "description": redact_approval_payload(description)
            if description
            else f'Agent wants to use the "{tool_name}" tool.',
            "approvalManagedBy": MANAGED_GATE_MARKER,
            "approvalOwnerId": self.owner_id,
            "approvalOwnerPid": self.owner_pid,
            "approvalExpiresAt": expires_at.isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            # Safe to expose: this covers the original input before the
            # display copy is redacted and contains no reversible content.
            "approvalActionDigest": canonical_action_digest(
                approval_action(self.request_method, tool_name, tool_input)
            ),
        }
        if turn_id:
            params["turnId"] = turn_id
        if tool_call_id:
            params["toolCallId"] = tool_call_id
        request = PendingServerRequest(
            id=f"approval-{uuid.uuid4().hex}",
            method=self.request_method,
            params=params,
            received_at=iso_now(),
        )
        key = str(request.id)
        self._pending[key] = request
        self._decisions[key] = asyncio.get_running_loop().create_future()
        try:
            record_shared_permission_request(request, self.requests_file)
        except BaseException:
            self._pending.pop(key, None)
            self._decisions.pop(key, None)
            raise
        return request

    async def _await_decision(self, request: PendingServerRequest) -> ApprovalOutcome:
        future = self._decisions[str(request.id)]
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.timeout_seconds
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return ApprovalOutcome("decline", "timeout", request)
            if self._stored_request_matches(request):
                shared = pop_shared_permission_decision(request.id, self.requests_file)
                if shared is not None:
                    decision = decision_from_answer(shared)
                    if decision is not None:
                        return ApprovalOutcome(decision, "answered", request)
            remaining = deadline - loop.time()
            if remaining <= 0:
                return ApprovalOutcome("decline", "timeout", request)
            try:
                decision = await asyncio.wait_for(
                    asyncio.shield(future),
                    timeout=min(DECISION_POLL_SECONDS, remaining),
                )
                return ApprovalOutcome(decision, "cancelled" if decision == "cancel" else "answered", request)
            except TimeoutError:
                continue

    def _stored_request_matches(self, request: PendingServerRequest) -> bool:
        stored = as_object(read_permission_store(self.requests_file).get("requests")).get(str(request.id))
        if not isinstance(stored, dict) or stored.get("method") != request.method:
            return False
        params = as_object(stored.get("params"))
        return (
            params.get("approvalOwnerId") == self.owner_id
            and params.get("threadId") == request.params.get("threadId")
            and params.get("turnId") == request.params.get("turnId")
            and params.get("toolCallId") == request.params.get("toolCallId")
        )


def _process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True
