"""Approval gating for the Claude Code backend.

Routes Claude Agent SDK ``can_use_tool`` permission checks through the shared
open-approvals queue, so approver surfaces that answer Codex app-server
approvals can answer Claude Code permission requests the same way: the gate
records a pending request, waits for a decision (answered in-process through
``answer_request`` or out-of-process through the shared store), and converts
it into the SDK permission result.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from open_approvals import iso_now

from super_agents.app_models import PendingServerRequest
from super_agents.app_permissions import (
    clear_shared_permission_request,
    pop_shared_permission_decision,
    record_shared_permission_request,
)

JsonObject = dict[str, Any]
TurnIdGetter = Callable[[], str | None]

CLAUDE_APPROVAL_METHOD = "claudeCode/requestApproval"
DECISION_POLL_SECONDS = 0.5

_DECISION_ALIASES = {
    "accept": "accept",
    "approve": "accept",
    "approved": "accept",
    "decline": "decline",
    "deny": "decline",
    "denied": "decline",
    "reject": "decline",
    "cancel": "cancel",
    "cancelled": "cancel",
}
_DENY_MESSAGES = {
    "decline": "Denied from Openbase approvals.",
    "cancel": "Cancelled from Openbase approvals.",
}


def decision_from_answer(result: JsonObject) -> str | None:
    """Extract an accept/decline/cancel decision from an answer payload.

    Accepts both approval answers (``{"decision": ...}``) and MCP elicitation
    answers (``{"action": ...}``).
    """
    raw = result.get("decision") or result.get("action")
    if not isinstance(raw, str):
        return None
    return _DECISION_ALIASES.get(raw.strip().lower())


class ClaudePermissionGate:
    """Tracks in-flight Claude Code permission requests and their decisions."""

    def __init__(self, requests_file: str | Path | None = None) -> None:
        self.requests_file = requests_file
        self._pending: dict[str, PendingServerRequest] = {}
        self._decisions: dict[str, asyncio.Future[str]] = {}

    def pending_requests(self) -> list[PendingServerRequest]:
        return list(self._pending.values())

    def resolve(self, request_id: str | int, decision: str) -> PendingServerRequest | None:
        """Answer one pending request in-process. Returns None when unknown."""
        key = str(request_id)
        request = self._pending.get(key)
        future = self._decisions.get(key)
        if request is None or future is None or future.done():
            return None
        future.set_result(decision)
        return request

    async def request_permission(
        self,
        *,
        tool_name: str,
        tool_input: JsonObject,
        thread_id: str,
        turn_id: str | None,
        description: str | None = None,
    ) -> str:
        """Queue one approval request and wait for its decision."""
        request = self._record_request(
            tool_name=tool_name,
            tool_input=tool_input,
            thread_id=thread_id,
            turn_id=turn_id,
            description=description,
        )
        key = str(request.id)
        try:
            return await self._await_decision(request)
        finally:
            self._pending.pop(key, None)
            self._decisions.pop(key, None)
            clear_shared_permission_request(request.id, self.requests_file)

    def _record_request(
        self,
        *,
        tool_name: str,
        tool_input: JsonObject,
        thread_id: str,
        turn_id: str | None,
        description: str | None,
    ) -> PendingServerRequest:
        params: JsonObject = {
            "name": f"Claude Code: {tool_name}",
            "toolName": tool_name,
            "input": tool_input,
            "threadId": thread_id,
            "description": description or f'Claude Code wants to use the "{tool_name}" tool.',
        }
        if turn_id:
            params["turnId"] = turn_id
        command = tool_input.get("command")
        if isinstance(command, str) and command:
            params["command"] = command
        request = PendingServerRequest(
            id=f"claude-{uuid.uuid4().hex}",
            method=CLAUDE_APPROVAL_METHOD,
            params=params,
            received_at=iso_now(),
        )
        self._pending[str(request.id)] = request
        self._decisions[str(request.id)] = asyncio.get_running_loop().create_future()
        record_shared_permission_request(request, self.requests_file)
        return request

    async def _await_decision(self, request: PendingServerRequest) -> str:
        future = self._decisions[str(request.id)]
        while True:
            shared = pop_shared_permission_decision(request.id, self.requests_file)
            if shared is not None:
                decision = decision_from_answer(shared)
                if decision is not None:
                    return decision
            try:
                return await asyncio.wait_for(asyncio.shield(future), DECISION_POLL_SECONDS)
            except TimeoutError:
                continue


def can_use_tool_handler(
    sdk: Any,
    gate: ClaudePermissionGate,
    thread_id: str,
    turn_id_getter: TurnIdGetter,
) -> Any:
    """Build a Claude Agent SDK ``can_use_tool`` callback backed by the gate."""

    async def can_use_tool(tool_name: str, tool_input: JsonObject, context: Any) -> Any:
        decision = await gate.request_permission(
            tool_name=tool_name,
            tool_input=tool_input,
            thread_id=thread_id,
            turn_id=turn_id_getter(),
            description=_context_description(context),
        )
        if decision == "accept":
            return sdk.PermissionResultAllow()
        return sdk.PermissionResultDeny(
            message=_DENY_MESSAGES.get(decision, _DENY_MESSAGES["decline"]),
            interrupt=decision == "cancel",
        )

    return can_use_tool


def _context_description(context: Any) -> str | None:
    for attribute in ("description", "title", "display_name"):
        value = getattr(context, attribute, None)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None
