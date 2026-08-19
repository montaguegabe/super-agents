from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from super_agents.app_models import PendingServerRequest
from super_agents.app_server_client import CodexAppServerClient
from super_agents.approval_gate import ToolApprovalGate
from super_agents.claude_permissions import CLAUDE_APPROVAL_METHOD, can_use_tool_handler
from super_agents.execution_control import (
    ApprovalAuthorizationRequest,
    AuthorizationDecision,
    ExecutionControlError,
    ExecutionRequest,
    canonical_action_digest,
)


class CapturingPolicyGuard:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed
        self.requests: list[ExecutionRequest] = []

    async def validate(self, request: ExecutionRequest) -> AuthorizationDecision:
        self.requests.append(request)
        return AuthorizationDecision(self.allowed, "blocked by test" if not self.allowed else None)


class CapturingAuthorizer:
    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed
        self.requests: list[ApprovalAuthorizationRequest] = []

    async def authorize(self, request: ApprovalAuthorizationRequest) -> AuthorizationDecision:
        self.requests.append(request)
        return AuthorizationDecision(self.allowed, "blocked by test" if not self.allowed else None)


class RacingSingleUseAuthorizer:
    def __init__(self) -> None:
        self.requests: list[ApprovalAuthorizationRequest] = []
        self._ready = asyncio.Event()
        self._lock = asyncio.Lock()
        self._consumed: set[tuple[str, str]] = set()

    async def authorize(self, request: ApprovalAuthorizationRequest) -> AuthorizationDecision:
        self.requests.append(request)
        if len(self.requests) == 2:
            self._ready.set()
        await self._ready.wait()
        key = (request.request_id, request.action_digest)
        async with self._lock:
            if key in self._consumed:
                return AuthorizationDecision(False, "already consumed")
            self._consumed.add(key)
            return AuthorizationDecision(True)


class StubCodexClient(CodexAppServerClient):
    def __init__(self, state_file: Path, **kwargs: Any) -> None:
        super().__init__(state_file=state_file, **kwargs)
        self.sent: list[dict[str, Any]] = []

    async def ensure_connected(self) -> None:
        return None

    async def _login_shell_config_override(self) -> dict[str, Any]:
        return {}

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout: float = 30,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        del timeout, context
        self.sent.append({"method": method, "params": params or {}})
        return {}

    async def send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)


class FakeSdk:
    class PermissionResultAllow:
        def __init__(self, *, updated_input: dict[str, Any]) -> None:
            self.behavior = "allow"
            self.updated_input = updated_input

    class PermissionResultDeny:
        def __init__(self, *, message: str, interrupt: bool) -> None:
            self.behavior = "deny"
            self.message = message
            self.interrupt = interrupt


def test_canonical_action_digest_is_sorted_compact_utf8_sha256() -> None:
    first = {"method": "tool/call", "input": {"z": "é", "a": [1, True]}}
    second = {"input": {"a": [1, True], "z": "é"}, "method": "tool/call"}

    assert canonical_action_digest(first) == canonical_action_digest(second)
    assert canonical_action_digest(first) == "4bbe65f006416f5a44a6115c2c5cb431c21b37341534b9d68d5397269a931306"


@pytest.mark.asyncio
async def test_codex_thread_policy_digest_is_same_for_direct_and_mcp_metadata(tmp_path: Path) -> None:
    guard = CapturingPolicyGuard()
    client = StubCodexClient(tmp_path / "state.json", execution_policy_guard=guard)
    base = {
        "name": "agent",
        "cwd": str(tmp_path),
        "approvalPolicy": "on-request",
        "sandbox": "read-only",
    }

    await client.start_thread(base)
    await client.start_thread({**base, "_mcpCallId": "transport-only"})

    assert [request.requested_policy for request in guard.requests] == [
        {"approvalPolicy": "on-request", "sandboxPolicy": {"type": "readOnly"}},
        {"approvalPolicy": "on-request", "sandboxPolicy": {"type": "readOnly"}},
    ]
    assert guard.requests[0].action_digest == guard.requests[1].action_digest


@pytest.mark.asyncio
async def test_required_missing_policy_guard_fails_closed(tmp_path: Path) -> None:
    client = StubCodexClient(tmp_path / "state.json", require_controls=True)

    with pytest.raises(ExecutionControlError, match="unavailable"):
        await client.start_thread({"name": "agent", "cwd": str(tmp_path)})

    assert client.sent == []


@pytest.mark.asyncio
async def test_codex_accept_authorizes_exact_scope_but_decline_does_not(tmp_path: Path) -> None:
    authorizer = CapturingAuthorizer()
    client = StubCodexClient(tmp_path / "state.json", approval_authorizer=authorizer)
    request = PendingServerRequest(
        id="approval-1",
        method="exec/requestApproval",
        params={
            "threadId": "thread-a",
            "turnId": "turn-a",
            "toolCallId": "tool-a",
            "command": "echo secret-value",
            "description": "volatile display text",
        },
        received_at="volatile timestamp",
    )
    client._pending_server_requests[request.id] = request

    await client.answer_request(request.id, {"decision": "accept", "capability": "caller-value"})

    assert authorizer.requests == [
        ApprovalAuthorizationRequest(
            backend="codex",
            request_id="approval-1",
            action_type="exec/requestApproval",
            action_digest=canonical_action_digest(
                {
                    "method": "exec/requestApproval",
                    "tool": "exec/requestApproval",
                    "input": {"command": "echo secret-value"},
                }
            ),
            thread_id="thread-a",
            turn_id="turn-a",
            tool_call_id="tool-a",
        )
    ]
    assert client.sent[-1]["result"] == {"decision": "accept", "capability": "caller-value"}

    declined = PendingServerRequest(
        id="approval-2",
        method="exec/requestApproval",
        params={"threadId": "thread-a", "turnId": "turn-a", "command": "false"},
        received_at="now",
    )
    client._pending_server_requests[declined.id] = declined
    await client.answer_request(declined.id, {"decision": "decline"})
    assert len(authorizer.requests) == 1


@pytest.mark.asyncio
async def test_concurrent_codex_accepts_allow_exactly_one_authorizer_consume(tmp_path: Path) -> None:
    authorizer = RacingSingleUseAuthorizer()
    client = StubCodexClient(tmp_path / "state.json", approval_authorizer=authorizer)
    request = PendingServerRequest(
        id="approval-race",
        method="exec/requestApproval",
        params={"threadId": "thread-a", "turnId": "turn-a", "command": "deploy"},
        received_at="now",
    )
    client._pending_server_requests[request.id] = request

    results = await asyncio.gather(
        client.answer_request(request.id, {"decision": "accept"}),
        client.answer_request(request.id, {"decision": "accept"}),
        return_exceptions=True,
    )

    assert sum(isinstance(result, ExecutionControlError) for result in results) == 1
    assert sum(isinstance(result, dict) for result in results) == 1
    assert sum(message.get("id") == request.id for message in client.sent) == 1
    assert len(authorizer.requests) == 2
    assert authorizer.requests[0] == authorizer.requests[1]


@pytest.mark.asyncio
async def test_claude_authorizes_immediately_before_allow_with_unredacted_input(tmp_path: Path) -> None:
    gate = ToolApprovalGate(
        backend="claude_code",
        request_method=CLAUDE_APPROVAL_METHOD,
        requests_file=tmp_path / "approvals.json",
        timeout_seconds=1,
    )
    authorizer = CapturingAuthorizer()
    context = SimpleNamespace(tool_use_id="tool-a", description="display only")
    handler = can_use_tool_handler(
        FakeSdk,
        gate,
        "thread-a",
        lambda: "turn-a",
        authorizer,
        backend="claude_code",
    )
    task = asyncio.create_task(handler("Bash", {"command": "echo secret-value"}, context))
    while not gate.pending_requests():
        await asyncio.sleep(0)
    pending = gate.pending_requests()[0]
    assert gate.resolve(pending.id, "accept", thread_id="thread-a", turn_id="turn-a")

    result = await task

    assert result.behavior == "allow"
    assert authorizer.requests == [
        ApprovalAuthorizationRequest(
            backend="claude_code",
            request_id=str(pending.id),
            action_type="Bash",
            action_digest=canonical_action_digest(
                {
                    "method": CLAUDE_APPROVAL_METHOD,
                    "tool": "Bash",
                    "input": {"command": "echo secret-value"},
                }
            ),
            thread_id="thread-a",
            turn_id="turn-a",
            tool_call_id="tool-a",
        )
    ]


@pytest.mark.asyncio
async def test_claude_authorizer_denial_never_allows_tool(tmp_path: Path) -> None:
    gate = ToolApprovalGate(
        backend="claude_code",
        request_method=CLAUDE_APPROVAL_METHOD,
        requests_file=tmp_path / "approvals.json",
        timeout_seconds=1,
    )
    authorizer = CapturingAuthorizer(allowed=False)
    handler = can_use_tool_handler(FakeSdk, gate, "thread-a", lambda: "turn-a", authorizer)
    task = asyncio.create_task(handler("Write", {"path": "secret.txt"}, object()))
    while not gate.pending_requests():
        await asyncio.sleep(0)
    pending = gate.pending_requests()[0]
    assert gate.resolve(pending.id, "accept")

    result = await task

    assert result.behavior == "deny"
    assert "blocked by test" in result.message
