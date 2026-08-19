from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from super_agents.agent_store import Store
from super_agents.approval_gate import ToolApprovalGate
from super_agents.claude_options import agent_options
from super_agents.claude_permissions import CLAUDE_APPROVAL_METHOD, can_use_tool_handler
from super_agents.claude_sdk import ClaudeAgentSdkClient


class FakePermissionResultAllow:
    behavior = "allow"

    def __init__(self, updated_input: dict[str, Any] | None = None) -> None:
        self.updated_input = updated_input


class FakePermissionResultDeny:
    behavior = "deny"

    def __init__(self, message: str, interrupt: bool) -> None:
        self.message = message
        self.interrupt = interrupt


class FakeOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


FAKE_SDK = SimpleNamespace(
    ClaudeAgentOptions=FakeOptions,
    PermissionResultAllow=FakePermissionResultAllow,
    PermissionResultDeny=FakePermissionResultDeny,
)


async def wait_for_pending(gate: ToolApprovalGate) -> None:
    async with asyncio.timeout(1):
        while not gate.pending_requests():
            await asyncio.sleep(0.005)


def test_agent_options_fail_closed_without_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_PERMISSION_MODE", "default")

    with pytest.raises(RuntimeError, match="no approval handler"):
        agent_options(FAKE_SDK, "/tmp", None, None, resume=None)


def test_agent_options_attach_gate_only_for_gated_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    handler = object()
    monkeypatch.delenv("SUPER_AGENTS_CLAUDE_PERMISSION_MODE", raising=False)
    options = agent_options(FAKE_SDK, "/tmp", None, None, resume=None, can_use_tool=handler)
    assert options.kwargs["permission_mode"] == "bypassPermissions"
    assert "can_use_tool" not in options.kwargs

    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_PERMISSION_MODE", "acceptEdits")
    options = agent_options(FAKE_SDK, "/tmp", None, None, resume=None, can_use_tool=handler)
    assert options.kwargs["permission_mode"] == "acceptEdits"
    assert options.kwargs["can_use_tool"] is handler


@pytest.mark.asyncio
async def test_denied_tool_never_executes(tmp_path: Path) -> None:
    gate = ToolApprovalGate(
        backend="claude_code",
        request_method=CLAUDE_APPROVAL_METHOD,
        requests_file=tmp_path / "approvals.json",
        timeout_seconds=1,
    )
    handler = can_use_tool_handler(FAKE_SDK, gate, "thread-a", lambda: "turn-a")
    executed: list[str] = []
    callback = asyncio.create_task(handler("Bash", {"command": "deploy"}, object()))
    await wait_for_pending(gate)
    request = gate.pending_requests()[0]
    assert gate.resolve(request.id, "decline", thread_id="thread-a", turn_id="turn-a")
    permission = await callback
    if permission.behavior == "allow":
        executed.append("deploy")

    assert permission.behavior == "deny"
    assert permission.interrupt is False
    assert executed == []


@pytest.mark.asyncio
async def test_timeout_is_sdk_denial(tmp_path: Path) -> None:
    gate = ToolApprovalGate(
        backend="claude_code",
        request_method=CLAUDE_APPROVAL_METHOD,
        requests_file=tmp_path / "approvals.json",
        timeout_seconds=0.02,
    )
    handler = can_use_tool_handler(FAKE_SDK, gate, "thread-a", lambda: "turn-a")

    permission = await handler("Write", {"file_path": "x"}, object())

    assert permission.behavior == "deny"
    assert permission.interrupt is False
    assert "timed out" in permission.message


@pytest.mark.asyncio
async def test_cancelled_scope_is_sdk_denial_and_leaves_no_runnable_request(tmp_path: Path) -> None:
    approvals_path = tmp_path / "approvals.json"
    gate = ToolApprovalGate(
        backend="claude_code",
        request_method=CLAUDE_APPROVAL_METHOD,
        requests_file=approvals_path,
        timeout_seconds=1,
    )
    handler = can_use_tool_handler(FAKE_SDK, gate, "thread-a", lambda: "turn-a")
    executed: list[str] = []
    callback = asyncio.create_task(handler("Bash", {"command": "deploy"}, object()))
    await wait_for_pending(gate)

    assert gate.cancel_scope(thread_id="thread-a", turn_id="turn-a") == 1
    permission = await callback
    if permission.behavior == "allow":
        executed.append("deploy")

    assert permission.behavior == "deny"
    assert permission.interrupt is True
    assert executed == []
    assert gate.pending_requests() == []
    assert not approvals_path.read_text(encoding="utf-8").count("approval-")


@pytest.mark.asyncio
async def test_client_exposes_and_answers_only_its_own_requests(tmp_path: Path) -> None:
    approvals = tmp_path / "approvals.json"
    first = ClaudeAgentSdkClient(
        store=Store(tmp_path / "first.sqlite3"),
        sdk_loader=lambda: FAKE_SDK,
        approval_requests_file=approvals,
    )
    second = ClaudeAgentSdkClient(
        store=Store(tmp_path / "second.sqlite3"),
        sdk_loader=lambda: FAKE_SDK,
        approval_requests_file=approvals,
    )
    task = asyncio.create_task(
        first._permission_gate.request_permission(
            tool_name="Bash",
            tool_input={"command": "deploy"},
            thread_id="thread-first",
            turn_id="turn-first",
        )
    )
    await wait_for_pending(first._permission_gate)
    request_id = first._permission_gate.pending_requests()[0].id

    status = await first.status()
    assert [item["id"] for item in status["pendingPermissionRequests"]] == [request_id]
    with pytest.raises(ValueError, match="No pending permission request"):
        await second.answer_request(request_id, {"decision": "accept"})
    assert not task.done()

    answered = await first.answer_request(request_id, {"decision": "accept"})
    assert answered["answered"] is True
    assert (await task).decision == "accept"
