from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest
from open_approvals import pending_requests, read_store, write_decision

from super_agents.agent_store import Store
from super_agents.claude_options import agent_options
from super_agents.claude_permissions import (
    CLAUDE_APPROVAL_METHOD,
    ClaudePermissionGate,
    can_use_tool_handler,
    decision_from_answer,
)
from super_agents.claude_sdk import ClaudeAgentSdkClient


class FakeClaudeAgentOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakePermissionResultAllow:
    def __init__(self, **kwargs: Any) -> None:
        self.behavior = "allow"
        self.kwargs = kwargs


class FakePermissionResultDeny:
    def __init__(self, message: str = "", interrupt: bool = False) -> None:
        self.behavior = "deny"
        self.message = message
        self.interrupt = interrupt


class FakeSdk:
    ClaudeAgentOptions = FakeClaudeAgentOptions
    PermissionResultAllow = FakePermissionResultAllow
    PermissionResultDeny = FakePermissionResultDeny


@pytest.fixture()
def approvals_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "approval-requests.json"
    monkeypatch.setenv("OPEN_APPROVALS_REQUESTS_FILE", str(path))
    return path


@pytest.mark.asyncio
async def test_gate_records_request_and_applies_shared_store_decision(approvals_file: Path) -> None:
    gate = ClaudePermissionGate()
    task = asyncio.create_task(
        gate.request_permission(
            tool_name="Bash",
            tool_input={"command": "rm -rf build"},
            thread_id="thread-1",
            turn_id="turn-1",
        )
    )
    await asyncio.sleep(0.05)

    pending = pending_requests()
    assert len(pending) == 1
    request = pending[0]
    assert request["method"] == CLAUDE_APPROVAL_METHOD
    assert request["params"]["toolName"] == "Bash"
    assert request["params"]["command"] == "rm -rf build"
    assert request["params"]["threadId"] == "thread-1"
    assert request["params"]["turnId"] == "turn-1"
    assert gate.pending_requests()[0].id == request["id"]

    assert write_decision(request["id"], "accept")
    assert await asyncio.wait_for(task, timeout=5) == "accept"
    assert gate.pending_requests() == []
    store = read_store()
    assert store["requests"] == {}
    assert store["decisions"] == {}


@pytest.mark.asyncio
async def test_gate_resolve_answers_in_process(approvals_file: Path) -> None:
    gate = ClaudePermissionGate()
    task = asyncio.create_task(
        gate.request_permission(
            tool_name="Write",
            tool_input={"file_path": "/tmp/x"},
            thread_id="thread-1",
            turn_id=None,
        )
    )
    await asyncio.sleep(0.05)
    request = gate.pending_requests()[0]

    assert gate.resolve(request.id, "decline") is not None
    assert await asyncio.wait_for(task, timeout=5) == "decline"
    assert gate.resolve(request.id, "decline") is None


@pytest.mark.asyncio
async def test_can_use_tool_handler_maps_decisions_to_sdk_results(approvals_file: Path) -> None:
    gate = ClaudePermissionGate()
    handler = can_use_tool_handler(FakeSdk, gate, "thread-9", lambda: "turn-9")

    async def answer(decision: str) -> None:
        while not gate.pending_requests():
            await asyncio.sleep(0.01)
        gate.resolve(gate.pending_requests()[0].id, decision)

    answer_task = asyncio.create_task(answer("accept"))
    allowed = await asyncio.wait_for(handler("Bash", {"command": "ls"}, object()), timeout=5)
    await answer_task
    assert allowed.behavior == "allow"

    answer_task = asyncio.create_task(answer("cancel"))
    denied = await asyncio.wait_for(handler("Bash", {"command": "ls"}, object()), timeout=5)
    await answer_task
    assert denied.behavior == "deny"
    assert denied.interrupt is True


@pytest.mark.asyncio
async def test_client_answer_request_resolves_gate_request(approvals_file: Path, tmp_path: Path) -> None:
    client = ClaudeAgentSdkClient(store=Store(tmp_path / "state.sqlite3"), sdk_loader=lambda: FakeSdk())
    gate = client._permission_gate
    task = asyncio.create_task(
        gate.request_permission(
            tool_name="Bash",
            tool_input={"command": "make deploy"},
            thread_id="thread-2",
            turn_id="turn-2",
        )
    )
    await asyncio.sleep(0.05)
    request_id = gate.pending_requests()[0].id

    status = await client.status()
    assert [item["id"] for item in status["pendingPermissionRequests"]] == [request_id]
    assert [item.id for item in client.pending_permission_requests()] == [request_id]

    answered = await client.answer_request(request_id, {"decision": "accept"})
    assert answered["answered"] is True
    assert await asyncio.wait_for(task, timeout=5) == "accept"

    with pytest.raises(ValueError):
        await client.answer_request(request_id, {"decision": "accept"})


@pytest.mark.asyncio
async def test_client_answer_request_rejects_unknown_decisions(tmp_path: Path) -> None:
    client = ClaudeAgentSdkClient(store=Store(tmp_path / "state.sqlite3"), sdk_loader=lambda: FakeSdk())
    with pytest.raises(ValueError):
        await client.answer_request("missing", {"decision": "maybe"})


def test_decision_from_answer_accepts_codex_and_elicitation_shapes() -> None:
    assert decision_from_answer({"decision": "accept"}) == "accept"
    assert decision_from_answer({"decision": "approved"}) == "accept"
    assert decision_from_answer({"action": "decline"}) == "decline"
    assert decision_from_answer({"action": "cancel"}) == "cancel"
    assert decision_from_answer({"decision": 7}) is None
    assert decision_from_answer({}) is None


def test_agent_options_only_gates_outside_bypass_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    async def can_use_tool(*args: Any) -> Any:  # pragma: no cover - never invoked
        raise AssertionError

    monkeypatch.delenv("SUPER_AGENTS_CLAUDE_PERMISSION_MODE", raising=False)
    options = agent_options(FakeSdk, "/tmp", None, None, resume=None, can_use_tool=can_use_tool)
    assert options.kwargs["permission_mode"] == "bypassPermissions"
    assert "can_use_tool" not in options.kwargs

    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_PERMISSION_MODE", "acceptEdits")
    options = agent_options(FakeSdk, "/tmp", None, None, resume=None, can_use_tool=can_use_tool)
    assert options.kwargs["permission_mode"] == "acceptEdits"
    assert options.kwargs["can_use_tool"] is can_use_tool
