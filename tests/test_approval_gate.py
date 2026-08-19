from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

import super_agents.approval_gate as approval_gate_module
from super_agents.app_models import PendingServerRequest
from super_agents.app_permissions import (
    read_permission_store,
    record_shared_permission_request,
    write_shared_permission_decision,
)
from super_agents.app_time import iso_now
from super_agents.approval_gate import (
    MANAGED_GATE_MARKER,
    MAX_APPROVAL_TIMEOUT_SECONDS,
    ToolApprovalGate,
)
from super_agents.approval_redaction import redact_approval_payload


def make_gate(path: Path, *, timeout: float = 1.0) -> ToolApprovalGate:
    return ToolApprovalGate(
        backend="test-backend",
        request_method="testBackend/requestApproval",
        requests_file=path,
        timeout_seconds=timeout,
    )


async def wait_for_pending(gate: ToolApprovalGate, count: int = 1) -> None:
    async with asyncio.timeout(1):
        while len(gate.pending_requests()) != count:
            await asyncio.sleep(0.005)


@pytest.mark.asyncio
async def test_native_store_decision_resolves_only_exact_request(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    gate = make_gate(path)
    first = asyncio.create_task(
        gate.request_permission(tool_name="Write", tool_input={}, thread_id="thread-a", turn_id="turn-a")
    )
    second = asyncio.create_task(
        gate.request_permission(tool_name="Bash", tool_input={}, thread_id="thread-b", turn_id="turn-b")
    )
    await wait_for_pending(gate, 2)
    by_thread = {request.params["threadId"]: request for request in gate.pending_requests()}

    assert write_shared_permission_decision(by_thread["thread-b"].id, "decline", path)
    assert (await asyncio.wait_for(second, 1)).decision == "decline"
    assert not first.done()
    assert (
        gate.resolve(
            by_thread["thread-a"].id,
            "accept",
            thread_id="thread-b",
            turn_id="turn-a",
        )
        is None
    )
    assert not first.done()
    assert (
        gate.resolve(
            by_thread["thread-a"].id,
            "accept",
            thread_id="thread-a",
            turn_id="turn-a",
        )
        is not None
    )
    assert (await asyncio.wait_for(first, 1)).decision == "accept"
    assert read_permission_store(path) == {"requests": {}, "decisions": {}}


@pytest.mark.asyncio
async def test_timeout_declines_and_cleans_request(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    gate = make_gate(path, timeout=0.02)

    outcome = await gate.request_permission(
        tool_name="Bash",
        tool_input={"command": "echo hello"},
        thread_id="thread-a",
        turn_id="turn-a",
    )

    assert outcome.decision == "decline"
    assert outcome.reason == "timeout"
    assert gate.pending_requests() == []
    assert read_permission_store(path) == {"requests": {}, "decisions": {}}


@pytest.mark.asyncio
async def test_task_cancellation_cleans_request(tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    gate = make_gate(path)
    task = asyncio.create_task(
        gate.request_permission(tool_name="Write", tool_input={}, thread_id="thread-a", turn_id="turn-a")
    )
    await wait_for_pending(gate)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert gate.pending_requests() == []
    assert read_permission_store(path) == {"requests": {}, "decisions": {}}


def test_redacts_sensitive_values_and_bounds_persisted_text() -> None:
    payload = redact_approval_payload(
        {
            "apiKey": "top-secret",
            "nested": {"accessToken": "also-secret"},
            "note": "Bearer abc.def.ghi",
            "content": "ghp_abcdefghijklmnopqrstuvwxyz123456 " + ("x" * 3_000),
        }
    )

    assert payload["apiKey"] == "[redacted]"
    assert payload["nested"]["accessToken"] == "[redacted]"
    assert "abc.def.ghi" not in payload["note"]
    assert "ghp_abcdefghijklmnopqrstuvwxyz123456" not in payload["content"]
    assert len(payload["content"]) < 2_100


def test_gate_clears_only_dead_owner_requests(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    path = tmp_path / "approvals.json"
    dead = PendingServerRequest(
        id="dead-request",
        method="testBackend/requestApproval",
        params={
            "backend": "test-backend",
            "approvalManagedBy": MANAGED_GATE_MARKER,
            "approvalOwnerId": "old-owner",
            "approvalOwnerPid": 999_999,
        },
        received_at=iso_now(),
    )
    live = PendingServerRequest(
        id="live-request",
        method="testBackend/requestApproval",
        params={
            "backend": "test-backend",
            "approvalManagedBy": MANAGED_GATE_MARKER,
            "approvalOwnerId": "live-owner",
            "approvalOwnerPid": os.getpid(),
        },
        received_at=iso_now(),
    )
    unrelated = PendingServerRequest(
        id="codex-request",
        method="exec/requestApproval",
        params={},
        received_at=iso_now(),
    )
    for request in (dead, live, unrelated):
        record_shared_permission_request(request, path)
    monkeypatch.setattr(approval_gate_module, "_process_is_alive", lambda pid: pid == os.getpid())

    make_gate(path)

    assert set(read_permission_store(path)["requests"]) == {"live-request", "codex-request"}


def test_gate_caps_timeout_and_rejects_nonpositive_timeout(tmp_path: Path) -> None:
    gate = make_gate(tmp_path / "approvals.json", timeout=MAX_APPROVAL_TIMEOUT_SECONDS * 2)
    assert gate.timeout_seconds == MAX_APPROVAL_TIMEOUT_SECONDS
    with pytest.raises(ValueError, match="greater than zero"):
        make_gate(tmp_path / "other.json", timeout=0)
