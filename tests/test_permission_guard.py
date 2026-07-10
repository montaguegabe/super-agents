from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_agents.mcp_server import clean_thread_input, clean_turn_input
from super_agents.permission_guard import (
    PERMISSION_GUARD_FILE_ENV,
    apply_permission_guard,
    permission_guard_restricted,
    write_permission_guard,
)


@pytest.fixture
def guard_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    path = tmp_path / "permission-guard.json"
    monkeypatch.setenv(PERMISSION_GUARD_FILE_ENV, str(path))
    return path


def test_missing_guard_leaves_input_untouched(guard_path: Path) -> None:
    input_data = {"approvalPolicy": "never", "sandbox": "danger-full-access"}
    assert apply_permission_guard(input_data) == input_data
    assert permission_guard_restricted() is False


def test_restricted_guard_downgrades_bypasses(guard_path: Path) -> None:
    write_permission_guard({"restricted": True})

    guarded = apply_permission_guard(
        {
            "approvalPolicy": "never",
            "sandbox": "danger-full-access",
            "permissionMode": "bypassPermissions",
        }
    )

    assert guarded["approvalPolicy"] == "on-request"
    assert guarded["sandboxPolicy"] == "workspace-write"
    assert guarded["permissionMode"] == "default"
    assert "sandbox" not in guarded
    assert permission_guard_restricted() is True


def test_restricted_guard_fills_omitted_permissions(guard_path: Path) -> None:
    write_permission_guard({"restricted": True})

    guarded = apply_permission_guard({"name": "agent"})

    assert guarded["approvalPolicy"] == "on-request"
    assert guarded["sandboxPolicy"] == "workspace-write"
    assert guarded["permissionMode"] == "default"


def test_restricted_guard_keeps_stricter_explicit_values(guard_path: Path) -> None:
    write_permission_guard({"restricted": True})

    guarded = apply_permission_guard(
        {
            "approvalPolicy": "untrusted",
            "sandboxPolicy": "read-only",
            "permissionMode": "acceptEdits",
        }
    )

    assert guarded["approvalPolicy"] == "untrusted"
    assert guarded["sandboxPolicy"] == "read-only"
    assert guarded["permissionMode"] == "acceptEdits"


def test_guard_overrides_come_from_the_file(guard_path: Path) -> None:
    write_permission_guard(
        {
            "restricted": True,
            "codex": {"approvalPolicy": "untrusted", "sandboxPolicy": "read-only"},
            "claude": {"permissionMode": "plan"},
        }
    )

    guarded = apply_permission_guard({"approvalPolicy": "never"})

    assert guarded["approvalPolicy"] == "untrusted"
    assert guarded["sandboxPolicy"] == "read-only"
    assert guarded["permissionMode"] == "plan"


def test_unrestricted_guard_file_is_passthrough(guard_path: Path) -> None:
    guard_path.write_text(json.dumps({"restricted": False}), encoding="utf-8")
    input_data = {"approvalPolicy": "never"}
    assert apply_permission_guard(input_data) == input_data


def test_clean_inputs_apply_the_guard(guard_path: Path) -> None:
    write_permission_guard({"restricted": True})

    thread_input = clean_thread_input({"name": "agent", "approvalPolicy": "never", "sandbox": "danger-full-access"})
    turn_input = clean_turn_input({"threadId": "t1", "prompt": "go", "sandboxType": "dangerFullAccess"})

    assert thread_input["approvalPolicy"] == "on-request"
    assert thread_input["sandboxPolicy"] == "workspace-write"
    assert thread_input["permissionMode"] == "default"
    assert "sandbox" not in thread_input
    assert turn_input["approvalPolicy"] == "on-request"
    assert turn_input["sandboxPolicy"] == "workspace-write"
    assert "sandboxType" not in turn_input
