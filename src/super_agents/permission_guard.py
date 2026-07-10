"""External permission guard for Super Agents launches.

A supervising process (an approvals UI, a voice control surface, a CI
harness) can drop a small JSON file next to the Super Agents state that
marks launches as *restricted*. While the guard is restricted, thread and
turn inputs flowing through the MCP server may not run with permission
bypasses: full-access sandboxes and never-ask approval policies are
downgraded to gated equivalents, on every backend. The supervisor — not the
calling agent — controls the file, so a prompt-injected agent cannot talk
itself into a bypass.

File shape (all fields optional except ``restricted``)::

    {
      "restricted": true,
      "codex": {"approvalPolicy": "on-request", "sandboxPolicy": "workspace-write"},
      "claude": {"permissionMode": "default"}
    }

A missing, empty, or unreadable file means unrestricted. Explicitly gated
values in the launch input (for example a ``read-only`` sandbox) are kept
even while restricted; only bypass values and omissions are rewritten.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

PERMISSION_GUARD_FILE_ENV = "SUPER_AGENTS_PERMISSION_GUARD_FILE"
DEFAULT_PERMISSION_GUARD_FILE = Path.home() / ".super-agents" / "permission-guard.json"

RESTRICTED_CODEX_APPROVAL_POLICY = "on-request"
RESTRICTED_CODEX_SANDBOX_POLICY = "workspace-write"
RESTRICTED_CLAUDE_PERMISSION_MODE = "default"

_BYPASS_APPROVAL_POLICIES = {"never"}
_BYPASS_SANDBOX_VALUES = {"danger-full-access", "dangerfullaccess"}
_BYPASS_PERMISSION_MODES = {"bypasspermissions"}


def permission_guard_file(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    configured = os.environ.get(PERMISSION_GUARD_FILE_ENV)
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_PERMISSION_GUARD_FILE


def read_permission_guard(path: str | Path | None = None) -> JsonObject:
    try:
        raw = json.loads(permission_guard_file(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return raw if isinstance(raw, dict) else {}


def write_permission_guard(guard: JsonObject, path: str | Path | None = None) -> None:
    guard_path = permission_guard_file(path)
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = guard_path.with_name(guard_path.name + ".tmp")
    tmp_path.write_text(json.dumps(guard, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, guard_path)


def permission_guard_restricted(path: str | Path | None = None) -> bool:
    return read_permission_guard(path).get("restricted") is True


def apply_permission_guard(input_data: JsonObject, path: str | Path | None = None) -> JsonObject:
    """Downgrade permission bypasses in a thread/turn input while restricted."""
    guard = read_permission_guard(path)
    if guard.get("restricted") is not True:
        return input_data
    codex = guard.get("codex") if isinstance(guard.get("codex"), dict) else {}
    claude = guard.get("claude") if isinstance(guard.get("claude"), dict) else {}
    guarded = dict(input_data)

    approval_policy = _lowered(guarded.get("approvalPolicy"))
    if approval_policy is None or approval_policy in _BYPASS_APPROVAL_POLICIES:
        guarded["approvalPolicy"] = _optional_str(codex.get("approvalPolicy")) or RESTRICTED_CODEX_APPROVAL_POLICY

    sandbox = (
        _lowered(guarded.get("sandboxPolicy"))
        or _lowered(guarded.get("sandbox"))
        or _lowered(guarded.get("sandboxType"))
    )
    if sandbox is None or sandbox in _BYPASS_SANDBOX_VALUES:
        guarded.pop("sandbox", None)
        guarded.pop("sandboxType", None)
        guarded["sandboxPolicy"] = _optional_str(codex.get("sandboxPolicy")) or RESTRICTED_CODEX_SANDBOX_POLICY

    permission_mode = _lowered(guarded.get("permissionMode"))
    if permission_mode is None or permission_mode in _BYPASS_PERMISSION_MODES:
        guarded["permissionMode"] = _optional_str(claude.get("permissionMode")) or RESTRICTED_CLAUDE_PERMISSION_MODE

    return guarded


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _lowered(value: Any) -> str | None:
    text = _optional_str(value)
    return text.strip().lower() if text else None
