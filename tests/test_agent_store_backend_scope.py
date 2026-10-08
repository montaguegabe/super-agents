from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from super_agents.agent_store import Store


def _insert(db: Path, session_id: str, backend: str | None) -> None:
    with sqlite3.connect(db) as conn:
        conn.execute(
            "insert into sessions (id, name, cwd, command_json, status, backend, created_at, updated_at)"
            " values (?, ?, '/tmp', '[]', 'completed', ?, '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z')",
            (session_id, session_id, backend),
        )


def _labels(db: Path) -> dict[str, str | None]:
    with sqlite3.connect(db) as conn:
        return dict(conn.execute("select id, backend from sessions").fetchall())


def test_store_is_scoped_by_execution_backend_without_relabelling(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite3"
    Store(db)  # schema
    _insert(db, "claude_a", "claude_code")
    _insert(db, "cloud_b", "openbase_cloud")
    _insert(db, "legacy", None)
    _insert(db, "codex_row", "codex")

    store = Store(db, backend="openbase_cloud")
    store.scope_backend("openbase_cloud")

    # Only legacy (unlabelled) rows are claimed; other identities keep theirs.
    assert _labels(db) == {
        "claude_a": "claude_code",
        "cloud_b": "openbase_cloud",
        "legacy": "openbase_cloud",
        "codex_row": "codex",
    }
    seen = {session.id for session in store.list_sessions(include_inactive=True)}
    assert seen == {"claude_a", "cloud_b", "legacy"}
    assert store.get_session("claude_a").backend == "claude_code"
    assert store.get_by_name("claude_a") is not None
    with pytest.raises(KeyError):
        store.get_session("codex_row")


def test_two_processes_with_different_claude_identities_do_not_fight(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite3"
    api = Store(db, backend="openbase_cloud")
    api.scope_backend("openbase_cloud")
    mcp = Store(db, backend="claude_code")
    mcp.scope_backend("claude_code")
    _insert(db, "imported", "openbase_cloud")  # a thread-sync import on the API side

    assert mcp.get_session("imported").backend == "openbase_cloud"
    assert api.get_session("imported").backend == "openbase_cloud"
    mcp.scope_backend("claude_code")
    api.scope_backend("openbase_cloud")
    assert _labels(db) == {"imported": "openbase_cloud"}
