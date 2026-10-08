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


def test_scoping_claims_rows_of_the_same_execution_backend(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite3"
    Store(db)  # schema
    _insert(db, "claude_a", "claude_code")
    _insert(db, "legacy", None)
    _insert(db, "codex_row", "codex")

    store = Store(db, backend="openbase_cloud")
    store.scope_backend("openbase_cloud")

    with sqlite3.connect(db) as conn:
        rows = dict(conn.execute("select id, backend from sessions").fetchall())
    assert rows == {"claude_a": "openbase_cloud", "legacy": "openbase_cloud", "codex_row": "codex"}
    assert {session.id for session in store.list_sessions(include_inactive=True)} == {"claude_a", "legacy"}


def test_get_session_claims_a_compatible_row_written_after_scoping(tmp_path: Path) -> None:
    db = tmp_path / "state.sqlite3"
    store = Store(db, backend="openbase_cloud")
    store.scope_backend("openbase_cloud")
    _insert(db, "claude_imported", "claude_code")  # a thread-sync import while running
    _insert(db, "codex_row", "codex")

    session = store.get_session("claude_imported")

    assert session.backend == "openbase_cloud"
    with pytest.raises(KeyError):
        store.get_session("codex_row")
    with sqlite3.connect(db) as conn:
        assert conn.execute("select backend from sessions where id = 'codex_row'").fetchone() == ("codex",)
