"""Session naming in the Claude home last-interaction index.

Claude Code's /rename appends a ``custom-title`` entry to the transcript;
the index must honor it at registration and propagate later renames on
refresh sweeps, without re-clobbering names set through the store's own
rename (``super_agents_rename``).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from super_agents.agent_store import Store
from super_agents.claude_home_index import refresh_last_interaction_index

SESSION_UUID = "55555555-5555-4555-8555-555555555555"


def _transcript_path(projects_dir: Path, cwd: str, session_id: str) -> Path:
    project_dir = projects_dir / cwd.replace("/", "-")
    project_dir.mkdir(parents=True, exist_ok=True)
    return project_dir / f"{session_id}.jsonl"


def _base_entries(cwd: str, prompt: str) -> list[dict]:
    return [
        {
            "type": "user",
            "cwd": cwd,
            "timestamp": "2026-09-09T12:00:00.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": prompt}]},
        },
        {
            "type": "assistant",
            "cwd": cwd,
            "timestamp": "2026-09-09T12:00:05.000Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": f"reply to {prompt}"}]},
        },
    ]


def _write_entries(path: Path, entries: list[dict], mtime: float) -> None:
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    os.utime(path, (mtime, mtime))


def _append_entry(path: Path, entry: dict, mtime: float) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry) + "\n")
    os.utime(path, (mtime, mtime))


def _custom_title(title: str) -> dict:
    return {"type": "custom-title", "customTitle": title, "sessionId": SESSION_UUID}


@pytest.fixture()
def projects(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    claude_home = tmp_path / "claude-home"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    return claude_home / "projects"


def test_registration_prefers_transcript_custom_title(projects: Path, tmp_path: Path) -> None:
    path = _transcript_path(projects, "/tmp/proj", SESSION_UUID)
    _write_entries(path, [*_base_entries("/tmp/proj", "first prompt"), _custom_title("my-title")], time.time())

    store = Store(tmp_path / "state.sqlite3", backend="claude_code")
    assert refresh_last_interaction_index(store, now=time.monotonic()) == 1

    session = store.list_sessions(include_inactive=True)[0]
    assert session.name == "my-title"


def test_transcript_rename_after_registration_propagates(projects: Path, tmp_path: Path) -> None:
    now = time.time()
    path = _transcript_path(projects, "/tmp/proj", SESSION_UUID)
    _write_entries(path, _base_entries("/tmp/proj", "first prompt"), now - 60)

    store = Store(tmp_path / "state.sqlite3", backend="claude_code")
    assert refresh_last_interaction_index(store, now=time.monotonic()) == 1
    assert store.list_sessions(include_inactive=True)[0].name == "first prompt"

    # The user runs /rename in Claude Code; a later /rename supersedes it.
    _append_entry(path, _custom_title("interim"), now - 30)
    _append_entry(path, _custom_title("renamed-later"), now)
    assert refresh_last_interaction_index(store, now=time.monotonic() + 100) == 1
    assert store.list_sessions(include_inactive=True)[0].name == "renamed-later"


def test_store_rename_is_not_reclobbered_by_stale_transcript_title(projects: Path, tmp_path: Path) -> None:
    now = time.time()
    path = _transcript_path(projects, "/tmp/proj", SESSION_UUID)
    _write_entries(path, [*_base_entries("/tmp/proj", "first prompt"), _custom_title("transcript-name")], now - 60)

    store = Store(tmp_path / "state.sqlite3", backend="claude_code")
    assert refresh_last_interaction_index(store, now=time.monotonic()) == 1
    session = store.list_sessions(include_inactive=True)[0]
    assert session.name == "transcript-name"

    # A store-side rename wins over the already-synced transcript title...
    store.rename_session(session.id, "store-name")
    os.utime(path, (now + 60, now + 60))
    assert refresh_last_interaction_index(store, now=time.monotonic() + 100) == 1
    assert store.list_sessions(include_inactive=True)[0].name == "store-name"

    # ...until the transcript records a *newer* rename.
    _append_entry(path, _custom_title("transcript-name-2"), now + 120)
    assert refresh_last_interaction_index(store, now=time.monotonic() + 200) == 1
    assert store.list_sessions(include_inactive=True)[0].name == "transcript-name-2"


def test_pre_title_sync_rows_are_backfilled_once_without_bumping_activity(projects: Path, tmp_path: Path) -> None:
    now = time.time()
    path = _transcript_path(projects, "/tmp/proj", SESSION_UUID)
    _write_entries(path, [*_base_entries("/tmp/proj", "first prompt"), _custom_title("backfilled")], now - 3600)

    store = Store(tmp_path / "state.sqlite3", backend="claude_code")
    assert refresh_last_interaction_index(store, now=time.monotonic()) == 1
    session = store.list_sessions(include_inactive=True)[0]

    # Simulate a row registered before title sync existed: generated name,
    # transcript_title never recorded.
    with store.connect() as conn:
        conn.execute(
            "update sessions set name = ?, transcript_title = null where id = ?",
            ("first prompt", session.id),
        )

    # The transcript is idle (mtime unchanged) but the backfill still runs...
    assert refresh_last_interaction_index(store, now=time.monotonic() + 100) == 1
    backfilled = store.list_sessions(include_inactive=True)[0]
    assert backfilled.name == "backfilled"
    assert backfilled.updated_at == session.updated_at

    # ...and only once: the recorded title makes the next idle sweep a no-op.
    assert refresh_last_interaction_index(store, now=time.monotonic() + 200) == 0


def test_transcript_titles_are_uniquified_against_other_sessions(projects: Path, tmp_path: Path) -> None:
    now = time.time()
    other_uuid = "66666666-6666-4666-8666-666666666666"
    path = _transcript_path(projects, "/tmp/proj-a", SESSION_UUID)
    _write_entries(path, [*_base_entries("/tmp/proj-a", "prompt a"), _custom_title("taken")], now - 60)
    other = _transcript_path(projects, "/tmp/proj-b", other_uuid)
    _write_entries(other, [*_base_entries("/tmp/proj-b", "prompt b"), _custom_title("taken")], now)

    store = Store(tmp_path / "state.sqlite3", backend="claude_code")
    assert refresh_last_interaction_index(store, now=time.monotonic()) == 2
    names = sorted(session.name for session in store.list_sessions(include_inactive=True))
    assert names == ["taken", "taken (2)"]

    # A refresh with an unchanged title must not suffix the session's own name.
    os.utime(path, (now + 60, now + 60))
    os.utime(other, (now + 60, now + 60))
    assert refresh_last_interaction_index(store, now=time.monotonic() + 100) == 2
    names = sorted(session.name for session in store.list_sessions(include_inactive=True))
    assert names == ["taken", "taken (2)"]
