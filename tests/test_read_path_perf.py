"""Read-path cost controls: incremental transcript parsing, incremental title
scans, batched latest-turn lookups, and reads that stay off the event loop."""

from __future__ import annotations

import asyncio
import json
import threading
from pathlib import Path

import pytest

from super_agents import claude_home_index, claude_transcript
from super_agents.agent_store import Store
from super_agents.claude_sdk import ClaudeAgentSdkClient
from super_agents.claude_transcript import _TRANSCRIPT_CACHE, transcript_turn_views


def _entry(kind: str, text: str, ts: str) -> str:
    return json.dumps(
        {
            "type": kind,
            "timestamp": ts,
            "message": {"role": kind, "content": [{"type": "text", "text": text}]},
        }
    )


@pytest.fixture(autouse=True)
def _clear_caches():
    _TRANSCRIPT_CACHE.clear()
    claude_home_index._TITLE_SCAN_STATE.clear()
    yield
    _TRANSCRIPT_CACHE.clear()
    claude_home_index._TITLE_SCAN_STATE.clear()


def _session(store: Store, tmp_path: Path, monkeypatch, backend_session_id: str):
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    cwd = str(tmp_path / "proj")
    session = store.create_session("imported", cwd=cwd, command=["claude"])
    session = store.update_session(session.id, backend_session_id=backend_session_id)
    path = claude_transcript.transcript_path(session)
    assert path is None
    projects = tmp_path / "claude" / "projects" / claude_transcript._project_dir_name(cwd)
    projects.mkdir(parents=True)
    return session, projects / f"{backend_session_id}.jsonl"


def test_transcript_append_is_parsed_incrementally(tmp_path, monkeypatch):
    store = Store(path=tmp_path / "state.sqlite3", backend="claude_code")
    session, path = _session(store, tmp_path, monkeypatch, "aaaaaaaa-0000-4000-8000-000000000001")
    path.write_text(
        _entry("user", "first prompt", "2026-09-01T00:00:00Z")
        + "\n"
        + _entry("assistant", "first reply", "2026-09-01T00:00:01Z")
        + "\n"
    )
    first = transcript_turn_views(session)
    assert [t["promptPreview"] for t in first] == ["first prompt"]
    assert first[0]["lastUsefulMessage"] == "first reply"

    full_parses: list[int] = []
    original = claude_transcript._parse_transcript_append

    def counting(p, parsed):
        full_parses.append(parsed.consumed)
        return original(p, parsed)

    monkeypatch.setattr(claude_transcript, "_parse_transcript_append", counting)

    # Append a reply to the open turn plus a partial (unterminated) line.
    with path.open("a") as handle:
        handle.write(_entry("assistant", "second reply", "2026-09-01T00:00:02Z") + "\n")
        handle.write(_entry("user", "second prompt", "2026-09-01T00:00:03Z"))  # no newline yet
    turns = transcript_turn_views(session)
    assert full_parses and full_parses[-1] > 0, "must resume from the consumed offset"
    assert len(turns) == 1
    assert turns[0]["lastUsefulMessage"] == "second reply"

    with path.open("a") as handle:
        handle.write("\n")
    turns = transcript_turn_views(session)
    assert [t["promptPreview"] for t in turns] == ["second prompt", "first prompt"]

    # A rewritten (shrunk) transcript is parsed from scratch.
    path.write_text(_entry("user", "only prompt", "2026-09-02T00:00:00Z") + "\n")
    turns = transcript_turn_views(session)
    assert [t["promptPreview"] for t in turns] == ["only prompt"]
    assert full_parses[-1] == 0


def test_latest_custom_title_scans_only_appended_bytes(tmp_path):
    path = tmp_path / "t.jsonl"
    path.write_text(json.dumps({"type": "custom-title", "customTitle": "one"}) + "\n")
    assert claude_home_index._latest_custom_title(path) == "one"
    scanned_after_first = claude_home_index._TITLE_SCAN_STATE[str(path)][0]
    assert scanned_after_first == path.stat().st_size

    with path.open("a") as handle:
        handle.write(_entry("user", "x", "2026-09-01T00:00:00Z") + "\n")
        handle.write(json.dumps({"type": "custom-title", "customTitle": "two"}))  # partial line
    assert claude_home_index._latest_custom_title(path) == "one"
    with path.open("a") as handle:
        handle.write("\n")
    assert claude_home_index._latest_custom_title(path) == "two"

    path.write_text(_entry("user", "fresh", "2026-09-01T00:00:00Z") + "\n")
    assert claude_home_index._latest_custom_title(path) is None


def test_latest_turns_by_session_matches_per_session_lookup(tmp_path):
    store = Store(path=tmp_path / "state.sqlite3", backend="claude_code")
    a = store.create_session("a", cwd=str(tmp_path), command=["claude"])
    b = store.create_session("b", cwd=str(tmp_path), command=["claude"])
    store.create_turn(a.id, "a1")
    store.create_turn(a.id, "a2")
    store.create_turn(b.id, "b1")
    latest = store.latest_turns_by_session()
    assert latest[a.id].id == store.list_turns(a.id, limit=1)[0].id
    assert latest[b.id].id == store.list_turns(b.id, limit=1)[0].id
    assert set(latest) == {a.id, b.id}


def test_sessions_listing_uses_one_latest_turn_query(tmp_path, monkeypatch):
    store = Store(path=tmp_path / "state.sqlite3", backend="claude_code")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    client = ClaudeAgentSdkClient(store=store)
    for name in ("a", "b", "c"):
        session = store.create_session(name, cwd=str(tmp_path), command=["claude"])
        store.create_turn(session.id, f"{name}-prompt", model="claude-x")
    per_session_calls: list[str] = []
    original = store.list_turns
    monkeypatch.setattr(
        store, "list_turns", lambda sid, limit=20: per_session_calls.append(sid) or original(sid, limit)
    )
    views = asyncio.run(client.sessions())
    assert len(views) == 3
    assert all(view["model"] == "claude-x" for view in views)
    assert per_session_calls == []


def test_reads_run_off_the_event_loop(tmp_path, monkeypatch):
    store = Store(path=tmp_path / "state.sqlite3", backend="claude_code")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude"))
    client = ClaudeAgentSdkClient(store=store)
    store.create_session("a", cwd=str(tmp_path), command=["claude"])
    threads: set[str] = set()
    original = store.list_sessions

    def recording(*args, **kwargs):
        threads.add(threading.current_thread().name)
        return original(*args, **kwargs)

    monkeypatch.setattr(store, "list_sessions", recording)

    async def run():
        loop_thread = threading.current_thread().name
        await client.sessions()
        return loop_thread

    loop_thread = asyncio.run(run())
    assert threads and loop_thread not in threads
