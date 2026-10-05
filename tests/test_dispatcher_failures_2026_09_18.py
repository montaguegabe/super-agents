"""Regression tests for the 2026-09-18 dispatcher voice-session failure.

A churning turn streamed app-server notifications at ~30/sec and every one
rewrote the entire multi-MB state file with no significant change (the "merge
storm"). These pin the coalescing behavior: volatile-only merges are flushed
at most once per interval, while significant changes always persist.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import super_agents.app_client_sessions as app_client_sessions
from super_agents.app_server_client import CodexAppServerClient
from super_agents.state import (
    StateFile,
    prune_state_file_sessions,
    read_state_file,
    update_state_file_when,
)


def _client(tmp_path: Path) -> CodexAppServerClient:
    return CodexAppServerClient(state_file=tmp_path / "state.json")


def _volatile_patch(turn_id: str, event_at: str, preview: str, event_count: int) -> dict:
    return {
        "activeTurnId": turn_id,
        "lastTurnId": turn_id,
        "lastStatus": "running",
        "lastEventAt": event_at,
        "lastUsefulMessage": preview,
        "turns": {
            turn_id: {
                "turnId": turn_id,
                "status": "running",
                "startedAt": "2026-09-18T18:32:00.000Z",
                "updatedAt": event_at,
                "lastUsefulMessage": preview,
                "eventCount": event_count,
            }
        },
    }


def test_volatile_only_merges_coalesce_to_one_write(tmp_path: Path) -> None:
    client = _client(tmp_path)
    state_file = tmp_path / "state.json"

    async def scenario() -> None:
        await client.merge_session("thread-1", _volatile_patch("turn-1", "2026-09-18T18:37:42.000Z", "chunk 1", 1))
        first_bytes = state_file.read_bytes()
        # A burst of notification-driven merges where only volatile fields
        # move must not rewrite the file again within the flush interval.
        for i in range(2, 32):
            await client.merge_session(
                "thread-1",
                _volatile_patch("turn-1", f"2026-09-18T18:37:42.{i:03d}Z", f"chunk {i}", i),
            )
        assert state_file.read_bytes() == first_bytes
        assert client._suppressed_merge_counts["thread-1"] == 30

    asyncio.run(scenario())


def test_significant_change_always_writes(tmp_path: Path) -> None:
    client = _client(tmp_path)
    state_file = tmp_path / "state.json"

    async def scenario() -> None:
        await client.merge_session("thread-1", _volatile_patch("turn-1", "2026-09-18T18:37:42.000Z", "chunk 1", 1))
        await client.merge_session("thread-1", _volatile_patch("turn-1", "2026-09-18T18:37:42.100Z", "chunk 2", 2))
        # Terminal status is significant: it must persist immediately even
        # inside the volatile flush interval, carrying the volatile fields.
        await client.merge_session(
            "thread-1",
            {
                "activeTurnId": None,
                "lastTurnId": "turn-1",
                "lastStatus": "completed",
                "lastEventAt": "2026-09-18T18:37:42.200Z",
                "turns": {
                    "turn-1": {
                        "turnId": "turn-1",
                        "status": "completed",
                        "startedAt": "2026-09-18T18:32:00.000Z",
                        "updatedAt": "2026-09-18T18:37:42.200Z",
                        "finishedAt": "2026-09-18T18:37:42.200Z",
                        "eventCount": 3,
                    }
                },
            },
            clear_fields=["activeTurnId"],
        )
        state = read_state_file(state_file)
        session = state.sessions["thread-1"]
        assert session.last_status == "completed"
        assert session.active_turn_id is None
        assert session.turns is not None
        assert session.turns["turn-1"].status == "completed"
        assert session.turns["turn-1"].event_count == 3
        assert client._suppressed_merge_counts.get("thread-1") is None

    asyncio.run(scenario())


def test_volatile_merge_flushes_after_interval(tmp_path: Path, monkeypatch) -> None:
    client = _client(tmp_path)
    state_file = tmp_path / "state.json"

    async def scenario() -> None:
        await client.merge_session("thread-1", _volatile_patch("turn-1", "2026-09-18T18:37:42.000Z", "chunk 1", 1))
        await client.merge_session("thread-1", _volatile_patch("turn-1", "2026-09-18T18:37:42.100Z", "chunk 2", 2))
        assert client._suppressed_merge_counts["thread-1"] == 1
        # Pretend the flush interval has elapsed: the next volatile merge
        # persists the freshest values and resets the suppressed counter.
        client._volatile_merge_write_times["thread-1"] -= app_client_sessions.VOLATILE_MERGE_FLUSH_SECONDS + 0.1
        await client.merge_session("thread-1", _volatile_patch("turn-1", "2026-09-18T18:37:43.000Z", "chunk 3", 3))
        assert client._suppressed_merge_counts.get("thread-1") is None
        state = read_state_file(state_file)
        session = state.sessions["thread-1"]
        assert session.last_event_at == "2026-09-18T18:37:43.000Z"
        assert session.last_useful_message == "chunk 3"

    asyncio.run(scenario())


def test_first_merge_for_new_thread_always_writes(tmp_path: Path) -> None:
    client = _client(tmp_path)
    state_file = tmp_path / "state.json"

    async def scenario() -> None:
        await client.merge_session("thread-new", _volatile_patch("turn-1", "2026-09-18T18:37:42.000Z", "hello", 1))
        state = read_state_file(state_file)
        assert "thread-new" in state.sessions

    asyncio.run(scenario())


def _write_state(tmp_path: Path, sessions: dict) -> Path:
    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({"sessions": sessions, "routines": {}}), encoding="utf-8")
    return state_file


def test_prune_drops_stale_sessions_and_caps_turns(tmp_path: Path) -> None:
    now = 1789600000.0  # 2026-09-18ish
    stale_turns = {
        f"turn-{i}": {
            "turnId": f"turn-{i}",
            "status": "completed",
            "startedAt": "2026-09-18T10:00:00.000Z",
            "updatedAt": f"2026-09-18T10:{i:02d}:00.000Z",
        }
        for i in range(50)
    }
    state_file = _write_state(
        tmp_path,
        {
            "old-running": {
                "threadId": "old-running",
                "lastStatus": "running",
                "updatedAt": "2026-05-26T00:00:00.000Z",
                "lastEventAt": "2026-05-26T00:00:00.000Z",
            },
            "fresh": {
                "threadId": "fresh",
                "lastStatus": "completed",
                "updatedAt": "2026-09-18T12:00:00.000Z",
                "activeTurnId": "turn-3",
                "turns": stale_turns,
            },
        },
    )
    summary = prune_state_file_sessions(state_file, max_age_days=30.0, max_turns_per_session=40, now=now)
    assert summary["removedSessions"] == 1
    assert summary["removedTurns"] == 9  # 50 - 40 newest - active turn-3 kept
    state = read_state_file(state_file)
    assert "old-running" not in state.sessions
    fresh = state.sessions["fresh"]
    assert fresh.turns is not None
    assert len(fresh.turns) == 41
    assert "turn-3" in fresh.turns  # active turn survives the cap
    assert "turn-49" in fresh.turns  # newest survives
    assert "turn-4" not in fresh.turns  # oldest non-active pruned


def test_prune_keeps_routine_referenced_sessions(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "sessions": {
                    "routine-thread": {
                        "threadId": "routine-thread",
                        "updatedAt": "2026-01-01T00:00:00.000Z",
                    }
                },
                "routines": {
                    "daily": {
                        "name": "daily",
                        "prompt": "p",
                        "time": "09:00",
                        "updatedAt": "2026-09-01T00:00:00.000Z",
                        "threadId": "routine-thread",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    summary = prune_state_file_sessions(state_file, max_age_days=30.0, now=1789600000.0)
    assert summary["removedSessions"] == 0
    assert "routine-thread" in read_state_file(state_file).sessions


def test_prune_noop_leaves_file_untouched(tmp_path: Path) -> None:
    state_file = _write_state(
        tmp_path,
        {"fresh": {"threadId": "fresh", "updatedAt": "2026-09-18T12:00:00.000Z"}},
    )
    before = state_file.read_bytes()
    summary = prune_state_file_sessions(state_file, max_age_days=30.0, now=1789600000.0)
    assert summary["removedSessions"] == 0
    assert summary["removedTurns"] == 0
    assert state_file.read_bytes() == before


def test_update_state_file_when_skips_write(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"

    def refuse(state: StateFile) -> bool:
        return False

    assert update_state_file_when(state_file, refuse) is False
    assert not state_file.exists()

    def accept(state: StateFile) -> bool:
        return True

    assert update_state_file_when(state_file, accept) is True
    assert state_file.exists()
