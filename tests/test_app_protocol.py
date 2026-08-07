from __future__ import annotations

from super_agents.app_protocol import collect_turns, find_latest_turn


def _claude_read_payload() -> dict:
    """Shape returned by the Claude backend's read_by_label."""
    return {
        "backend": "claude_code",
        "threadId": "s_session123",
        "session": {
            "id": "s_session123",
            "name": "dispatcher",
            "cwd": "/Users/someone",
            "command": ["claude-agent-sdk"],
            "status": "running",
            "activeTurnId": "t_turn2",
            "updatedAt": "2026-08-07T15:00:00.000Z",
        },
        "turns": [
            {
                "turnId": "t_turn1",
                "sessionId": "s_session123",
                "status": "completed",
                "updatedAt": "2026-08-07T14:00:00.000Z",
            },
            {
                "turnId": "t_turn2",
                "sessionId": "s_session123",
                "status": "running",
                "updatedAt": "2026-08-07T14:59:00.000Z",
            },
        ],
    }


def test_find_latest_turn_returns_turn_not_session_for_claude_payloads():
    """A running session must not be mistaken for its own active turn.

    Regression: console steering on the Claude backend resolved the SESSION
    id as the active turn id ("Expected active turn id `s_...` but found
    `t_...`") because the session dict has id+status while Claude turns
    serialize as turnId.
    """
    turn = find_latest_turn(_claude_read_payload(), active_only=True)

    assert turn is not None
    assert turn["id"] == "t_turn2"


def test_collect_turns_normalizes_turn_id_key():
    turns = collect_turns(_claude_read_payload())

    ids = sorted(turn["id"] for turn in turns)
    assert ids == ["t_turn1", "t_turn2"]


def test_collect_turns_keeps_raw_app_server_turn_shape():
    """Codex app-server turns use a plain `id` key and must keep working."""
    payload = {
        "thread": {
            "turns": [
                {"id": "t_codex1", "status": "completed", "updatedAt": "2026-08-07T14:00:00.000Z"},
                {"id": "t_codex2", "status": "inProgress", "updatedAt": "2026-08-07T14:59:00.000Z"},
            ]
        }
    }

    active = find_latest_turn(payload, active_only=True)
    everything = collect_turns(payload)

    assert active is not None
    assert active["id"] == "t_codex2"
    assert sorted(turn["id"] for turn in everything) == ["t_codex1", "t_codex2"]


def test_collect_turns_ignores_session_shaped_dicts_without_turns():
    payload = {
        "session": {
            "id": "s_only_session",
            "cwd": "/tmp",
            "command": [],
            "status": "running",
        }
    }

    assert collect_turns(payload) == []
    assert find_latest_turn(payload, active_only=True) is None
