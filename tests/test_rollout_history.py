from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from super_agents.app_server_client import CodexAppServerClient
from super_agents.rollout_history import (
    needs_rollout_turn_fallback,
    rollout_fallback_turns,
)


def _write_rollout(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    return path


def _event(payload: dict[str, Any], timestamp: str = "2026-08-12T01:06:41.990Z") -> dict[str, Any]:
    return {"timestamp": timestamp, "type": "event_msg", "payload": payload}


SAMPLE_RECORDS = [
    {"timestamp": "2026-08-12T01:05:40.939Z", "type": "session_meta", "payload": {"id": "t-1"}},
    _event({"type": "task_started", "turn_id": "turn-1", "started_at": 1786496801}),
    _event(
        {
            "type": "item_completed",
            "turn_id": "turn-1",
            "item": {
                "type": "UserMessage",
                "id": "item-1",
                "content": [{"type": "text", "text": "Make me a train game", "text_elements": []}],
            },
        }
    ),
    _event(
        {
            "type": "item_completed",
            "turn_id": "turn-1",
            "item": {
                "type": "AgentMessage",
                "id": "item-2",
                "content": [{"type": "Text", "text": "Checking the local setup first."}],
                "phase": "commentary",
            },
        }
    ),
    {"timestamp": "2026-08-12T01:07:00.000Z", "type": "response_item", "payload": {"type": "message"}},
    _event({"type": "token_count", "turn_id": "turn-1"}),
    _event(
        {
            "type": "item_completed",
            "turn_id": "turn-1",
            "item": {
                "type": "FileChange",
                "id": "call-1",
                "changes": {"/tmp/app.tsx": {"type": "add", "content": "export {};"}},
            },
        }
    ),
    _event(
        {
            "type": "item_completed",
            "turn_id": "turn-1",
            "item": {
                "type": "AgentMessage",
                "id": "item-3",
                "content": [{"type": "Text", "text": "Done. The game is running."}],
                "phase": "final_answer",
            },
        }
    ),
    _event({"type": "task_complete", "turn_id": "turn-1", "last_agent_message": "Done."}, "2026-08-12T01:26:52.528Z"),
    _event({"type": "task_started", "turn_id": "turn-2", "started_at": 1786498000}, "2026-08-12T01:26:53.000Z"),
    _event(
        {
            "type": "item_completed",
            "turn_id": "turn-2",
            "item": {
                "type": "UserMessage",
                "id": "item-4",
                "content": [{"type": "text", "text": "Stop", "text_elements": []}],
            },
        },
        "2026-08-12T01:26:54.000Z",
    ),
    _event(
        {"type": "turn_aborted", "turn_id": "turn-2", "reason": "interrupted", "completed_at": 1786498012},
        "2026-08-12T01:26:55.000Z",
    ),
]


def test_rollout_fallback_turns_rebuilds_turns(tmp_path: Path) -> None:
    rollout = _write_rollout(tmp_path / "rollout.jsonl", SAMPLE_RECORDS)

    turns = rollout_fallback_turns(rollout)

    assert [turn["id"] for turn in turns] == ["turn-1", "turn-2"]
    first, second = turns

    assert first["status"] == "completed"
    assert first["startedAt"] == 1786496801
    assert first["completedAt"] == 1786498012  # 2026-08-12T01:26:52.528Z
    assert [item["type"] for item in first["items"]] == [
        "userMessage",
        "agentMessage",
        "fileChange",
        "agentMessage",
    ]

    user_message = first["items"][0]
    assert user_message["content"] == [{"type": "text", "text": "Make me a train game", "text_elements": []}]

    commentary = first["items"][1]
    assert commentary["text"] == "Checking the local setup first."
    assert commentary["phase"] == "commentary"
    assert "content" not in commentary

    file_change = first["items"][2]
    assert file_change["changes"] == [{"path": "/tmp/app.tsx", "kind": {"type": "add"}, "diff": "export {};"}]

    final = first["items"][3]
    assert final["text"] == "Done. The game is running."
    assert final["phase"] == "final_answer"

    assert second["status"] == "interrupted"
    assert second["completedAt"] == 1786498012


def test_rollout_fallback_turns_tolerates_bad_input(tmp_path: Path) -> None:
    assert rollout_fallback_turns(tmp_path / "missing.jsonl") == []

    rollout = tmp_path / "garbled.jsonl"
    rollout.write_text('not json\n{"type": "event_msg"}\n{"type": "event_msg", "payload": 5}\n', encoding="utf-8")
    assert rollout_fallback_turns(rollout) == []


def test_needs_rollout_turn_fallback_conditions() -> None:
    thread = {
        "historyMode": "paginated",
        "status": {"type": "notLoaded"},
        "turns": [],
        "path": "/tmp/rollout.jsonl",
    }
    assert needs_rollout_turn_fallback(thread, include_turns=True)
    assert not needs_rollout_turn_fallback(thread, include_turns=False)
    assert not needs_rollout_turn_fallback(None, include_turns=True)
    assert not needs_rollout_turn_fallback({**thread, "turns": [{"id": "turn-1"}]}, include_turns=True)
    assert not needs_rollout_turn_fallback({**thread, "historyMode": "legacy"}, include_turns=True)
    assert not needs_rollout_turn_fallback({**thread, "status": {"type": "idle"}}, include_turns=True)
    assert not needs_rollout_turn_fallback({**thread, "path": ""}, include_turns=True)


class RolloutFallbackClient(CodexAppServerClient):
    def __init__(self, thread: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._thread = thread

    async def ensure_connected(self) -> None:
        return None

    async def request(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        timeout_seconds: float = 30,
        **kwargs: Any,
    ) -> dict[str, Any]:
        assert method == "thread/read"
        return {"thread": self._thread}


@pytest.mark.asyncio
async def test_read_thread_falls_back_to_rollout_for_paginated_threads(tmp_path: Path) -> None:
    rollout = _write_rollout(tmp_path / "rollout.jsonl", SAMPLE_RECORDS)
    thread = {
        "id": "t-1",
        "historyMode": "paginated",
        "status": {"type": "notLoaded"},
        "turns": [],
        "path": str(rollout),
    }
    client = RolloutFallbackClient(thread)

    result = await client.read_thread("t-1", True)

    turns = result["thread"]["turns"]
    assert [turn["id"] for turn in turns] == ["turn-1", "turn-2"]
    assert turns[0]["items"][0]["content"][0]["text"] == "Make me a train game"


@pytest.mark.asyncio
async def test_read_thread_leaves_loaded_threads_untouched(tmp_path: Path) -> None:
    thread = {
        "id": "t-1",
        "historyMode": "paginated",
        "status": {"type": "idle"},
        "turns": [],
        "path": str(tmp_path / "rollout.jsonl"),
    }
    client = RolloutFallbackClient(thread)

    result = await client.read_thread("t-1", True)

    assert result["thread"]["turns"] == []
