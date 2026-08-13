"""Tests for transcript-JSONL turn synthesis for imported Claude sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient
from super_agents.claude_transcript import (
    _TRANSCRIPT_CACHE,
    transcript_path,
    transcript_turn_views,
)

SESSION_UUID = "49e25a93-88f8-43ad-93b0-000000000001"


@pytest.fixture(autouse=True)
def clear_transcript_cache() -> None:
    _TRANSCRIPT_CACHE.clear()


def _user_entry(text: str, *, uuid: str, timestamp: str, **extra: Any) -> dict:
    return {
        "type": "user",
        "uuid": uuid,
        "timestamp": timestamp,
        "message": {"role": "user", "content": text},
        **extra,
    }


def _assistant_entry(text: str, *, uuid: str, timestamp: str, model: str | None = None, **extra: Any) -> dict:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    }
    if model:
        message["model"] = model
    return {
        "type": "assistant",
        "uuid": uuid,
        "timestamp": timestamp,
        "message": message,
        **extra,
    }


def _write_transcript(config_dir: Path, cwd: str, entries: list[dict], session_uuid: str = SESSION_UUID) -> Path:
    project_dir = config_dir / "projects" / "".join(c if c.isalnum() else "-" for c in cwd)
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_uuid}.jsonl"
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries), encoding="utf-8")
    return path


def _imported_session(store: Store, cwd: str, session_uuid: str = SESSION_UUID):
    session = store.create_session("imported", cwd=cwd, command=["claude", "--resume", session_uuid])
    return store.update_session(session.id, backend_session_id=session_uuid, status="completed")


def _transcript_entries() -> list[dict]:
    return [
        {"type": "custom-title", "customTitle": "imported"},
        _user_entry("meta noise", uuid="u0", timestamp="2026-08-06T15:00:00Z", isMeta=True),
        _user_entry("first question", uuid="u1", timestamp="2026-08-06T15:05:00Z"),
        _assistant_entry("thinking about it", uuid="a1", timestamp="2026-08-06T15:05:30Z", model="claude-fable-5"),
        _assistant_entry("first answer", uuid="a2", timestamp="2026-08-06T15:06:00Z", model="claude-fable-5"),
        _user_entry("sidechain question", uuid="u2", timestamp="2026-08-06T15:07:00Z", isSidechain=True),
        _assistant_entry("sidechain reply", uuid="a3", timestamp="2026-08-06T15:07:30Z", isSidechain=True),
        {
            "type": "user",
            "uuid": "u3",
            "timestamp": "2026-08-06T15:08:00Z",
            "message": {
                "role": "user",
                "content": [{"type": "tool_result", "content": "tool output"}],
            },
        },
        _user_entry("second question", uuid="u4", timestamp="2026-08-06T15:09:00Z"),
        _assistant_entry("second answer", uuid="a4", timestamp="2026-08-06T15:09:45Z", model="claude-fable-5"),
    ]


def test_transcript_turns_parse_prompts_and_replies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "claude_config"
    cwd = str(tmp_path / "project")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    _write_transcript(config_dir, cwd, _transcript_entries())
    store = Store(tmp_path / "state.sqlite3")
    session = _imported_session(store, cwd)

    turns = transcript_turn_views(session, limit=20)

    assert [turn["promptPreview"] for turn in turns] == ["second question", "first question"]
    newest, oldest = turns
    assert newest["turnId"] == "u4"
    assert newest["status"] == "completed"
    assert newest["source"] == "transcript"
    assert newest["model"] == "claude-fable-5"
    assert newest["lastUsefulMessage"] == "second answer"
    assert newest["createdAt"] == "2026-08-06T15:09:00Z"
    assert newest["finishedAt"] == "2026-08-06T15:09:45Z"
    assert newest["items"] == [
        {"type": "userMessage", "content": [{"type": "text", "text": "second question"}]},
        {"type": "agentMessage", "text": "second answer"},
    ]
    # Intermediate assistant text stays in items so full history renders.
    assert [item["text"] for item in oldest["items"] if item["type"] == "agentMessage"] == [
        "thinking about it",
        "first answer",
    ]


def test_transcript_turns_respect_limit_newest_first(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "claude_config"
    cwd = str(tmp_path / "project")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    _write_transcript(config_dir, cwd, _transcript_entries())
    store = Store(tmp_path / "state.sqlite3")
    session = _imported_session(store, cwd)

    turns = transcript_turn_views(session, limit=1)

    assert [turn["promptPreview"] for turn in turns] == ["second question"]


def test_transcript_path_falls_back_to_project_scan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "claude_config"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    # Store the transcript under a project dir that does not match the
    # session cwd's mangled name.
    _write_transcript(config_dir, "/some/other/place", _transcript_entries())
    store = Store(tmp_path / "state.sqlite3")
    session = _imported_session(store, str(tmp_path / "project"))

    assert transcript_path(session) is not None
    assert transcript_turn_views(session, limit=20)


def test_transcript_turns_empty_without_backend_session_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude_config"))
    store = Store(tmp_path / "state.sqlite3")
    session = store.create_session("local", cwd=str(tmp_path), command=["claude-agent-sdk"])

    assert transcript_turn_views(session, limit=20) == []


def test_transcript_cache_invalidates_on_append(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "claude_config"
    cwd = str(tmp_path / "project")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    path = _write_transcript(config_dir, cwd, _transcript_entries())
    store = Store(tmp_path / "state.sqlite3")
    session = _imported_session(store, cwd)

    assert len(transcript_turn_views(session, limit=20)) == 2
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(_user_entry("third question", uuid="u5", timestamp="2026-08-06T15:10:00Z")) + "\n")
        handle.write(json.dumps(_assistant_entry("third answer", uuid="a5", timestamp="2026-08-06T15:10:30Z")) + "\n")

    turns = transcript_turn_views(session, limit=20)
    assert [turn["promptPreview"] for turn in turns][0] == "third question"
    assert len(turns) == 3


@pytest.mark.asyncio
async def test_read_by_label_falls_back_to_transcript_turns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "claude_config"
    cwd = str(tmp_path / "project")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    _write_transcript(config_dir, cwd, _transcript_entries())
    store = Store(tmp_path / "state.sqlite3")
    session = _imported_session(store, cwd)
    client = ClaudeAgentSdkClient(store=store)

    detail = await client.read_by_label(LabelQueryInput(thread_id=session.id), include_turns=True)
    assert [turn["promptPreview"] for turn in detail["turns"]] == ["second question", "first question"]

    compact = await client.read_by_label(LabelQueryInput(thread_id=session.id))
    assert [turn["promptPreview"] for turn in compact["recentTurns"]] == ["second question", "first question"]


@pytest.mark.asyncio
async def test_session_list_view_includes_latest_turn_model_and_effort(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "claude_config"))
    store = Store(tmp_path / "state.sqlite3")
    session = store.create_session("worked", cwd=str(tmp_path), command=["claude-agent-sdk"])
    store.create_turn(
        session.id,
        "do things",
        status="completed",
        model="claude-fable-5",
        reasoning_effort="high",
    )
    client = ClaudeAgentSdkClient(store=store)

    sessions = await client.sessions()
    view = next(item for item in sessions if item["id"] == session.id)
    assert view["model"] == "claude-fable-5"
    assert view["reasoningEffort"] == "high"
