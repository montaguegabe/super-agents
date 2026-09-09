"""Regression tests for the 2026-09-09 dispatcher voice-session failures.

Covers: model-first launch with provider-style aliases, rejection of unknown
slugs like "seoul", caller-model inheritance, failed-turn surfacing (the
"astra completed with error" incident), and last-interaction session
indexing of the local Claude home.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from super_agents.agent_store import Store
from super_agents.app_protocol import normalize_turn_status, turn_error_message
from super_agents.app_server_client import CodexAppServerClient
from super_agents.backend_config import (
    CLAUDE_CODE_BACKEND,
    CODEX_BACKEND,
    UnknownModelError,
    execution_backend_for_model,
    resolve_model,
)
from super_agents.claude_home_index import refresh_last_interaction_index
from super_agents.defaults import default_super_agents_model

# --- Model catalog and aliases -------------------------------------------------


def test_catalog_models_resolve_to_their_backends() -> None:
    assert resolve_model("sol") == ("sol", CODEX_BACKEND)
    assert resolve_model("astra") == ("astra", CODEX_BACKEND)
    assert resolve_model("gpt-5.5") == ("gpt-5.5", CODEX_BACKEND)
    assert resolve_model("fable") == ("fable", CLAUDE_CODE_BACKEND)
    assert resolve_model("opus") == ("opus", CLAUDE_CODE_BACKEND)


def test_provider_style_aliases_normalize_to_canonical_slugs() -> None:
    assert resolve_model("openai-sol") == ("sol", CODEX_BACKEND)
    assert resolve_model("open-ai-sol") == ("sol", CODEX_BACKEND)
    assert resolve_model("openai/sol") == ("sol", CODEX_BACKEND)
    assert resolve_model("OpenAI Astra") == ("astra", CODEX_BACKEND)
    assert resolve_model("claude-fable-5") == ("fable", CLAUDE_CODE_BACKEND)
    assert resolve_model("anthropic-fable") == ("fable", CLAUDE_CODE_BACKEND)


def test_unknown_bare_slug_is_rejected_with_suggestions() -> None:
    with pytest.raises(UnknownModelError) as excinfo:
        resolve_model("seoul")
    message = str(excinfo.value)
    assert "seoul" in message
    assert "sol" in message


def test_provider_prefixed_unknown_models_pass_through_by_family() -> None:
    slug, backend = resolve_model("gpt-6-preview")
    assert slug == "gpt-6-preview"
    assert backend == CODEX_BACKEND
    slug, backend = resolve_model("claude-nova-9")
    assert backend == CLAUDE_CODE_BACKEND


def test_execution_backend_for_model_covers_new_codex_models() -> None:
    assert execution_backend_for_model("astra") == CODEX_BACKEND
    assert execution_backend_for_model("sol") == CODEX_BACKEND
    assert execution_backend_for_model("fable") == CLAUDE_CODE_BACKEND
    assert execution_backend_for_model("seoul") is None


def test_extra_models_env_extends_catalog(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SUPER_AGENTS_EXTRA_MODELS", json.dumps({"codex": ["nova"]}))
    assert resolve_model("nova") == ("nova", CODEX_BACKEND)


# --- Caller-model inheritance --------------------------------------------------


@pytest.fixture()
def isolated_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "SUPER_AGENTS_DEFAULT_CONFIG_PATH",
        str(tmp_path / "missing-dispatcher-config.json"),
    )
    monkeypatch.delenv("SUPER_AGENTS_CALLER_MODEL", raising=False)
    monkeypatch.delenv("AGENT_MODEL", raising=False)


def test_caller_model_becomes_default_for_matching_backend(
    monkeypatch: pytest.MonkeyPatch, isolated_defaults: None
) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")
    monkeypatch.setenv("AGENT_MODEL", "astra")
    assert default_super_agents_model(backend="codex") == "astra"


def test_caller_model_is_ignored_for_incompatible_backend(
    monkeypatch: pytest.MonkeyPatch, isolated_defaults: None
) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")
    monkeypatch.setenv("AGENT_MODEL", "fable")
    assert default_super_agents_model(backend="codex") is None


def test_explicit_caller_model_env_wins_over_agent_model(
    monkeypatch: pytest.MonkeyPatch, isolated_defaults: None
) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")
    monkeypatch.setenv("AGENT_MODEL", "gpt-5.5")
    monkeypatch.setenv("SUPER_AGENTS_CALLER_MODEL", "sol")
    assert default_super_agents_model(backend="codex") == "sol"


# --- Failed-turn surfacing -----------------------------------------------------

ASTRA_ERROR_ENVELOPE = json.dumps(
    {
        "type": "error",
        "status": 400,
        "error": {
            "type": "invalid_request_error",
            "message": "The 'astra' model is not supported when using Codex with a ChatGPT account.",
        },
    }
)


def test_turn_error_message_unwraps_provider_envelope() -> None:
    turn = {"id": "turn-1", "status": "completed", "error": {"message": ASTRA_ERROR_ENVELOPE}}
    message = turn_error_message(turn)
    assert message == "The 'astra' model is not supported when using Codex with a ChatGPT account."


def test_completed_turn_with_error_normalizes_to_failed() -> None:
    turn = {"id": "turn-1", "status": "completed", "error": {"message": "boom"}}
    assert normalize_turn_status(turn) == "failed"
    assert normalize_turn_status({"id": "turn-2", "status": "completed", "error": None}) == "completed"


@pytest.mark.asyncio
async def test_turn_completed_notification_with_error_records_failure(tmp_path: Path) -> None:
    client = CodexAppServerClient(state_file=tmp_path / "state.json")
    try:
        client.ensure_turn("thread-err", "turn-err")
        client.handle_notification(
            "turn/completed",
            {
                "threadId": "thread-err",
                "turnId": "turn-err",
                "turn": {"id": "turn-err", "status": "completed", "error": {"message": ASTRA_ERROR_ENVELOPE}},
            },
        )
        await asyncio.sleep(0.05)
        session = await client.get_session("thread-err")
        assert session is not None
        assert session.last_status == "failed"
        assert "astra" in (session.last_error or "")
        assert "astra" in (session.last_useful_message or "")
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_clean_completion_still_records_completed(tmp_path: Path) -> None:
    client = CodexAppServerClient(state_file=tmp_path / "state.json")
    try:
        client.ensure_turn("thread-ok", "turn-ok")
        client.handle_notification(
            "turn/completed",
            {"threadId": "thread-ok", "turnId": "turn-ok"},
        )
        await asyncio.sleep(0.05)
        session = await client.get_session("thread-ok")
        assert session is not None
        assert session.last_status == "completed"
        assert session.last_error is None
    finally:
        await client.close()


# --- Last-interaction Claude home index ----------------------------------------


def _write_transcript(projects_dir: Path, cwd: str, session_id: str, text: str, mtime: float) -> Path:
    project_dir = projects_dir / cwd.replace("/", "-")
    project_dir.mkdir(parents=True, exist_ok=True)
    path = project_dir / f"{session_id}.jsonl"
    entries = [
        {
            "type": "user",
            "cwd": cwd,
            "timestamp": "2026-09-09T12:00:00.000Z",
            "message": {"role": "user", "content": [{"type": "text", "text": text}]},
        },
        {
            "type": "assistant",
            "cwd": cwd,
            "timestamp": "2026-09-09T12:00:05.000Z",
            "message": {"role": "assistant", "content": [{"type": "text", "text": f"reply to {text}"}]},
        },
    ]
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    os.utime(path, (mtime, mtime))
    return path


def test_local_claude_sessions_are_indexed_by_last_interaction(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    claude_home = tmp_path / "claude-home"
    projects = claude_home / "projects"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    now = time.time()
    _write_transcript(projects, "/tmp/proj-a", "11111111-1111-4111-8111-111111111111", "old task", now - 3600)
    _write_transcript(projects, "/tmp/proj-b", "22222222-2222-4222-8222-222222222222", "work on notifications", now)

    store = Store(tmp_path / "state.sqlite3", backend="claude_code")
    changed = refresh_last_interaction_index(store, now=time.monotonic())
    assert changed == 2

    sessions = store.list_sessions(include_inactive=True)
    assert [session.name for session in sessions] == ["work on notifications", "old task"]
    assert sessions[0].backend_session_id == "22222222-2222-4222-8222-222222222222"
    assert sessions[0].cwd == "/tmp/proj-b"


def test_index_refreshes_registered_sessions_to_transcript_mtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    claude_home = tmp_path / "claude-home"
    projects = claude_home / "projects"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    now = time.time()
    session_id = "33333333-3333-4333-8333-333333333333"
    path = _write_transcript(projects, "/tmp/proj-c", session_id, "long-lived session", now - 86400)

    store = Store(tmp_path / "state.sqlite3", backend="claude_code")
    assert refresh_last_interaction_index(store, now=time.monotonic()) == 1
    stale = store.list_sessions(include_inactive=True)[0]

    # The user talks to the session again much later: only mtime moves.
    os.utime(path, (now, now))
    assert refresh_last_interaction_index(store, now=time.monotonic() + 100) == 1
    fresh = store.list_sessions(include_inactive=True)[0]
    assert fresh.id == stale.id
    assert fresh.updated_at > stale.updated_at


def test_sidechain_only_transcripts_are_not_indexed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    claude_home = tmp_path / "claude-home"
    projects = claude_home / "projects"
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_home))
    project_dir = projects / "tool-agents"
    project_dir.mkdir(parents=True)
    entry = {
        "type": "user",
        "isSidechain": True,
        "message": {"role": "user", "content": [{"type": "text", "text": "subagent prompt"}]},
    }
    (project_dir / "44444444-4444-4444-8444-444444444444.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")

    store = Store(tmp_path / "state.sqlite3", backend="claude_code")
    assert refresh_last_interaction_index(store, now=time.monotonic()) == 0
    assert store.list_sessions(include_inactive=True) == []
