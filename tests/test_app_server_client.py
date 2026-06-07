from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Callable

import pytest
import websockets

import super_agents.app_server_client as app_server_client
from super_agents.app_client_transport import websocket_max_size
from super_agents.app_server_client import CodexAppServerClient
from super_agents.mcp_server import (
    build_tools,
    clean_queue_turn_input,
    clean_start_turn_by_name_input,
    clean_thread_input,
    clean_turn_input,
    default_super_agent_instructions_path,
)


class ReadyClient(CodexAppServerClient):
    async def check_ready(self) -> bool:
        return True


def test_websocket_max_size_defaults_above_codex_default(monkeypatch) -> None:
    monkeypatch.delenv("SUPER_AGENTS_WEBSOCKET_MAX_SIZE", raising=False)

    assert websocket_max_size() == 16 * 1024 * 1024


def test_websocket_max_size_can_be_disabled(monkeypatch) -> None:
    monkeypatch.setenv("SUPER_AGENTS_WEBSOCKET_MAX_SIZE", "unlimited")

    assert websocket_max_size() is None


def test_turn_input_does_not_default_approval_or_sandbox() -> None:
    cleaned = clean_turn_input(
        {
            "threadId": "thread-1",
            "prompt": 'Wait, then run /Users/gabemontague/.local/bin/openbase-coder user say Dottie "Done".',
            "approvalPolicy": "never",
            "sandboxType": "dangerFullAccess",
        }
    )

    assert "approvalPolicy" not in cleaned
    assert "sandboxType" not in cleaned
    assert cleaned["reasoningEffort"] == "high"
    assert cleaned["serviceTier"] == "fast"


def test_turn_input_accepts_hidden_reasoning_and_service_tier_overrides() -> None:
    cleaned = clean_turn_input(
        {
            "threadId": "thread-1",
            "prompt": "work",
            "reasoningEffort": "low",
            "serviceTier": "standard",
        }
    )

    assert cleaned["reasoningEffort"] == "low"
    assert cleaned["serviceTier"] == "standard"


def test_thread_input_does_not_default_approval_or_sandbox() -> None:
    cleaned = clean_thread_input(
        {
            "name": "new-agent",
            "agentName": "Dottie",
            "approvalPolicy": "never",
            "sandbox": "danger-full-access",
        }
    )

    assert "approvalPolicy" not in cleaned
    assert "sandbox" not in cleaned
    assert cleaned["agentName"] == "Dottie"


def test_thread_input_rejects_missing_cwd(tmp_path: Path) -> None:
    missing_cwd = tmp_path / "missing"

    with pytest.raises(ValueError, match="cwd must be an existing directory"):
        clean_thread_input({"name": "new-agent", "cwd": str(missing_cwd)})


@pytest.mark.asyncio
async def test_start_tool_rejects_missing_cwd_before_client_start(tmp_path: Path) -> None:
    class RecordingClient(CodexAppServerClient):
        called = False

        async def start_thread(self, input_data: dict[str, Any]) -> dict[str, Any]:
            self.called = True
            return {"threadId": "unexpected"}

    missing_cwd = tmp_path / "missing"
    client = RecordingClient("ws://127.0.0.1:1", tmp_path / "state.json", "gpt-test")
    tool = next(tool for tool in build_tools(client) if tool.name == "super_agents_start")

    with pytest.raises(ValueError, match="cwd must be an existing directory"):
        await tool.handler({"name": "new-agent", "cwd": str(missing_cwd)})

    assert client.called is False


def test_turn_input_uses_openbase_super_agents_reasoning_default(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(json.dumps({"super_agents_reasoning_effort": "low"}), encoding="utf-8")
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))

    cleaned = clean_turn_input({"threadId": "thread-1", "prompt": "work"})

    assert cleaned["reasoningEffort"] == "low"


def test_turn_input_ignores_legacy_shared_reasoning_key(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(json.dumps({"reasoning_effort": "low"}), encoding="utf-8")
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))

    cleaned = clean_turn_input({"threadId": "thread-1", "prompt": "work"})

    assert cleaned["reasoningEffort"] == "high"


def test_super_agent_instructions_default_to_configured_path(monkeypatch, tmp_path: Path) -> None:
    instructions_path = tmp_path / "SUPER_AGENT_INSTRUCTIONS.md"
    instructions_path.write_text("The random animal is raccoon.\n", encoding="utf-8")
    monkeypatch.setenv("CODEX_SUPER_AGENT_INSTRUCTIONS_PATH", str(instructions_path))

    thread_input = clean_thread_input({"name": "new-agent"})
    turn_input = clean_start_turn_by_name_input({"name": "new-agent", "prompt": "work"})
    queue_input = clean_queue_turn_input({"name": "new-agent", "prompt": "later"})

    assert default_super_agent_instructions_path() == instructions_path
    assert thread_input["developerInstructions"] == "The random animal is raccoon.\n"
    assert turn_input["developerInstructions"] == "The random animal is raccoon.\n"
    assert queue_input["developerInstructions"] == "The random animal is raccoon.\n"


def test_super_agent_instructions_default_to_codex_home(monkeypatch, tmp_path: Path) -> None:
    codex_home = tmp_path / "codex_home"
    codex_home.mkdir()
    instructions_path = codex_home / "SUPER_AGENT_INSTRUCTIONS.md"
    instructions_path.write_text("Use the Super Agent instructions.\n", encoding="utf-8")
    monkeypatch.delenv("CODEX_SUPER_AGENT_INSTRUCTIONS_PATH", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    cleaned = clean_thread_input({"name": "new-agent"})

    assert default_super_agent_instructions_path() == instructions_path
    assert cleaned["developerInstructions"] == "Use the Super Agent instructions.\n"


def test_explicit_developer_instructions_override_super_agent_default(monkeypatch, tmp_path: Path) -> None:
    instructions_path = tmp_path / "SUPER_AGENT_INSTRUCTIONS.md"
    instructions_path.write_text("Default instructions.\n", encoding="utf-8")
    monkeypatch.setenv("CODEX_SUPER_AGENT_INSTRUCTIONS_PATH", str(instructions_path))

    thread_input = clean_thread_input({"name": "new-agent", "developerInstructions": "Explicit instructions."})
    turn_input = clean_start_turn_by_name_input(
        {"name": "new-agent", "prompt": "work", "developerInstructions": "Explicit turn instructions."}
    )

    assert thread_input["developerInstructions"] == "Explicit instructions."
    assert turn_input["developerInstructions"] == "Explicit turn instructions."


def test_start_turn_by_name_input_ignores_agent_name() -> None:
    cleaned = clean_start_turn_by_name_input({"name": "new-agent", "prompt": "work", "agentName": "Dottie"})

    assert "agentName" not in cleaned


def test_explicit_null_suppresses_super_agent_default_for_turns(monkeypatch, tmp_path: Path) -> None:
    instructions_path = tmp_path / "SUPER_AGENT_INSTRUCTIONS.md"
    instructions_path.write_text("Default instructions.\n", encoding="utf-8")
    monkeypatch.setenv("CODEX_SUPER_AGENT_INSTRUCTIONS_PATH", str(instructions_path))

    turn_input = clean_start_turn_by_name_input({"name": "new-agent", "prompt": "work", "developerInstructions": None})
    queue_input = clean_queue_turn_input({"name": "new-agent", "prompt": "later", "developerInstructions": None})

    assert "developerInstructions" not in turn_input
    assert "developerInstructions" not in queue_input


@pytest.mark.asyncio
async def test_steer_turn_sends_expected_turn_id(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    server = await start_fake_app_server(captured)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.steer_turn("thread-1", "turn-1", "narrow the scope")
        steer_request = next(message for message in captured if message.get("method") == "turn/steer")
        assert steer_request["params"] == {
            "threadId": "thread-1",
            "expectedTurnId": "turn-1",
            "input": [{"type": "text", "text": "narrow the scope"}],
        }
        assert "turnId" not in steer_request["params"]
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_label_tools_resolve_latest_active_session_and_list_active_agents(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    thread_index = 0
    turn_index = 0

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        nonlocal thread_index, turn_index
        if message.get("method") == "thread/start":
            thread_index += 1
            return {"threadId": f"thread-{thread_index}", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            turn_index += 1
            return {"turnId": f"turn-{turn_index}", "text": f"started {turn_index}"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "build", "cwd": "/tmp/one"})
        await client.start_turn({"threadId": "thread-1", "label": "build", "prompt": "first"})
        await asyncio.sleep(0.005)
        await client.start_thread({"label": "build", "cwd": "/tmp/two"})
        await client.start_turn({"threadId": "thread-2", "label": "build", "prompt": "second"})

        resolved = await client.resolve_label(type_query(label="build"))
        assert resolved["threadId"] == "thread-2"
        assert resolved["turnId"] == "turn-2"
        assert resolved["status"] == "running"

        active = await client.active(type_query(label="build"))
        assert active["count"] == 2
        assert active["agents"][0]["threadId"] == "thread-2"
        assert active["agents"][0]["runningTurnId"] == "turn-2"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_start_turn_sets_super_agent_identity_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    async def fake_login_shell_environment() -> dict[str, str]:
        return {
            "PATH": "/usr/bin",
            "SHELL": "/bin/zsh",
            "HOME": "/Users/example",
            "USER": "example",
            "LOGNAME": "example",
        }

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "turn/start":
            return {"turnId": "turn-1"}
        return {"ok": True}

    monkeypatch.setattr(app_server_client, "login_shell_environment", fake_login_shell_environment)
    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_turn(
            {
                "threadId": "thread-1",
                "label": "Build",
                "agentName": "Dottie",
                "prompt": "work",
            }
        )

        resume_request = next(message for message in captured if message.get("method") == "thread/resume")
        resume_env = resume_request["params"]["config"]["shell_environment_policy"]["set"]
        assert resume_request["params"]["threadId"] == "thread-1"
        assert "OPENBASE_SUPER_AGENT_THREAD_ID" not in resume_env
        assert "OPENBASE_SUPER_AGENT_LABEL" not in resume_env
        assert "OPENBASE_SUPER_AGENT_AGENT_NAME" not in resume_env

        start_request = next(message for message in captured if message.get("method") == "turn/start")
        set_env = start_request["params"]["config"]["shell_environment_policy"]["set"]
        assert "OPENBASE_SUPER_AGENT_THREAD_ID" not in set_env
        assert "OPENBASE_SUPER_AGENT_LABEL" not in set_env
        assert "OPENBASE_SUPER_AGENT_AGENT_NAME" not in set_env
        assert set_env["PATH"] == "/usr/bin"
        assert (
            start_request["params"]["collaborationMode"]["settings"]["developer_instructions"]
            == "Super Agent thread name: Build\nSuper Agent thread id: thread-1\nYour name is Dottie."
        )
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_start_turn_continues_when_new_thread_has_no_rollout_to_resume(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    async def fake_login_shell_environment() -> dict[str, str]:
        return {"PATH": "/usr/bin"}

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/resume":
            return {"__error__": {"code": -32600, "message": "no rollout found for thread id thread-new"}}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-1"}
        return {"ok": True}

    monkeypatch.setattr(app_server_client, "login_shell_environment", fake_login_shell_environment)
    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        result = await client.start_turn(
            {"threadId": "thread-new", "label": "Build", "agentName": "Dottie", "prompt": "work"}
        )

        assert result["turnId"] == "turn-1"
        start_request = next(message for message in captured if message.get("method") == "turn/start")
        assert start_request["params"]["collaborationMode"]["settings"]["developer_instructions"] == (
            "Super Agent thread name: Build\nSuper Agent thread id: thread-new\nYour name is Dottie."
        )
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_login_shell_config_excludes_legacy_super_agent_identity_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_login_shell_environment() -> dict[str, str]:
        return {
            "PATH": "/usr/bin",
            "OPENBASE_SUPER_AGENT_THREAD_ID": "stale-thread",
            "OPENBASE_SUPER_AGENT_LABEL": "Stale",
            "OPENBASE_SUPER_AGENT_AGENT_NAME": "Stale Name",
        }

    monkeypatch.setenv("OPENBASE_SUPER_AGENT_THREAD_ID", "parent-thread")
    monkeypatch.setenv("OPENBASE_SUPER_AGENT_LABEL", "Parent")
    monkeypatch.setenv("OPENBASE_SUPER_AGENT_AGENT_NAME", "Parent Name")
    monkeypatch.setattr(app_server_client, "login_shell_environment", fake_login_shell_environment)

    first = await app_server_client.login_shell_config_override(
        thread_id="thread-a",
        label="Agent A",
        agent_name="Dottie",
    )
    second = await app_server_client.login_shell_config_override(
        thread_id="thread-b",
        label="Agent B",
    )
    cleared = await app_server_client.login_shell_config_override()

    for config in (first, second, cleared):
        set_env = config["shell_environment_policy"]["set"]
        assert "OPENBASE_SUPER_AGENT_THREAD_ID" not in set_env
        assert "OPENBASE_SUPER_AGENT_LABEL" not in set_env
        assert "OPENBASE_SUPER_AGENT_AGENT_NAME" not in set_env
    assert os.environ["OPENBASE_SUPER_AGENT_THREAD_ID"] == "parent-thread"
    assert os.environ["OPENBASE_SUPER_AGENT_AGENT_NAME"] == "Parent Name"


def test_super_agent_identity_instructions_replace_stale_identity_lines() -> None:
    assert app_server_client.with_super_agent_identity_instructions(
        "Base instructions.\n\nSuper Agent thread name: Old\nSuper Agent thread id: old-thread\nYour name is Old.",
        "New Agent",
        "new-thread",
        "Dottie",
    ) == "Base instructions.\n\nSuper Agent thread name: New Agent\nSuper Agent thread id: new-thread\nYour name is Dottie."


@pytest.mark.asyncio
async def test_thread_start_does_not_clear_super_agent_identity_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []

    async def fake_login_shell_environment() -> dict[str, str]:
        return {
            "PATH": "/usr/bin",
            "OPENBASE_SUPER_AGENT_THREAD_ID": "parent-thread",
            "OPENBASE_SUPER_AGENT_LABEL": "Parent",
            "OPENBASE_SUPER_AGENT_AGENT_NAME": "Parent Name",
        }

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"thread": {"id": "thread-new", "cwd": message["params"]["cwd"], "model": "gpt-test"}}
        return {"ok": True}

    monkeypatch.setattr(app_server_client, "login_shell_environment", fake_login_shell_environment)
    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"name": "New Agent", "cwd": "/tmp/new"})

        start_request = next(message for message in captured if message.get("method") == "thread/start")
        set_env = start_request["params"]["config"]["shell_environment_policy"]["set"]
        assert "OPENBASE_SUPER_AGENT_THREAD_ID" not in set_env
        assert "OPENBASE_SUPER_AGENT_LABEL" not in set_env
        assert "OPENBASE_SUPER_AGENT_AGENT_NAME" not in set_env
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_start_thread_appends_super_agent_identity_to_explicit_instructions(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"thread": {"id": "thread-identity", "cwd": message["params"]["cwd"], "model": "gpt-test"}}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread(
            {
                "name": "Identity Agent",
                "agentName": "Dottie",
                "cwd": "/tmp/identity",
                "developerInstructions": "Explicit instructions.",
            }
        )

        start_request = next(message for message in captured if message.get("method") == "thread/start")
        assert start_request["params"]["developerInstructions"] == (
            "Explicit instructions.\n\nSuper Agent thread name: Identity Agent\nYour name is Dottie."
        )
        state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
        assert state["sessions"]["thread-identity"]["agentName"] == "Dottie"
        assert not [message for message in captured if message.get("method") == "thread/resume"]
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_start_thread_sets_native_app_server_thread_name(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"thread": {"id": "thread-native", "cwd": message["params"]["cwd"], "model": "gpt-test"}}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"name": "native-name", "agentName": "Carl", "cwd": "/tmp/native"})

        start_request = next(message for message in captured if message.get("method") == "thread/start")
        assert "approvalPolicy" not in start_request["params"]
        assert "sandbox" not in start_request["params"]

        rename_request = next(message for message in captured if message.get("method") == "thread/name/set")
        assert rename_request["params"] == {"threadId": "thread-native", "name": "native-name"}

        state = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
        assert state["sessions"]["thread-native"]["label"] == "native-name"
        assert state["sessions"]["thread-native"]["agentName"] == "Carl"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_start_turn_by_name_resolves_native_app_server_thread_name(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/list":
            return {
                "data": [
                    {
                        "id": "thread-native",
                        "name": "native-name",
                        "cwd": "/tmp/native",
                        "updatedAt": 100,
                        "status": "completed",
                    }
                ],
                "nextCursor": None,
                "backwardsCursor": None,
            }
        if message.get("method") == "turn/start":
            return {"turnId": "turn-native"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        caplog.set_level("INFO", logger="super_agents.app_server_client")
        result = await client.start_turn_by_label(
            type_query(label="native-name"),
            {"label": "native-name", "prompt": "continue", "_mcpCallId": "mcp-test"},
        )

        assert result["threadId"] == "thread-native"
        start_request = next(message for message in captured if message.get("method") == "turn/start")
        assert start_request["params"]["threadId"] == "thread-native"
        assert start_request["params"]["cwd"] == "/tmp/native"
        assert start_request["params"]["serviceTier"] == "fast"
        assert "approvalPolicy" not in start_request["params"]
        assert "sandboxPolicy" not in start_request["params"]
        messages = [record.getMessage() for record in caplog.records]
        assert any("stage=super_agents_resolve_start" in message for message in messages)
        assert any("stage=super_agents_resolve_end" in message for message in messages)
        assert any("stage=app_server_turn_start_request" in message for message in messages)
        assert any("stage=app_server_turn_start_response" in message for message in messages)
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_resume_thread_does_not_override_codex_approval_or_sandbox_defaults(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/resume":
            return {"thread": {"id": "thread-resume", "model": "gpt-test"}}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.resume_thread("thread-resume", label="Resume Agent", agent_name="Dottie")

        resume_request = next(message for message in captured if message.get("method") == "thread/resume")
        assert "approvalPolicy" not in resume_request["params"]
        assert "sandbox" not in resume_request["params"]
        set_env = resume_request["params"]["config"]["shell_environment_policy"]["set"]
        assert "OPENBASE_SUPER_AGENT_THREAD_ID" not in set_env
        assert "OPENBASE_SUPER_AGENT_LABEL" not in set_env
        assert "OPENBASE_SUPER_AGENT_AGENT_NAME" not in set_env
        assert resume_request["params"]["developerInstructions"] == (
            "Super Agent thread name: Resume Agent\nSuper Agent thread id: thread-resume\nYour name is Dottie."
        )
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_rename_by_name_uses_native_app_server_thread_name(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/list":
            return {
                "data": [
                    {
                        "id": "thread-native",
                        "name": "old-name",
                        "cwd": "/tmp/native",
                        "updatedAt": 100,
                        "status": "completed",
                    }
                ],
                "nextCursor": None,
                "backwardsCursor": None,
            }
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        result = await client.rename_by_label(type_query(label="old-name"), "new-name")

        assert result["renamed"] is True
        rename_request = next(message for message in captured if message.get("method") == "thread/name/set")
        assert rename_request["params"] == {"threadId": "thread-native", "name": "new-name"}
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_progress_by_label_reuses_resolved_thread_and_turn_ids(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-progress", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-progress"}
        if message.get("method") == "thread/read":
            return {
                "threadId": message["params"]["threadId"],
                "turns": [{"id": "turn-progress", "status": "inProgress", "message": "still working"}],
            }
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "progress"})
        await client.start_turn({"threadId": "thread-progress", "label": "progress", "prompt": "work"})
        progress = await client.progress_by_label(type_query(label="progress"))

        assert progress["threadId"] == "thread-progress"
        assert progress["turnId"] == "turn-progress"
        assert progress["status"] == "running"
        assert "turn" not in progress
        assert "events" not in progress.get("trackedTurn", {})
        assert progress["summary"]["lastUsefulMessage"] == "still working"
        assert any(
            message.get("method") == "thread/read" and message["params"]["threadId"] == "thread-progress"
            for message in captured
        )
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_steer_and_cancel_by_label_call_existing_id_based_app_server_methods(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-control", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-control"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "control"})
        await client.start_turn({"threadId": "thread-control", "label": "control", "prompt": "work"})
        await client.steer_by_label(type_query(label="control"), "adjust")
        await client.cancel_by_label(type_query(label="control"))

        steer_request = next(message for message in captured if message.get("method") == "turn/steer")
        assert steer_request["params"] == {
            "threadId": "thread-control",
            "expectedTurnId": "turn-control",
            "input": [{"type": "text", "text": "adjust"}],
        }
        cancel_request = next(message for message in captured if message.get("method") == "turn/interrupt")
        assert cancel_request["params"] == {"threadId": "thread-control", "turnId": "turn-control"}

        active = await client.active(type_query(label="control"))
        assert active["count"] == 0
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_start_turn_by_label_can_start_follow_up_on_latest_inactive_label(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-follow-up", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-follow-up"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "follow-up", "cwd": "/tmp/project"})
        result = await client.start_turn_by_label(
            type_query(label="follow-up"), {"label": "follow-up", "prompt": "continue"}
        )
        assert result["threadId"] == "thread-follow-up"
        assert result["turnId"] == "turn-follow-up"
        start_request = next(message for message in captured if message.get("method") == "turn/start")
        assert start_request["params"]["threadId"] == "thread-follow-up"
        assert start_request["params"]["cwd"] == "/tmp/project"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_start_turn_by_name_on_running_thread_queues_prompt(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-active", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-active"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "active", "cwd": "/tmp/project"})
        await client.start_turn({"threadId": "thread-active", "label": "active", "prompt": "work"})

        result = await client.start_turn_by_label(type_query(label="active"), {"label": "active", "prompt": "queued"})

        start_requests = [message for message in captured if message.get("method") == "turn/start"]
        assert result["queued"] is True
        assert result["drain"] == "waiting_for_active_turn"
        assert result["queueDepth"] == 1
        assert result["item"]["promptPreview"] == "queued"
        assert len(start_requests) == 1
        assert client.queued_turn_summary()[0]["queueDepth"] == 1
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_queue_turn_waits_for_active_turn_completion_then_starts_next_turn(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    turn_index = 0

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        nonlocal turn_index
        if message.get("method") == "thread/start":
            return {"threadId": "thread-queue", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            turn_index += 1
            return {"turnId": f"turn-{turn_index}"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "queue", "cwd": "/tmp/project"})
        await client.start_turn({"threadId": "thread-queue", "label": "queue", "prompt": "first"})

        queued = await client.queue_turn_by_label(type_query(label="queue"), {"label": "queue", "prompt": "second"})

        assert queued["queued"] is True
        assert queued["drain"] == "waiting_for_active_turn"
        assert [message.get("method") for message in captured].count("turn/start") == 1
        assert client.queued_turn_summary()[0]["queueDepth"] == 1

        client.handle_notification("turn/completed", {"threadId": "thread-queue", "turnId": "turn-1"})
        await asyncio.sleep(0.05)

        start_requests = [message for message in captured if message.get("method") == "turn/start"]
        assert len(start_requests) == 2
        assert start_requests[-1]["params"]["input"] == [{"type": "text", "text": "second"}]
        assert start_requests[-1]["params"]["threadId"] == "thread-queue"
        assert client.queued_turn_summary() == []
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_queue_drain_ignores_running_turn_with_finished_at(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    turn_index = 0

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        nonlocal turn_index
        if message.get("method") == "thread/start":
            return {"threadId": "thread-stale-queue", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            turn_index += 1
            return {"turnId": f"turn-{turn_index}"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "stale-queue", "cwd": "/tmp/project"})
        await client.start_turn({"threadId": "thread-stale-queue", "label": "stale-queue", "prompt": "first"})

        queued = await client.queue_turn_by_label(
            type_query(label="stale-queue"),
            {"label": "stale-queue", "prompt": "second"},
        )
        await asyncio.sleep(0.05)

        assert queued["queued"] is True
        assert [message.get("method") for message in captured].count("turn/start") == 1
        assert client.queued_turn_summary()[0]["queueDepth"] == 1

        stale_turn = client.ensure_turn("thread-stale-queue", "turn-1")
        stale_turn.status = "running"
        stale_turn.started_at = "2020-01-01T00:00:00.000Z"
        stale_turn.finished_at = "2026-01-01T00:00:00.000Z"
        client.schedule_queue_drain("thread-stale-queue")

        for _ in range(20):
            if [message.get("method") for message in captured].count("turn/start") == 2:
                break
            await asyncio.sleep(0.01)

        start_requests = [message for message in captured if message.get("method") == "turn/start"]
        assert len(start_requests) == 2
        assert start_requests[-1]["params"]["input"] == [{"type": "text", "text": "second"}]
        assert start_requests[-1]["params"]["threadId"] == "thread-stale-queue"
        assert client.queued_turn_summary() == []

        status = await client.status()
        active_turn_ids = {turn["turnId"] for turn in status["activeTurns"]}
        assert "turn-1" not in active_turn_ids
        assert "turn-2" in active_turn_ids
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_thread_queue_is_persisted_across_mcp_clients(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    turn_index = 0
    state_file = tmp_path / "state.json"

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        nonlocal turn_index
        if message.get("method") == "thread/start":
            return {"threadId": "thread-persisted-queue", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            turn_index += 1
            return {"turnId": f"turn-{turn_index}"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    owner = ReadyClient(server.ws_url, state_file, "gpt-test")
    enqueuer = ReadyClient(server.ws_url, state_file, "gpt-test")
    try:
        await owner.start_thread({"label": "persisted-queue", "cwd": "/tmp/project"})
        await owner.start_turn(
            {
                "threadId": "thread-persisted-queue",
                "label": "persisted-queue",
                "prompt": "first",
            }
        )

        queued = await enqueuer.queue_turn_by_label(
            type_query(thread_id="thread-persisted-queue"),
            {"prompt": "second"},
        )

        assert queued["queued"] is True
        assert enqueuer.queued_turn_summary()[0]["threadId"] == "thread-persisted-queue"
        assert enqueuer.queued_turn_summary()[0]["queueDepth"] == 1
        assert (tmp_path / "queues").is_dir()
        assert len([message for message in captured if message.get("method") == "turn/start"]) == 1

        owner.handle_notification(
            "turn/completed",
            {"threadId": "thread-persisted-queue", "turnId": "turn-1"},
        )
        await asyncio.sleep(0.05)

        start_requests = [message for message in captured if message.get("method") == "turn/start"]
        assert len(start_requests) == 2
        assert start_requests[-1]["params"]["input"] == [{"type": "text", "text": "second"}]
        assert owner.queued_turn_summary() == []
    finally:
        await owner.close()
        await enqueuer.close()
        await server.close()


@pytest.mark.asyncio
async def test_queue_turn_on_idle_thread_starts_without_waiting(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-idle", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-idle"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "idle", "cwd": "/tmp/project"})

        queued = await client.queue_turn_by_label(type_query(label="idle"), {"label": "idle", "prompt": "start now"})

        assert queued["queued"] is False
        assert queued["startedImmediately"] is True
        assert queued["drain"] == "started_immediately"
        assert queued["turnId"] == "turn-idle"
        start_request = next(message for message in captured if message.get("method") == "turn/start")
        assert start_request["params"]["input"] == [{"type": "text", "text": "start now"}]
        assert client.queued_turn_summary() == []
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_old_state_files_are_tolerated_by_recent_and_label_resolution(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "sessions": {
                    "thread-old": {
                        "label": "old",
                        "threadId": "thread-old",
                        "cwd": "/tmp/old",
                        "model": "gpt-test",
                        "lastTurnId": "turn-old",
                        "updatedAt": "2026-01-01T00:00:00.000Z",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    client = ReadyClient("ws://127.0.0.1:1", state_file, "gpt-test")

    recent = await client.recent(type_query(label="old", include_inactive=True))
    assert recent["count"] == 1
    assert recent["agents"][0]["threadId"] == "thread-old"

    resolved = await client.resolve_label(type_query(label="old", prefer="latest_any"))
    assert resolved["threadId"] == "thread-old"
    assert resolved["turnId"] == "turn-old"
    assert resolved["status"] == "unknown"


@pytest.mark.asyncio
async def test_recent_can_filter_local_openbase_favorites(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENBASE_CODER_CLI_DATA_DIR", str(tmp_path / "openbase"))
    (tmp_path / "openbase").mkdir()
    (tmp_path / "openbase" / "thread-favorites.json").write_text(
        json.dumps(
            {
                "threads": {
                    "thread-favorite": {
                        "thread_id": "thread-favorite",
                        "favorited_at": "2026-06-10T12:00:00Z",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "sessions": {
                    "thread-favorite": {
                        "label": "favorite",
                        "threadId": "thread-favorite",
                        "updatedAt": "2026-01-02T00:00:00.000Z",
                        "lastStatus": "completed",
                    },
                    "thread-normal": {
                        "label": "normal",
                        "threadId": "thread-normal",
                        "updatedAt": "2026-01-01T00:00:00.000Z",
                        "lastStatus": "completed",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    client = ReadyClient("ws://127.0.0.1:1", state_file, "gpt-test")

    recent = await client.recent(type_query(favorite=True, include_inactive=True))
    normal = await client.recent(type_query(favorite=False, include_inactive=True))

    assert [agent["threadId"] for agent in recent["agents"]] == ["thread-favorite"]
    assert recent["agents"][0]["isFavorite"] is True
    assert recent["agents"][0]["favoritedAt"] == "2026-06-10T12:00:00Z"
    assert [agent["threadId"] for agent in normal["agents"]] == ["thread-normal"]


def test_thread_favorite_reads_local_openbase_favorite(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OPENBASE_CODER_CLI_DATA_DIR", str(tmp_path))
    (tmp_path / "thread-favorites.json").write_text(
        json.dumps(
            {
                "threads": {
                    "thread-1": {
                        "thread_id": "thread-1",
                        "favorited_at": "2026-06-10T12:00:00Z",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    client = ReadyClient("ws://127.0.0.1:1", tmp_path / "state.json", "gpt-test")

    assert asyncio.run(client.thread_favorite("thread-1")) == {
        "threadId": "thread-1",
        "isFavorite": True,
        "favoritedAt": "2026-06-10T12:00:00Z",
    }
    assert asyncio.run(client.thread_favorite("thread-2")) == {
        "threadId": "thread-2",
        "isFavorite": False,
        "favoritedAt": None,
    }


@pytest.mark.asyncio
async def test_session_state_is_enriched_with_turn_metadata_compatibly(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-state", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-state", "text": "started state work"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    state_file = tmp_path / "state.json"
    client = ReadyClient(server.ws_url, state_file, "gpt-test")
    try:
        await client.start_thread({"label": "state", "group": "batch", "cwd": "/tmp/state"})
        await client.start_turn(
            {"threadId": "thread-state", "label": "state", "group": "batch", "prompt": "state prompt"}
        )

        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["sessions"]["thread-state"]["label"] == "state"
        assert state["sessions"]["thread-state"]["group"] == "batch"
        assert state["sessions"]["thread-state"]["activeTurnId"] == "turn-state"
        assert state["sessions"]["thread-state"]["turns"]["turn-state"]["promptPreview"] == "state prompt"
        assert state["sessions"]["thread-state"]["turns"]["turn-state"]["reasoningEffort"] == "high"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_turn_reasoning_effort_is_sent_persisted_and_visible(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(json.dumps({"super_agents_reasoning_effort": "low"}), encoding="utf-8")
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-reasoning", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-reasoning"}
        if message.get("method") == "thread/read":
            return {
                "threadId": message["params"]["threadId"],
                "turns": [{"id": "turn-reasoning", "status": "inProgress", "message": "still working"}],
            }
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    state_file = tmp_path / "state.json"
    client = ReadyClient(server.ws_url, state_file, "gpt-test")
    try:
        await client.start_thread({"label": "reasoning", "cwd": "/tmp/reasoning"})
        result = await client.start_turn(
            clean_turn_input({"threadId": "thread-reasoning", "label": "reasoning", "prompt": "work"})
        )

        start_request = next(message for message in captured if message.get("method") == "turn/start")
        assert start_request["params"]["collaborationMode"]["settings"]["reasoning_effort"] == "low"
        assert result["reasoningEffort"] == "low"

        state = json.loads(state_file.read_text(encoding="utf-8"))
        turn = state["sessions"]["thread-reasoning"]["turns"]["turn-reasoning"]
        assert turn["reasoningEffort"] == "low"

        status = await client.compact_status(type_query(label="reasoning"))
        assert status["agents"][0]["reasoningEffort"] == "low"

        progress = await client.progress_by_label(type_query(label="reasoning"))
        assert progress["summary"]["reasoningEffort"] == "low"
        assert progress["trackedTurn"]["reasoningEffort"] == "low"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_progress_full_opt_in_returns_raw_turn_and_tracked_events(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/read":
            return {
                "threadId": message["params"]["threadId"],
                "turns": [{"id": "turn-full", "status": "inProgress", "diff": "x" * 5000}],
            }
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        client.ensure_turn("thread-full", "turn-full").events.append(
            {"method": "turn/item", "params": {"text": "event"}, "receivedAt": "2026-01-01T00:00:00.000Z"}
        )
        progress = await client.progress_by_label(type_query(thread_id="thread-full", turn_id="turn-full", full=True))

        assert progress["turn"]["diff"] == "x" * 5000
        assert progress["trackedTurn"]["events"][0]["method"] == "turn/item"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_active_preview_defaults_to_short_and_can_be_omitted(tmp_path: Path) -> None:
    client = ReadyClient("ws://127.0.0.1:1", tmp_path / "state.json", "gpt-test")
    long_message = " ".join(["word"] * 80)
    await client.remember_session(
        "thread-preview",
        {
            "label": "preview",
            "threadId": "thread-preview",
            "lastStatus": "running",
            "activeTurnId": "turn-preview",
            "lastUsefulMessage": long_message,
        },
    )

    active = await client.active(type_query(label="preview", include_preview=True))
    assert len(active["agents"][0]["lastUsefulMessage"]) <= 160

    without_preview = await client.active(type_query(label="preview", include_preview=False))
    assert "lastUsefulMessage" not in without_preview["agents"][0]


@pytest.mark.asyncio
async def test_compact_status_has_no_preview_or_transcript_fields(tmp_path: Path) -> None:
    client = ReadyClient("ws://127.0.0.1:1", tmp_path / "state.json", "gpt-test")
    await client.remember_session(
        "thread-status",
        {
            "label": "status",
            "threadId": "thread-status",
            "cwd": "/tmp/status",
            "lastStatus": "running",
            "activeTurnId": "turn-status",
            "lastUsefulMessage": "verbose output should not appear",
        },
    )

    status = await client.compact_status(type_query(label="status"))
    item = status["agents"][0]
    assert item["name"] == "status"
    assert item["threadId"] == "thread-status"
    assert item["turnId"] == "turn-status"
    assert "lastUsefulMessage" not in item
    assert "preview" not in item
    assert "turn" not in item


@pytest.mark.asyncio
async def test_callback_requests_are_recorded_and_answered(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    server_request_id = "request-1"

    async def after_message(message: dict[str, Any], websocket: Any) -> None:
        if message.get("method") == "turn/start":
            await websocket.send(
                json.dumps(
                    {
                        "id": server_request_id,
                        "method": "plan/question",
                        "params": {"threadId": "thread-callback", "turnId": "turn-callback", "text": "Choose"},
                    }
                )
            )

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-callback", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-callback"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler, after_message)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "callback"})
        await client.start_turn({"threadId": "thread-callback", "label": "callback", "prompt": "work"})
        await asyncio.sleep(0.05)

        status = await client.status()
        assert status["pendingRequests"][0]["id"] == server_request_id
        assert status["pendingPermissionRequests"] == []
        assert status["activeTurns"][0]["status"] == "waiting"

        answered = await client.answer_request(server_request_id, {"decision": "accept"})
        await asyncio.sleep(0.05)
        assert answered["answered"] is True
        assert captured[-1] == {"id": server_request_id, "result": {"decision": "accept"}}
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_permission_callback_can_answer_approval_request(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    received: list[dict[str, Any]] = []
    server_request_id = "approval-1"

    async def after_message(message: dict[str, Any], websocket: Any) -> None:
        if message.get("method") == "turn/start":
            await websocket.send(
                json.dumps(
                    {
                        "id": server_request_id,
                        "method": "exec/requestApproval",
                        "params": {
                            "threadId": "thread-approval",
                            "turnId": "turn-approval",
                            "command": "make test",
                        },
                    }
                )
            )

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-approval", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-approval"}
        return {"ok": True}

    async def permission_callback(request: app_server_client.PendingServerRequest) -> dict[str, str]:
        received.append(request.to_json())
        return {"decision": "accept"}

    server = await start_fake_app_server(captured, handler, after_message)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    client.register_permission_callback(permission_callback)
    try:
        await client.start_thread({"label": "approval"})
        await client.start_turn({"threadId": "thread-approval", "label": "approval", "prompt": "work"})

        for _ in range(20):
            if any(message == {"id": server_request_id, "result": {"decision": "accept"}} for message in captured):
                break
            await asyncio.sleep(0.01)

        assert received[0]["id"] == server_request_id
        assert received[0]["method"] == "exec/requestApproval"
        assert {"id": server_request_id, "result": {"decision": "accept"}} in captured
        status = await client.status()
        assert status["pendingRequests"] == []
        assert status["pendingPermissionRequests"] == []
        assert status["activeTurns"][0]["status"] == "running"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_status_lists_pending_permission_requests_across_threads(tmp_path: Path) -> None:
    client = ReadyClient("ws://127.0.0.1:1", tmp_path / "state.json", "gpt-test")
    client.handle_server_request(
        "approval-1",
        "exec/requestApproval",
        {"threadId": "thread-one", "turnId": "turn-one", "command": "make test"},
    )
    client.handle_server_request(
        "question-1",
        "plan/question",
        {"threadId": "thread-two", "turnId": "turn-two", "text": "Choose"},
    )

    status = await client.status()

    assert [item["id"] for item in status["pendingRequests"]] == ["approval-1", "question-1"]
    assert [item["id"] for item in status["pendingPermissionRequests"]] == ["approval-1"]
    assert [item.id for item in client.pending_permission_requests()] == ["approval-1"]


@pytest.mark.asyncio
async def test_permission_callback_ignores_non_approval_requests(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    received: list[dict[str, Any]] = []
    server_request_id = "question-1"

    async def after_message(message: dict[str, Any], websocket: Any) -> None:
        if message.get("method") == "turn/start":
            await websocket.send(
                json.dumps(
                    {
                        "id": server_request_id,
                        "method": "plan/question",
                        "params": {"threadId": "thread-question", "turnId": "turn-question", "text": "Choose"},
                    }
                )
            )

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-question", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-question"}
        return {"ok": True}

    def permission_callback(request: app_server_client.PendingServerRequest) -> None:
        received.append(request.to_json())
        return None

    server = await start_fake_app_server(captured, handler, after_message)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test", permission_callback)
    try:
        await client.start_thread({"label": "question"})
        await client.start_turn({"threadId": "thread-question", "label": "question", "prompt": "work"})
        await asyncio.sleep(0.05)

        assert received == []
        status = await client.status()
        assert status["pendingRequests"][0]["id"] == server_request_id
        assert status["pendingPermissionRequests"] == []
        assert not any(message == {"id": server_request_id, "result": None} for message in captured)
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_routine_is_persisted_and_due_runner_starts_turn(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "turn/start":
            return {"turnId": "turn-routine"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        saved = await client.save_routine(
            {
                "name": "daily-check",
                "prompt": "Inspect project health.",
                "time": "00:00",
                "timezone": "UTC",
                "threadId": "thread-routine",
                "cwd": "/tmp/routine",
                "mode": "plan",
                "model": "gpt-test",
                "reasoningEffort": "low",
            }
        )
        assert saved["nativeSupport"] is False
        assert saved["routine"]["name"] == "daily-check"

        result = await client.run_due_routines()
        assert result["count"] == 1
        assert result["results"][0]["threadId"] == "thread-routine"
        assert result["results"][0]["turnId"] == "turn-routine"

        start_request = next(message for message in captured if message.get("method") == "turn/start")
        params = start_request["params"]
        assert params["threadId"] == "thread-routine"
        assert params["cwd"] == "/tmp/routine"
        assert "approvalPolicy" not in params
        assert "sandboxPolicy" not in params
        assert params["collaborationMode"]["mode"] == "plan"
        assert params["collaborationMode"]["settings"]["model"] == "gpt-test"
        assert params["collaborationMode"]["settings"]["reasoning_effort"] == "low"
        assert params["input"] == [{"type": "text", "text": "Inspect project health."}]

        second = await client.run_due_routines()
        assert second["count"] == 0

        stored = await client.read_routine("daily-check")
        assert stored["routine"]["lastRunDate"]
        assert stored["routine"]["lastThreadId"] == "thread-routine"
        assert stored["routine"]["lastTurnId"] == "turn-routine"
        assert stored["routine"]["lastStatus"] == "started"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_routine_reservation_prevents_duplicate_due_runs(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    first_client = ReadyClient("ws://127.0.0.1:1", state_file, "gpt-test")
    second_client = ReadyClient("ws://127.0.0.1:1", state_file, "gpt-test")
    try:
        await first_client.save_routine(
            {
                "name": "daily-check",
                "prompt": "Inspect project health.",
                "time": "00:00",
                "timezone": "UTC",
                "threadId": "thread-routine",
                "cwd": "/tmp/routine",
            }
        )

        reserved = await first_client.reserve_due_routines()
        assert [routine.name for routine in reserved] == ["daily-check"]

        second = await second_client.run_due_routines()
        assert second["count"] == 0

        stored = await second_client.read_routine("daily-check")
        assert stored["routine"]["lastRunDate"]
        assert stored["routine"]["lastStatus"] == "starting"
        assert state_file.with_suffix(".json.lock").exists()
    finally:
        await first_client.close()
        await second_client.close()


@pytest.mark.asyncio
async def test_client_does_not_spawn_managed_app_server_when_unavailable(tmp_path: Path) -> None:
    class NotReadyClient(CodexAppServerClient):
        async def check_ready(self) -> bool:
            return False

    client = NotReadyClient("ws://127.0.0.1:1", tmp_path / "state.json", "gpt-test")
    try:
        with pytest.raises(RuntimeError, match="Codex app-server is not running"):
            await client.ensure_connected()
        status = await client.status()
        assert status["managedProcess"] is False
    finally:
        await client.close()


def test_tool_surface_preserves_current_names_and_schemas() -> None:
    tools = build_tools(CodexAppServerClient("ws://127.0.0.1:1"))
    names = [tool.name for tool in tools]
    assert names == [
        "codex_app_server_status",
        "super_agents_start",
        "super_agents_resume",
        "super_agents_read",
        "super_agents_rename",
        "codex_answer_request",
        "super_agents_sessions",
        "super_agents_thread_favorite",
        "super_agents_active",
        "super_agents_status",
        "super_agents_resolve",
        "super_agents_progress",
        "super_agents_steer",
        "super_agents_cancel",
        "super_agents_start_turn",
        "super_agents_queue_turn",
        "super_agents_recent",
    ]
    by_name = {tool.name: tool for tool in tools}
    assert by_name["super_agents_start"].input_schema["required"] == ["name"]
    assert by_name["super_agents_start_turn"].input_schema["required"] == ["name", "prompt"]
    assert by_name["super_agents_queue_turn"].input_schema["required"] == ["prompt"]
    assert by_name["super_agents_thread_favorite"].input_schema["required"] == ["threadId"]
    assert "favorite" in by_name["super_agents_recent"].input_schema["properties"]
    assert "approvalPolicy" not in by_name["super_agents_start"].input_schema["properties"]
    assert "sandbox" not in by_name["super_agents_start"].input_schema["properties"]
    assert "approvalPolicy" not in by_name["super_agents_start_turn"].input_schema["properties"]
    assert "sandboxType" not in by_name["super_agents_start_turn"].input_schema["properties"]
    assert "approvalPolicy" not in by_name["super_agents_queue_turn"].input_schema["properties"]
    assert "sandboxType" not in by_name["super_agents_queue_turn"].input_schema["properties"]
    assert "per-thread filesystem queue" in by_name["super_agents_queue_turn"].description
    assert "no separate queued-next-turn API" in by_name["super_agents_start_turn"].description
    assert "threadId" not in by_name["super_agents_start_turn"].input_schema["properties"]
    assert "agentName" not in by_name["super_agents_start_turn"].input_schema["properties"]
    assert "reasoningEffort" not in by_name["super_agents_start_turn"].input_schema["properties"]
    assert "reasoningEffort" not in by_name["super_agents_queue_turn"].input_schema["properties"]
    assert "serviceTier" not in by_name["super_agents_start_turn"].input_schema["properties"]
    assert "serviceTier" not in by_name["super_agents_queue_turn"].input_schema["properties"]
    assert by_name["super_agents_read"].input_schema["properties"]["includeTurns"]["default"] is False
    assert "required" not in by_name["super_agents_progress"].input_schema
    assert "threadId" in by_name["super_agents_progress"].input_schema["properties"]
    assert by_name["super_agents_active"].input_schema["properties"]["previewLength"]["default"] == 160
    assert by_name["super_agents_active"].annotations == {"readOnlyHint": True, "idempotentHint": True}
    assert by_name["codex_answer_request"].input_schema["required"] == ["requestId", "result"]


class FakeServer:
    def __init__(self, server: Any, ws_url: str) -> None:
        self.server = server
        self.ws_url = ws_url

    async def close(self) -> None:
        self.server.close()
        await self.server.wait_closed()


async def start_fake_app_server(
    captured: list[dict[str, Any]],
    handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    after_message: Callable[[dict[str, Any], Any], Any] | None = None,
) -> FakeServer:
    def default_handler(_message: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    response_handler = handler or default_handler

    async def ws_handler(websocket: Any) -> None:
        async for raw in websocket:
            message = json.loads(raw)
            captured.append(message)
            if "id" not in message:
                continue
            response = response_handler(message)
            if "__error__" in response:
                await websocket.send(json.dumps({"id": message["id"], "error": response["__error__"]}))
            else:
                await websocket.send(json.dumps({"id": message["id"], "result": response}))
            if after_message:
                result = after_message(message, websocket)
                if hasattr(result, "__await__"):
                    await result

    server = await websockets.serve(ws_handler, "127.0.0.1", 0)
    socket = server.sockets[0]
    host, port = socket.getsockname()[:2]
    return FakeServer(server, f"ws://{host}:{port}")


def type_query(**kwargs: Any) -> Any:
    from super_agents.app_server_client import LabelQueryInput

    return LabelQueryInput(**kwargs)
