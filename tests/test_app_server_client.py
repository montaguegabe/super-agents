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
from super_agents.app_models import TurnState
from super_agents.app_server_client import CodexAppServerClient
from super_agents.app_time import turn_key
from super_agents.mcp_server import (
    build_tools,
    clean_queue_turn_input,
    clean_start_turn_by_name_input,
    clean_thread_input,
    clean_turn_input,
    default_service_tier,
    default_super_agent_instructions_path,
    default_super_agents_model,
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


def test_default_model_ignores_model_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(tmp_path / "missing.json"))
    monkeypatch.delenv("SUPER_AGENTS_MODEL", raising=False)
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude-code")
    monkeypatch.setenv("CODEX_CLAUDE_MODEL", "claude-custom")
    monkeypatch.setenv("SUPER_AGENTS_MODEL", "gpt-custom")

    client = CodexAppServerClient(state_file=tmp_path / "state.json")

    assert client.default_model == app_server_client.DEFAULT_MODEL


@pytest.mark.asyncio
async def test_shared_tags_apply_to_threads_and_reports(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENBASE_CODER_CLI_DATA_DIR", str(tmp_path))
    client = CodexAppServerClient(state_file=tmp_path / "state.json")

    thread = await client.thread_tags("thread-1", ["Needs Review"])
    report = await client.report_tags("/tmp/project", "summary.md", ["needs review"])
    options = await client.tags()

    assert thread["tags"] == ["Needs Review"]
    assert report["tags"] == ["Needs Review"]
    assert options["tags"] == [
        {
            "slug": "needs-review",
            "label": "Needs Review",
            "created_at": thread["tagOptions"][0]["created_at"],
            "updated_at": report["tagOptions"][0]["updated_at"],
            "usageCount": 2,
        }
    ]


def test_tag_tools_are_registered(tmp_path: Path) -> None:
    client = CodexAppServerClient(state_file=tmp_path / "state.json")
    tool_names = {tool.name for tool in build_tools(client)}

    assert {
        "super_agents_tags",
        "super_agents_thread_tags",
        "super_agents_report_tags",
    }.issubset(tool_names)


def test_turn_input_preserves_explicit_approval_and_sandbox() -> None:
    cleaned = clean_turn_input(
        {
            "threadId": "thread-1",
            "prompt": 'Wait, then run /Users/gabemontague/.local/bin/openbase-coder user say Dottie "Done".',
            "approvalPolicy": "never",
            "sandboxType": "dangerFullAccess",
        }
    )

    assert cleaned["approvalPolicy"] == "never"
    assert cleaned["sandboxType"] == "dangerFullAccess"
    assert cleaned["reasoningEffort"] == "high"
    assert cleaned["serviceTier"] == "standard"


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


def test_thread_input_preserves_explicit_approval_and_sandbox() -> None:
    cleaned = clean_thread_input(
        {
            "name": "new-agent",
            "agentName": "Dottie",
            "approvalPolicy": "never",
            "sandbox": "danger-full-access",
        }
    )

    assert cleaned["approvalPolicy"] == "never"
    assert cleaned["sandbox"] == "danger-full-access"
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


def test_turn_input_uses_openbase_codex_service_tier_default(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(json.dumps({"codex_service_tier": "standard"}), encoding="utf-8")
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("CODEX_SERVICE_TIER", "fast")

    cleaned = clean_turn_input({"threadId": "thread-1", "prompt": "work"})

    assert cleaned["serviceTier"] == "standard"
    assert default_service_tier() == "standard"


def test_turn_input_ignores_legacy_super_agents_model_default(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(json.dumps({"super_agents_model": "opus"}), encoding="utf-8")
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("SUPER_AGENTS_MODEL", "sonnet")
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")

    cleaned = clean_turn_input({"threadId": "thread-1", "prompt": "work"})

    assert "model" not in cleaned
    assert default_super_agents_model() is None


def test_turn_input_uses_backend_specific_super_agents_model(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(
        json.dumps(
            {
                "backend_models": {
                    "codex": {"super_agents": "gpt-5.5"},
                    "claude_code": {"super_agents": "sonnet"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")

    cleaned = clean_turn_input({"threadId": "thread-1", "prompt": "work"})

    assert cleaned["model"] == "sonnet"
    assert default_super_agents_model() == "sonnet"


def test_turn_input_uses_claude_fable_model(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(
        json.dumps({"backend_models": {"claude_code": {"super_agents": "fable"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")

    cleaned = clean_turn_input({"threadId": "thread-1", "prompt": "work"})

    assert cleaned["model"] == "fable"
    assert default_super_agents_model() == "fable"


def test_turn_input_ignores_claude_model_default_for_codex(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(json.dumps({"super_agents_model": "opus"}), encoding="utf-8")
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")
    monkeypatch.delenv("SUPER_AGENTS_MODEL", raising=False)

    cleaned = clean_turn_input({"threadId": "thread-1", "prompt": "work"})

    assert "model" not in cleaned
    assert default_super_agents_model() is None


def test_turn_input_uses_backend_specific_codex_model(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(
        json.dumps({"backend_models": {"codex": {"super_agents": "gpt-5.5"}}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")

    cleaned = clean_turn_input({"threadId": "thread-1", "prompt": "work"})

    assert cleaned["model"] == "gpt-5.5"
    assert default_super_agents_model() == "gpt-5.5"


def test_openbase_cloud_uses_codex_model_defaults(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(
        json.dumps(
            {
                "backend_models": {
                    "codex": {"super_agents": "gpt-5.5"},
                    "openbase_cloud": {"super_agents": "openbase-codex"},
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "openbase_cloud")

    cleaned = clean_turn_input({"threadId": "thread-1", "prompt": "work"})

    assert cleaned["model"] == "gpt-5.5"
    assert default_super_agents_model() == "gpt-5.5"


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


def test_explicit_developer_instructions_extend_super_agent_default(monkeypatch, tmp_path: Path) -> None:
    instructions_path = tmp_path / "SUPER_AGENT_INSTRUCTIONS.md"
    instructions_path.write_text("Default instructions.\n", encoding="utf-8")
    monkeypatch.setenv("CODEX_SUPER_AGENT_INSTRUCTIONS_PATH", str(instructions_path))

    thread_input = clean_thread_input({"name": "new-agent", "developerInstructions": "Explicit instructions."})
    turn_input = clean_start_turn_by_name_input(
        {"name": "new-agent", "prompt": "work", "developerInstructions": "Explicit turn instructions."}
    )

    assert thread_input["developerInstructions"] == "Default instructions.\n\nExplicit instructions."
    assert turn_input["developerInstructions"] == "Default instructions.\n\nExplicit turn instructions."


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
async def test_python_client_uses_super_agents_model_and_reasoning_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[dict[str, Any]] = []
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(
        json.dumps(
            {
                "super_agents_reasoning_effort": "low",
                "backend_models": {"codex": {"super_agents": "gpt-default"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-defaults", "cwd": message["params"]["cwd"]}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-defaults"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    state_file = tmp_path / "state.json"
    client = ReadyClient(server.ws_url, state_file)
    try:
        await client.start_thread({"label": "defaults", "cwd": "/tmp/defaults"})
        result = await client.start_turn({"threadId": "thread-defaults", "label": "defaults", "prompt": "work"})

        start_request = next(message for message in captured if message.get("method") == "turn/start")
        settings = start_request["params"]["collaborationMode"]["settings"]
        assert settings["model"] == "gpt-default"
        assert settings["reasoning_effort"] == "low"
        assert result["reasoningEffort"] == "low"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_python_client_explicit_model_and_reasoning_override_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[dict[str, Any]] = []
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(
        json.dumps(
            {
                "super_agents_reasoning_effort": "low",
                "backend_models": {"codex": {"super_agents": "gpt-default"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-override", "cwd": message["params"]["cwd"]}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-override"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    state_file = tmp_path / "state.json"
    client = ReadyClient(server.ws_url, state_file)
    try:
        await client.start_thread({"label": "override", "cwd": "/tmp/override"})
        result = await client.start_turn(
            {
                "threadId": "thread-override",
                "label": "override",
                "prompt": "work",
                "model": "gpt-explicit",
                "reasoningEffort": "xhigh",
            }
        )

        start_request = next(message for message in captured if message.get("method") == "turn/start")
        settings = start_request["params"]["collaborationMode"]["settings"]
        assert settings["model"] == "gpt-explicit"
        assert settings["reasoning_effort"] == "xhigh"
        assert result["reasoningEffort"] == "xhigh"
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
async def test_active_list_does_not_report_idle_native_thread_as_running(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-idle", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-idle"}
        if message.get("method") == "thread/list":
            return {
                "threads": [
                    {
                        "id": "thread-idle",
                        "name": "idle-task",
                        "cwd": "/tmp/project",
                        "updatedAt": 1782343247,
                        "status": {"type": "idle"},
                    }
                ]
            }
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "idle-task", "cwd": "/tmp/project"})
        await client.start_turn({"threadId": "thread-idle", "label": "idle-task", "prompt": "work"})

        recent = await client.recent(type_query(label="idle-task", include_inactive=True))
        assert recent["agents"][0]["status"] == "completed"

        active = await client.active(type_query(label="idle-task"))
        assert active["count"] == 0
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

    config = await app_server_client.login_shell_config_override()

    set_env = config["shell_environment_policy"]["set"]
    assert set_env["PATH"] == "/usr/bin"
    assert "OPENBASE_SUPER_AGENT_THREAD_ID" not in set_env
    assert "OPENBASE_SUPER_AGENT_LABEL" not in set_env
    assert "OPENBASE_SUPER_AGENT_AGENT_NAME" not in set_env
    assert os.environ["OPENBASE_SUPER_AGENT_THREAD_ID"] == "parent-thread"
    assert os.environ["OPENBASE_SUPER_AGENT_AGENT_NAME"] == "Parent Name"


def test_super_agent_identity_instructions_replace_stale_identity_lines() -> None:
    assert (
        app_server_client.with_super_agent_identity_instructions(
            "Base instructions.\n\nSuper Agent thread name: Old\nSuper Agent thread id: old-thread\nYour name is Old.",
            "New Agent",
            "new-thread",
            "Dottie",
        )
        == "Base instructions.\n\nSuper Agent thread name: New Agent\nSuper Agent thread id: new-thread\nYour name is Dottie."
    )


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
            {
                "label": "native-name",
                "prompt": "continue",
                "approvalPolicy": "never",
                "sandboxType": "dangerFullAccess",
                "_mcpCallId": "mcp-test",
            },
        )

        assert result["threadId"] == "thread-native"
        start_request = next(message for message in captured if message.get("method") == "turn/start")
        assert start_request["params"]["threadId"] == "thread-native"
        assert start_request["params"]["cwd"] == "/tmp/native"
        assert start_request["params"]["serviceTier"] == "standard"
        assert start_request["params"]["approvalPolicy"] == "never"
        assert start_request["params"]["sandboxPolicy"] == {"type": "dangerFullAccess"}
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
async def test_steer_by_label_starts_when_no_active_turn_exists(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-steer-idle", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-steer-idle"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "steer-idle", "cwd": "/tmp/project"})

        result = await client.steer_by_label(type_query(label="steer-idle"), "start from steer")

        assert result["queued"] is False
        assert result["startedImmediately"] is True
        assert result["turnId"] == "turn-steer-idle"
        assert not any(message.get("method") == "turn/steer" for message in captured)
        start_request = next(message for message in captured if message.get("method") == "turn/start")
        assert start_request["params"]["input"] == [{"type": "text", "text": "start from steer"}]
        assert start_request["params"]["threadId"] == "thread-steer-idle"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_start_turn_by_name_on_running_thread_steers_prompt(tmp_path: Path) -> None:
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

        result = await client.start_turn_by_label(type_query(label="active"), {"label": "active", "prompt": "adjust"})

        start_requests = [message for message in captured if message.get("method") == "turn/start"]
        steer_request = next(message for message in captured if message.get("method") == "turn/steer")
        assert result["queued"] is False
        assert result["steered"] is True
        assert result["drain"] == "steered_active_turn"
        assert len(start_requests) == 1
        assert steer_request["params"] == {
            "threadId": "thread-active",
            "expectedTurnId": "turn-active",
            "input": [{"type": "text", "text": "adjust"}],
        }
        assert client.queued_turn_summary() == []
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_start_turn_by_name_ignores_stale_runtime_last_turn(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        captured.append(message)
        if message.get("method") == "thread/list":
            return {"data": []}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-new"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.remember_session(
            "thread-stale",
            {
                "label": "stale",
                "threadId": "thread-stale",
                "cwd": "/tmp/stale",
                "lastTurnId": "turn-stale",
                "lastStatus": "unknown",
            },
        )
        client._turns[turn_key("thread-stale", "turn-stale")] = TurnState(
            thread_id="thread-stale",
            turn_id="turn-stale",
            status="running",
            started_at="2026-06-18T16:00:00.000Z",
        )

        result = await client.start_turn_by_label(
            type_query(label="stale"),
            {"label": "stale", "prompt": "new work"},
        )

        assert result["queued"] is False
        assert result["startedImmediately"] is True
        assert result["turnId"] == "turn-new"
        assert not any(message.get("method") == "turn/steer" for message in captured)
        start_request = next(message for message in captured if message.get("method") == "turn/start")
        assert start_request["params"]["threadId"] == "thread-stale"
        assert start_request["params"]["input"] == [{"type": "text", "text": "new work"}]
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_start_turn_by_name_ignores_queue_item_as_active_turn(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        captured.append(message)
        if message.get("method") == "thread/list":
            return {"data": []}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-real"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.remember_session(
            "thread-queue-id",
            {
                "label": "queue-id",
                "threadId": "thread-queue-id",
                "cwd": "/tmp/project",
                "lastTurnId": "q_stale-queue-item",
                "lastStatus": "running",
            },
        )

        result = await client.start_turn_by_label(
            type_query(label="queue-id"),
            {"label": "queue-id", "prompt": "follow up"},
        )

        assert result["queued"] is False
        assert result["startedImmediately"] is True
        assert result["turnId"] == "turn-real"
        assert not client.queued_turn_summary()
        assert not any(message.get("method") == "turn/steer" for message in captured)
        start_request = next(message for message in captured if message.get("method") == "turn/start")
        assert start_request["params"]["threadId"] == "thread-queue-id"
        assert start_request["params"]["input"] == [{"type": "text", "text": "follow up"}]
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_start_turn_by_name_warns_and_starts_after_orphaned_active_turn(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        captured.append(message)
        if message.get("method") == "thread/list":
            return {"data": []}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-new"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.remember_session(
            "thread-orphan",
            {
                "label": "orphan",
                "threadId": "thread-orphan",
                "cwd": "/tmp/project",
                "activeTurnId": "turn-orphan",
                "lastTurnId": "turn-orphan",
                "lastStatus": "running",
                "turns": {
                    "turn-orphan": {
                        "turnId": "turn-orphan",
                        "status": "running",
                        "startedAt": "2026-06-30T22:13:13.000Z",
                        "updatedAt": "2026-06-30T22:13:13.000Z",
                    },
                },
            },
        )

        session = await client.get_session("thread-orphan")
        assert session is not None
        assert client.session_status(session) == "unknown"
        assert client.session_view(session)["statusWarning"] == "stale_active_turn"

        result = await client.start_turn_by_label(
            type_query(label="orphan"),
            {"label": "orphan", "prompt": "start fresh"},
        )

        assert result["queued"] is False
        assert result["startedImmediately"] is True
        assert result["turnId"] == "turn-new"
        assert not any(message.get("method") == "turn/steer" for message in captured)
        start_request = next(message for message in captured if message.get("method") == "turn/start")
        assert start_request["params"]["threadId"] == "thread-orphan"
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
    client.handle_server_request(
        "elicitation-1",
        "mcpServer/elicitation/request",
        {"threadId": "thread-three", "turnId": "turn-three", "serverName": "super_agents"},
    )

    status = await client.status()

    assert [item["id"] for item in status["pendingRequests"]] == [
        "approval-1",
        "question-1",
        "elicitation-1",
    ]
    assert [item["id"] for item in status["pendingPermissionRequests"]] == [
        "approval-1",
        "elicitation-1",
    ]
    assert [item.id for item in client.pending_permission_requests()] == [
        "approval-1",
        "elicitation-1",
    ]


@pytest.mark.asyncio
async def test_permission_callback_can_answer_mcp_elicitation_request(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    received: list[dict[str, Any]] = []
    server_request_id = "elicitation-1"

    async def after_message(message: dict[str, Any], websocket: Any) -> None:
        if message.get("method") == "turn/start":
            await websocket.send(
                json.dumps(
                    {
                        "id": server_request_id,
                        "method": "mcpServer/elicitation/request",
                        "params": {
                            "threadId": "thread-elicitation",
                            "turnId": "turn-elicitation",
                            "serverName": "super_agents",
                        },
                    }
                )
            )

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-elicitation", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-elicitation"}
        return {"ok": True}

    async def permission_callback(request: app_server_client.PendingServerRequest) -> dict[str, Any]:
        received.append(request.to_json())
        return {"action": "accept", "content": None, "_meta": None}

    server = await start_fake_app_server(captured, handler, after_message)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    client.register_permission_callback(permission_callback)
    try:
        await client.start_thread({"label": "elicitation"})
        await client.start_turn({"threadId": "thread-elicitation", "label": "elicitation", "prompt": "work"})

        expected = {"id": server_request_id, "result": {"action": "accept", "content": None, "_meta": None}}
        for _ in range(20):
            if expected in captured:
                break
            await asyncio.sleep(0.01)

        assert received[0]["id"] == server_request_id
        assert received[0]["method"] == "mcpServer/elicitation/request"
        assert expected in captured
        status = await client.status()
        assert status["pendingRequests"] == []
        assert status["pendingPermissionRequests"] == []
        assert status["activeTurns"][0]["status"] == "running"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_permission_callback_normalizes_elicitation_decision(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    server_request_id = "elicitation-1"

    async def after_message(message: dict[str, Any], websocket: Any) -> None:
        if message.get("method") == "turn/start":
            await websocket.send(
                json.dumps(
                    {
                        "id": server_request_id,
                        "method": "mcpServer/elicitation/request",
                        "params": {
                            "threadId": "thread-elicitation",
                            "turnId": "turn-elicitation",
                            "serverName": "super_agents",
                        },
                    }
                )
            )

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {
                "threadId": "thread-elicitation",
                "cwd": message["params"]["cwd"],
                "model": "gpt-test",
            }
        if message.get("method") == "turn/start":
            return {"turnId": "turn-elicitation"}
        return {"ok": True}

    async def permission_callback(
        _request: app_server_client.PendingServerRequest,
    ) -> dict[str, Any]:
        return {"decision": "accept"}

    server = await start_fake_app_server(captured, handler, after_message)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    client.register_permission_callback(permission_callback)
    try:
        await client.start_thread({"label": "elicitation"})
        await client.start_turn(
            {
                "threadId": "thread-elicitation",
                "label": "elicitation",
                "prompt": "work",
            }
        )

        expected = {
            "id": server_request_id,
            "result": {"action": "accept", "content": None, "_meta": None},
        }
        for _ in range(20):
            if expected in captured:
                break
            await asyncio.sleep(0.01)

        assert expected in captured
    finally:
        await client.close()
        await server.close()


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
                "approvalPolicy": "never",
                "sandboxType": "dangerFullAccess",
                "mode": "plan",
                "model": "gpt-test",
                "reasoningEffort": "low",
            }
        )
        assert saved["nativeSupport"] is False
        assert saved["routine"]["name"] == "daily-check"
        assert saved["routine"]["kind"] == "agent"

        result = await client.run_due_routines()
        assert result["count"] == 1
        assert result["results"][0]["threadId"] == "thread-routine"
        assert result["results"][0]["turnId"] == "turn-routine"

        start_request = next(message for message in captured if message.get("method") == "turn/start")
        params = start_request["params"]
        assert params["threadId"] == "thread-routine"
        assert params["cwd"] == "/tmp/routine"
        assert params["approvalPolicy"] == "never"
        assert params["sandboxPolicy"] == {"type": "dangerFullAccess"}
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
        assert not any(message.get("method") == "thread/start" for message in captured)
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_command_routine_runs_local_command_without_app_server(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        captured.append(message)
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.save_routine(
            {
                "name": "discover-prs",
                "kind": "command",
                "command": "printf '{\"eligible\":[]}'",
                "time": "00:00",
                "timezone": "UTC",
                "cwd": str(tmp_path),
            }
        )

        result = await client.run_due_routines(name="discover-prs", force=True)
        assert result["count"] == 1
        assert result["results"][0]["kind"] == "command"
        assert result["results"][0]["exitCode"] == 0
        assert result["results"][0]["stdout"] == '{"eligible":[]}'
        assert not any(message.get("method") in {"thread/start", "turn/start"} for message in captured)

        stored = await client.read_routine("discover-prs")
        assert stored["routine"]["lastStatus"] == "completed"
        assert "lastThreadId" not in stored["routine"]
        assert "lastTurnId" not in stored["routine"]
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_command_routine_records_nonzero_exit_as_failed(tmp_path: Path) -> None:
    client = ReadyClient("ws://127.0.0.1:1", tmp_path / "state.json", "gpt-test")
    try:
        await client.save_routine(
            {
                "name": "failing-command",
                "kind": "command",
                "command": "exit 7",
                "time": "00:00",
                "timezone": "UTC",
            }
        )

        result = await client.run_due_routines(name="failing-command", force=True)
        assert result["results"][0]["exitCode"] == 7
        stored = await client.read_routine("failing-command")
        assert stored["routine"]["lastStatus"] == "failed"
        assert stored["routine"]["lastError"] == "Command routine exited with code 7."
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_routine_marks_immediately_interrupted_turn_failed(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "turn/start":
            return {"turnId": "turn-interrupted"}
        return {"ok": True}

    async def after_message(message: dict[str, Any], websocket: Any) -> None:
        if message.get("method") == "turn/start":
            await websocket.send(
                json.dumps(
                    {
                        "method": "turn/interrupted",
                        "params": {
                            "threadId": "thread-routine",
                            "turnId": "turn-interrupted",
                            "status": "interrupted",
                        },
                    }
                )
            )

    server = await start_fake_app_server(captured, handler, after_message)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.save_routine(
            {
                "name": "daily-check",
                "prompt": "Inspect project health.",
                "time": "00:00",
                "timezone": "UTC",
                "threadId": "thread-routine",
            }
        )

        result = await client.run_due_routines()
        assert result["count"] == 1
        assert result["results"][0]["turnId"] == "turn-interrupted"

        stored = await client.read_routine("daily-check")
        assert stored["routine"]["lastThreadId"] == "thread-routine"
        assert stored["routine"]["lastTurnId"] == "turn-interrupted"
        assert stored["routine"]["lastStatus"] == "failed"
        assert stored["routine"]["lastError"] == "Routine turn became cancelled immediately after launch."

        await asyncio.sleep(0.01)
        session = await client.get_session("thread-routine")
        assert session is not None
        assert session.last_status == "cancelled"
        assert session.active_turn_id is None
        assert session.turns is not None
        assert session.turns["turn-interrupted"].status == "cancelled"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_fresh_thread_per_run_routine_starts_new_named_thread(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-fresh"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-fresh"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.remember_session(
            "thread-existing-target",
            {
                "label": "ignored-target",
                "agentName": "Dottie",
                "threadId": "thread-existing-target",
                "cwd": "/tmp/routine",
                "lastStatus": "completed",
            },
        )
        await client.save_routine(
            {
                "name": "daily-check",
                "prompt": "Inspect project health.",
                "time": "00:00",
                "timezone": "UTC",
                "threadId": "ignored-thread",
                "targetName": "ignored-target",
                "freshThreadPerRun": True,
                "cwd": "/tmp/routine",
                "approvalPolicy": "never",
                "sandboxType": "dangerFullAccess",
                "mode": "plan",
                "model": "gpt-test",
                "reasoningEffort": "low",
                "developerInstructions": "routine instructions",
            }
        )

        result = await client.run_due_routines()
        assert result["count"] == 1
        assert result["results"][0]["threadId"] == "thread-fresh"
        assert result["results"][0]["turnId"] == "turn-fresh"

        thread_start = next(message for message in captured if message.get("method") == "thread/start")
        thread_params = thread_start["params"]
        assert thread_params["cwd"] == "/tmp/routine"
        assert thread_params["approvalPolicy"] == "never"
        assert thread_params["sandboxPolicy"] == {"type": "dangerFullAccess"}
        assert "Super Agent thread name: daily-check-" in thread_params["developerInstructions"]
        assert "Your name is Dottie." in thread_params["developerInstructions"]
        assert "routine instructions" in thread_params["developerInstructions"]

        turn_start = next(message for message in captured if message.get("method") == "turn/start")
        turn_params = turn_start["params"]
        assert turn_params["threadId"] == "thread-fresh"
        assert turn_params["cwd"] == "/tmp/routine"
        assert turn_params["approvalPolicy"] == "never"
        assert turn_params["sandboxPolicy"] == {"type": "dangerFullAccess"}
        assert turn_params["collaborationMode"]["mode"] == "plan"
        assert turn_params["collaborationMode"]["settings"]["model"] == "gpt-test"
        assert turn_params["collaborationMode"]["settings"]["reasoning_effort"] == "low"
        turn_instructions = turn_params["collaborationMode"]["settings"]["developer_instructions"]
        assert "Super Agent thread name: daily-check-" in turn_instructions
        assert turn_params["input"] == [{"type": "text", "text": "Inspect project health."}]
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
async def test_read_routine_reconciles_terminal_turn_status(tmp_path: Path) -> None:
    client = ReadyClient("ws://127.0.0.1:1", tmp_path / "state.json", "gpt-test")
    try:
        await client.save_routine(
            {
                "name": "daily-check",
                "prompt": "Inspect project health.",
                "time": "00:00",
                "timezone": "UTC",
                "threadId": "thread-routine",
                "lastThreadId": "thread-routine",
                "lastTurnId": "turn-routine",
                "lastStatus": "starting",
            }
        )
        await client.remember_session(
            "thread-routine",
            {
                "threadId": "thread-routine",
                "lastTurnId": "turn-routine",
                "lastStatus": "cancelled",
                "turns": {
                    "turn-routine": {
                        "turnId": "turn-routine",
                        "status": "cancelled",
                        "startedAt": "2026-07-02T13:16:49.142Z",
                        "updatedAt": "2026-07-02T13:17:00.000Z",
                    }
                },
            },
        )

        stored = await client.read_routine("daily-check")
        assert stored["routine"]["lastStatus"] == "failed"
        assert stored["routine"]["lastError"] == "Routine turn became cancelled."
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_interval_routine_runs_after_interval_elapsed(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "turn/start":
            return {"turnId": f"turn-{len(captured)}"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.save_routine(
            {
                "name": "priority-poller",
                "prompt": "Poll Notion prioritized tasks.",
                "scheduleType": "interval",
                "intervalSeconds": 60,
                "threadId": "thread-routine",
            }
        )

        first = await client.run_due_routines()
        assert first["count"] == 1
        stored = await client.read_routine("priority-poller")
        assert stored["routine"]["lastRunAt"]
        assert stored["routine"]["nextRunAt"]

        second = await client.run_due_routines()
        assert second["count"] == 0

        await client.save_routine(
            {
                "name": "priority-poller",
                "lastRunAt": "2000-01-01T00:00:00.000Z",
                "lastStatus": "completed",
            }
        )
        third = await client.run_due_routines()
        assert third["count"] == 1
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_interval_routine_does_not_overlap_active_run(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "turn/start":
            return {"turnId": f"turn-{len(captured)}"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.save_routine(
            {
                "name": "open-pr-review-routine",
                "prompt": "Review open PRs.",
                "scheduleType": "interval",
                "intervalSeconds": 60,
                "threadId": "thread-routine",
                "lastRunAt": "2000-01-01T00:00:00.000Z",
                "lastStatus": "started",
            }
        )

        skipped = await client.run_due_routines()
        assert skipped["count"] == 0
        assert not any(message.get("method") == "turn/start" for message in captured)

        await client.save_routine({"name": "open-pr-review-routine", "lastStatus": "completed"})
        started = await client.run_due_routines()
        assert started["count"] == 1
    finally:
        await client.close()
        await server.close()


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
        "super_agents_tags",
        "super_agents_thread_tags",
        "super_agents_report_tags",
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
    assert by_name["super_agents_thread_tags"].input_schema["required"] == ["threadId"]
    assert by_name["super_agents_report_tags"].input_schema["required"] == [
        "projectPath",
        "path",
    ]
    assert "favorite" in by_name["super_agents_recent"].input_schema["properties"]
    assert "approvalPolicy" in by_name["super_agents_start"].input_schema["properties"]
    assert "sandbox" in by_name["super_agents_start"].input_schema["properties"]
    assert "approvalPolicy" in by_name["super_agents_start_turn"].input_schema["properties"]
    assert "sandboxType" in by_name["super_agents_start_turn"].input_schema["properties"]
    assert "approvalPolicy" in by_name["super_agents_queue_turn"].input_schema["properties"]
    assert "sandboxType" in by_name["super_agents_queue_turn"].input_schema["properties"]
    assert "per-thread filesystem queue" in by_name["super_agents_queue_turn"].description
    assert "steers the active turn" in by_name["super_agents_start_turn"].description
    assert "super_agents_queue_turn" in by_name["super_agents_start_turn"].description
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


@pytest.mark.asyncio
async def test_recent_finds_exact_name_missed_by_search_term(tmp_path: Path) -> None:
    """searchTerm is fuzzy content search; exact-name lookups page the full list."""
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") != "thread/list":
            return {"ok": True}
        params = message.get("params") or {}
        if params.get("searchTerm"):
            # The app server's content search misses the exact-named thread.
            return {"data": [{"id": "thread-noise", "name": "other-thread", "cwd": "/tmp/x", "updatedAt": 300}]}
        if not params.get("cursor"):
            return {
                "data": [
                    {"id": "thread-a", "name": "alpha", "cwd": "/tmp/x", "updatedAt": 900},
                    {"id": "thread-b", "name": "beta", "cwd": "/tmp/x", "updatedAt": 800},
                ],
                "nextCursor": "page-2",
            }
        return {"data": [{"id": "thread-target", "name": "spirit-tech", "cwd": "/tmp/x", "updatedAt": 700}]}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        found = await client.scan_threads_for_name("spirit-tech")
        assert found["id"] == "thread-target"
        cursors = [
            (message.get("params") or {}).get("cursor")
            for message in captured
            if message.get("method") == "thread/list"
        ]
        assert "page-2" in cursors

        recent = await client.recent(type_query(label="spirit-tech", include_inactive=True))
        assert recent["count"] == 1
        assert recent["agents"][0]["threadId"] == "thread-target"

        resolved = await client.resolve_thread_name("spirit-tech", type_query(label="spirit-tech"))
        assert resolved["id"] == "thread-target"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_name_scan_gives_up_after_thread_cap(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    page_index = 0

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        nonlocal page_index
        if message.get("method") != "thread/list":
            return {"ok": True}
        page_index += 1
        return {
            "data": [
                {"id": f"thread-{page_index}-{item}", "name": f"noise-{page_index}-{item}", "cwd": "/tmp/x"}
                for item in range(50)
            ],
            "nextCursor": f"page-{page_index + 1}",
        }

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        found = await client.scan_threads_for_name("never-present")
        assert found == {}
        list_calls = [message for message in captured if message.get("method") == "thread/list"]
        assert len(list_calls) == 10
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_recent_ranks_across_creation_ordered_pages(tmp_path: Path) -> None:
    """A thread created long ago but touched today must top the recent view."""
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") != "thread/list":
            return {"ok": True}
        params = message.get("params") or {}
        if not params.get("cursor"):
            return {
                "data": [{"id": "thread-new", "name": "created-recently", "cwd": "/tmp/x", "updatedAt": 1_778_200_100}],
                "nextCursor": "page-2",
            }
        return {"data": [{"id": "thread-old", "name": "old-but-active", "cwd": "/tmp/x", "updatedAt": 1_778_300_000}]}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        recent = await client.recent(type_query(include_inactive=True))
        assert [agent["threadId"] for agent in recent["agents"]] == [
            "thread-old",
            "thread-new",
        ]
        list_params = [message.get("params") or {} for message in captured if message.get("method") == "thread/list"]
        # Provider-agnostic by default: the empty list disables the app
        # server's active-provider filter.
        assert all(params.get("modelProviders") == [] for params in list_params)
    finally:
        await client.close()
        await server.close()
