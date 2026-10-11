from __future__ import annotations

import json
import os
import sys
from unittest.mock import AsyncMock

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent


@pytest.mark.asyncio
async def test_cli_entrypoint_serves_tools_over_stdio(tmp_path) -> None:
    env = {
        **{
            key: value
            for key, value in os.environ.items()
            if not (key.startswith("SUPER_AGENTS_") or key == "CODEX_APP_SERVER_URL")
        },
        "OPENBASE_CODING_BACKEND": "codex",
        "SUPER_AGENTS_WS_URL": "ws://127.0.0.1:1",
        "SUPER_AGENTS_STATE_FILE": str(tmp_path / "state.json"),
        # The multi-backend entrypoint also opens Claude storage and the
        # ownership index; never read or modify the operator's real sessions.
        "SUPER_AGENTS_CLAUDE_CODE_HOME": str(tmp_path / "claude"),
        "SUPER_AGENTS_BACKEND_PROVENANCE_FILE": str(tmp_path / "provenance.json"),
    }
    params = StdioServerParameters(command=sys.executable, args=["-m", "super_agents"], env=env)

    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            tool_names = [tool.name for tool in tools.tools]
            assert "codex_app_server_status" in tool_names
            assert "super_agents_start_turn" in tool_names

            result = await session.call_tool("codex_app_server_status", {})
            assert result.isError is False
            assert isinstance(result.content[0], TextContent)
            payload = json.loads(result.content[0].text)
            assert payload["ready"] is False
            assert payload["websocketUrl"] == "ws://127.0.0.1:1"


@pytest.mark.asyncio
async def test_steer_tool_forwards_explicit_interrupt_without_changing_default():
    from super_agents.mcp_server import _tool_super_agents_steer

    class Client:
        async def steer_by_label(self, query, prompt, turn_input):
            return turn_input

    tool = _tool_super_agents_steer(Client())
    ordinary = await tool.handler({"name": "elm", "prompt": "new context"})
    replacing = await tool.handler({"name": "elm", "prompt": "stop the tool", "interruptCurrentWork": True})
    assert ordinary["interruptCurrentWork"] is False
    assert replacing["interruptCurrentWork"] is True


class _StartClient:
    backend = "claude_code"

    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    async def start_thread(self, input_data):
        self.calls.append(("start_thread", input_data))
        return {"backend": self.backend, "threadId": "s_new", "session": {"id": "s_new", "name": input_data["name"]}}

    async def start_turn_by_label(self, query, turn_input):
        self.calls.append(("start_turn_by_label", (query, turn_input)))
        return {"threadId": "s_new", "turnId": "t_1", "startedImmediately": True}


@pytest.mark.asyncio
async def test_start_with_prompt_starts_the_first_turn_on_the_new_thread(tmp_path, monkeypatch):
    # Regression (2026-10-08 staging voice demo): the dispatcher created a
    # thread with the task in developerInstructions and never started a turn;
    # the thread showed no messages and the agent never ran.
    from super_agents.mcp_server import _tool_super_agents_start

    monkeypatch.setenv("SUPER_AGENTS_STATE_FILE", str(tmp_path / "state.json"))
    client = _StartClient()
    tool = _tool_super_agents_start(client)
    assert "prompt" in tool.input_schema["properties"]
    assert "sandboxType" in tool.input_schema["properties"]
    assert "starts no work" in tool.description

    result = await tool.handler(
        {
            "name": "tic-tac-toe",
            "cwd": str(tmp_path),
            "agentName": "Renee",
            "developerInstructions": "Keep it in one HTML file.",
            "prompt": "Build a tic tac toe game.",
            "sandboxType": "dangerFullAccess",
        }
    )

    assert [name for name, _ in client.calls] == ["start_thread", "start_turn_by_label"]
    query, turn_input = client.calls[1][1]
    assert query.label == "tic-tac-toe"
    assert query.thread_id == "s_new"
    assert query.backend == "claude_code"
    assert turn_input["prompt"] == "Build a tic tac toe game."
    assert turn_input["name"] == "tic-tac-toe"
    assert turn_input["sandboxType"] == "dangerFullAccess"
    assert turn_input["developerInstructions"].endswith("Keep it in one HTML file.")
    assert "threadId" not in turn_input
    assert result["threadId"] == "s_new"
    assert result["turnStarted"] is True
    assert result["turn"]["turnId"] == "t_1"


@pytest.mark.asyncio
async def test_start_without_prompt_says_no_turn_started(tmp_path, monkeypatch):
    from super_agents.mcp_server import START_TURN_NEXT_STEP, _tool_super_agents_start

    monkeypatch.setenv("SUPER_AGENTS_STATE_FILE", str(tmp_path / "state.json"))
    client = _StartClient()
    result = await _tool_super_agents_start(client).handler(
        {"name": "tic-tac-toe", "developerInstructions": "Build a tic tac toe game."}
    )

    assert [name for name, _ in client.calls] == ["start_thread"]
    assert result["turnStarted"] is False
    assert result["nextStep"] == START_TURN_NEXT_STEP
    assert "super_agents_start_turn" in result["nextStep"]


@pytest.mark.asyncio
async def test_start_with_prompt_targets_new_codex_thread_when_name_listing_is_stale(tmp_path, monkeypatch):
    from super_agents.app_server_client import CodexAppServerClient
    from super_agents.mcp_server import _tool_super_agents_start

    client = CodexAppServerClient("ws://unused", tmp_path / "state.json", "gpt-test")
    monkeypatch.setattr(client, "ensure_connected", AsyncMock())
    monkeypatch.setattr(client, "_login_shell_config_override", AsyncMock(return_value={}))
    monkeypatch.setattr(client, "start_or_queue_turn", AsyncMock(return_value={"turnId": "turn-new"}))

    async def request(method, params=None, **kwargs):
        if method == "thread/start":
            return {"thread": {"id": "thread-new", "cwd": str(tmp_path)}}
        if method == "thread/list":
            return {"data": [{"id": "thread-old", "name": "same-name", "status": "completed"}]}
        return {}

    monkeypatch.setattr(client, "request", request)
    result = await _tool_super_agents_start(client).handler(
        {"name": "same-name", "cwd": str(tmp_path), "prompt": "Do the new task."}
    )

    target, turn_input = client.start_or_queue_turn.call_args.args
    assert target.session.thread_id == "thread-new"
    assert turn_input["prompt"] == "Do the new task."
    assert result["turnStarted"] is True


@pytest.mark.asyncio
async def test_first_turn_failure_identifies_created_thread_and_recovery(tmp_path, monkeypatch):
    from super_agents.mcp_server import _tool_super_agents_start

    client = _StartClient()
    monkeypatch.setattr(client, "start_turn_by_label", AsyncMock(side_effect=RuntimeError("backend unavailable")))
    with pytest.raises(RuntimeError) as raised:
        await _tool_super_agents_start(client).handler({"name": "new-task", "prompt": "Build it."})

    message = str(raised.value)
    assert "s_new" in message
    assert "super_agents_start_turn" in message
    assert "backend unavailable" in message
