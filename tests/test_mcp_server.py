from __future__ import annotations

import json
import os
import sys

import pytest
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import TextContent


@pytest.mark.asyncio
async def test_cli_entrypoint_serves_tools_over_stdio(tmp_path) -> None:
    env = {
        **{key: value for key, value in os.environ.items()
            if not (key.startswith("SUPER_AGENTS_") or key == "CODEX_APP_SERVER_URL")},
        "OPENBASE_CODING_BACKEND": "codex",
        "SUPER_AGENTS_WS_URL": "ws://127.0.0.1:1",
        "SUPER_AGENTS_STATE_FILE": str(tmp_path / "state.json"),
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
