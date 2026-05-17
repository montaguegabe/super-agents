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
        **os.environ,
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
