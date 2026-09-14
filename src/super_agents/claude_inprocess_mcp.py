"""Keep recursive Super Agents delegation inside the owning SDK process."""

from __future__ import annotations

import re
from typing import Any


def replace_super_agents_stdio_server(options: Any, *, client: Any) -> None:
    """Replace configured Super Agents stdio entries with SDK servers.

    Claude Code owns configured stdio MCP subprocesses. When an agent starts a
    background turn and then finishes its own response, Claude shuts down that
    subprocess and interrupts the delegated turn. Wiring the same generic MCP
    server in-process keeps nested turns owned by the caller's long-lived
    ``ClaudeAgentSdkClient`` instead.
    """
    servers = _mcp_servers(options)
    if not isinstance(servers, dict):
        return

    matching_names = [name for name in servers if _is_super_agents_name(name)]
    if not matching_names:
        return

    from super_agents.mcp_server import create_server

    updated = dict(servers)
    for name in matching_names:
        updated[name] = {
            "type": "sdk",
            "name": name,
            "instance": create_server(client),
        }
    _set_mcp_servers(options, updated)


def _is_super_agents_name(name: object) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", "-", str(name).strip().lower()).strip("-")
    return normalized in {"super-agents", "mcp-super-agents"}


def _mcp_servers(options: Any) -> Any:
    if hasattr(options, "mcp_servers"):
        return options.mcp_servers
    kwargs = getattr(options, "kwargs", None)
    return kwargs.get("mcp_servers") if isinstance(kwargs, dict) else None


def _set_mcp_servers(options: Any, servers: dict[str, Any]) -> None:
    if hasattr(options, "mcp_servers"):
        options.mcp_servers = servers
        return
    kwargs = getattr(options, "kwargs", None)
    if isinstance(kwargs, dict):
        kwargs["mcp_servers"] = servers
