"""Claude Agent SDK option building for the Claude Code backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

CLAUDE_PERMISSION_MODE = "bypassPermissions"
CLAUDE_PERMISSION_MODE_ENV = "SUPER_AGENTS_CLAUDE_PERMISSION_MODE"
CLAUDE_CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"
CLAUDE_CONFIG_FILENAME = ".claude.json"
CLAUDE_SETTINGS_FILENAME = "settings.json"
CLAUDE_INSTRUCTIONS_FILENAME = "CLAUDE.md"
CLAUDE_SERVICE_TIER_EFFORTS = {
    "fast": "low",
    "standard": "high",
    "slow": "high",
}
# The Claude Code backend authenticates through the Claude Code CLI login, not
# Anthropic API keys. The SDK merges these values over the inherited process
# environment, so an empty override keeps ANTHROPIC_API_KEY away from the
# spawned CLI without mutating this process's environment.
CLAUDE_SDK_ENV_OVERRIDES = {"ANTHROPIC_API_KEY": ""}


def resolve_permission_mode() -> str:
    """Permission mode for Claude sessions, overridable via the environment.

    Gated modes (anything other than ``bypassPermissions``) route the SDK's
    permission prompts through the shared open-approvals queue.
    """
    return os.environ.get(CLAUDE_PERMISSION_MODE_ENV, "").strip() or CLAUDE_PERMISSION_MODE


def agent_options(
    sdk: Any,
    cwd: str,
    model: str | None,
    reasoning_effort: str | None,
    *,
    resume: str | None,
    can_use_tool: Any | None = None,
) -> Any:
    managed_options = managed_claude_config_options()
    permission_mode = resolve_permission_mode()
    kwargs: JsonObject = {
        "cwd": cwd,
        "permission_mode": permission_mode,
        **managed_options,
        "env": {**managed_options.get("env", {}), **CLAUDE_SDK_ENV_OVERRIDES},
    }
    if can_use_tool is not None and permission_mode != "bypassPermissions":
        kwargs["can_use_tool"] = can_use_tool
    if model:
        kwargs["model"] = model
    if reasoning_effort:
        kwargs["effort"] = reasoning_effort
    if resume:
        kwargs["resume"] = resume
    return sdk.ClaudeAgentOptions(**kwargs)


def claude_effort(reasoning_effort: str | None, service_tier: str | None) -> str | None:
    if reasoning_effort and reasoning_effort != "high":
        return reasoning_effort
    tier_effort = CLAUDE_SERVICE_TIER_EFFORTS.get((service_tier or "").strip().lower())
    return tier_effort or reasoning_effort


def managed_claude_config_options() -> JsonObject:
    config_dir_value = os.environ.get(CLAUDE_CONFIG_DIR_ENV)
    if not config_dir_value:
        return {}

    config_dir = Path(config_dir_value).expanduser()
    options: JsonObject = {
        "env": {CLAUDE_CONFIG_DIR_ENV: str(config_dir)},
        # "user" scope resolves against CLAUDE_CONFIG_DIR (exported in "env"
        # above), not the host's ~/.claude, so including it loads the managed
        # config dir's skills/ and agents/ without leaking the host's personal
        # settings into the session. Omitting "user" silently hides everything
        # installed under CLAUDE_CONFIG_DIR that is user-scoped.
        "setting_sources": ["user", "project"],
    }

    settings_path = config_dir / CLAUDE_SETTINGS_FILENAME
    if settings_path.exists():
        options["settings"] = str(settings_path)

    instructions_path = config_dir / CLAUDE_INSTRUCTIONS_FILENAME
    if instructions_path.exists():
        options["system_prompt"] = {
            "type": "file",
            "path": str(instructions_path),
        }

    mcp_servers = _managed_claude_mcp_servers(config_dir)
    if mcp_servers:
        options["mcp_servers"] = mcp_servers

    return options


def _managed_claude_mcp_servers(config_dir: Path) -> JsonObject | None:
    config_path = config_dir / CLAUDE_CONFIG_FILENAME
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    mcp_servers = payload.get("mcpServers")
    return mcp_servers if isinstance(mcp_servers, dict) and mcp_servers else None
