"""Claude Agent SDK option building for the Claude Code backend."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from .backend_config import OPENBASE_CLOUD_BACKEND, configured_backend_from_environment, normalize_backend

JsonObject = dict[str, Any]

CLAUDE_PERMISSION_MODE = "bypassPermissions"
CLAUDE_PERMISSION_MODE_ENV = "SUPER_AGENTS_CLAUDE_PERMISSION_MODE"
CLAUDE_CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"
# JSON object of extra Claude Code CLI flags, e.g. {"chrome": null} for
# --chrome. Values map to flag arguments; null means a bare flag.
CLAUDE_EXTRA_ARGS_ENV = "SUPER_AGENTS_CLAUDE_EXTRA_ARGS"
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
# The Claude Code SDK expands family aliases to whatever ids are current
# (including dated snapshots); the Cloud proxy allowlist is keyed by these
# public ids, so pin aliases before they reach the SDK on the Cloud backend.
OPENBASE_CLOUD_CLAUDE_MODEL_MAP = {
    "fable": "claude-fable-5",
    "opus": "claude-opus-4-8",
    "sonnet": "claude-sonnet-5",
    "haiku": "claude-haiku-4-5",
}
# With no model configured, the Claude Code CLI falls back to its own family
# default (Opus/Fable), which the Cloud proxy rejects for trial accounts —
# pin the proxy's trial-safe default instead of leaving the choice to the SDK.
OPENBASE_CLOUD_DEFAULT_CLAUDE_MODEL = "claude-sonnet-5"
OPENBASE_CLOUD_DEFAULT_BASE_URL = "https://app.openbase.cloud"
OPENBASE_CLOUD_ANTHROPIC_PATH = "/api/openbase/llm/anthropic"
OPENBASE_CLOUD_ANTHROPIC_BASE_URL_ENV = "OPENBASE_CLOUD_ANTHROPIC_BASE_URL"
OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN_ENV = "OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN"
OPENBASE_CODER_CLI_WEB_BACKEND_URL_ENV = "OPENBASE_CODER_CLI_WEB_BACKEND_URL"


def resolve_permission_mode() -> str:
    """Resolve the process-wide Claude permission posture."""
    return os.environ.get(CLAUDE_PERMISSION_MODE_ENV, "").strip() or CLAUDE_PERMISSION_MODE


def agent_options(
    sdk: Any,
    cwd: str,
    model: str | None,
    reasoning_effort: str | None,
    *,
    resume: str | None,
    can_use_tool: Any | None = None,
    backend: str | None = None,
) -> Any:
    managed_options = managed_claude_config_options()
    permission_mode = resolve_permission_mode()
    kwargs: JsonObject = {
        "cwd": cwd,
        "permission_mode": permission_mode,
        **managed_options,
        "env": {
            **managed_options.get("env", {}),
            **CLAUDE_SDK_ENV_OVERRIDES,
            **openbase_cloud_claude_env(backend),
        },
    }
    if permission_mode != "bypassPermissions":
        if can_use_tool is None:
            raise RuntimeError("Claude permission gating was requested but no approval handler is available.")
        kwargs["can_use_tool"] = can_use_tool
    if resolved_model := openbase_cloud_claude_model(model, backend):
        kwargs["model"] = resolved_model
    if reasoning_effort:
        kwargs["effort"] = reasoning_effort
    if resume:
        kwargs["resume"] = resume
    if extra_args := claude_extra_args():
        kwargs["extra_args"] = extra_args
    return sdk.ClaudeAgentOptions(**kwargs)


def openbase_cloud_claude_model(model: str | None, backend: str | None = None) -> str | None:
    if _configured_backend(backend) != OPENBASE_CLOUD_BACKEND:
        return model
    if not model:
        return OPENBASE_CLOUD_DEFAULT_CLAUDE_MODEL
    return OPENBASE_CLOUD_CLAUDE_MODEL_MAP.get(model.strip().lower(), model)


def openbase_cloud_claude_env(backend: str | None = None) -> dict[str, str]:
    if _configured_backend(backend) != OPENBASE_CLOUD_BACKEND:
        return {}
    return {
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_BASE_URL": _openbase_cloud_anthropic_base_url(),
        "ANTHROPIC_AUTH_TOKEN": _openbase_cloud_anthropic_auth_token(),
    }


def _configured_backend(backend: str | None) -> str:
    return normalize_backend(backend) if backend else configured_backend_from_environment()


def _openbase_cloud_anthropic_base_url() -> str:
    configured = (
        os.environ.get(OPENBASE_CLOUD_ANTHROPIC_BASE_URL_ENV)
        or os.environ.get(OPENBASE_CODER_CLI_WEB_BACKEND_URL_ENV)
        or OPENBASE_CLOUD_DEFAULT_BASE_URL
    ).rstrip("/")
    if configured.endswith(f"{OPENBASE_CLOUD_ANTHROPIC_PATH}/v1"):
        return configured[: -len("/v1")]
    if configured.endswith(OPENBASE_CLOUD_ANTHROPIC_PATH):
        return configured
    return f"{configured}{OPENBASE_CLOUD_ANTHROPIC_PATH}"


def _openbase_cloud_anthropic_auth_token() -> str:
    configured = os.environ.get(OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN_ENV, "").strip()
    if configured:
        return configured
    try:
        result = subprocess.run(
            ["openbase-coder", "auth", "print-machine-token"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(
            "Unable to get an Openbase Cloud machine token. Run `openbase-coder login`, then restart services."
        ) from exc
    token = result.stdout.strip()
    if result.returncode != 0 or not token:
        raise RuntimeError(
            "Unable to get an Openbase Cloud machine token. Run `openbase-coder login`, then restart services."
        )
    return token


def claude_extra_args() -> dict[str, str | None] | None:
    raw = os.environ.get(CLAUDE_EXTRA_ARGS_ENV, "").strip()
    if not raw:
        return None
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or not payload:
        return None
    return {str(flag): (None if value is None else str(value)) for flag, value in payload.items()}


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
