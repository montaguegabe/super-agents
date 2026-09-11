"""Optional session-scoped configuration for embedding applications."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

CODEX_PROFILE_PATH_ENV = "SUPER_AGENTS_CODEX_PROFILE_PATH"
CLAUDE_SETTINGS_PATH_ENV = "SUPER_AGENTS_CLAUDE_SETTINGS_PATH"
CLAUDE_MCP_CONFIG_PATH_ENV = "SUPER_AGENTS_CLAUDE_MCP_CONFIG_PATH"


def codex_profile_config(backend: str = "codex") -> dict[str, Any]:
    """Load the selected TOML profile for thread/start and thread/resume.

    Codex accepts native profiles in its TUI/exec commands, but not on
    app-server. Its thread config overrides accept the same TOML structure.
    A configured, missing or invalid profile must fail rather than fall back
    to the user's unrelated defaults.
    """
    key = f"SUPER_AGENTS_{backend.upper()}_PROFILE_PATH"
    configured = os.environ.get(key) or os.environ.get(CODEX_PROFILE_PATH_ENV)
    if not configured:
        return {}
    with Path(configured).expanduser().open("rb") as stream:
        return tomllib.load(stream)


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overrides.items():
        previous = result.get(key)
        result[key] = merge_config(previous, value) if isinstance(previous, dict) and isinstance(value, dict) else value
    return result
