from __future__ import annotations

import os
from pathlib import Path


APP_DIR_ENV = "SUPER_AGENTS_CLAUDE_TUI_HOME"
LEGACY_APP_DIR_ENV = "SUPER_AGENTS_CLAUDE_HOME"


def app_dir() -> Path:
    configured = os.environ.get(APP_DIR_ENV) or os.environ.get(LEGACY_APP_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "super-agents-claude-tui"


def database_path() -> Path:
    return app_dir() / "state.sqlite3"


def logs_dir() -> Path:
    return app_dir() / "logs"
