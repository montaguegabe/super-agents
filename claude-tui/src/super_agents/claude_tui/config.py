from __future__ import annotations

import os
import shlex
from pathlib import Path


CLAUDE_CMD_ENV = "SUPER_AGENTS_CLAUDE_TUI_CMD"
CLAUDE_ARGS_ENV = "SUPER_AGENTS_CLAUDE_TUI_ARGS"
CLAUDE_MODEL_ENV = "SUPER_AGENTS_CLAUDE_TUI_MODEL"
SMOKE_MODEL_ENV = "SUPER_AGENTS_CLAUDE_TUI_SMOKE_MODEL"
LEGACY_CLAUDE_CMD_ENV = "SUPER_AGENTS_CLAUDE_CMD"
LEGACY_CLAUDE_ARGS_ENV = "SUPER_AGENTS_CLAUDE_ARGS"
LEGACY_CLAUDE_MODEL_ENV = "SUPER_AGENTS_CLAUDE_MODEL"
LEGACY_SMOKE_MODEL_ENV = "SUPER_AGENTS_CLAUDE_SMOKE_MODEL"
DEFAULT_SMOKE_MODEL = "sonnet"
SMOKE_MODELS_TO_AVOID = ("fable",)


def default_cwd() -> str:
    return str(Path.home())


def build_claude_command(model: str | None = None) -> list[str]:
    """Build the command used for one interactive Claude TUI/CLI session."""
    command = shlex.split(os.environ.get(CLAUDE_CMD_ENV) or os.environ.get(LEGACY_CLAUDE_CMD_ENV, "claude"))
    extra_args = shlex.split(os.environ.get(CLAUDE_ARGS_ENV) or os.environ.get(LEGACY_CLAUDE_ARGS_ENV, ""))
    chosen_model = model or os.environ.get(CLAUDE_MODEL_ENV) or os.environ.get(LEGACY_CLAUDE_MODEL_ENV)
    if chosen_model:
        extra_args.extend(["--model", chosen_model])
    return command + extra_args


def recommended_smoke_model() -> str:
    return os.environ.get(SMOKE_MODEL_ENV) or os.environ.get(LEGACY_SMOKE_MODEL_ENV, DEFAULT_SMOKE_MODEL)
