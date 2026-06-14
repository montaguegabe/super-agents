from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

from .app_models import LabelQueryInput
from .app_server_client import CodexAppServerClient

JsonObject = dict[str, Any]

CODEX_BACKEND = "codex"
OPENBASE_CLOUD_BACKEND = "openbase_cloud"
CLAUDE_CODE_BACKEND = "claude_code"
CODEX_COMPATIBLE_BACKENDS = {CODEX_BACKEND, OPENBASE_CLOUD_BACKEND}
CODING_BACKEND_ENV_KEY = "OPENBASE_CODING_BACKEND"
LEGACY_CODEX_BACKEND_ENV_KEY = "OPENBASE_CODEX_BACKEND"
DEFAULT_ENV_FILE = Path.home() / ".openbase" / ".env"
BACKEND_ALIASES = {
    "": CODEX_BACKEND,
    "openai": CODEX_BACKEND,
    "codex": CODEX_BACKEND,
    "openbase": OPENBASE_CLOUD_BACKEND,
    "openbase-cloud": OPENBASE_CLOUD_BACKEND,
    "openbase_cloud": OPENBASE_CLOUD_BACKEND,
    "cloud": OPENBASE_CLOUD_BACKEND,
    "claude": CLAUDE_CODE_BACKEND,
    "claude-code": CLAUDE_CODE_BACKEND,
    "claude_code": CLAUDE_CODE_BACKEND,
    "claude-agent": CLAUDE_CODE_BACKEND,
    "claude-agent-sdk": CLAUDE_CODE_BACKEND,
    "claude_agent_sdk": CLAUDE_CODE_BACKEND,
    "claude-sdk": CLAUDE_CODE_BACKEND,
    "claude-tui": CLAUDE_CODE_BACKEND,
    "claude-code-tui": CLAUDE_CODE_BACKEND,
}


class SuperAgentsClient(Protocol):
    async def status(self) -> JsonObject: ...
    async def start_thread(self, input_data: JsonObject) -> JsonObject: ...
    async def resume_by_label(
        self,
        input_data: LabelQueryInput,
        *,
        developer_instructions: str | None = None,
    ) -> JsonObject: ...
    async def read_by_label(self, input_data: LabelQueryInput, include_turns: bool = False) -> JsonObject: ...
    async def rename_by_label(self, input_data: LabelQueryInput, new_name: str) -> JsonObject: ...
    async def answer_request(self, request_id: str | int, result: JsonObject) -> JsonObject: ...
    async def sessions(self) -> list[JsonObject]: ...
    async def thread_favorite(self, thread_id: str) -> JsonObject: ...
    async def tags(self) -> JsonObject: ...
    async def thread_tags(self, thread_id: str, tags: list[Any] | None = None) -> JsonObject: ...
    async def report_tags(self, project_path: str, path: str, tags: list[Any] | None = None) -> JsonObject: ...
    async def active(self, input_data: LabelQueryInput | None = None) -> JsonObject: ...
    async def recent(self, input_data: LabelQueryInput | None = None) -> JsonObject: ...
    async def compact_status(self, input_data: LabelQueryInput | None = None) -> JsonObject: ...
    async def resolve_label(self, input_data: LabelQueryInput) -> JsonObject: ...
    async def progress_by_label(self, input_data: LabelQueryInput) -> JsonObject: ...
    async def steer_by_label(
        self,
        input_data: LabelQueryInput,
        prompt: str,
        turn_input: JsonObject | None = None,
    ) -> JsonObject: ...
    async def cancel_by_label(self, input_data: LabelQueryInput) -> JsonObject: ...
    async def start_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject: ...
    async def queue_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject: ...


def normalize_backend(value: str | None) -> str:
    raw = (value or "").strip().lower()
    try:
        return BACKEND_ALIASES[raw]
    except KeyError as exc:
        supported = ", ".join(
            sorted({CODEX_BACKEND, OPENBASE_CLOUD_BACKEND, CLAUDE_CODE_BACKEND})
        )
        raise ValueError(f"Unsupported {CODING_BACKEND_ENV_KEY}: {value}. Supported backends: {supported}.") from exc


def backend_from_environment() -> str:
    env_values = _env_file_values(DEFAULT_ENV_FILE)
    return execution_backend(
        normalize_backend(
            os.environ.get(CODING_BACKEND_ENV_KEY)
            or os.environ.get(LEGACY_CODEX_BACKEND_ENV_KEY)
            or env_values.get(CODING_BACKEND_ENV_KEY)
            or env_values.get(LEGACY_CODEX_BACKEND_ENV_KEY)
        )
    )


def execution_backend(backend: str) -> str:
    return CODEX_BACKEND if backend in CODEX_COMPATIBLE_BACKENDS else backend


def configured_backend_from_environment() -> str:
    env_values = _env_file_values(DEFAULT_ENV_FILE)
    return normalize_backend(
        os.environ.get(CODING_BACKEND_ENV_KEY)
        or os.environ.get(LEGACY_CODEX_BACKEND_ENV_KEY)
        or env_values.get(CODING_BACKEND_ENV_KEY)
        or env_values.get(LEGACY_CODEX_BACKEND_ENV_KEY)
    )


def _env_file_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def client_from_environment() -> SuperAgentsClient:
    backend = backend_from_environment()
    if backend == CLAUDE_CODE_BACKEND:
        from super_agents.claude_sdk import ClaudeAgentSdkClient

        return ClaudeAgentSdkClient()
    return CodexAppServerClient()
