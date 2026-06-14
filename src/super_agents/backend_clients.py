from __future__ import annotations

import os
from typing import Any, Protocol

from .app_models import LabelQueryInput
from .app_server_client import CodexAppServerClient

JsonObject = dict[str, Any]

CODEX_BACKEND = "codex"
CLAUDE_CODE_PROXY_BACKEND = "claude-code-proxy"
CLAUDE_TUI_BACKEND = "claude-tui"
BACKEND_ALIASES = {
    "": CODEX_BACKEND,
    "openai": CODEX_BACKEND,
    "codex": CODEX_BACKEND,
    "claude": CLAUDE_CODE_PROXY_BACKEND,
    "claude-code": CLAUDE_CODE_PROXY_BACKEND,
    "claude-code-proxy": CLAUDE_CODE_PROXY_BACKEND,
    "claude-proxy": CLAUDE_CODE_PROXY_BACKEND,
    "claude-tui": CLAUDE_TUI_BACKEND,
    "claude-code-tui": CLAUDE_TUI_BACKEND,
}


class SuperAgentsClient(Protocol):
    async def status(self) -> JsonObject: ...
    async def start_thread(self, input_data: JsonObject) -> JsonObject: ...
    async def resume_by_label(self, input_data: LabelQueryInput) -> JsonObject: ...
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
    async def steer_by_label(self, input_data: LabelQueryInput, prompt: str) -> JsonObject: ...
    async def cancel_by_label(self, input_data: LabelQueryInput) -> JsonObject: ...
    async def start_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject: ...
    async def queue_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject: ...


def normalize_backend(value: str | None) -> str:
    raw = (value or "").strip().lower()
    try:
        return BACKEND_ALIASES[raw]
    except KeyError as exc:
        supported = ", ".join(sorted({CODEX_BACKEND, CLAUDE_CODE_PROXY_BACKEND, CLAUDE_TUI_BACKEND}))
        raise ValueError(f"Unsupported OPENBASE_CODEX_BACKEND: {value}. Supported backends: {supported}.") from exc


def backend_from_environment() -> str:
    return normalize_backend(os.environ.get("OPENBASE_CODEX_BACKEND"))


def client_from_environment() -> SuperAgentsClient:
    backend = backend_from_environment()
    if backend == CLAUDE_TUI_BACKEND:
        from super_agents.claude_tui.client import ClaudeTuiClient

        return ClaudeTuiClient()
    return CodexAppServerClient()
