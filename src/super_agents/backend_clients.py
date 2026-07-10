from __future__ import annotations

from typing import Any, Protocol

from .app_models import LabelQueryInput
from .app_server_client import CodexAppServerClient
from .backend_config import (  # noqa: F401  (re-exported for compatibility)
    BACKEND_ALIASES,
    CLAUDE_CODE_BACKEND,
    CODEX_BACKEND,
    CODEX_COMPATIBLE_BACKENDS,
    CODING_BACKEND_ENV_KEY,
    DEFAULT_ENV_FILE,
    OPENBASE_CLOUD_BACKEND,
    backend_from_environment,
    configured_backend_from_environment,
    execution_backend,
    normalize_backend,
)

JsonObject = dict[str, Any]


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


def client_from_environment() -> SuperAgentsClient:
    backend = backend_from_environment()
    if backend == CLAUDE_CODE_BACKEND:
        from super_agents.claude_sdk import ClaudeAgentSdkClient

        return ClaudeAgentSdkClient()
    return CodexAppServerClient()


def multi_client_from_environment() -> SuperAgentsClient:
    """Client that defaults to the configured backend but can launch on any."""
    from super_agents.multi_backend import MultiBackendClient

    return MultiBackendClient()
