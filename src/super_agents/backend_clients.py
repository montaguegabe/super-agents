from __future__ import annotations

import os
from typing import Any, Protocol

from .app_models import LabelQueryInput, QueueCancelInput
from .app_endpoint import AppServerEndpointError
from .app_server_client import CodexAppServerClient
from .backend_config import (  # noqa: F401  (re-exported for compatibility)
    BACKEND_ALIASES,
    CLAUDE_CODE_BACKEND,
    CODEX_BACKEND,
    CODEX_COMPATIBLE_BACKENDS,
    CODING_BACKEND_ENV_KEY,
    DEFAULT_BACKEND_ENV_KEY,
    DEFAULT_ENV_FILE,
    OPENBASE_CLOUD_BACKEND,
    OPENBASE_CLOUD_CODEX_BACKEND,
    backend_from_environment,
    configured_backend_from_environment,
    default_backend_from_environment,
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
        replace_developer_instructions: bool = False,
    ) -> JsonObject: ...
    async def read_by_label(self, input_data: LabelQueryInput, include_turns: bool = False) -> JsonObject: ...
    async def rename_by_label(self, input_data: LabelQueryInput, new_name: str) -> JsonObject: ...
    async def answer_request(
        self,
        request_id: str | int,
        result: JsonObject,
        *,
        backend: str | None = None,
    ) -> JsonObject: ...
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
    async def cancel_queued_turn(self, input_data: QueueCancelInput) -> JsonObject: ...


def client_from_environment() -> SuperAgentsClient:
    return client_for_backend(configured_backend_from_environment())


def client_for_backend(backend: str) -> SuperAgentsClient:
    """Build a client for one configured backend identity.

    Configured identity is deliberately kept separate from execution kind:
    Cloud-backed identities use the same protocol clients while retaining
    their own model, credential, approval, and provenance semantics.
    """
    identity = normalize_backend(backend)
    if execution_backend(identity) == CLAUDE_CODE_BACKEND:
        from super_agents.claude_sdk import ClaudeAgentSdkClient

        return ClaudeAgentSdkClient(backend_identity=identity)
    from super_agents.defaults import default_super_agents_model

    endpoint_key = f"SUPER_AGENTS_{identity.upper()}_APP_SERVER_ENDPOINT"
    legacy_key = f"SUPER_AGENTS_{identity.upper()}_WS_URL"
    endpoint_value = os.environ.get(endpoint_key, "").strip()
    legacy_value = os.environ.get(legacy_key, "").strip()
    if endpoint_value and legacy_value and endpoint_value != legacy_value:
        raise AppServerEndpointError(
            f"Conflicting Codex app-server endpoints select different owners: "
            f"{endpoint_key}={endpoint_value}, {legacy_key}={legacy_value}. "
            "Configure exactly one endpoint."
        )
    backend_endpoint = endpoint_value or legacy_value or None
    return CodexAppServerClient(
        endpoint=backend_endpoint,
        backend_identity=identity,
        default_model=default_super_agents_model(backend=identity),
    )


def multi_client_from_environment() -> SuperAgentsClient:
    from super_agents.multi_backend import MultiBackendClient

    return MultiBackendClient()
