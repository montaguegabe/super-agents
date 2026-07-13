"""Route Super Agents operations across coding backends chosen per launch.

``MultiBackendClient`` implements the ``SuperAgentsClient`` protocol while
letting each thread launch pick its own backend: ``super_agents_start`` may
name ``codex``, ``openbase_cloud``, or ``claude_code`` and the thread runs
there; everything else defaults to the default backend — the
``default_backend`` constructor argument, else ``SUPER_AGENTS_DEFAULT_BACKEND``
in the environment, else the configured backend. Name- and id-addressed
operations route to the backend that owns the thread — an explicit
``backend`` on the query pins the route, otherwise engaged backends are
tried in order (default backend first).

A backend is "engaged" once this process has built a client for it or when
its on-disk state shows it has been used before, so threads started on a
non-default backend stay reachable across restarts.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Awaitable, Callable

from .app_models import LabelQueryInput
from .backend_config import (
    CLAUDE_CODE_BACKEND,
    CODEX_BACKEND,
    default_backend_from_environment,
    execution_backend,
    normalize_backend,
)
from .defaults import default_super_agents_model

JsonObject = dict[str, Any]
logger = logging.getLogger(__name__)

# Status keys whose values are lists of per-item payloads that can be merged
# across backends without changing their shape.
_STATUS_LIST_KEYS = (
    "pendingRequests",
    "pendingPermissionRequests",
    "queuedTurns",
    "activeTurns",
)


class MultiBackendClient:
    """SuperAgentsClient that multiplexes per-launch backend choices."""

    def __init__(
        self,
        default_backend: str | None = None,
        clients: dict[str, Any] | None = None,
    ) -> None:
        configured = (
            normalize_backend(default_backend) if default_backend is not None else default_backend_from_environment()
        )
        self._default_backend = execution_backend(configured)
        self._clients: dict[str, Any] = dict(clients or {})
        self.client_for(None)

    @property
    def backend(self) -> str:
        return self._default_backend

    def client_for(self, backend: str | None) -> Any:
        key = self.resolve_backend(backend)
        if key not in self._clients:
            self._clients[key] = self._create_client(key)
        return self._clients[key]

    def resolve_backend(self, backend: str | None) -> str:
        if backend is None or not str(backend).strip():
            return self._default_backend
        return execution_backend(normalize_backend(str(backend)))

    def _create_client(self, backend: str) -> Any:
        if backend == CLAUDE_CODE_BACKEND:
            from .claude_sdk import ClaudeAgentSdkClient

            return ClaudeAgentSdkClient()
        from .app_server_client import CodexAppServerClient

        return CodexAppServerClient()

    def engaged_clients(self) -> list[tuple[str, Any]]:
        """Backends to consult, configured backend first."""
        backends = [self._default_backend]
        for backend in (CODEX_BACKEND, CLAUDE_CODE_BACKEND):
            if backend in backends:
                continue
            if backend in self._clients or _backend_has_local_state(backend):
                backends.append(backend)
        return [(backend, self.client_for(backend)) for backend in backends]

    # -- thread lifecycle -------------------------------------------------

    async def start_thread(self, input_data: JsonObject) -> JsonObject:
        data = dict(input_data)
        backend = self.resolve_backend(_optional_str(data.pop("backend", None)))
        client = self.client_for(backend)
        result = await client.start_thread(data)
        if isinstance(result, dict):
            result.setdefault("backend", backend)
        return result

    async def resume_by_label(
        self,
        input_data: LabelQueryInput,
        *,
        developer_instructions: str | None = None,
    ) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda backend, client: client.resume_by_label(input_data, developer_instructions=developer_instructions),
        )

    async def read_by_label(self, input_data: LabelQueryInput, include_turns: bool = False) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda backend, client: client.read_by_label(input_data, include_turns),
        )

    async def rename_by_label(self, input_data: LabelQueryInput, new_name: str) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda backend, client: client.rename_by_label(input_data, new_name),
        )

    async def resolve_label(self, input_data: LabelQueryInput) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda backend, client: client.resolve_label(input_data),
        )

    async def progress_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda backend, client: client.progress_by_label(input_data),
        )

    async def steer_by_label(
        self,
        input_data: LabelQueryInput,
        prompt: str,
        turn_input: JsonObject | None = None,
    ) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda backend, client: client.steer_by_label(
                input_data, prompt, self._turn_input_for(backend, turn_input) if turn_input else turn_input
            ),
        )

    async def cancel_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda backend, client: client.cancel_by_label(input_data),
        )

    async def start_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda backend, client: client.start_turn_by_label(input_data, self._turn_input_for(backend, turn_input)),
        )

    async def queue_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda backend, client: client.queue_turn_by_label(input_data, self._turn_input_for(backend, turn_input)),
        )

    async def answer_request(self, request_id: str | int, result: JsonObject) -> JsonObject:
        first_error: Exception | None = None
        for _backend, client in self.engaged_clients():
            try:
                return await client.answer_request(request_id, result)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        raise first_error or ValueError(f"No pending request found for id {request_id}.")

    # -- aggregations ------------------------------------------------------

    async def status(self) -> JsonObject:
        engaged = self.engaged_clients()
        if len(engaged) == 1:
            return await engaged[0][1].status()
        statuses: dict[str, JsonObject] = {}
        errors: dict[str, str] = {}
        for backend, client in engaged:
            try:
                statuses[backend] = await client.status()
            except Exception as exc:
                errors[backend] = str(exc)
        base = statuses.get(self._default_backend) or next(iter(statuses.values()), {})
        merged = dict(base)
        for key in _STATUS_LIST_KEYS:
            items: list[Any] = []
            for backend, status in statuses.items():
                for item in status.get(key) or []:
                    if isinstance(item, dict):
                        item.setdefault("backend", backend)
                    items.append(item)
            merged[key] = items
        merged["backend"] = self._default_backend
        merged["engagedBackends"] = [backend for backend, _client in engaged]
        if errors:
            merged["backendErrors"] = errors
        return merged

    async def sessions(self) -> list[JsonObject]:
        items: list[JsonObject] = []
        for backend, client in self.engaged_clients():
            try:
                sessions = await client.sessions()
            except Exception:
                logger.exception("Failed to list %s Super Agents sessions.", backend)
                continue
            for session in sessions:
                if isinstance(session, dict):
                    session.setdefault("backend", backend)
                items.append(session)
        return items

    async def active(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        return await self._merged_agents("active", input_data)

    async def recent(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        return await self._merged_agents("recent", input_data)

    async def compact_status(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        return await self._merged_agents("compact_status", input_data)

    # -- local Openbase Coder metadata --------------------------------------

    # Favorites and tags read/write local Openbase Coder files; the Codex
    # client implements them without needing a server connection, so they
    # work for threads from any backend.

    async def thread_favorite(self, thread_id: str) -> JsonObject:
        return await self.client_for(CODEX_BACKEND).thread_favorite(thread_id)

    async def tags(self) -> JsonObject:
        return await self.client_for(CODEX_BACKEND).tags()

    async def thread_tags(self, thread_id: str, tags: list[Any] | None = None) -> JsonObject:
        return await self.client_for(CODEX_BACKEND).thread_tags(thread_id, tags)

    async def report_tags(self, project_path: str, path: str, tags: list[Any] | None = None) -> JsonObject:
        return await self.client_for(CODEX_BACKEND).report_tags(project_path, path, tags)

    # -- routing internals ---------------------------------------------------

    async def _route_label(
        self,
        input_data: LabelQueryInput,
        op: Callable[[str, Any], Awaitable[JsonObject]],
    ) -> JsonObject:
        pinned = getattr(input_data, "backend", None)
        if pinned:
            backend = self.resolve_backend(pinned)
            return await op(backend, self.client_for(backend))
        engaged = self.engaged_clients()
        first_error: Exception | None = None
        for backend, client in engaged:
            try:
                return await op(backend, client)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
        assert first_error is not None
        raise first_error

    def _turn_input_for(self, backend: str, turn_input: JsonObject) -> JsonObject:
        """Fix up a turn input for the backend the thread actually lives on.

        The MCP layer defaults the model for the configured backend before
        the owning backend is known; when the route lands elsewhere, that
        default would name a model from the wrong backend family.
        """
        if backend == self._default_backend:
            return turn_input
        model = turn_input.get("model")
        if not model or model != default_super_agents_model(backend=self._default_backend):
            return turn_input
        adjusted = dict(turn_input)
        routed_default = default_super_agents_model(backend=backend)
        if routed_default:
            adjusted["model"] = routed_default
        else:
            adjusted.pop("model", None)
        return adjusted

    async def _merged_agents(self, method: str, input_data: LabelQueryInput | None) -> JsonObject:
        pinned = getattr(input_data, "backend", None) if input_data else None
        if pinned:
            backend = self.resolve_backend(pinned)
            return await getattr(self.client_for(backend), method)(input_data)
        engaged = self.engaged_clients()
        if len(engaged) == 1:
            return await getattr(engaged[0][1], method)(input_data)
        agents: list[JsonObject] = []
        errors: dict[str, str] = {}
        for backend, client in engaged:
            try:
                result = await getattr(client, method)(input_data)
            except Exception as exc:
                errors[backend] = str(exc)
                continue
            for item in result.get("agents") or []:
                if isinstance(item, dict):
                    item.setdefault("backend", backend)
                agents.append(item)
        merged: JsonObject = {
            "backend": self._default_backend,
            "engagedBackends": [backend for backend, _client in engaged],
            "count": len(agents),
            "agents": agents,
        }
        if errors:
            merged["backendErrors"] = errors
        return merged


def _backend_has_local_state(backend: str) -> bool:
    if backend == CLAUDE_CODE_BACKEND:
        from .agent_store import database_path

        return database_path().exists()
    from .app_server_client import DEFAULT_STATE_FILE

    state_file = os.environ.get("SUPER_AGENTS_STATE_FILE")
    return (Path(state_file).expanduser() if state_file else DEFAULT_STATE_FILE).exists()


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
