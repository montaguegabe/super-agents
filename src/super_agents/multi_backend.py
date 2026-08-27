"""Deterministic per-thread routing across configured backend identities."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from .app_models import LabelQueryInput, QueueCancelInput
from .backend_clients import client_for_backend
from .backend_config import (
    CLAUDE_CODE_BACKEND,
    CODEX_BACKEND,
    default_backend_from_environment,
    execution_backend,
    execution_backend_for_model,
    normalize_backend,
)
from .backend_provenance import BackendProvenanceStore
from .backend_routing import (
    AmbiguousBackendError,
    BackendEndpointConflictError,
    BackendNotFoundError,
    BackendResolutionError,
    annotated_items as _annotated_items,
    backend_has_local_state as _backend_has_local_state,
    client_has_pending_request as _client_has_pending_request,
    collect_result_ids as _collect_result_ids,
    optional_string as _optional_string,
    require_one_backend as _require_one_backend,
    session_matches as _session_matches,
    session_thread_id as _session_thread_id,
)
from .defaults import default_super_agents_model

JsonObject = dict[str, Any]
RoutedOperation = Callable[[Any], Awaitable[JsonObject]]

_STATUS_LIST_KEYS = (
    "pendingRequests",
    "pendingPermissionRequests",
    "queuedTurns",
    "activeTurns",
)


class MultiBackendClient:
    """Route each operation to the configured identity that owns its thread."""

    def __init__(
        self,
        default_backend: str | None = None,
        clients: dict[str, Any] | None = None,
        *,
        provenance_store: BackendProvenanceStore | None = None,
    ) -> None:
        self._default_backend = normalize_backend(
            default_backend if default_backend is not None else default_backend_from_environment()
        )
        self._clients = {normalize_backend(key): value for key, value in (clients or {}).items()}
        self._factory_created: set[str] = set()
        self.provenance = provenance_store or BackendProvenanceStore()
        self.client_for(self._default_backend)
        # Claim legacy state through its canonical local backend before an
        # explicit Cloud launch can instantiate a second identity that shares
        # the same execution client and store.
        for identity in (CODEX_BACKEND, CLAUDE_CODE_BACKEND):
            if execution_backend(identity) == execution_backend(self._default_backend):
                continue
            if identity not in self._clients and _backend_has_local_state(identity):
                self.client_for(identity)

    @property
    def backend(self) -> str:
        return self._default_backend

    def client_for(self, backend: str | None) -> Any:
        identity = self.resolve_backend(backend)
        if identity not in self._clients:
            candidate = client_for_backend(identity)
            self._validate_endpoint(identity, candidate)
            self._clients[identity] = candidate
            self._factory_created.add(identity)
        return self._clients[identity]

    def _validate_endpoint(self, identity: str, candidate: Any) -> None:
        if execution_backend(identity) != CODEX_BACKEND:
            return
        endpoint = getattr(candidate, "ws_url", None)
        for other_identity in self._factory_created:
            if execution_backend(other_identity) != CODEX_BACKEND:
                continue
            other_endpoint = getattr(self._clients[other_identity], "ws_url", None)
            if endpoint == other_endpoint:
                env_key = f"SUPER_AGENTS_{identity.upper()}_WS_URL"
                raise BackendEndpointConflictError(
                    f"Backends {other_identity} and {identity} target the same Codex app-server endpoint "
                    f"({endpoint}). Configure a distinct {env_key} before using both identities in one process."
                )

    def resolve_backend(self, backend: str | None) -> str:
        return normalize_backend(backend) if backend and backend.strip() else self._default_backend

    def engaged_backends(self) -> list[str]:
        identities = [self._default_backend]
        known = self.provenance.engaged_backends() | set(self._clients)
        for identity in sorted(known):
            if identity not in identities:
                identities.append(identity)

        default_execution = execution_backend(self._default_backend)
        for identity in (CODEX_BACKEND, CLAUDE_CODE_BACKEND):
            if execution_backend(identity) == default_execution or identity in identities:
                continue
            if _backend_has_local_state(identity):
                identities.append(identity)
        return identities

    async def start_thread(self, input_data: JsonObject) -> JsonObject:
        data = dict(input_data)
        explicit_backend = _optional_string(data.pop("backend", None))
        identity = self.resolve_backend(explicit_backend)
        if explicit_backend is None:
            # An explicit model names the runtime the caller wants: "fable"
            # must not be handed to Codex just because Codex is the default.
            identity = self._backend_for_model(_optional_string(data.get("model")), identity)
        result = await self.client_for(identity).start_thread(data)
        return self._remember_result(identity, result)

    def _backend_for_model(self, model: str | None, fallback: str) -> str:
        family = execution_backend_for_model(model)
        if family is None or execution_backend(fallback) == family:
            return fallback
        for identity in self.engaged_backends():
            if execution_backend(identity) == family:
                return identity
        return family

    async def resume_by_label(
        self,
        input_data: LabelQueryInput,
        *,
        developer_instructions: str | None = None,
        replace_developer_instructions: bool = False,
    ) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda client: client.resume_by_label(
                input_data,
                developer_instructions=developer_instructions,
                replace_developer_instructions=replace_developer_instructions,
            ),
        )

    async def read_by_label(
        self,
        input_data: LabelQueryInput,
        include_turns: bool = False,
    ) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda client: client.read_by_label(input_data, include_turns),
        )

    async def rename_by_label(self, input_data: LabelQueryInput, new_name: str) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda client: client.rename_by_label(input_data, new_name),
        )

    async def resolve_label(self, input_data: LabelQueryInput) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda client: client.resolve_label(input_data),
        )

    async def progress_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda client: client.progress_by_label(input_data),
        )

    async def steer_by_label(
        self,
        input_data: LabelQueryInput,
        prompt: str,
        turn_input: JsonObject | None = None,
    ) -> JsonObject:
        identity = await self._backend_for_label_query(input_data)
        adjusted = self._turn_input_for(identity, turn_input) if turn_input else None
        result = await self.client_for(identity).steer_by_label(input_data, prompt, adjusted)
        return self._remember_result(identity, result)

    async def cancel_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        return await self._route_label(
            input_data,
            lambda client: client.cancel_by_label(input_data),
        )

    async def start_turn_by_label(
        self,
        input_data: LabelQueryInput,
        turn_input: JsonObject,
    ) -> JsonObject:
        identity = await self._backend_for_label_query(input_data)
        result = await self.client_for(identity).start_turn_by_label(
            input_data,
            self._turn_input_for(identity, turn_input),
        )
        return self._remember_result(identity, result)

    async def queue_turn_by_label(
        self,
        input_data: LabelQueryInput,
        turn_input: JsonObject,
    ) -> JsonObject:
        identity = await self._backend_for_label_query(input_data)
        result = await self.client_for(identity).queue_turn_by_label(
            input_data,
            self._turn_input_for(identity, turn_input),
        )
        return self._remember_result(identity, result)

    async def cancel_queued_turn(self, input_data: QueueCancelInput) -> JsonObject:
        identity = await self._backend_for_queue_query(input_data)
        result = await self.client_for(identity).cancel_queued_turn(input_data)
        return self._remember_result(identity, result)

    async def answer_request(
        self,
        request_id: str | int,
        result: JsonObject,
        *,
        backend: str | None = None,
    ) -> JsonObject:
        known = self.provenance.backends_for_request(request_id)
        if backend:
            identity = self.resolve_backend(backend)
            if known and identity not in known:
                owners = ", ".join(sorted(known))
                raise BackendNotFoundError(f"Pending request {request_id} belongs to {owners}, not {identity}.")
            if not _client_has_pending_request(self.client_for(identity), request_id):
                raise BackendNotFoundError(f"Backend {identity} has no pending request {request_id}.")
        elif len(known) > 1:
            owners = ", ".join(sorted(known))
            raise AmbiguousBackendError(
                f"Pending request {request_id} exists on multiple backends: {owners}. Provide backend."
            )
        elif known:
            identity = next(iter(known))
        else:
            matches = [
                candidate
                for candidate in self.engaged_backends()
                if _client_has_pending_request(self.client_for(candidate), request_id)
            ]
            identity = _require_one_backend(
                matches,
                subject=f"pending request {request_id}",
            )
            self.provenance.remember(identity, request_ids={str(request_id)})
        response = await self.client_for(identity).answer_request(request_id, result)
        return self._remember_result(identity, response)

    async def status(self) -> JsonObject:
        statuses: list[tuple[str, JsonObject]] = []
        for identity in self.engaged_backends():
            status = await self.client_for(identity).status()
            statuses.append((identity, self._remember_result(identity, status)))
        if len(statuses) == 1:
            return statuses[0][1]
        merged = dict(statuses[0][1])
        for key in _STATUS_LIST_KEYS:
            merged[key] = [
                item for identity, status in statuses for item in _annotated_items(status.get(key), identity)
            ]
        merged["backend"] = self._default_backend
        merged["engagedBackends"] = [identity for identity, _status in statuses]
        return merged

    async def sessions(self) -> list[JsonObject]:
        sessions: list[JsonObject] = []
        for identity in self.engaged_backends():
            sessions.extend(await self._sessions_for_backend(identity))
        return sessions

    async def active(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        return await self._agents_result("active", input_data)

    async def recent(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        return await self._agents_result("recent", input_data)

    async def compact_status(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        return await self._agents_result("compact_status", input_data)

    async def thread_favorite(self, thread_id: str) -> JsonObject:
        identity = await self._backend_for_thread_id(thread_id)
        return await self.client_for(identity).thread_favorite(thread_id)

    async def tags(self) -> JsonObject:
        return await self.client_for(self._default_backend).tags()

    async def thread_tags(
        self,
        thread_id: str,
        tags: list[Any] | None = None,
    ) -> JsonObject:
        identity = await self._backend_for_thread_id(thread_id)
        return await self.client_for(identity).thread_tags(thread_id, tags)

    async def report_tags(
        self,
        project_path: str,
        path: str,
        tags: list[Any] | None = None,
    ) -> JsonObject:
        return await self.client_for(self._default_backend).report_tags(project_path, path, tags)

    async def _route_label(
        self,
        input_data: LabelQueryInput,
        operation: RoutedOperation,
    ) -> JsonObject:
        identity = await self._backend_for_label_query(input_data)
        result = await operation(self.client_for(identity))
        return self._remember_result(identity, result)

    async def _backend_for_label_query(self, input_data: LabelQueryInput) -> str:
        if input_data.backend:
            identity = self.resolve_backend(input_data.backend)
            self._validate_explicit_id_owner(identity, input_data)
            return identity
        if input_data.thread_id:
            return await self._backend_for_thread_id(input_data.thread_id)
        if input_data.turn_id:
            identity = self.provenance.backend_for_turn(input_data.turn_id)
            if identity:
                return identity
        if input_data.label:
            matches: list[str] = []
            for identity in self.engaged_backends():
                sessions = await self._sessions_for_backend(identity)
                if any(_session_matches(item, input_data) for item in sessions):
                    matches.append(identity)
            return _require_one_backend(matches, subject=f'name "{input_data.label}"')
        return self._default_backend

    async def _backend_for_thread_id(self, thread_id: str) -> str:
        identity = self.provenance.backend_for_thread(thread_id)
        if identity:
            return identity
        matches: list[str] = []
        for backend in self.engaged_backends():
            sessions = await self._sessions_for_backend(backend)
            if any(_session_thread_id(item) == thread_id for item in sessions):
                matches.append(backend)
        identity = _require_one_backend(matches, subject=f"thread {thread_id}")
        self.provenance.remember(identity, thread_ids={thread_id})
        return identity

    async def _backend_for_queue_query(self, input_data: QueueCancelInput) -> str:
        if input_data.backend:
            identity = self.resolve_backend(input_data.backend)
            if input_data.thread_id:
                owner = self.provenance.backend_for_thread(input_data.thread_id)
                if owner and owner != identity:
                    raise BackendResolutionError(f"Thread {input_data.thread_id} belongs to {owner}, not {identity}.")
            if input_data.queue_item_id:
                owner = self.provenance.backend_for_turn(input_data.queue_item_id)
                if owner and owner != identity:
                    raise BackendResolutionError(
                        f"Queued item {input_data.queue_item_id} belongs to {owner}, not {identity}."
                    )
            return identity
        if input_data.thread_id:
            return await self._backend_for_thread_id(input_data.thread_id)
        if input_data.queue_item_id:
            identity = self.provenance.backend_for_turn(input_data.queue_item_id)
            if identity:
                return identity
        if input_data.label:
            return await self._backend_for_label_query(LabelQueryInput(label=input_data.label, cwd=input_data.cwd))
        raise BackendNotFoundError(
            "Cannot route queued-turn cancellation without a known queueItemId, threadId, name, or backend."
        )

    async def _sessions_for_backend(self, identity: str) -> list[JsonObject]:
        raw_sessions = await self.client_for(identity).sessions()
        provenance = self.provenance.read()
        legacy_identity = self._legacy_identity_for_execution(execution_backend(identity))
        migrated: set[str] = set()
        sessions: list[JsonObject] = []
        for raw in raw_sessions:
            if not isinstance(raw, dict):
                continue
            thread_id = _session_thread_id(raw)
            owner = provenance.threads.get(thread_id) if thread_id else None
            if owner and owner != identity:
                continue
            if thread_id and owner is None:
                if identity != legacy_identity:
                    continue
                migrated.add(thread_id)
            sessions.append({**raw, "backend": identity})
        if migrated:
            self.provenance.remember(identity, thread_ids=migrated)
        return sessions

    def _legacy_identity_for_execution(self, execution: str) -> str:
        if execution_backend(self._default_backend) == execution:
            return self._default_backend
        return execution

    def _validate_explicit_id_owner(self, identity: str, query: LabelQueryInput) -> None:
        if query.thread_id:
            owner = self.provenance.backend_for_thread(query.thread_id)
            if owner and owner != identity:
                raise BackendResolutionError(f"Thread {query.thread_id} belongs to {owner}, not {identity}.")
        if query.turn_id:
            owner = self.provenance.backend_for_turn(query.turn_id)
            if owner and owner != identity:
                raise BackendResolutionError(f"Turn {query.turn_id} belongs to {owner}, not {identity}.")

    async def _agents_result(
        self,
        method: str,
        input_data: LabelQueryInput | None,
    ) -> JsonObject:
        query = input_data or LabelQueryInput()
        if query.backend or query.thread_id or query.turn_id or query.label:
            identity = await self._backend_for_label_query(query)
            result = await getattr(self.client_for(identity), method)(query)
            return self._remember_result(identity, result)

        agents: list[JsonObject] = []
        identities = self.engaged_backends()
        for identity in identities:
            result = await getattr(self.client_for(identity), method)(query)
            remembered = self._remember_result(identity, result)
            agents.extend(_annotated_items(remembered.get("agents"), identity))
        return {
            "backend": self._default_backend,
            "engagedBackends": identities,
            "count": len(agents),
            "agents": agents,
        }

    def _turn_input_for(self, identity: str, turn_input: JsonObject) -> JsonObject:
        requested = turn_input.get("model")
        family = execution_backend_for_model(requested if isinstance(requested, str) else None)
        if family is not None and family != execution_backend(identity):
            raise BackendResolutionError(
                f"Model {requested!r} runs on the {family} backend, but this thread "
                f"lives on the {identity} backend, which cannot execute it. "
                "Start a NEW thread (super_agents_start) with this model instead, "
                "or drop the model to continue this thread on its own backend."
            )
        if identity == self._default_backend:
            return turn_input
        model = turn_input.get("model")
        if not model or model != default_super_agents_model(backend=self._default_backend):
            return turn_input
        adjusted = dict(turn_input)
        routed_default = default_super_agents_model(backend=identity)
        if routed_default:
            adjusted["model"] = routed_default
        else:
            adjusted.pop("model", None)
        return adjusted

    def _remember_result(self, identity: str, result: JsonObject) -> JsonObject:
        annotated = {**result, "backend": identity}
        thread_ids: set[str] = set()
        turn_ids: set[str] = set()
        request_ids: set[str] = set()
        _collect_result_ids(
            annotated,
            thread_ids=thread_ids,
            turn_ids=turn_ids,
            request_ids=request_ids,
        )
        if thread_ids or turn_ids or request_ids:
            self.provenance.remember(
                identity,
                thread_ids=thread_ids,
                turn_ids=turn_ids,
                request_ids=request_ids,
            )
        return annotated
