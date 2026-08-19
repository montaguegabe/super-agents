from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from super_agents.app_models import LabelQueryInput, QueueCancelInput
from super_agents.backend_provenance import BackendProvenanceStore
from super_agents.mcp_server import build_tools, clean_name_query_input, clean_thread_input
from super_agents.multi_backend import (
    AmbiguousBackendError,
    BackendEndpointConflictError,
    BackendResolutionError,
    MultiBackendClient,
)

JsonObject = dict[str, Any]


class FakeBackendClient:
    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.sessions_by_id: dict[str, JsonObject] = {}
        self.pending: list[JsonObject] = []
        self.calls: list[tuple[str, Any]] = []
        self._next_thread = 1
        self._next_turn = 1

    def add_session(self, thread_id: str, name: str, cwd: str = "/repo") -> None:
        self.sessions_by_id[thread_id] = {
            "threadId": thread_id,
            "name": name,
            "cwd": cwd,
            "status": "completed",
        }

    async def start_thread(self, input_data: JsonObject) -> JsonObject:
        self.calls.append(("start_thread", dict(input_data)))
        thread_id = f"{self.backend}-thread-{self._next_thread}"
        self._next_thread += 1
        self.add_session(thread_id, str(input_data["name"]), str(input_data.get("cwd") or "/repo"))
        return {"threadId": thread_id, "name": input_data["name"]}

    async def sessions(self) -> list[JsonObject]:
        self.calls.append(("sessions", None))
        return list(self.sessions_by_id.values())

    async def status(self) -> JsonObject:
        self.calls.append(("status", None))
        return {
            "ready": True,
            "pendingRequests": self.pending,
            "pendingPermissionRequests": self.pending,
            "queuedTurns": [],
            "activeTurns": list(self.sessions_by_id.values()),
        }

    def pending_permission_requests(self) -> list[JsonObject]:
        return self.pending

    async def answer_request(self, request_id: str | int, result: JsonObject) -> JsonObject:
        self.calls.append(("answer_request", (request_id, result)))
        return {"answered": True, "request": {"id": request_id}}

    async def read_by_label(
        self,
        input_data: LabelQueryInput,
        include_turns: bool = False,
    ) -> JsonObject:
        self.calls.append(("read_by_label", (input_data, include_turns)))
        return self._thread_result(input_data)

    async def resume_by_label(self, input_data: LabelQueryInput, **_kwargs: Any) -> JsonObject:
        self.calls.append(("resume_by_label", input_data))
        return self._thread_result(input_data)

    async def rename_by_label(self, input_data: LabelQueryInput, new_name: str) -> JsonObject:
        self.calls.append(("rename_by_label", (input_data, new_name)))
        result = self._thread_result(input_data)
        self.sessions_by_id[result["threadId"]]["name"] = new_name
        return {**result, "name": new_name, "renamed": True}

    async def resolve_label(self, input_data: LabelQueryInput) -> JsonObject:
        self.calls.append(("resolve_label", input_data))
        return self._thread_result(input_data)

    async def progress_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        self.calls.append(("progress_by_label", input_data))
        return self._thread_result(input_data)

    async def steer_by_label(
        self,
        input_data: LabelQueryInput,
        prompt: str,
        turn_input: JsonObject | None = None,
    ) -> JsonObject:
        self.calls.append(("steer_by_label", (input_data, prompt, turn_input)))
        return self._new_turn_result(input_data)

    async def cancel_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        self.calls.append(("cancel_by_label", input_data))
        return {**self._thread_result(input_data), "cancelled": True}

    async def start_turn_by_label(
        self,
        input_data: LabelQueryInput,
        turn_input: JsonObject,
    ) -> JsonObject:
        self.calls.append(("start_turn_by_label", (input_data, turn_input)))
        return self._new_turn_result(input_data)

    async def queue_turn_by_label(
        self,
        input_data: LabelQueryInput,
        turn_input: JsonObject,
    ) -> JsonObject:
        self.calls.append(("queue_turn_by_label", (input_data, turn_input)))
        return {**self._new_turn_result(input_data), "queued": True}

    async def cancel_queued_turn(self, input_data: QueueCancelInput) -> JsonObject:
        self.calls.append(("cancel_queued_turn", input_data))
        thread = self._find(input_data.thread_id, input_data.label)
        return {
            "threadId": thread["threadId"],
            "queueItemId": input_data.queue_item_id,
            "cancelled": True,
        }

    async def active(self, _input_data: LabelQueryInput | None = None) -> JsonObject:
        return {"agents": list(self.sessions_by_id.values()), "count": len(self.sessions_by_id)}

    async def recent(self, _input_data: LabelQueryInput | None = None) -> JsonObject:
        return {"agents": list(self.sessions_by_id.values()), "count": len(self.sessions_by_id)}

    async def compact_status(self, _input_data: LabelQueryInput | None = None) -> JsonObject:
        self.calls.append(("compact_status", _input_data))
        return {"agents": list(self.sessions_by_id.values()), "count": len(self.sessions_by_id)}

    async def thread_favorite(self, thread_id: str) -> JsonObject:
        return {"threadId": thread_id, "favorite": False}

    async def tags(self) -> JsonObject:
        return {"tags": []}

    async def thread_tags(self, thread_id: str, tags: list[Any] | None = None) -> JsonObject:
        return {"threadId": thread_id, "tags": tags or []}

    async def report_tags(self, project_path: str, path: str, tags: list[Any] | None = None) -> JsonObject:
        return {"projectPath": project_path, "path": path, "tags": tags or []}

    def _thread_result(self, query: LabelQueryInput) -> JsonObject:
        return dict(self._find(query.thread_id, query.label))

    def _new_turn_result(self, query: LabelQueryInput) -> JsonObject:
        thread = self._find(query.thread_id, query.label)
        turn_id = f"{self.backend}-turn-{self._next_turn}"
        self._next_turn += 1
        return {"threadId": thread["threadId"], "turnId": turn_id}

    def _find(self, thread_id: str | None, name: str | None) -> JsonObject:
        if thread_id:
            return self.sessions_by_id[thread_id]
        return next(item for item in self.sessions_by_id.values() if item["name"] == name)


@pytest.fixture
def clients() -> dict[str, FakeBackendClient]:
    return {
        "codex": FakeBackendClient("codex"),
        "claude_code": FakeBackendClient("claude_code"),
        "openbase_cloud": FakeBackendClient("openbase_cloud"),
        "openbase_cloud_codex": FakeBackendClient("openbase_cloud_codex"),
    }


def make_client(
    tmp_path: Path,
    clients: dict[str, FakeBackendClient],
    default: str = "codex",
) -> MultiBackendClient:
    return MultiBackendClient(
        default_backend=default,
        clients=clients,
        provenance_store=BackendProvenanceStore(tmp_path / "provenance.json"),
    )


@pytest.mark.asyncio
async def test_launch_default_override_and_all_configured_identities_are_preserved(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    clients: dict[str, FakeBackendClient],
) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_BACKEND", "claude_code")
    client = MultiBackendClient(
        clients=clients,
        provenance_store=BackendProvenanceStore(tmp_path / "provenance.json"),
    )

    default_result = await client.start_thread({"name": "default"})
    cloud_result = await client.start_thread({"name": "cloud", "backend": "openbase_cloud"})
    cloud_codex_result = await client.start_thread({"name": "cloud-codex", "backend": "openbase_cloud_codex"})

    assert default_result["backend"] == "claude_code"
    assert cloud_result["backend"] == "openbase_cloud"
    assert cloud_codex_result["backend"] == "openbase_cloud_codex"
    assert clients["claude_code"].calls[0][1] == {"name": "default"}
    assert clients["openbase_cloud"].calls[0][1] == {"name": "cloud"}
    assert clients["openbase_cloud_codex"].calls[0][1] == {"name": "cloud-codex"}


@pytest.mark.asyncio
async def test_local_claude_and_codex_coexist_and_route_every_follow_up_by_id(
    tmp_path: Path,
    clients: dict[str, FakeBackendClient],
) -> None:
    client = make_client(tmp_path, clients)
    codex = await client.start_thread({"name": "codex-agent"})
    claude = await client.start_thread({"name": "claude-agent", "backend": "claude_code"})

    query = LabelQueryInput(thread_id=claude["threadId"])
    await client.read_by_label(query)
    await client.resume_by_label(query)
    await client.progress_by_label(query)
    await client.steer_by_label(query, "steer")
    turn = await client.start_turn_by_label(query, {"prompt": "turn"})
    queued = await client.queue_turn_by_label(query, {"prompt": "queued"})
    await client.cancel_by_label(query)
    await client.compact_status(query)
    await client.cancel_queued_turn(
        QueueCancelInput(
            queue_item_id=queued["turnId"],
            thread_id=claude["threadId"],
        )
    )

    assert client.provenance.backend_for_thread(codex["threadId"]) == "codex"
    assert client.provenance.backend_for_thread(claude["threadId"]) == "claude_code"
    assert client.provenance.backend_for_turn(turn["turnId"]) == "claude_code"
    routed_methods = {name for name, _payload in clients["claude_code"].calls}
    assert {
        "read_by_label",
        "resume_by_label",
        "progress_by_label",
        "steer_by_label",
        "start_turn_by_label",
        "queue_turn_by_label",
        "cancel_by_label",
        "compact_status",
        "cancel_queued_turn",
    } <= routed_methods
    codex_methods = {name for name, _payload in clients["codex"].calls}
    assert not routed_methods.intersection(codex_methods - {"start_thread", "sessions"})

    with pytest.raises(BackendResolutionError, match="belongs to claude_code"):
        await client.read_by_label(LabelQueryInput(thread_id=claude["threadId"], backend="codex"))


@pytest.mark.asyncio
async def test_restart_uses_persisted_provenance_without_wrong_backend_probe(
    tmp_path: Path,
    clients: dict[str, FakeBackendClient],
) -> None:
    first = make_client(tmp_path, clients)
    launched = await first.start_thread({"name": "persistent", "backend": "claude_code"})
    restarted_clients = {
        "codex": FakeBackendClient("codex"),
        "claude_code": clients["claude_code"],
    }
    restarted = make_client(tmp_path, restarted_clients)

    result = await restarted.read_by_label(LabelQueryInput(thread_id=launched["threadId"]))

    assert result["backend"] == "claude_code"
    assert not any(name == "read_by_label" for name, _payload in restarted_clients["codex"].calls)


@pytest.mark.asyncio
async def test_duplicate_name_requires_backend_or_authoritative_id(
    tmp_path: Path,
    clients: dict[str, FakeBackendClient],
) -> None:
    clients["codex"].add_session("codex-shared", "shared")
    clients["claude_code"].add_session("claude-shared", "shared")
    client = make_client(tmp_path, clients)

    with pytest.raises(AmbiguousBackendError, match="Provide backend or an authoritative threadId"):
        await client.read_by_label(LabelQueryInput(label="shared"))

    explicit = await client.read_by_label(LabelQueryInput(label="shared", backend="claude_code"))
    by_id = await client.read_by_label(LabelQueryInput(thread_id="codex-shared"))

    assert explicit["backend"] == "claude_code"
    assert by_id["backend"] == "codex"


@pytest.mark.asyncio
async def test_approval_request_routes_by_persisted_owner_after_status(
    tmp_path: Path,
    clients: dict[str, FakeBackendClient],
) -> None:
    clients["claude_code"].pending = [{"id": "approval-1", "method": "requestApproval"}]
    client = make_client(tmp_path, clients)
    await client.start_thread({"name": "claude", "backend": "claude_code"})

    status = await client.status()
    result = await client.answer_request("approval-1", {"decision": "accept"})

    assert any(item["backend"] == "claude_code" for item in status["pendingRequests"])
    assert result["backend"] == "claude_code"
    assert client.provenance.backend_for_request("approval-1") == "claude_code"
    assert not any(name == "answer_request" for name, _payload in clients["codex"].calls)


@pytest.mark.asyncio
async def test_colliding_approval_ids_require_explicit_backend(
    tmp_path: Path,
    clients: dict[str, FakeBackendClient],
) -> None:
    request = {"id": "shared-approval", "method": "requestApproval"}
    clients["codex"].pending = [request]
    clients["claude_code"].pending = [request]
    client = make_client(tmp_path, clients)
    await client.start_thread({"name": "claude", "backend": "claude_code"})
    await client.status()

    with pytest.raises(AmbiguousBackendError, match="Provide backend"):
        await client.answer_request("shared-approval", {"decision": "accept"})

    result = await client.answer_request(
        "shared-approval",
        {"decision": "accept"},
        backend="claude_code",
    )

    assert result["backend"] == "claude_code"
    assert client.provenance.backends_for_request("shared-approval") == {"codex", "claude_code"}
    assert not any(name == "answer_request" for name, _payload in clients["codex"].calls)


@pytest.mark.asyncio
async def test_legacy_session_without_backend_is_claimed_by_safe_default(
    tmp_path: Path,
    clients: dict[str, FakeBackendClient],
) -> None:
    clients["openbase_cloud"].add_session("legacy-thread", "legacy")
    client = make_client(tmp_path, clients, default="openbase_cloud")

    result = await client.read_by_label(LabelQueryInput(label="legacy"))

    assert result["backend"] == "openbase_cloud"
    assert client.provenance.backend_for_thread("legacy-thread") == "openbase_cloud"


def test_mcp_backend_schema_and_cleaners_are_additive(
    tmp_path: Path,
    clients: dict[str, FakeBackendClient],
) -> None:
    client = make_client(tmp_path, clients)
    start = next(tool for tool in build_tools(client) if tool.name == "super_agents_start")
    read = next(tool for tool in build_tools(client) if tool.name == "super_agents_read")

    assert start.input_schema["properties"]["backend"]["enum"] == [
        "claude_code",
        "codex",
        "openbase_cloud",
        "openbase_cloud_codex",
    ]
    assert "backend" in read.input_schema["properties"]
    answer = next(tool for tool in build_tools(client) if tool.name == "codex_answer_request")
    assert "backend" in answer.input_schema["properties"]
    assert clean_thread_input({"name": "agent", "backend": "Openbase Cloud"})["backend"] == "openbase_cloud"
    assert clean_name_query_input({"name": "agent", "backend": "Claude Code"}).backend == "claude_code"


def test_same_codex_endpoint_is_rejected_but_distinct_endpoints_are_supported(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_CODE_HOME", str(tmp_path / "claude"))
    monkeypatch.setenv("SUPER_AGENTS_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setenv("SUPER_AGENTS_WS_URL", "ws://127.0.0.1:4500")
    first = MultiBackendClient(
        default_backend="codex",
        provenance_store=BackendProvenanceStore(tmp_path / "first-provenance.json"),
    )

    with pytest.raises(BackendEndpointConflictError, match="target the same Codex app-server endpoint"):
        first.client_for("openbase_cloud_codex")

    monkeypatch.setenv("SUPER_AGENTS_CODEX_WS_URL", "ws://127.0.0.1:4500")
    monkeypatch.setenv("SUPER_AGENTS_OPENBASE_CLOUD_CODEX_WS_URL", "ws://127.0.0.1:4600")
    second = MultiBackendClient(
        default_backend="codex",
        provenance_store=BackendProvenanceStore(tmp_path / "second-provenance.json"),
    )

    assert second.client_for("codex").ws_url == "ws://127.0.0.1:4500"
    assert second.client_for("openbase_cloud_codex").ws_url == "ws://127.0.0.1:4600"
