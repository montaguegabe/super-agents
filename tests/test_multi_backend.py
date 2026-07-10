from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from super_agents.app_models import LabelQueryInput
from super_agents.mcp_server import clean_name_query_input, clean_thread_input
from super_agents.multi_backend import MultiBackendClient

JsonObject = dict[str, Any]


class FakeBackendClient:
    def __init__(self, backend: str, names: set[str] | None = None) -> None:
        self.backend = backend
        self.names = names or set()
        self.calls: list[tuple[str, Any]] = []

    async def start_thread(self, input_data: JsonObject) -> JsonObject:
        self.calls.append(("start_thread", input_data))
        self.names.add(str(input_data.get("name")))
        return {"threadId": f"{self.backend}-thread"}

    def _require(self, input_data: LabelQueryInput) -> None:
        if input_data.label not in self.names:
            raise KeyError(f"No session named {input_data.label}")

    async def read_by_label(self, input_data: LabelQueryInput, include_turns: bool = False) -> JsonObject:
        self._require(input_data)
        return {"backend": self.backend, "name": input_data.label}

    async def start_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject:
        self._require(input_data)
        self.calls.append(("start_turn_by_label", turn_input))
        return {"backend": self.backend, "turnId": f"{self.backend}-turn"}

    async def active(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        return {"backend": self.backend, "count": 1, "agents": [{"name": f"{self.backend}-agent"}]}

    async def status(self) -> JsonObject:
        return {"ready": True, "backend": self.backend, "activeTurns": [{"threadId": f"{self.backend}-t"}]}

    async def sessions(self) -> list[JsonObject]:
        return [{"name": name} for name in sorted(self.names)]

    async def answer_request(self, request_id: str | int, result: JsonObject) -> JsonObject:
        if self.backend != "claude_code":
            raise ValueError(f"No pending app-server request found for id {request_id}.")
        return {"answered": True, "backend": self.backend}


@pytest.fixture(autouse=True)
def isolated_backend_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUPER_AGENTS_STATE_FILE", str(tmp_path / "state.json"))
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_CODE_HOME", str(tmp_path / "claude-home"))
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(tmp_path / "missing-config.json"))


def multi_client(default: str = "codex") -> tuple[MultiBackendClient, FakeBackendClient, FakeBackendClient]:
    codex = FakeBackendClient("codex")
    claude = FakeBackendClient("claude_code")
    client = MultiBackendClient(default_backend=default, clients={"codex": codex, "claude_code": claude})
    return client, codex, claude


@pytest.mark.asyncio
async def test_start_thread_routes_by_backend_param() -> None:
    client, codex, claude = multi_client()

    default_result = await client.start_thread({"name": "on-default"})
    claude_result = await client.start_thread({"name": "on-claude", "backend": "claude code"})

    assert default_result == {"threadId": "codex-thread", "backend": "codex"}
    assert claude_result == {"threadId": "claude_code-thread", "backend": "claude_code"}
    assert codex.calls[0][1] == {"name": "on-default"}
    # The backend selector is consumed by the router, not forwarded.
    assert claude.calls[0][1] == {"name": "on-claude"}


@pytest.mark.asyncio
async def test_openbase_cloud_launch_uses_codex_execution_backend() -> None:
    client, codex, _claude = multi_client()

    result = await client.start_thread({"name": "cloud", "backend": "openbase_cloud"})

    assert result["backend"] == "codex"
    assert codex.calls


@pytest.mark.asyncio
async def test_label_ops_route_to_owning_backend() -> None:
    client, _codex, _claude = multi_client()
    await client.start_thread({"name": "claude-only", "backend": "claude_code"})

    result = await client.read_by_label(LabelQueryInput(label="claude-only"))

    assert result == {"backend": "claude_code", "name": "claude-only"}


@pytest.mark.asyncio
async def test_label_ops_honor_backend_pin() -> None:
    client, codex, claude = multi_client()
    codex.names.add("shared")
    claude.names.add("shared")

    pinned = await client.read_by_label(LabelQueryInput(label="shared", backend="claude_code"))
    unpinned = await client.read_by_label(LabelQueryInput(label="shared"))

    assert pinned["backend"] == "claude_code"
    assert unpinned["backend"] == "codex"


@pytest.mark.asyncio
async def test_label_ops_raise_first_error_when_missing_everywhere() -> None:
    client, _codex, _claude = multi_client()

    with pytest.raises(KeyError, match="missing"):
        await client.read_by_label(LabelQueryInput(label="missing"))


@pytest.mark.asyncio
async def test_turn_model_default_is_rewritten_for_the_routed_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(
        json.dumps(
            {
                "backend_models": {
                    "codex": {"super_agents": "gpt-5.5-codex"},
                    "claude_code": {"super_agents": "sonnet"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))
    client, codex, claude = multi_client()
    await client.start_thread({"name": "claude-only", "backend": "claude_code"})

    # The MCP layer defaulted the model for the configured (codex) backend.
    await client.start_turn_by_label(
        LabelQueryInput(label="claude-only"),
        {"prompt": "go", "model": "gpt-5.5-codex"},
    )

    assert claude.calls[-1] == ("start_turn_by_label", {"prompt": "go", "model": "sonnet"})
    # Explicit non-default models pass through untouched.
    await client.start_turn_by_label(
        LabelQueryInput(label="claude-only"),
        {"prompt": "go", "model": "opus"},
    )
    assert claude.calls[-1] == ("start_turn_by_label", {"prompt": "go", "model": "opus"})
    assert all(call[0] != "start_turn_by_label" for call in codex.calls)


@pytest.mark.asyncio
async def test_aggregations_merge_engaged_backends() -> None:
    client, _codex, _claude = multi_client()
    await client.start_thread({"name": "a"})
    await client.start_thread({"name": "b", "backend": "claude_code"})

    active = await client.active()
    status = await client.status()
    sessions = await client.sessions()

    assert active["count"] == 2
    assert {agent["backend"] for agent in active["agents"]} == {"codex", "claude_code"}
    assert status["engagedBackends"] == ["codex", "claude_code"]
    assert {item["backend"] for item in status["activeTurns"]} == {"codex", "claude_code"}
    assert {session["backend"] for session in sessions} == {"codex", "claude_code"}


@pytest.mark.asyncio
async def test_answer_request_falls_through_to_owning_backend() -> None:
    client, _codex, _claude = multi_client()
    await client.start_thread({"name": "b", "backend": "claude_code"})

    result = await client.answer_request("req-1", {"decision": "accept"})

    assert result == {"answered": True, "backend": "claude_code"}


def test_clean_thread_input_normalizes_backend_and_permission_mode() -> None:
    cleaned = clean_thread_input(
        {
            "name": "agent",
            "backend": "Claude Code",
            "permissionMode": "default",
        }
    )
    assert cleaned["backend"] == "claude_code"
    assert cleaned["permissionMode"] == "default"

    with pytest.raises(ValueError, match="Supported backends"):
        clean_thread_input({"name": "agent", "backend": "gemini"})


def test_clean_name_query_input_carries_backend_pin() -> None:
    query = clean_name_query_input({"name": "agent", "backend": "openbase cloud"})
    assert query.backend == "openbase_cloud"
    assert clean_name_query_input({"name": "agent"}).backend is None
