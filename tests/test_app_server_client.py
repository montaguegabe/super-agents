from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Callable

import pytest
import websockets

from super_agents.app_server_client import CodexAppServerClient
from super_agents.mcp_server import build_tools


class ReadyClient(CodexAppServerClient):
    async def check_ready(self) -> bool:
        return True


@pytest.mark.asyncio
async def test_steer_turn_sends_expected_turn_id(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    server = await start_fake_app_server(captured)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.steer_turn("thread-1", "turn-1", "narrow the scope")
        steer_request = next(message for message in captured if message.get("method") == "turn/steer")
        assert steer_request["params"] == {
            "threadId": "thread-1",
            "expectedTurnId": "turn-1",
            "input": [{"type": "text", "text": "narrow the scope"}],
        }
        assert "turnId" not in steer_request["params"]
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_label_tools_resolve_latest_active_session_and_list_active_agents(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    thread_index = 0
    turn_index = 0

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        nonlocal thread_index, turn_index
        if message.get("method") == "thread/start":
            thread_index += 1
            return {"threadId": f"thread-{thread_index}", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            turn_index += 1
            return {"turnId": f"turn-{turn_index}", "text": f"started {turn_index}"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "build", "cwd": "/tmp/one"})
        await client.start_turn({"threadId": "thread-1", "label": "build", "prompt": "first"})
        await asyncio.sleep(0.005)
        await client.start_thread({"label": "build", "cwd": "/tmp/two"})
        await client.start_turn({"threadId": "thread-2", "label": "build", "prompt": "second"})

        resolved = await client.resolve_label(type_query(label="build"))
        assert resolved["threadId"] == "thread-2"
        assert resolved["turnId"] == "turn-2"
        assert resolved["status"] == "running"

        active = await client.active(type_query(label="build"))
        assert active["count"] == 2
        assert active["agents"][0]["threadId"] == "thread-2"
        assert active["agents"][0]["runningTurnId"] == "turn-2"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_progress_by_label_reuses_resolved_thread_and_turn_ids(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-progress", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-progress"}
        if message.get("method") == "thread/read":
            return {
                "threadId": message["params"]["threadId"],
                "turns": [{"id": "turn-progress", "status": "inProgress", "message": "still working"}],
            }
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "progress"})
        await client.start_turn({"threadId": "thread-progress", "label": "progress", "prompt": "work"})
        progress = await client.progress_by_label(type_query(label="progress"))

        assert progress["threadId"] == "thread-progress"
        assert progress["turnId"] == "turn-progress"
        assert progress["status"] == "running"
        assert any(
            message.get("method") == "thread/read" and message["params"]["threadId"] == "thread-progress"
            for message in captured
        )
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_steer_and_cancel_by_label_call_existing_id_based_app_server_methods(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-control", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-control"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "control"})
        await client.start_turn({"threadId": "thread-control", "label": "control", "prompt": "work"})
        await client.steer_by_label(type_query(label="control"), "adjust")
        await client.cancel_by_label(type_query(label="control"))

        steer_request = next(message for message in captured if message.get("method") == "turn/steer")
        assert steer_request["params"] == {
            "threadId": "thread-control",
            "expectedTurnId": "turn-control",
            "input": [{"type": "text", "text": "adjust"}],
        }
        cancel_request = next(message for message in captured if message.get("method") == "turn/interrupt")
        assert cancel_request["params"] == {"threadId": "thread-control", "turnId": "turn-control"}

        active = await client.active(type_query(label="control"))
        assert active["count"] == 0
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_start_turn_by_label_can_start_follow_up_on_latest_inactive_label(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-follow-up", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-follow-up"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "follow-up", "cwd": "/tmp/project"})
        result = await client.start_turn_by_label(type_query(label="follow-up"), {"label": "follow-up", "prompt": "continue"})
        assert result["threadId"] == "thread-follow-up"
        assert result["turnId"] == "turn-follow-up"
        start_request = next(message for message in captured if message.get("method") == "turn/start")
        assert start_request["params"]["threadId"] == "thread-follow-up"
        assert start_request["params"]["cwd"] == "/tmp/project"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_old_state_files_are_tolerated_by_recent_and_label_resolution(tmp_path: Path) -> None:
    state_file = tmp_path / "state.json"
    state_file.write_text(
        json.dumps(
            {
                "sessions": {
                    "thread-old": {
                        "label": "old",
                        "threadId": "thread-old",
                        "cwd": "/tmp/old",
                        "model": "gpt-test",
                        "lastTurnId": "turn-old",
                        "updatedAt": "2026-01-01T00:00:00.000Z",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    client = ReadyClient("ws://127.0.0.1:1", state_file, "gpt-test")

    recent = await client.recent(type_query(label="old", include_inactive=True))
    assert recent["count"] == 1
    assert recent["agents"][0]["threadId"] == "thread-old"

    resolved = await client.resolve_label(type_query(label="old", prefer="latest_any"))
    assert resolved["threadId"] == "thread-old"
    assert resolved["turnId"] == "turn-old"
    assert resolved["status"] == "unknown"


@pytest.mark.asyncio
async def test_session_state_is_enriched_with_turn_metadata_compatibly(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-state", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-state", "text": "started state work"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler)
    state_file = tmp_path / "state.json"
    client = ReadyClient(server.ws_url, state_file, "gpt-test")
    try:
        await client.start_thread({"label": "state", "group": "batch", "cwd": "/tmp/state"})
        await client.start_turn({"threadId": "thread-state", "label": "state", "group": "batch", "prompt": "state prompt"})

        state = json.loads(state_file.read_text(encoding="utf-8"))
        assert state["sessions"]["thread-state"]["label"] == "state"
        assert state["sessions"]["thread-state"]["group"] == "batch"
        assert state["sessions"]["thread-state"]["activeTurnId"] == "turn-state"
        assert state["sessions"]["thread-state"]["turns"]["turn-state"]["promptPreview"] == "state prompt"
    finally:
        await client.close()
        await server.close()


@pytest.mark.asyncio
async def test_callback_requests_are_recorded_and_answered(tmp_path: Path) -> None:
    captured: list[dict[str, Any]] = []
    server_request_id = "request-1"

    async def after_message(message: dict[str, Any], websocket: Any) -> None:
        if message.get("method") == "turn/start":
            await websocket.send(
                json.dumps(
                    {
                        "id": server_request_id,
                        "method": "plan/question",
                        "params": {"threadId": "thread-callback", "turnId": "turn-callback", "text": "Choose"},
                    }
                )
            )

    def handler(message: dict[str, Any]) -> dict[str, Any]:
        if message.get("method") == "thread/start":
            return {"threadId": "thread-callback", "cwd": message["params"]["cwd"], "model": "gpt-test"}
        if message.get("method") == "turn/start":
            return {"turnId": "turn-callback"}
        return {"ok": True}

    server = await start_fake_app_server(captured, handler, after_message)
    client = ReadyClient(server.ws_url, tmp_path / "state.json", "gpt-test")
    try:
        await client.start_thread({"label": "callback"})
        await client.start_turn({"threadId": "thread-callback", "label": "callback", "prompt": "work"})
        await asyncio.sleep(0.05)

        status = await client.status()
        assert status["pendingRequests"][0]["id"] == server_request_id
        assert status["activeTurns"][0]["status"] == "waiting"

        answered = await client.answer_request(server_request_id, {"decision": "accept"})
        await asyncio.sleep(0.05)
        assert answered["answered"] is True
        assert captured[-1] == {"id": server_request_id, "result": {"decision": "accept"}}
    finally:
        await client.close()
        await server.close()


def test_tool_surface_preserves_current_names_and_schemas() -> None:
    tools = build_tools(CodexAppServerClient("ws://127.0.0.1:1"))
    names = [tool.name for tool in tools]
    assert names == [
        "codex_app_server_status",
        "codex_thread_start",
        "codex_thread_resume",
        "codex_thread_list",
        "codex_thread_read",
        "codex_turn_start",
        "codex_turn_progress",
        "codex_turn_steer",
        "codex_turn_cancel",
        "codex_answer_request",
        "super_agents_sessions",
        "super_agents_active",
        "super_agents_resolve",
        "super_agents_progress",
        "super_agents_steer",
        "super_agents_cancel",
        "super_agents_start_turn",
        "super_agents_recent",
    ]
    by_name = {tool.name: tool for tool in tools}
    assert by_name["codex_turn_start"].input_schema["required"] == ["threadId", "prompt"]
    assert by_name["super_agents_active"].annotations == {"readOnlyHint": True, "idempotentHint": True}
    assert by_name["codex_answer_request"].input_schema["required"] == ["requestId", "result"]


class FakeServer:
    def __init__(self, server: Any, ws_url: str) -> None:
        self.server = server
        self.ws_url = ws_url

    async def close(self) -> None:
        self.server.close()
        await self.server.wait_closed()


async def start_fake_app_server(
    captured: list[dict[str, Any]],
    handler: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    after_message: Callable[[dict[str, Any], Any], Any] | None = None,
) -> FakeServer:
    def default_handler(_message: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}

    response_handler = handler or default_handler

    async def ws_handler(websocket: Any) -> None:
        async for raw in websocket:
            message = json.loads(raw)
            captured.append(message)
            if "id" not in message:
                continue
            await websocket.send(json.dumps({"id": message["id"], "result": response_handler(message)}))
            if after_message:
                result = after_message(message, websocket)
                if hasattr(result, "__await__"):
                    await result

    server = await websockets.serve(ws_handler, "127.0.0.1", 0)
    socket = server.sockets[0]
    host, port = socket.getsockname()[:2]
    return FakeServer(server, f"ws://{host}:{port}")


def type_query(**kwargs: Any) -> Any:
    from super_agents.app_server_client import LabelQueryInput

    return LabelQueryInput(**kwargs)
