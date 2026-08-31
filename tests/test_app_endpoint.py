from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any

import pytest
import websockets
from websockets.exceptions import InvalidHandshake

import super_agents.app_endpoint as app_endpoint
from super_agents.app_endpoint import (
    DEFAULT_UNIX_HANDSHAKE_URI,
    AppServerEndpointError,
    configured_app_server_endpoint,
    open_app_server_connection,
    parse_app_server_endpoint,
    standard_app_server_socket,
)
from super_agents.app_server_client import CodexAppServerClient


def _short_socket_path() -> Path:
    return Path("/tmp") / f"sa-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"


def test_default_endpoint_prefers_standard_unix_socket(tmp_path: Path) -> None:
    env = {"CODEX_HOME": str(tmp_path / "codex-home")}

    endpoint = configured_app_server_endpoint(env=env, platform="darwin")

    assert endpoint.transport == "unix"
    assert endpoint.source == "standard"
    assert endpoint.value == "unix://"
    assert endpoint.socket_path == standard_app_server_socket(env)
    assert endpoint.handshake_uri == DEFAULT_UNIX_HANDSHAKE_URI


def test_explicit_unix_and_websocket_endpoints_are_distinct(tmp_path: Path) -> None:
    socket_path = tmp_path / "control.sock"

    explicit_unix = parse_app_server_endpoint(f"unix://{socket_path}")
    websocket = parse_app_server_endpoint("wss://codex.example/rpc")

    assert explicit_unix.transport == "unix"
    assert explicit_unix.socket_path == socket_path
    assert websocket.transport == "websocket"
    assert websocket.value == "wss://codex.example/rpc"


def test_legacy_websocket_default_remains_the_windows_fallback() -> None:
    endpoint = configured_app_server_endpoint(env={}, platform="win32")

    assert endpoint.transport == "websocket"
    assert endpoint.value == "ws://127.0.0.1:4500"
    assert endpoint.source == "platform-default"


def test_environment_rejects_multiple_owner_endpoints() -> None:
    with pytest.raises(AppServerEndpointError, match="select different owners"):
        configured_app_server_endpoint(
            env={
                "SUPER_AGENTS_APP_SERVER_ENDPOINT": "unix://",
                "CODEX_APP_SERVER_URL": "ws://127.0.0.1:4500",
            },
            platform="darwin",
        )


@pytest.mark.asyncio
async def test_unix_connector_receives_path_uri_size_and_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str | None, dict[str, object]]] = []

    class FakeConnection:
        async def send(self, _payload: str) -> None:
            return None

        async def recv(self) -> str:
            return json.dumps({"id": 0, "result": {}})

        async def close(self) -> None:
            return None

    async def fake_unix_connect(
        path: str,
        uri: str | None = None,
        **kwargs: object,
    ) -> FakeConnection:
        calls.append((path, uri, kwargs))
        return FakeConnection()

    monkeypatch.setattr(app_endpoint.websockets, "unix_connect", fake_unix_connect)
    client = CodexAppServerClient(
        endpoint=f"unix://{tmp_path / 'control.sock'}",
        state_file=tmp_path / "state.json",
    )

    assert await client.check_ready() is True
    assert calls == [
        (
            str(tmp_path / "control.sock"),
            DEFAULT_UNIX_HANDSHAKE_URI,
            {
                "max_size": 16 * 1024 * 1024,
                "open_timeout": 5,
                "compression": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_tcp_connector_keeps_the_configured_websocket_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    connection = object()

    async def fake_connect(uri: str, **kwargs: object) -> object:
        calls.append((uri, kwargs))
        return connection

    monkeypatch.setattr(app_endpoint.websockets, "connect", fake_connect)

    result = await open_app_server_connection(
        parse_app_server_endpoint("wss://codex.example/rpc"),
        max_size=123,
    )

    assert result is connection
    assert calls == [("wss://codex.example/rpc", {"max_size": 123})]


@pytest.mark.asyncio
async def test_unix_connection_and_readiness_complete_initialize_handshake(
    tmp_path: Path,
) -> None:
    socket_path = _short_socket_path()
    captured: list[dict[str, Any]] = []

    async def handler(websocket: Any) -> None:
        async for raw in websocket:
            message = json.loads(raw)
            captured.append(message)
            if message.get("method") == "initialize":
                await websocket.send(
                    json.dumps(
                        {
                            "id": message["id"],
                            "result": {"userAgent": "codex-app-server/test"},
                        }
                    )
                )

    server = await websockets.unix_serve(handler, str(socket_path))
    client = CodexAppServerClient(
        endpoint=f"unix://{socket_path}",
        state_file=tmp_path / "state.json",
    )
    try:
        assert await client.check_ready() is True
        await client.ensure_connected()

        status = await client.status()
        assert status["transport"] == "unix"
        assert status["websocketUrl"] is None
        assert status["websocketConnected"] is True
        assert status["appServerVersion"] == "codex-app-server/test"
        assert [message["method"] for message in captured].count("initialize") >= 2
        assert any(message.get("method") == "initialized" for message in captured)
    finally:
        await client.close()
        server.close()
        await server.wait_closed()
        socket_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_unix_client_reconnects_after_owner_disconnect_without_duplicate_reader(
    tmp_path: Path,
) -> None:
    socket_path = _short_socket_path()
    initialized_count = 0
    first_connection_closed = asyncio.Event()
    second_connection_initialized = asyncio.Event()

    async def handler(websocket: Any) -> None:
        nonlocal initialized_count
        async for raw in websocket:
            message = json.loads(raw)
            if message.get("method") == "initialize":
                await websocket.send(json.dumps({"id": message["id"], "result": {}}))
            elif message.get("method") == "initialized":
                initialized_count += 1
                if initialized_count == 1:
                    await websocket.close()
                    first_connection_closed.set()
                else:
                    second_connection_initialized.set()

    class ReadyUnixClient(CodexAppServerClient):
        async def check_ready(self) -> bool:
            return True

    server = await websockets.unix_serve(handler, str(socket_path))
    client = ReadyUnixClient(
        endpoint=f"unix://{socket_path}",
        state_file=tmp_path / "state.json",
    )
    try:
        await client.ensure_connected()
        await asyncio.wait_for(first_connection_closed.wait(), timeout=2)
        for _ in range(20):
            if client._ws is None:
                break
            await asyncio.sleep(0.01)

        assert client._ws is None
        assert client._reader_task is None

        await client.ensure_connected()
        await asyncio.wait_for(second_connection_initialized.wait(), timeout=2)

        assert initialized_count == 2
        assert client._reader_task is not None
        assert not client._reader_task.done()
    finally:
        await client.close()
        server.close()
        await server.wait_closed()
        socket_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_missing_unix_socket_has_actionable_diagnostic(tmp_path: Path) -> None:
    socket_path = _short_socket_path()
    client = CodexAppServerClient(
        endpoint=f"unix://{socket_path}",
        state_file=tmp_path / "state.json",
    )

    assert await client.check_ready() is False
    status = await client.status()
    assert "missing" in status["lastConnectionError"].lower()

    with pytest.raises(RuntimeError, match="Unix control socket is missing"):
        await client.ensure_connected()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (PermissionError(), "Permission denied opening Unix control socket"),
        (ConnectionRefusedError(), "exists but is not accepting connections"),
        (InvalidHandshake("invalid response"), "did not accept a WebSocket handshake"),
    ],
)
async def test_unix_readiness_failures_have_actionable_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    error: Exception,
    expected: str,
) -> None:
    async def fail_connect(*_args: object, **_kwargs: object) -> object:
        raise error

    monkeypatch.setattr(app_endpoint.websockets, "unix_connect", fail_connect)
    client = CodexAppServerClient(
        endpoint=f"unix://{tmp_path / 'control.sock'}",
        state_file=tmp_path / "state.json",
    )

    assert await client.check_ready() is False
    assert expected in (await client.status())["lastConnectionError"]
