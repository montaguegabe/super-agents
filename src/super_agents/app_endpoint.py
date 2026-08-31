from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlsplit

import websockets

DEFAULT_WEBSOCKET_ENDPOINT = "ws://127.0.0.1:4500"
DEFAULT_UNIX_HANDSHAKE_URI = "ws://localhost/rpc"
STANDARD_SOCKET_RELATIVE_PATH = Path("app-server-control") / "app-server-control.sock"
ENDPOINT_ENV = "SUPER_AGENTS_APP_SERVER_ENDPOINT"
LEGACY_ENDPOINT_ENVS = ("SUPER_AGENTS_WS_URL", "CODEX_APP_SERVER_URL")


class AppServerEndpointError(ValueError):
    """Raised when app-server endpoint configuration is invalid or ambiguous."""


@dataclass(frozen=True)
class AppServerEndpoint:
    value: str
    transport: str
    source: str
    socket_path: Path | None = None
    handshake_uri: str | None = None

    @property
    def is_unix(self) -> bool:
        return self.transport == "unix"

    @property
    def description(self) -> str:
        if self.socket_path is None:
            return self.value
        return _display_path(self.socket_path)


def codex_home(env: dict[str, str] | None = None) -> Path:
    values = env if env is not None else os.environ
    configured = values.get("CODEX_HOME", "").strip()
    return Path(configured).expanduser() if configured else Path.home() / ".codex"


def standard_app_server_socket(env: dict[str, str] | None = None) -> Path:
    return codex_home(env) / STANDARD_SOCKET_RELATIVE_PATH


def default_app_server_endpoint(*, platform: str | None = None) -> str:
    current_platform = platform or sys.platform
    return "unix://" if current_platform != "win32" else DEFAULT_WEBSOCKET_ENDPOINT


def _display_path(path: Path) -> str:
    expanded = path.expanduser()
    try:
        relative = expanded.relative_to(Path.home())
    except ValueError:
        return str(expanded)
    return str(Path("~") / relative)


def parse_app_server_endpoint(
    value: str,
    *,
    env: dict[str, str] | None = None,
    source: str = "explicit",
) -> AppServerEndpoint:
    configured = value.strip()
    if configured in {"ws://", "wss://", "unix:"}:
        raise AppServerEndpointError(f"Invalid Codex app-server endpoint: {configured!r}.")
    if configured.startswith(("ws://", "wss://")):
        parsed = urlsplit(configured)
        if not parsed.hostname:
            raise AppServerEndpointError(f"Invalid Codex app-server endpoint: {configured!r}.")
        return AppServerEndpoint(
            value=configured,
            transport="websocket",
            source=source,
        )
    if configured == "unix://":
        path = standard_app_server_socket(env)
        return AppServerEndpoint(
            value="unix://",
            transport="unix",
            source=source,
            socket_path=path,
            handshake_uri=DEFAULT_UNIX_HANDSHAKE_URI,
        )
    if configured.startswith("unix://"):
        raw_path = unquote(configured.removeprefix("unix://"))
        if not raw_path.startswith("/"):
            raise AppServerEndpointError(
                "An explicit unix:// Codex app-server endpoint must use an absolute socket path."
            )
        path = Path(raw_path)
        return AppServerEndpoint(
            value=f"unix://{path}",
            transport="unix",
            source=source,
            socket_path=path,
            handshake_uri=DEFAULT_UNIX_HANDSHAKE_URI,
        )
    raise AppServerEndpointError("Codex app-server endpoint must use ws://, wss://, or unix://.")


def configured_app_server_endpoint(
    explicit: str | None = None,
    *,
    env: dict[str, str] | None = None,
    platform: str | None = None,
) -> AppServerEndpoint:
    values = env if env is not None else os.environ
    if explicit and explicit.strip():
        return parse_app_server_endpoint(explicit, env=values, source="explicit")

    configured = [
        (key, values[key].strip()) for key in (ENDPOINT_ENV, *LEGACY_ENDPOINT_ENVS) if values.get(key, "").strip()
    ]
    distinct = {value for _key, value in configured}
    if len(distinct) > 1:
        detail = ", ".join(f"{key}={value}" for key, value in configured)
        raise AppServerEndpointError(
            f"Conflicting Codex app-server endpoints select different owners: {detail}. Configure exactly one endpoint."
        )
    if configured:
        key, value = configured[0]
        return parse_app_server_endpoint(value, env=values, source=key)

    default = default_app_server_endpoint(platform=platform)
    source = "standard" if default == "unix://" else "platform-default"
    return parse_app_server_endpoint(default, env=values, source=source)


async def open_app_server_connection(
    endpoint: AppServerEndpoint,
    **kwargs: object,
):
    if endpoint.is_unix:
        assert endpoint.socket_path is not None
        # Codex's local control socket uses tungstenite without WebSocket
        # extension negotiation. websockets enables per-message deflate by
        # default, which Codex 0.151 rejects during the HTTP Upgrade.
        kwargs.setdefault("compression", None)
        return await websockets.unix_connect(
            str(endpoint.socket_path),
            uri=endpoint.handshake_uri,
            **kwargs,
        )
    return await websockets.connect(endpoint.value, **kwargs)
