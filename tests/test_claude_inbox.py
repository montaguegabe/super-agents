"""Tests for the Claude Code inbox-socket steering client.

The fake server here mirrors Claude Code's inbox wire contract (optional auth
line, one message frame, session_id gating, EOF-on-reject) so these tests also
pin the frame format the client emits: if the shape drifts, the server stops
parsing it and the assertions fail.
"""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from super_agents.claude_inbox import (
    InboxRecord,
    deliver_steer,
    forget_inbox,
    inbox_registry_dir,
    resolve_inbox,
)


@pytest.fixture
def short_sock_dir():
    """A short-pathed temp dir for AF_UNIX sockets.

    macOS caps a Unix socket path at ~104 chars; the worktree ``tmp_path`` is
    already longer than that, so server sockets must bind under a short root.
    """
    with tempfile.TemporaryDirectory(prefix="cci-", dir="/tmp") as d:
        yield Path(d)


class FakeInboxServer:
    """Minimal stand-in for a Claude Code session's inbox socket.

    Accepts a connection, reads newline-delimited JSON, records the frames, and
    either keeps the connection open (accepted) or closes it immediately
    (rejected) to mimic auth/session-mismatch handling.
    """

    def __init__(self, path: Path, *, expect_token: str | None, session_id: str) -> None:
        self._path = path
        self._expect_token = expect_token
        self._session_id = session_id
        self.received: list[dict] = []
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_unix_server(self._handle, path=str(self._path))

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            frames: list[dict] = []
            # Read up to two lines (auth + message) with a short grace period.
            for _ in range(2):
                try:
                    line = await asyncio.wait_for(reader.readline(), timeout=0.5)
                except TimeoutError:
                    break
                if not line:
                    break
                try:
                    frames.append(json.loads(line))
                except ValueError:
                    self._reject(writer)
                    return
                if frames[-1].get("type") == "user":
                    break
            self.received.extend(frames)
            auth = next((f for f in frames if f.get("type") == "auth"), None)
            msg = next((f for f in frames if f.get("type") == "user"), None)
            if self._expect_token is not None and (auth is None or auth.get("token") != self._expect_token):
                self._reject(writer)
                return
            if msg is None or msg.get("session_id") != self._session_id:
                self._reject(writer)
                return
            # Accepted: hold the connection open briefly like the real server.
            await asyncio.sleep(0.3)
        finally:
            with pytest_suppress():
                writer.close()

    def _reject(self, writer: asyncio.StreamWriter) -> None:
        with pytest_suppress():
            writer.write_eof()
            writer.close()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            with pytest_suppress():
                await self._server.wait_closed()


class pytest_suppress:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return True


@pytest.mark.asyncio
async def test_deliver_steer_writes_expected_frame(short_sock_dir: Path) -> None:
    sock = short_sock_dir / "sess.sock"
    server = FakeInboxServer(sock, expect_token="tok123", session_id="claude-abc")
    await server.start()
    try:
        record = InboxRecord(session_id="claude-abc", socket=str(sock), token="tok123")
        result = await deliver_steer(
            record,
            "stop and refactor instead",
            target_session_id="claude-abc",
            from_name="openbase",
            priority="now",
        )
    finally:
        await server.stop()

    assert result.written is True
    assert result.reason is None
    # Auth line first, then the user frame, in order.
    assert server.received[0] == {"type": "auth", "token": "tok123"}
    msg = server.received[1]
    assert msg["type"] == "user"
    assert msg["session_id"] == "claude-abc"
    assert msg["priority"] == "now"
    assert msg["from"] == "openbase"
    assert msg["message"] == {"role": "user", "content": "stop and refactor instead"}
    assert isinstance(msg["msg_id"], str) and msg["msg_id"]


@pytest.mark.asyncio
async def test_deliver_steer_without_token_sends_no_auth_line(short_sock_dir: Path) -> None:
    sock = short_sock_dir / "sess.sock"
    server = FakeInboxServer(sock, expect_token=None, session_id="claude-abc")
    await server.start()
    try:
        record = InboxRecord(session_id="claude-abc", socket=str(sock), token=None)
        result = await deliver_steer(record, "go", target_session_id="claude-abc")
    finally:
        await server.stop()

    assert result.written is True
    assert all(f.get("type") != "auth" for f in server.received)


@pytest.mark.asyncio
async def test_deliver_steer_session_mismatch_is_reported_as_rejected(short_sock_dir: Path) -> None:
    sock = short_sock_dir / "sess.sock"
    server = FakeInboxServer(sock, expect_token=None, session_id="the-real-one")
    await server.start()
    try:
        record = InboxRecord(session_id="wrong-id", socket=str(sock), token=None)
        result = await deliver_steer(record, "go", target_session_id="wrong-id")
    finally:
        await server.stop()

    assert result.written is False
    assert result.reason == "rejected_by_peer"


@pytest.mark.asyncio
async def test_deliver_steer_dead_socket_is_unreachable(tmp_path: Path) -> None:
    record = InboxRecord(session_id="x", socket=str(tmp_path / "nope.sock"), token=None)
    result = await deliver_steer(record, "go", target_session_id="x")
    assert result.written is False
    assert result.reason == "socket_unreachable"


@pytest.mark.asyncio
async def test_deliver_steer_empty_text_short_circuits(tmp_path: Path) -> None:
    record = InboxRecord(session_id="x", socket=str(tmp_path / "unused.sock"), token=None)
    result = await deliver_steer(record, "", target_session_id="x")
    assert result.written is False
    assert result.reason == "empty_text"


def test_resolve_inbox_reads_hook_written_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    registry = tmp_path / "inbox-registry"
    registry.mkdir()
    monkeypatch.setenv("CLAUDE_INBOX_REGISTRY_DIR", str(registry))
    (registry / "sid-1.json").write_text(
        json.dumps(
            {
                "sessionId": "sid-1",
                "socket": "/tmp/cc-socks/123.sock",
                "token": "abc",
                "cwd": "/work",
                "recordedAt": 1789000000.0,
            }
        )
    )
    assert inbox_registry_dir() == registry
    record = resolve_inbox("sid-1")
    assert record is not None
    assert record.socket == "/tmp/cc-socks/123.sock"
    assert record.token == "abc"
    assert record.cwd == "/work"


def test_resolve_inbox_missing_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_INBOX_REGISTRY_DIR", str(tmp_path))
    assert resolve_inbox("absent") is None
    assert resolve_inbox("") is None


def test_resolve_inbox_malformed_record_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_INBOX_REGISTRY_DIR", str(tmp_path))
    (tmp_path / "bad.json").write_text("{not json")
    assert resolve_inbox("bad") is None
    # A record without a socket is unusable.
    (tmp_path / "nosock.json").write_text(json.dumps({"sessionId": "nosock"}))
    assert resolve_inbox("nosock") is None


def test_forget_inbox_removes_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_INBOX_REGISTRY_DIR", str(tmp_path))
    (tmp_path / "gone.json").write_text(json.dumps({"sessionId": "gone", "socket": "/x.sock"}))
    assert resolve_inbox("gone") is not None
    forget_inbox("gone")
    assert resolve_inbox("gone") is None
    # Idempotent.
    forget_inbox("gone")


def test_registry_dir_defaults_into_claude_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CLAUDE_INBOX_REGISTRY_DIR", raising=False)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / ".claude"))
    assert inbox_registry_dir() == tmp_path / ".claude" / "inbox-registry"
