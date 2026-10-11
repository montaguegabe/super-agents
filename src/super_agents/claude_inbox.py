"""Deliver a steering message into a live Claude Code session's inbox socket.

Claude Code has no shared app-server daemon (unlike Codex), so a turn started
from a plain terminal runs in a process this library never launched and cannot
reach through the SDK. Claude Code's cross-session-messaging feature (v2.1.224+)
does expose one external delivery channel: each session binds a per-session
AF_UNIX *inbox socket* and, per the official docs, a script or hook may post a
message into it. A message posted this way is read by the receiving Claude
*between tool calls during the active turn* — i.e. the same steering semantics
Codex gives through ``turn/steer``.

This module is the client for that channel. It is deliberately generic (no
Openbase-specific paths or domain concepts) so it stays usable in standalone
super-agents:

* The socket path and per-session auth token are exported by Claude Code only
  to the session's own hooks/children (``CLAUDE_CODE_MESSAGING_SOCKET`` /
  ``CLAUDE_CODE_MESSAGING_TOKEN``) and there is no discoverable registry file,
  so an out-of-process steerer needs them recorded at SessionStart. This module
  reads that recording from a registry directory (one JSON file per session);
  writing it is the integrator's job (Openbase ships a SessionStart hook).
* The registry lives in a generic location inside the Claude home
  (``$CLAUDE_CONFIG_DIR/inbox-registry`` or ``~/.claude/inbox-registry``),
  overridable with ``CLAUDE_INBOX_REGISTRY_DIR``.

Wire format (line-delimited JSON over the socket), pinned empirically against
claude 2.1.270 and stable across the 2.1.x series the registry targets:

* optional first line (required on Windows, optional on macOS/Linux)::

      {"type": "auth", "token": "<CLAUDE_CODE_MESSAGING_TOKEN>"}

* one message line::

      {"type": "user", "from": "<name>", "priority": "now",
       "msg_id": "<id>", "session_id": "<target session id>",
       "message": {"role": "user", "content": "<text>"}}

  ``session_id`` must equal the target session's id or the frame is dropped.

Delivery is best-effort and cannot be *synchronously* confirmed: whether the
receiving Claude actually surfaces the message depends on its inbound policy
(a ``bypassPermissions`` session holds an unattested external message unless
``crossSessionInbound`` is ``accept``). The caller acks on a successful socket
write, not on delivery, and handles the held case out of band.
"""

from __future__ import annotations

import asyncio
import contextlib
import errno
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from super_agents.claude_transcript import claude_config_dir

logger = logging.getLogger(__name__)

# Env override for the registry directory; otherwise a generic location inside
# the Claude home, matching how claude_config_dir() resolves the shared home.
INBOX_REGISTRY_DIR_ENV = "CLAUDE_INBOX_REGISTRY_DIR"
_REGISTRY_DIR_NAME = "inbox-registry"

# Claude Code closes a connection that hasn't sent a complete line within its
# first-line deadline (~30s); we only connect with the payload ready, so a
# short timeout is plenty and keeps a dead socket from stalling a turn.
_CONNECT_TIMEOUT_SECONDS = 2.0
_WRITE_TIMEOUT_SECONDS = 2.0

# Accepted message priorities (Claude Code enum). "now" jumps the queue so a
# steer is read at the next tool boundary rather than after other peer traffic.
_VALID_PRIORITIES = ("now", "next", "later")


def inbox_registry_dir() -> Path:
    configured = os.environ.get(INBOX_REGISTRY_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return claude_config_dir() / _REGISTRY_DIR_NAME


@dataclass(frozen=True)
class InboxRecord:
    """A session's recorded inbox coordinates."""

    session_id: str
    socket: str
    token: str | None = None
    cwd: str | None = None
    recorded_at: float | None = None

    @property
    def path_exists(self) -> bool:
        with contextlib.suppress(OSError):
            return Path(self.socket).exists()
        return False


@dataclass(frozen=True)
class InboxDeliveryResult:
    """Outcome of a delivery attempt.

    ``written`` means the local writer drained, not that the model consumed
    the frame. A write error or EOF after submission is ambiguous; callers
    must not retry when ``may_have_been_written`` is true. There is currently
    no correlated peer ACK, so ``confirmed`` remains false.
    """

    written: bool
    reason: str | None = None
    confirmed: bool = False
    status: str | None = None
    message_id: str | None = None
    may_have_been_written: bool = False

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "written": self.written,
            "confirmed": self.confirmed,
            "mayHaveBeenWritten": self.written or self.may_have_been_written,
        }
        if self.message_id is not None:
            payload["messageId"] = self.message_id
        if self.reason is not None:
            payload["reason"] = self.reason
        if self.status is not None:
            payload["deliveryStatus"] = self.status
        return payload


def _record_path(registry_dir: Path, backend_session_id: str) -> Path:
    return registry_dir / f"{backend_session_id}.json"


def resolve_inbox(backend_session_id: str, *, registry_dir: Path | None = None) -> InboxRecord | None:
    """Read the recorded inbox coordinates for a Claude session id, or None.

    ``backend_session_id`` is Claude Code's own session id (the transcript
    stem / ``--resume`` id), which is what the SessionStart hook names the file.
    """
    if not backend_session_id:
        return None
    directory = registry_dir or inbox_registry_dir()
    path = _record_path(directory, backend_session_id)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("Malformed Claude inbox record at %s", path)
        return None
    socket_path = data.get("socket")
    session_id = data.get("sessionId") or backend_session_id
    if not isinstance(socket_path, str) or not socket_path:
        return None
    token = data.get("token")
    cwd = data.get("cwd")
    recorded_at = data.get("recordedAt")
    return InboxRecord(
        session_id=str(session_id),
        socket=socket_path,
        token=token if isinstance(token, str) and token else None,
        cwd=cwd if isinstance(cwd, str) and cwd else None,
        recorded_at=float(recorded_at) if isinstance(recorded_at, (int, float)) else None,
    )


def forget_inbox(backend_session_id: str, *, registry_dir: Path | None = None) -> None:
    """Drop a stale record (its socket is gone / the session ended)."""
    directory = registry_dir or inbox_registry_dir()
    with contextlib.suppress(OSError):
        _record_path(directory, backend_session_id).unlink()


def _auth_line(token: str | None) -> bytes:
    return (json.dumps({"type": "auth", "token": token}) + "\n").encode("utf-8")


def _message_line(
    *,
    target_session_id: str,
    text: str,
    from_name: str,
    priority: str,
    msg_id: str,
) -> bytes:
    frame = {
        "type": "user",
        "from": from_name,
        "priority": priority,
        "msg_id": msg_id,
        "session_id": target_session_id,
        "message": {"role": "user", "content": text},
    }
    return (json.dumps(frame) + "\n").encode("utf-8")


async def deliver_steer(
    record: InboxRecord,
    text: str,
    *,
    target_session_id: str | None = None,
    from_name: str = "openbase",
    priority: str = "now",
    connect_timeout: float = _CONNECT_TIMEOUT_SECONDS,
) -> InboxDeliveryResult:
    """Post ``text`` as a steering message into ``record``'s inbox socket.

    Returns an :class:`InboxDeliveryResult`. A ``written`` result means the
    frame reached the socket; it does not guarantee the model surfaces it (see
    module docstring on inbound policy). Never raises for an unreachable or
    dead socket — those come back as ``written=False`` so the caller can fall
    back to a deferred/queued steer.
    """
    if priority not in _VALID_PRIORITIES:
        priority = "now"
    if not text:
        return InboxDeliveryResult(written=False, reason="empty_text")

    session_id = target_session_id or record.session_id
    message_id = f"super-agents-steer-{uuid.uuid4().hex}"
    payload = _message_line(
        target_session_id=session_id,
        text=text,
        from_name=from_name,
        priority=priority,
        msg_id=message_id,
    )

    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_unix_connection(record.socket),
            timeout=connect_timeout,
        )
    except (TimeoutError, OSError) as exc:
        # Socket gone (session ended) or refused. Caller falls back.
        logger.info("Claude inbox socket unreachable at %s: %s", record.socket, exc)
        # Only missing/refused sockets prove this endpoint is dead. A timeout
        # or permission error must not authorize resuming a live conversation.
        dead = isinstance(exc, OSError) and exc.errno in {errno.ENOENT, errno.ECONNREFUSED}
        return InboxDeliveryResult(
            written=False,
            reason="socket_unreachable" if dead else "connect_unavailable",
            message_id=message_id,
        )

    try:
        if record.token:
            writer.write(_auth_line(record.token))
        writer.write(payload)
        await asyncio.wait_for(writer.drain(), timeout=_WRITE_TIMEOUT_SECONDS)
    except (TimeoutError, OSError) as exc:
        logger.info("Claude inbox write failed at %s: %s", record.socket, exc)
        _close_writer(writer)
        return InboxDeliveryResult(
            written=False,
            reason="write_failed",
            message_id=message_id,
            may_have_been_written=True,
        )

    # A peer can consume the frame and then close without an ACK. EOF cannot
    # distinguish that from rejection; it must not authorize a second delivery.
    try:
        closed = await _peer_closed_immediately(reader)
    finally:
        _close_writer(writer)
    return InboxDeliveryResult(
        written=True,
        message_id=message_id,
        may_have_been_written=True,
        reason="peer_closed_without_ack" if closed else None,
    )


async def _peer_closed_immediately(reader: asyncio.StreamReader, *, window: float = 0.15) -> bool:
    """Observe EOF only; a byte or an open connection is not a correlated ACK."""
    try:
        data = await asyncio.wait_for(reader.read(1), timeout=window)
    except TimeoutError:
        return False
    except OSError:
        return True
    return data == b""


def _close_writer(writer: asyncio.StreamWriter) -> None:
    with contextlib.suppress(Exception):
        writer.close()


def is_recent(record: InboxRecord, *, max_age_seconds: float | None = None) -> bool:
    """Whether a record is fresh enough to trust without probing the socket.

    A ``None`` age (older hook) or no bound is treated as recent; callers that
    care prove liveness by attempting delivery, which fails cleanly on a dead
    socket.
    """
    if max_age_seconds is None or record.recorded_at is None:
        return True
    return (time.time() - record.recorded_at) <= max_age_seconds
