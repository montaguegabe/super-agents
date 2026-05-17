from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeVar

import websockets

from .state import (
    JsonObject,
    SessionRecord,
    StateFile,
    StoredStatus,
    TrackedStatus,
    TurnSummary,
    as_mode,
    as_stored_status,
    get_string,
    read_state_file,
    write_state_file,
)

DEFAULT_WS_URL = "ws://127.0.0.1:4500"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_STATE_FILE = Path.home() / ".super-agents" / "state.json"
LOGIN_ENV_TIMEOUT_SECONDS = 5

T = TypeVar("T")
LabelResolutionPrefer = Literal["latest_active", "latest_any"]
Mode = Literal["default", "plan"]


@dataclass(slots=True)
class PendingServerRequest:
    id: str | int
    method: str
    params: JsonObject
    received_at: str

    def to_json(self) -> JsonObject:
        return {
            "id": self.id,
            "method": self.method,
            "params": self.params,
            "receivedAt": self.received_at,
        }


@dataclass(slots=True)
class TurnState:
    thread_id: str
    turn_id: str
    status: TrackedStatus
    started_at: str
    events: list[JsonObject] = field(default_factory=list)
    pending_requests: list[PendingServerRequest] = field(default_factory=list)
    finished_at: str | None = None

    def to_json(self) -> JsonObject:
        return without_none(
            {
                "threadId": self.thread_id,
                "turnId": self.turn_id,
                "status": self.status,
                "startedAt": self.started_at,
                "finishedAt": self.finished_at,
                "events": self.events,
                "pendingRequests": [request.to_json() for request in self.pending_requests],
            }
        )


@dataclass(slots=True)
class LabelQueryInput:
    label: str | None = None
    cwd: str | None = None
    group: str | None = None
    status: str | None = None
    limit: int | None = None
    include_inactive: bool | None = None
    prefer: LabelResolutionPrefer | None = None
    turn_id: str | None = None


@dataclass(slots=True)
class ResolvedSession:
    session: SessionRecord
    status: StoredStatus
    turn_id: str | None = None


class CodexAppServerClient:
    def __init__(
        self,
        ws_url: str | None = None,
        state_file: str | Path | None = None,
        default_model: str | None = None,
    ) -> None:
        self.ws_url = ws_url or os.environ.get("SUPER_AGENTS_WS_URL") or DEFAULT_WS_URL
        self.state_file = Path(state_file or os.environ.get("SUPER_AGENTS_STATE_FILE") or DEFAULT_STATE_FILE)
        self.default_model = default_model or os.environ.get("SUPER_AGENTS_MODEL") or DEFAULT_MODEL
        self._ws: Any | None = None
        self._next_id = 1
        self._pending: dict[str | int, asyncio.Future[Any]] = {}
        self._pending_server_requests: dict[str | int, PendingServerRequest] = {}
        self._turns: dict[str, TurnState] = {}
        self._child: asyncio.subprocess.Process | None = None
        self._connect_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._state_lock = asyncio.Lock()

    async def status(self) -> JsonObject:
        ready = await self.check_ready()
        return {
            "ready": ready,
            "websocketUrl": self.ws_url,
            "websocketConnected": websocket_is_open(self._ws),
            "managedProcess": bool(self._child and self._child.returncode is None),
            "pendingRequests": [request.to_json() for request in self._pending_server_requests.values()],
            "activeTurns": [
                turn.to_json() for turn in self._turns.values() if turn.status in {"running", "waiting"}
            ],
        }

    async def ensure_connected(self) -> None:
        if websocket_is_open(self._ws):
            return
        async with self._connect_lock:
            if websocket_is_open(self._ws):
                return
            await self._connect()

    async def start_thread(self, input_data: JsonObject) -> JsonObject:
        await self.ensure_connected()
        params: JsonObject = {
            "cwd": input_data.get("cwd") or str(Path.home()),
            "approvalPolicy": input_data.get("approvalPolicy") or "never",
            "sandbox": input_data.get("sandbox") or "danger-full-access",
            "config": await login_shell_config_override(),
        }
        if input_data.get("developerInstructions"):
            params["developerInstructions"] = input_data["developerInstructions"]

        result = await self.request("thread/start", params)
        thread_id = extract_thread_id(result)
        if thread_id:
            now = iso_now()
            await self.remember_session(
                thread_id,
                {
                    "label": input_data.get("label"),
                    "threadId": thread_id,
                    "cwd": extract_thread_cwd(result) or str(params["cwd"]),
                    "group": input_data.get("group"),
                    "model": extract_model(result) or self.default_model,
                    "createdAt": now,
                    "lastStatus": "unknown",
                },
            )
        return result

    async def resume_thread(self, thread_id: str) -> JsonObject:
        await self.ensure_connected()
        result = await self.request(
            "thread/resume",
            {
                "threadId": thread_id,
                "approvalPolicy": "never",
                "sandbox": "danger-full-access",
                "config": await login_shell_config_override(),
            },
        )
        await self.merge_session(
            thread_id,
            {
                "threadId": thread_id,
                "model": extract_model(result) or self.default_model,
                "lastUsefulMessage": text_preview(result),
            },
        )
        return result

    async def list_threads(self, use_state_db_only: bool = True) -> JsonObject:
        await self.ensure_connected()
        return await self.request("thread/list", {"useStateDbOnly": use_state_db_only})

    async def read_thread(self, thread_id: str, include_turns: bool = True) -> JsonObject:
        await self.ensure_connected()
        return await self.request("thread/read", {"threadId": thread_id, "includeTurns": include_turns})

    async def start_turn(self, input_data: JsonObject) -> JsonObject:
        await self.ensure_connected()
        session = await self.get_session(str(input_data["threadId"]))
        mode: Mode = input_data.get("mode") if input_data.get("mode") in {"default", "plan"} else "default"
        model = input_data.get("model") or (session.model if session else None) or self.default_model
        params: JsonObject = {
            "threadId": input_data["threadId"],
            "cwd": input_data.get("cwd") or (session.cwd if session else None) or str(Path.home()),
            "approvalPolicy": input_data.get("approvalPolicy") or "never",
            "sandboxPolicy": {"type": input_data.get("sandboxType") or "dangerFullAccess"},
            "collaborationMode": collaboration_mode(
                mode,
                str(model),
                input_data.get("reasoningEffort"),
                input_data.get("developerInstructions") if "developerInstructions" in input_data else None,
            ),
            "input": [{"type": "text", "text": input_data["prompt"]}],
        }
        result = await self.request("turn/start", params)
        thread_id = str(input_data["threadId"])
        turn_id = extract_turn_id(result) or f"{thread_id}:unknown:{int(time.time() * 1000)}"
        now = iso_now()
        key = turn_key(thread_id, turn_id)
        existing_turn = self._turns.get(key)
        turn = existing_turn or TurnState(thread_id=thread_id, turn_id=turn_id, status="running", started_at=now)
        if existing_turn is None:
            self._turns[key] = turn
        status: TrackedStatus = "waiting" if turn.pending_requests or turn.status == "waiting" else "running"
        turn.status = status
        await self.merge_session(
            thread_id,
            {
                "label": input_data.get("label"),
                "threadId": thread_id,
                "cwd": str(params["cwd"]),
                "group": input_data.get("group"),
                "model": model,
                "lastTurnId": turn_id,
                "activeTurnId": turn_id,
                "lastStartedAt": now,
                "lastStatus": status,
                "lastUsefulMessage": text_preview(result),
                "turns": {
                    turn_id: {
                        "turnId": turn_id,
                        "status": status,
                        "mode": mode,
                        "startedAt": turn.started_at,
                        "updatedAt": now,
                        "promptPreview": preview_text(str(input_data["prompt"])),
                        "lastUsefulMessage": text_preview(result),
                        "pendingRequestIds": [request.id for request in turn.pending_requests],
                        "eventCount": len(turn.events),
                    }
                },
            },
        )
        return {**result, "threadId": thread_id, "turnId": turn_id, "mode": mode}

    async def steer_turn(self, thread_id: str, turn_id: str, prompt: str) -> JsonObject:
        await self.ensure_connected()
        return await self.request(
            "turn/steer",
            {
                "threadId": thread_id,
                "expectedTurnId": turn_id,
                "input": [{"type": "text", "text": prompt}],
            },
        )

    async def turn_progress(self, thread_id: str, turn_id: str) -> JsonObject:
        await self.ensure_connected()
        key = turn_key(thread_id, turn_id)
        tracked_turn = self._turns.get(key)
        thread = await self.read_thread(thread_id, True)
        persisted_turn = find_turn(thread, turn_id)
        pending_requests = [
            request
            for request in self._pending_server_requests.values()
            if request.params.get("threadId") == thread_id and request.params.get("turnId") == turn_id
        ]
        status = (
            "waiting"
            if pending_requests
            else normalize_turn_status(persisted_turn) or (tracked_turn.status if tracked_turn else "unknown")
        )
        if tracked_turn and status != "unknown":
            tracked_turn.status = to_tracked_turn_status(status)
            if tracked_turn.status in {"completed", "failed"} and tracked_turn.finished_at is None:
                tracked_turn.finished_at = iso_now()
        await self.record_turn_progress(thread_id, turn_id, status, persisted_turn, pending_requests)
        return {
            "status": status,
            "threadId": thread_id,
            "turnId": turn_id,
            "turn": persisted_turn,
            "trackedTurn": tracked_turn.to_json() if tracked_turn else None,
            "pendingRequests": [request.to_json() for request in pending_requests],
        }

    async def cancel_turn(self, thread_id: str, turn_id: str) -> JsonObject:
        await self.ensure_connected()
        result = await self.request("turn/interrupt", {"threadId": thread_id, "turnId": turn_id})
        turn = self.ensure_turn(thread_id, turn_id)
        turn.status = "cancelled"
        turn.finished_at = iso_now()
        await self.merge_session(
            thread_id,
            {
                "threadId": thread_id,
                "lastTurnId": turn_id,
                "lastStatus": "cancelled",
                "lastFinishedAt": turn.finished_at,
                "turns": {
                    turn_id: {
                        "turnId": turn_id,
                        "status": "cancelled",
                        "startedAt": turn.started_at,
                        "updatedAt": turn.finished_at,
                        "finishedAt": turn.finished_at,
                        "eventCount": len(turn.events),
                        "pendingRequestIds": [],
                    }
                },
            },
            clear_fields=["activeTurnId"],
        )
        return {"cancelled": True, "threadId": thread_id, "turnId": turn_id, "result": result}

    async def answer_request(self, request_id: str | int, result: JsonObject) -> JsonObject:
        await self.ensure_connected()
        if request_id not in self._pending_server_requests:
            raise ValueError(f"No pending app-server request found for id {request_id}.")
        await self.send({"id": request_id, "result": result})
        request = self._pending_server_requests.pop(request_id)
        self.remove_pending_request_from_turn(request_id)
        return {"answered": True, "request": request.to_json()}

    async def sessions(self) -> list[JsonObject]:
        state = await self.read_state()
        return [session.to_json() for session in sorted(state.sessions.values(), key=lambda item: item.updated_at, reverse=True)]

    async def active(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        sessions = await self.filtered_sessions(query)
        items = [
            self.session_view(session)
            for session in sessions
            if is_active_status(str(self.session_view(session).get("status")))
        ][: query.limit or 50]
        return {"count": len(items), "agents": items}

    async def recent(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        sessions = await self.filtered_sessions(query)
        items = [
            self.session_view(session)
            for session in sessions
            if query.include_inactive or is_active_status(str(self.session_view(session).get("status")))
        ][: query.limit or 20]
        return {"count": len(items), "agents": items}

    async def resolve_label(self, input_data: LabelQueryInput) -> JsonObject:
        resolved = await self.resolve_session(required_label(input_data), input_data)
        return {
            "label": resolved.session.label,
            "group": resolved.session.group,
            "cwd": resolved.session.cwd,
            "threadId": resolved.session.thread_id,
            "turnId": resolved.turn_id,
            "status": resolved.status,
            "updatedAt": resolved.session.updated_at,
            "lastUsefulMessage": resolved.session.last_useful_message,
        }

    async def progress_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        resolved = await self.resolve_session(required_label(input_data), input_data)
        turn_id = input_data.turn_id or resolved.turn_id
        if not turn_id:
            raise ValueError(f"No turn is known for label {required_label(input_data)}.")
        return await self.turn_progress(resolved.session.thread_id, turn_id)

    async def steer_by_label(self, input_data: LabelQueryInput, prompt: str) -> JsonObject:
        resolved = await self.resolve_session(required_label(input_data), input_data)
        turn_id = input_data.turn_id or resolved.turn_id
        if not turn_id or not is_active_status(resolved.status):
            raise ValueError(f"No active turn is known for label {required_label(input_data)}.")
        return await self.steer_turn(resolved.session.thread_id, turn_id, prompt)

    async def cancel_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        resolved = await self.resolve_session(required_label(input_data), input_data)
        turn_id = input_data.turn_id or resolved.turn_id
        if not turn_id or not is_active_status(resolved.status):
            raise ValueError(f"No active turn is known for label {required_label(input_data)}.")
        return await self.cancel_turn(resolved.session.thread_id, turn_id)

    async def start_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject:
        resolved = await self.resolve_session(
            required_label(input_data),
            replace(input_data, prefer=input_data.prefer or "latest_any"),
        )
        return await self.start_turn({**turn_input, "threadId": resolved.session.thread_id})

    async def close(self) -> None:
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None
        if self._child and self._child.returncode is None:
            self._child.terminate()
            try:
                await asyncio.wait_for(self._child.wait(), timeout=2)
            except TimeoutError:
                self._child.kill()
        self._child = None

    async def _connect(self) -> None:
        if not await self.check_ready():
            await self.start_managed_server()
        self._ws = await asyncio.wait_for(websockets.connect(self.ws_url), timeout=5)
        self._reader_task = asyncio.create_task(self._reader_loop())
        await self.request(
            "initialize",
            {
                "clientInfo": {"name": "super-agents-mcp", "title": "Super Agents MCP", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        await self.send({"method": "initialized", "params": {}})

    async def start_managed_server(self) -> None:
        env = await login_shell_environment()
        self._child = await asyncio.create_subprocess_exec(
            "codex",
            "app-server",
            "--listen",
            self.ws_url,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        asyncio.create_task(self._drain_child_stderr(self._child))
        started = time.monotonic()
        while time.monotonic() - started < 10:
            if await self.check_ready():
                return
            await asyncio.sleep(0.25)
        raise RuntimeError("Codex app-server did not become ready.")

    async def check_ready(self) -> bool:
        ready_url = self.ws_url.replace("ws:", "http:", 1).replace("wss:", "https:", 1).rstrip("/") + "/readyz"
        try:
            return await asyncio.to_thread(_check_ready_sync, ready_url)
        except Exception:
            return False

    async def request(self, method: str, params: JsonObject | None = None, timeout_seconds: float = 30) -> JsonObject:
        request_id = self._next_id
        self._next_id += 1
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        await self.send({"id": request_id, "method": method, "params": params or {}})
        try:
            result = await asyncio.wait_for(future, timeout=timeout_seconds)
        except TimeoutError:
            self._pending.pop(request_id, None)
            raise TimeoutError(f"Timed out waiting for app-server response to {method}.") from None
        return result if isinstance(result, dict) else {"result": result}

    async def send(self, message: JsonObject) -> None:
        if not websocket_is_open(self._ws):
            raise RuntimeError("Codex app-server websocket is not connected.")
        await self._ws.send(json.dumps(message))

    async def _reader_loop(self) -> None:
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                self.handle_message(str(raw))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.reject_pending(exc)
            self._ws = None

    def handle_message(self, raw: str) -> None:
        try:
            message = json.loads(raw)
        except Exception:
            return
        if not isinstance(message, dict):
            return
        message_id = message.get("id")
        method = message.get("method") if isinstance(message.get("method"), str) else None
        if isinstance(message_id, str | int) and method:
            self.handle_server_request(message_id, method, as_object(message.get("params")))
            return
        if isinstance(message_id, str | int):
            self.handle_rpc_response(message_id, message)
            return
        if method:
            self.handle_notification(method, as_object(message.get("params")))

    def handle_rpc_response(self, request_id: str | int, message: JsonObject) -> None:
        pending = self._pending.pop(request_id, None)
        if pending is None or pending.done():
            return
        if message.get("error"):
            pending.set_exception(RuntimeError(json.dumps(message["error"])))
        else:
            pending.set_result(message.get("result"))

    def handle_server_request(self, request_id: str | int, method: str, params: JsonObject) -> None:
        pending_request = PendingServerRequest(id=request_id, method=method, params=params, received_at=iso_now())
        self._pending_server_requests[request_id] = pending_request
        thread_id = extract_notification_thread_id(params)
        turn_id = extract_notification_turn_id(params)
        if thread_id and turn_id:
            turn = self.ensure_turn(thread_id, turn_id)
            turn.status = "waiting"
            turn.pending_requests.append(pending_request)
            asyncio.create_task(
                self.merge_session(
                    thread_id,
                    {
                        "threadId": thread_id,
                        "activeTurnId": turn_id,
                        "lastTurnId": turn_id,
                        "lastStatus": "waiting",
                        "lastEventAt": pending_request.received_at,
                        "lastUsefulMessage": text_preview(params) or method,
                        "turns": {
                            turn_id: {
                                "turnId": turn_id,
                                "status": "waiting",
                                "startedAt": turn.started_at,
                                "updatedAt": pending_request.received_at,
                                "lastUsefulMessage": text_preview(params) or method,
                                "pendingRequestIds": [request.id for request in turn.pending_requests],
                                "eventCount": len(turn.events),
                            }
                        },
                    },
                )
            )

    def handle_notification(self, method: str, params: JsonObject) -> None:
        thread_id = extract_notification_thread_id(params)
        turn_id = extract_notification_turn_id(params)
        if not thread_id or not turn_id:
            return
        turn = self.ensure_turn(thread_id, turn_id)
        received_at = iso_now()
        turn.events.append({"method": method, "params": params, "receivedAt": received_at})
        if len(turn.events) > 200:
            turn.events.pop(0)
        if method == "turn/completed":
            turn.status = "completed"
            turn.finished_at = iso_now()
        elif method == "turn/failed":
            turn.status = "failed"
            turn.finished_at = iso_now()
        elif turn.status not in {"waiting", "completed", "failed"}:
            turn.status = "running"

        last_useful_message = text_preview(params) or method
        clear_fields = ["activeTurnId"] if turn.status in {"completed", "failed"} else []
        asyncio.create_task(
            self.merge_session(
                thread_id,
                {
                    "threadId": thread_id,
                    "activeTurnId": None if clear_fields else turn_id,
                    "lastTurnId": turn_id,
                    "lastStatus": turn.status,
                    "lastEventAt": received_at,
                    "lastFinishedAt": turn.finished_at,
                    "lastUsefulMessage": last_useful_message,
                    "turns": {
                        turn_id: {
                            "turnId": turn_id,
                            "status": turn.status,
                            "startedAt": turn.started_at,
                            "updatedAt": received_at,
                            "finishedAt": turn.finished_at,
                            "lastUsefulMessage": last_useful_message,
                            "pendingRequestIds": [request.id for request in turn.pending_requests],
                            "eventCount": len(turn.events),
                        }
                    },
                },
                clear_fields=clear_fields,
            )
        )

    def ensure_turn(self, thread_id: str, turn_id: str) -> TurnState:
        key = turn_key(thread_id, turn_id)
        if key not in self._turns:
            self._turns[key] = TurnState(thread_id=thread_id, turn_id=turn_id, status="running", started_at=iso_now())
        return self._turns[key]

    def remove_pending_request_from_turn(self, request_id: str | int) -> None:
        for turn in self._turns.values():
            turn.pending_requests = [request for request in turn.pending_requests if request.id != request_id]
            if not turn.pending_requests and turn.status == "waiting":
                turn.status = "running"

    def reject_pending(self, error: Exception) -> None:
        for request_id, future in list(self._pending.items()):
            if not future.done():
                future.set_exception(error)
            self._pending.pop(request_id, None)

    async def filtered_sessions(self, input_data: LabelQueryInput) -> list[SessionRecord]:
        state = await self.read_state()
        sessions = list(state.sessions.values())
        sessions = [session for session in sessions if not input_data.label or session.label == input_data.label]
        sessions = [session for session in sessions if not input_data.cwd or session.cwd == input_data.cwd]
        sessions = [session for session in sessions if not input_data.group or session.group == input_data.group]
        sessions = [
            session for session in sessions if not input_data.status or self.session_status(session) == input_data.status
        ]
        return sorted(sessions, key=session_recency, reverse=True)

    async def resolve_session(self, label: str, input_data: LabelQueryInput) -> ResolvedSession:
        prefer = input_data.prefer or "latest_active"
        candidates = await self.filtered_sessions(replace(input_data, label=label))
        if not candidates:
            raise ValueError(f"No Super Agents session found for label {label}.")
        active_candidates = [session for session in candidates if is_active_status(self.session_status(session))]
        scoped_candidates = candidates if prefer == "latest_any" else active_candidates
        if not scoped_candidates:
            recent = [self.session_view(session) for session in candidates[:5]]
            raise ValueError(f"No active Super Agents session found for label {label}. Recent inactive candidates: {json.dumps(recent)}")
        first = scoped_candidates[0]
        if len(scoped_candidates) > 1 and session_recency(first) == session_recency(scoped_candidates[1]):
            candidates_json = json.dumps([self.session_view(session) for session in scoped_candidates[:5]])
            raise ValueError(f"Ambiguous Super Agents label {label}. Candidates: {candidates_json}")
        return ResolvedSession(
            session=first,
            turn_id=input_data.turn_id or first.active_turn_id or first.last_turn_id,
            status=self.session_status(first),
        )

    def session_view(self, session: SessionRecord) -> JsonObject:
        status = self.session_status(session)
        running_turn_id = session.active_turn_id or session.last_turn_id if is_active_status(status) else None
        started_at = session.last_started_at or session.updated_at
        return without_none(
            {
                "label": session.label,
                "group": session.group,
                "cwd": session.cwd,
                "threadId": session.thread_id,
                "runningTurnId": running_turn_id,
                "lastTurnId": session.last_turn_id,
                "status": status,
                "ageMs": int(time.time() * 1000) - parse_iso_ms(started_at),
                "updatedAt": session.updated_at,
                "lastUsefulMessage": session.last_useful_message,
                "pendingRequestCount": self.pending_request_count(session.thread_id, running_turn_id),
            }
        )

    def session_status(self, session: SessionRecord) -> StoredStatus:
        turn_id = session.active_turn_id or session.last_turn_id
        runtime_turn = self._turns.get(turn_key(session.thread_id, turn_id)) if turn_id else None
        return runtime_turn.status if runtime_turn else session.last_status or "unknown"

    def pending_request_count(self, thread_id: str, turn_id: str | None = None) -> int:
        return len(
            [
                request
                for request in self._pending_server_requests.values()
                if request.params.get("threadId") == thread_id
                and (not turn_id or request.params.get("turnId") == turn_id)
            ]
        )

    async def record_turn_progress(
        self,
        thread_id: str,
        turn_id: str,
        status: str,
        persisted_turn: JsonObject | None,
        pending_requests: list[PendingServerRequest],
    ) -> None:
        tracked_status: StoredStatus = "unknown" if status == "unknown" else to_tracked_turn_status(status)
        tracked_turn = self._turns.get(turn_key(thread_id, turn_id))
        finished_at = (
            (tracked_turn.finished_at if tracked_turn else None) or iso_now()
            if tracked_status in {"completed", "failed", "cancelled"}
            else None
        )
        await self.merge_session(
            thread_id,
            {
                "threadId": thread_id,
                "activeTurnId": turn_id if is_active_status(tracked_status) else None,
                "lastTurnId": turn_id,
                "lastStatus": tracked_status,
                "lastFinishedAt": finished_at,
                "lastUsefulMessage": text_preview(persisted_turn),
                "turns": {
                    turn_id: {
                        "turnId": turn_id,
                        "status": "running" if tracked_status == "unknown" else tracked_status,
                        "startedAt": tracked_turn.started_at if tracked_turn else iso_now(),
                        "updatedAt": iso_now(),
                        "finishedAt": finished_at,
                        "lastUsefulMessage": text_preview(persisted_turn),
                        "pendingRequestIds": [request.id for request in pending_requests],
                        "eventCount": len(tracked_turn.events) if tracked_turn else 0,
                    }
                },
            },
            clear_fields=[] if is_active_status(tracked_status) else ["activeTurnId"],
        )

    async def remember_session(self, thread_id: str, patch: JsonObject) -> None:
        async with self._state_lock:
            state = read_state_file(self.state_file)
            now = iso_now()
            state.sessions[thread_id] = session_from_patch({"threadId": thread_id, "createdAt": now, **patch, "updatedAt": now})
            write_state_file(self.state_file, state)

    async def merge_session(
        self,
        thread_id: str,
        patch: JsonObject,
        clear_fields: list[str] | None = None,
    ) -> None:
        async with self._state_lock:
            state = read_state_file(self.state_file)
            now = iso_now()
            current = state.sessions.get(thread_id) or SessionRecord(thread_id=thread_id, created_at=now, updated_at=now)
            merged_json = {**current.to_json(), **without_none(patch), "threadId": thread_id, "updatedAt": now}
            merged_json["createdAt"] = current.created_at or now
            merged_json["turns"] = merge_turns(current.turns, patch.get("turns"))
            for field_name in clear_fields or []:
                merged_json.pop(field_name, None)
            state.sessions[thread_id] = session_from_patch(merged_json)
            write_state_file(self.state_file, state)

    async def get_session(self, thread_id: str) -> SessionRecord | None:
        state = await self.read_state()
        return state.sessions.get(thread_id)

    async def read_state(self) -> StateFile:
        async with self._state_lock:
            return read_state_file(self.state_file)

    async def _drain_child_stderr(self, child: asyncio.subprocess.Process) -> None:
        if child.stderr is None:
            return
        while True:
            line = await child.stderr.readline()
            if not line:
                return
            print(f"[codex app-server] {line.decode('utf-8', errors='replace').strip()}", file=os.sys.stderr)


def _check_ready_sync(url: str) -> bool:
    with urllib.request.urlopen(url, timeout=1) as response:
        return 200 <= response.status < 300


_login_shell_environment_task: asyncio.Task[dict[str, str]] | None = None


async def login_shell_config_override() -> JsonObject:
    env = await login_shell_environment()
    set_values = {key: value for key in ["PATH", "SHELL", "HOME", "USER", "LOGNAME"] if (value := env.get(key))}
    return {"shell_environment_policy": {"inherit": "all", "set": set_values}}


async def login_shell_environment() -> dict[str, str]:
    global _login_shell_environment_task
    if _login_shell_environment_task is None:
        _login_shell_environment_task = asyncio.create_task(read_login_shell_environment())
    try:
        return await _login_shell_environment_task
    except Exception as exc:
        print(f"[super-agents] Failed to read login shell environment: {exc}", file=os.sys.stderr)
        return dict(os.environ)


async def read_login_shell_environment() -> dict[str, str]:
    shell = os.environ.get("SHELL") if os.environ.get("SHELL", "").startswith("/") else "/bin/zsh"

    def run() -> dict[str, str]:
        proc = subprocess.run(
            [shell, "-lic", "/usr/bin/env -0"],
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=LOGIN_ENV_TIMEOUT_SECONDS,
            check=True,
        )
        return {**os.environ, **parse_null_separated_env(proc.stdout.decode("utf-8", errors="replace"))}

    return await asyncio.to_thread(run)


def parse_null_separated_env(output: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for entry in output.split("\0"):
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        if key:
            env[key] = value
    return env


def collaboration_mode(mode: Mode, model: str, reasoning_effort: str | None, developer_instructions: str | None) -> JsonObject:
    settings: JsonObject = {"model": model, "developer_instructions": developer_instructions}
    if mode == "plan":
        settings["reasoning_effort"] = reasoning_effort or "medium"
    return {"mode": mode, "settings": settings}


def extract_model(value: JsonObject) -> str | None:
    return get_string(value, "model") or get_string(as_object(value.get("thread")), "model")


def extract_thread_id(value: JsonObject) -> str | None:
    return get_string(value, "threadId") or get_string(value, "id") or get_string(as_object(value.get("thread")), "id")


def extract_thread_cwd(value: JsonObject) -> str | None:
    return get_string(value, "cwd") or get_string(as_object(value.get("thread")), "cwd")


def extract_turn_id(value: JsonObject) -> str | None:
    return get_string(value, "turnId") or get_string(value, "id") or get_string(as_object(value.get("turn")), "id")


def extract_notification_thread_id(value: JsonObject) -> str | None:
    return (
        get_string(value, "threadId")
        or get_string(as_object(value.get("thread")), "id")
        or get_string(as_object(value.get("turn")), "threadId")
        or get_string(as_object(value.get("item")), "threadId")
    )


def extract_notification_turn_id(value: JsonObject) -> str | None:
    return (
        get_string(value, "turnId")
        or get_string(as_object(value.get("turn")), "id")
        or get_string(as_object(value.get("item")), "turnId")
    )


def find_turn(value: Any, turn_id: str) -> JsonObject | None:
    if isinstance(value, list):
        for item in value:
            result = find_turn(item, turn_id)
            if result:
                return result
        return None
    if not isinstance(value, dict):
        return None
    if get_string(value, "id") == turn_id:
        return value
    for item in value.values():
        result = find_turn(item, turn_id)
        if result:
            return result
    return None


def normalize_turn_status(turn: JsonObject | None) -> str | None:
    status = get_string(turn, "status") if turn else None
    if not status:
        return None
    if status in {"inProgress", "active"}:
        return "running"
    return status


def to_tracked_turn_status(status: str) -> TrackedStatus:
    if status in {"completed", "failed", "waiting", "cancelled"}:
        return status  # type: ignore[return-value]
    return "running"


def is_active_status(status: str | None) -> bool:
    return status in {"running", "waiting"}


def required_label(input_data: LabelQueryInput) -> str:
    if not input_data.label:
        raise ValueError("label must be a non-empty string.")
    return input_data.label


def session_recency(session: SessionRecord) -> int:
    return parse_iso_ms(session.last_event_at or session.updated_at)


def merge_turns(current: dict[str, TurnSummary] | None, patch: Any) -> JsonObject | None:
    if not current and not patch:
        return None
    result: JsonObject = {key: value.to_json() for key, value in (current or {}).items()}
    if isinstance(patch, dict):
        for turn_id, summary in patch.items():
            if isinstance(summary, dict):
                result[str(turn_id)] = {**as_object(result.get(str(turn_id))), **summary, "turnId": str(turn_id)}
    return result


def session_from_patch(value: JsonObject) -> SessionRecord:
    turns = None
    if isinstance(value.get("turns"), dict):
        turns = {}
        for turn_id, raw in value["turns"].items():
            if isinstance(raw, dict):
                status = as_stored_status(get_string(raw, "status"))
                if status and status != "unknown":
                    turns[str(turn_id)] = TurnSummary(
                        turn_id=get_string(raw, "turnId") or str(turn_id),
                        status=status,
                        mode=as_mode(get_string(raw, "mode")),
                        started_at=get_string(raw, "startedAt") or get_string(raw, "updatedAt") or "1970-01-01T00:00:00.000Z",
                        updated_at=get_string(raw, "updatedAt") or "1970-01-01T00:00:00.000Z",
                        finished_at=get_string(raw, "finishedAt"),
                        prompt_preview=get_string(raw, "promptPreview"),
                        last_useful_message=get_string(raw, "lastUsefulMessage"),
                        pending_request_ids=[
                            item for item in raw.get("pendingRequestIds", []) if isinstance(item, str | int) and not isinstance(item, bool)
                        ]
                        if isinstance(raw.get("pendingRequestIds"), list)
                        else None,
                        event_count=raw.get("eventCount") if isinstance(raw.get("eventCount"), int) else None,
                    )
        if not turns:
            turns = None
    return SessionRecord(
        label=get_string(value, "label"),
        thread_id=get_string(value, "threadId") or "",
        cwd=get_string(value, "cwd"),
        group=get_string(value, "group"),
        model=get_string(value, "model"),
        last_turn_id=get_string(value, "lastTurnId"),
        active_turn_id=get_string(value, "activeTurnId"),
        created_at=get_string(value, "createdAt"),
        last_started_at=get_string(value, "lastStartedAt"),
        last_finished_at=get_string(value, "lastFinishedAt"),
        last_status=as_stored_status(get_string(value, "lastStatus")),
        last_useful_message=get_string(value, "lastUsefulMessage"),
        last_event_at=get_string(value, "lastEventAt"),
        turns=turns,
        updated_at=get_string(value, "updatedAt") or iso_now(),
    )


def preview_text(value: str, max_length: int = 240) -> str:
    normalized = " ".join(value.split())
    return f"{normalized[: max_length - 3]}..." if len(normalized) > max_length else normalized


def text_preview(value: Any) -> str | None:
    text = find_useful_text(value)
    return preview_text(text) if text else None


def find_useful_text(value: Any, depth: int = 0) -> str | None:
    if depth > 6 or value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if len(trimmed) >= 8 else None
    if isinstance(value, list):
        for item in value:
            result = find_useful_text(item, depth + 1)
            if result:
                return result
        return None
    if not isinstance(value, dict):
        return None
    for key in ["text", "message", "content", "summary", "output", "preview"]:
        result = find_useful_text(value.get(key), depth + 1)
        if result:
            return result
    for item in value.values():
        result = find_useful_text(item, depth + 1)
        if result:
            return result
    return None


def as_object(value: Any) -> JsonObject:
    return value if isinstance(value, dict) else {}


def without_none(value: JsonObject) -> JsonObject:
    return {key: item for key, item in value.items() if item is not None}


def websocket_is_open(ws: Any | None) -> bool:
    if ws is None:
        return False
    closed = getattr(ws, "closed", None)
    if isinstance(closed, bool):
        return not closed
    state = getattr(ws, "state", None)
    return state == 1 or str(state).endswith(".OPEN")


def turn_key(thread_id: str, turn_id: str | None) -> str:
    return f"{thread_id}:{turn_id}"


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso_ms(value: str | None) -> int:
    if not value:
        return 0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0
    return int(parsed.timestamp() * 1000)
