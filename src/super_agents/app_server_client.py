from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import subprocess
import tempfile
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, TypeVar
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import websockets

from .state import (
    JsonObject,
    RoutineRecord,
    SessionRecord,
    StateFile,
    StoredStatus,
    TrackedStatus,
    TurnSummary,
    as_mode,
    as_stored_status,
    get_string,
    read_state_file_locked,
    routine_record_from_json,
    update_state_file,
)

DEFAULT_WS_URL = "ws://127.0.0.1:4500"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_STATE_FILE = Path.home() / ".super-agents" / "state.json"
DEFAULT_APPROVAL_REQUESTS_FILE = Path.home() / ".super-agents" / "approval-requests.json"
LOGIN_ENV_TIMEOUT_SECONDS = 5
OPENBASE_SUPER_AGENT_THREAD_ID_ENV = "OPENBASE_SUPER_AGENT_THREAD_ID"
OPENBASE_SUPER_AGENT_LABEL_ENV = "OPENBASE_SUPER_AGENT_LABEL"
SUPER_AGENT_IDENTITY_INSTRUCTION_PREFIX = "Super Agent name:"
DEFAULT_ROUTINE_TIMEZONE = "America/New_York"
DEFAULT_ROUTINE_POLL_SECONDS = 30
logger = logging.getLogger(__name__)

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


PermissionRequestCallback = Callable[[PendingServerRequest], JsonObject | Awaitable[JsonObject | None] | None]


@dataclass(slots=True)
class TurnState:
    thread_id: str
    turn_id: str
    status: TrackedStatus
    started_at: str
    reasoning_effort: str | None = None
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
                "reasoningEffort": self.reasoning_effort,
                "finishedAt": self.finished_at,
                "events": self.events,
                "pendingRequests": [request.to_json() for request in self.pending_requests],
            }
        )


@dataclass(slots=True)
class QueuedTurn:
    id: int
    thread_id: str
    label: str | None
    input_data: JsonObject
    queued_at: str

    def to_json(self) -> JsonObject:
        return without_none(
            {
                "id": self.id,
                "threadId": self.thread_id,
                "label": self.label,
                "queuedAt": self.queued_at,
                "promptPreview": preview_text(str(self.input_data.get("prompt") or "")),
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
    thread_id: str | None = None
    turn_id: str | None = None
    include_turn: bool | None = None
    include_items: bool | None = None
    full: bool | None = None
    final_only: bool | None = None
    max_items: int | None = None
    max_output_chars: int | None = None
    include_preview: bool | None = None
    preview_length: int | None = None
    fields: list[str] | None = None


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
        permission_callback: PermissionRequestCallback | None = None,
        approval_requests_file: str | Path | None = None,
    ) -> None:
        self.ws_url = ws_url or os.environ.get("SUPER_AGENTS_WS_URL") or DEFAULT_WS_URL
        self.state_file = Path(state_file or os.environ.get("SUPER_AGENTS_STATE_FILE") or DEFAULT_STATE_FILE)
        self.approval_requests_file = Path(
            approval_requests_file
            or os.environ.get("SUPER_AGENTS_APPROVAL_REQUESTS_FILE")
            or self.state_file.with_name("approval-requests.json")
        )
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
        self._queue_lock = asyncio.Lock()
        self._queued_turns: dict[str, deque[QueuedTurn]] = {}
        self._queue_tasks: dict[str, asyncio.Task[None]] = {}
        self._queue_sequence = 0
        self._routine_scheduler_task: asyncio.Task[None] | None = None
        self._permission_callback = permission_callback
        self._permission_callback_tasks: set[asyncio.Task[None]] = set()
        self._permission_decision_tasks: set[asyncio.Task[None]] = set()

    async def status(self) -> JsonObject:
        ready = await self.check_ready()
        return {
            "ready": ready,
            "websocketUrl": self.ws_url,
            "websocketConnected": websocket_is_open(self._ws),
            "managedProcess": bool(self._child and self._child.returncode is None),
            "pendingRequests": [request.to_json() for request in self._pending_server_requests.values()],
            "pendingPermissionRequests": [
                request.to_json() for request in self.pending_permission_requests()
            ],
            "queuedTurns": self.queued_turn_summary(),
            "activeTurns": [
                self.compact_tracked_turn(turn)
                for turn in self._turns.values()
                if turn.status in {"running", "waiting"}
            ],
            "routines": await self.routine_status_summary(),
        }

    def start_routine_scheduler(self) -> None:
        if self._routine_scheduler_task and not self._routine_scheduler_task.done():
            return
        self._routine_scheduler_task = asyncio.create_task(self._routine_scheduler_loop())

    def register_permission_callback(
        self, callback: PermissionRequestCallback | None
    ) -> PermissionRequestCallback | None:
        """Register a Python callback for app-server approval requests.

        The callback receives a PendingServerRequest. Return a JSON object such as
        {"decision": "accept"} to answer the request, or return None to leave it
        pending for the existing MCP codex_answer_request flow.
        """
        self._permission_callback = callback
        return callback

    def clear_permission_callback(self) -> None:
        self._permission_callback = None

    def pending_permission_requests(self) -> list[PendingServerRequest]:
        return [
            request
            for request in self._pending_server_requests.values()
            if is_permission_request(request.method)
        ]

    async def _routine_scheduler_loop(self) -> None:
        while True:
            try:
                await self.run_due_routines()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Failed to run due Super Agents routines.")
            await asyncio.sleep(routine_poll_seconds())

    async def ensure_connected(self) -> None:
        if websocket_is_open(self._ws):
            return
        async with self._connect_lock:
            if websocket_is_open(self._ws):
                return
            await self._connect()

    async def start_thread(self, input_data: JsonObject) -> JsonObject:
        dispatch_id = str(input_data.get("_mcpCallId") or "")
        started = time.monotonic()
        logger.info(
            "dispatch_timing stage=super_agents_thread_start_request dispatch_id=%s name=%s cwd_basename=%s",
            dispatch_id,
            input_data.get("name") or input_data.get("label") or "",
            path_basename(str(input_data.get("cwd") or Path.home())),
        )
        await self.ensure_connected()
        name = super_agent_label(input_data.get("name") or input_data.get("label"))
        params: JsonObject = {
            "cwd": input_data.get("cwd") or str(Path.home()),
            "config": await login_shell_config_override(),
        }
        if developer_instructions := with_super_agent_identity_instructions(
            get_string(input_data, "developerInstructions"),
            name,
        ):
            params["developerInstructions"] = developer_instructions

        result = await self.request("thread/start", params)
        thread_id = extract_thread_id(result)
        logger.info(
            "dispatch_timing stage=super_agents_thread_start_response dispatch_id=%s thread_id=%s elapsed_ms=%d",
            dispatch_id,
            thread_id or "",
            int((time.monotonic() - started) * 1000),
        )
        if thread_id:
            if name:
                await self.set_thread_name(thread_id, name)
            now = iso_now()
            await self.remember_session(
                thread_id,
                {
                    "label": name,
                    "threadId": thread_id,
                    "cwd": extract_thread_cwd(result) or str(params["cwd"]),
                    "group": input_data.get("group"),
                    "model": extract_model(result) or self.default_model,
                    "createdAt": now,
                    "lastStatus": "unknown",
                },
            )
        return result

    async def set_thread_name(self, thread_id: str, name: str) -> JsonObject:
        await self.ensure_connected()
        return await self.request("thread/name/set", {"threadId": thread_id, "name": name})

    async def resume_thread(
        self,
        thread_id: str,
        *,
        label: str | None = None,
        developer_instructions: str | None = None,
    ) -> JsonObject:
        started = time.monotonic()
        logger.info(
            "dispatch_timing stage=super_agents_thread_resume_request thread_id=%s",
            thread_id,
        )
        await self.ensure_connected()
        params: JsonObject = {
            "threadId": thread_id,
            "config": await login_shell_config_override(),
        }
        if identity_instructions := with_super_agent_identity_instructions(developer_instructions, label):
            params["developerInstructions"] = identity_instructions
        result = await self.request(
            "thread/resume",
            params,
        )
        await self.merge_session(
            thread_id,
            {
                "threadId": thread_id,
                "model": extract_model(result) or self.default_model,
                "lastUsefulMessage": text_preview(result),
            },
        )
        logger.info(
            "dispatch_timing stage=super_agents_thread_resume_response thread_id=%s elapsed_ms=%d",
            thread_id,
            int((time.monotonic() - started) * 1000),
        )
        return result

    async def list_threads(
        self,
        use_state_db_only: bool = True,
        search_term: str | None = None,
        cwd: str | None = None,
        limit: int | None = None,
    ) -> JsonObject:
        await self.ensure_connected()
        return await self.request(
            "thread/list",
            without_none(
                {
                    "useStateDbOnly": use_state_db_only,
                    "searchTerm": search_term,
                    "cwd": cwd,
                    "limit": limit,
                }
            ),
        )

    async def read_thread(self, thread_id: str, include_turns: bool = True) -> JsonObject:
        started = time.monotonic()
        await self.ensure_connected()
        result = await self.request("thread/read", {"threadId": thread_id, "includeTurns": include_turns})
        logger.info(
            "dispatch_timing stage=super_agents_thread_read_response thread_id=%s include_turns=%s elapsed_ms=%d",
            thread_id,
            include_turns,
            int((time.monotonic() - started) * 1000),
        )
        return result

    async def start_turn(self, input_data: JsonObject) -> JsonObject:
        dispatch_id = str(input_data.get("_mcpCallId") or "")
        started = time.monotonic()
        logger.info(
            "dispatch_timing stage=super_agents_turn_start_request dispatch_id=%s "
            "thread_id=%s name=%s cwd_basename=%s mode=%s",
            dispatch_id,
            input_data.get("threadId"),
            input_data.get("name") or input_data.get("label") or "",
            path_basename(str(input_data.get("cwd") or Path.home())),
            input_data.get("mode") or "default",
        )
        await self.ensure_connected()
        session = await self.get_session(str(input_data["threadId"]))
        mode: Mode = input_data.get("mode") if input_data.get("mode") in {"default", "plan"} else "default"
        model = input_data.get("model") or (session.model if session else None) or self.default_model
        reasoning_effort = effective_reasoning_effort(input_data)
        label = super_agent_label(
            input_data.get("label")
            if isinstance(input_data.get("label"), str)
            else input_data.get("name")
            if isinstance(input_data.get("name"), str)
            else session.label
            if session
            else None
        )
        developer_instructions = with_super_agent_identity_instructions(
            input_data.get("developerInstructions") if "developerInstructions" in input_data else None,
            label,
        )
        params: JsonObject = {
            "threadId": input_data["threadId"],
            "cwd": input_data.get("cwd") or (session.cwd if session else None) or str(Path.home()),
            "serviceTier": input_data.get("serviceTier") or "fast",
            "config": await login_shell_config_override(
                thread_id=str(input_data["threadId"]),
                label=label,
            ),
            "collaborationMode": collaboration_mode(
                mode,
                str(model),
                reasoning_effort,
                developer_instructions,
            ),
            "input": [{"type": "text", "text": input_data["prompt"]}],
        }
        logger.info(
            "dispatch_timing stage=app_server_turn_start_request dispatch_id=%s "
            "thread_id=%s cwd_basename=%s mode=%s reasoning_effort=%s",
            dispatch_id,
            input_data["threadId"],
            path_basename(str(params["cwd"])),
            mode,
            reasoning_effort,
        )
        result = await self.request("turn/start", params)
        thread_id = str(input_data["threadId"])
        turn_id = extract_turn_id(result) or f"{thread_id}:unknown:{int(time.time() * 1000)}"
        logger.info(
            "dispatch_timing stage=app_server_turn_start_response dispatch_id=%s thread_id=%s turn_id=%s elapsed_ms=%d",
            dispatch_id,
            thread_id,
            turn_id,
            int((time.monotonic() - started) * 1000),
        )
        now = iso_now()
        key = turn_key(thread_id, turn_id)
        existing_turn = self._turns.get(key)
        turn = existing_turn or TurnState(
            thread_id=thread_id,
            turn_id=turn_id,
            status="running",
            started_at=now,
            reasoning_effort=reasoning_effort,
        )
        if existing_turn is None:
            self._turns[key] = turn
        elif turn.reasoning_effort is None:
            turn.reasoning_effort = reasoning_effort
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
                        "reasoningEffort": reasoning_effort,
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
        return {**result, "threadId": thread_id, "turnId": turn_id, "mode": mode, "reasoningEffort": reasoning_effort}

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

    async def turn_progress(
        self, thread_id: str, turn_id: str, input_data: LabelQueryInput | None = None
    ) -> JsonObject:
        options = input_data or LabelQueryInput()
        started = time.monotonic()
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
        logger.info(
            "dispatch_timing stage=super_agents_progress_response thread_id=%s "
            "turn_id=%s status=%s pending_requests=%d elapsed_ms=%d",
            thread_id,
            turn_id,
            status,
            len(pending_requests),
            int((time.monotonic() - started) * 1000),
        )
        if options.full:
            return {
                "status": status,
                "threadId": thread_id,
                "turnId": turn_id,
                "turn": persisted_turn,
                "trackedTurn": tracked_turn.to_json() if tracked_turn else None,
                "pendingRequests": [request.to_json() for request in pending_requests],
            }
        result = {
            "status": status,
            "threadId": thread_id,
            "turnId": turn_id,
            "summary": compact_turn_summary(
                persisted_turn,
                tracked_turn,
                include_items=bool(options.include_items),
                final_only=bool(options.final_only),
                max_items=options.max_items or 10,
                max_output_chars=options.max_output_chars or 1200,
            ),
            "trackedTurn": self.compact_tracked_turn(tracked_turn) if tracked_turn else None,
            "pendingRequests": [request.to_json() for request in pending_requests],
        }
        if options.include_turn:
            result["turn"] = compact_json(
                persisted_turn,
                max_chars=options.max_output_chars or 4000,
                max_items=options.max_items or 20,
                include_diff=False,
            )
        return apply_field_selection(result, options.fields)

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
                        "reasoningEffort": turn.reasoning_effort,
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
        self.schedule_queue_drain(thread_id)
        return {"cancelled": True, "threadId": thread_id, "turnId": turn_id, "result": result}

    async def answer_request(self, request_id: str | int, result: JsonObject) -> JsonObject:
        await self.ensure_connected()
        if request_id not in self._pending_server_requests:
            raise ValueError(f"No pending app-server request found for id {request_id}.")
        await self.send({"id": request_id, "result": result})
        request = self._pending_server_requests.pop(request_id)
        self.remove_pending_request_from_turn(request_id)
        clear_shared_permission_request(request_id, self.approval_requests_file)
        return {"answered": True, "request": request.to_json()}

    async def sessions(self) -> list[JsonObject]:
        try:
            threads = await self.list_threads(True)
        except Exception:
            threads = {}
        native_threads = extract_threads(threads)
        if native_threads:
            return [self.thread_view(thread) for thread in native_threads]
        state = await self.read_state()
        return [
            self.session_view(session)
            for session in sorted(state.sessions.values(), key=lambda item: item.updated_at, reverse=True)
        ]

    async def active(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        items = [item for item in await self.recent_items(query) if is_active_status(str(item.get("status")))][
            : query.limit or 50
        ]
        items = [self.compact_agent_item(item, query) for item in items]
        return {"count": len(items), "agents": items}

    async def recent(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        items = [
            item
            for item in await self.recent_items(query)
            if query.include_inactive or is_active_status(str(item.get("status")))
        ][: query.limit or 20]
        items = [self.compact_agent_item(item, query) for item in items]
        return {"count": len(items), "agents": items}

    async def compact_status(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        items = [
            self.status_item(item)
            for item in await self.recent_items(query)
            if query.include_inactive or is_active_status(str(item.get("status")))
        ][: query.limit or 50]
        return {"count": len(items), "agents": [apply_field_selection(item, query.fields) for item in items]}

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

    async def read_by_label(self, input_data: LabelQueryInput, include_turns: bool = False) -> JsonObject:
        if input_data.thread_id:
            result = await self.read_thread(input_data.thread_id, include_turns)
            session = await self.get_session(input_data.thread_id)
            return without_none(
                {"threadId": input_data.thread_id, "trackedSession": session.to_json() if session else None, **result}
            )
        resolved = await self.resolve_session(required_label(input_data), replace(input_data, prefer="latest_any"))
        result = await self.read_thread(resolved.session.thread_id, include_turns)
        return without_none({"name": resolved.session.label, "trackedSession": resolved.session.to_json(), **result})

    async def resume_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        resolved = await self.resolve_session(required_label(input_data), replace(input_data, prefer="latest_any"))
        result = await self.resume_thread(resolved.session.thread_id, label=resolved.session.label)
        return {"name": resolved.session.label, **result}

    async def rename_by_label(self, input_data: LabelQueryInput, new_name: str) -> JsonObject:
        resolved = await self.resolve_session(required_label(input_data), replace(input_data, prefer="latest_any"))
        result = await self.set_thread_name(resolved.session.thread_id, new_name)
        await self.merge_session(
            resolved.session.thread_id, {"label": new_name, "threadId": resolved.session.thread_id}
        )
        return {"renamed": True, "name": new_name, "previousName": resolved.session.label, "result": result}

    async def progress_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        if input_data.thread_id:
            turn_id = input_data.turn_id or await self.latest_turn_id(input_data.thread_id)
            if not turn_id:
                raise ValueError(f"No turn is known for thread {input_data.thread_id}.")
            return await self.turn_progress(input_data.thread_id, turn_id, input_data)
        resolved = await self.resolve_session(required_label(input_data), input_data)
        turn_id = input_data.turn_id or resolved.turn_id or await self.latest_turn_id(resolved.session.thread_id)
        if not turn_id:
            raise ValueError(f"No turn is known for label {required_label(input_data)}.")
        return await self.turn_progress(resolved.session.thread_id, turn_id, input_data)

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
        dispatch_id = str(turn_input.get("_mcpCallId") or "")
        started = time.monotonic()
        logger.info(
            "dispatch_timing stage=super_agents_resolve_start dispatch_id=%s name=%s prefer=%s",
            dispatch_id,
            required_label(input_data),
            input_data.prefer or "latest_any",
        )
        resolved = await self.resolve_session(
            required_label(input_data),
            replace(input_data, prefer=input_data.prefer or "latest_any"),
        )
        logger.info(
            "dispatch_timing stage=super_agents_resolve_end dispatch_id=%s thread_id=%s status=%s elapsed_ms=%d",
            dispatch_id,
            resolved.session.thread_id,
            resolved.status,
            int((time.monotonic() - started) * 1000),
        )
        return await self.start_or_queue_turn(resolved, turn_input)

    async def queue_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject:
        target = await self.resolve_queue_target(input_data)
        return await self.start_or_queue_turn(target, turn_input)

    async def start_or_queue_turn(self, target: ResolvedSession, turn_input: JsonObject) -> JsonObject:
        if is_active_status(target.status) or self.thread_has_active_turn(target.session.thread_id):
            return await self.enqueue_turn(target, turn_input)
        result = await self.start_turn({"cwd": target.session.cwd, **turn_input, "threadId": target.session.thread_id})
        return {**result, "queued": False, "startedImmediately": True, "drain": "started_immediately"}

    async def enqueue_turn(self, target: ResolvedSession, turn_input: JsonObject) -> JsonObject:
        async with self._queue_lock:
            self._queue_sequence += 1
            queued = QueuedTurn(
                id=self._queue_sequence,
                thread_id=target.session.thread_id,
                label=target.session.label,
                input_data={"cwd": target.session.cwd, **turn_input},
                queued_at=iso_now(),
            )
            queue = self._queued_turns.setdefault(target.session.thread_id, deque())
            queue.append(queued)
            position = len(queue)
        self.schedule_queue_drain(target.session.thread_id)
        return {
            "queued": True,
            "threadId": target.session.thread_id,
            "name": target.session.label,
            "position": position,
            "queueDepth": position,
            "item": queued.to_json(),
            "drain": "scheduled" if not is_active_status(target.status) else "waiting_for_active_turn",
        }

    async def save_routine(self, input_data: JsonObject) -> JsonObject:
        name = str(input_data["name"])
        async with self._state_lock:
            def update(state: StateFile) -> JsonObject:
                now = iso_now()
                current = state.routines.get(name)
                raw = current.to_json() if current else {"name": name, "createdAt": now}
                for key, value in input_data.items():
                    if value is not None:
                        raw[key] = value
                raw["name"] = name
                raw["updatedAt"] = now
                raw.setdefault("createdAt", now)
                raw.setdefault("enabled", True)
                raw.setdefault("timezone", DEFAULT_ROUTINE_TIMEZONE)
                state.routines[name] = routine_from_patch(raw)
                return state.routines[name].to_json()

            routine = update_state_file(self.state_file, update)
        return {
            "routine": routine,
            "nativeSupport": False,
            "scheduler": "super-agents-local-wrapper",
        }

    async def list_routines(self) -> JsonObject:
        state = await self.read_state()
        routines = sorted(state.routines.values(), key=lambda routine: routine.updated_at, reverse=True)
        return {
            "count": len(routines),
            "routines": [routine_with_next_run(routine) for routine in routines],
            "nativeSupport": False,
            "scheduler": "super-agents-local-wrapper",
        }

    async def read_routine(self, name: str) -> JsonObject:
        state = await self.read_state()
        routine = state.routines.get(name)
        if routine is None:
            raise ValueError(f"No Super Agents routine found for name {name}.")
        return {
            "routine": routine_with_next_run(routine),
            "nativeSupport": False,
            "scheduler": "super-agents-local-wrapper",
        }

    async def delete_routine(self, name: str) -> JsonObject:
        async with self._state_lock:
            def update(state: StateFile) -> JsonObject:
                if name not in state.routines:
                    raise ValueError(f"No Super Agents routine found for name {name}.")
                removed = state.routines.pop(name)
                return removed.to_json()

            removed = update_state_file(self.state_file, update)
        return {"deleted": True, "routine": removed}

    async def routine_status_summary(self) -> JsonObject:
        state = await self.read_state()
        enabled = [routine for routine in state.routines.values() if routine.enabled]
        return {
            "count": len(state.routines),
            "enabledCount": len(enabled),
            "nextRuns": [
                routine_next_run_summary(routine) for routine in sorted(enabled, key=routine_next_run_sort_key)[:5]
            ],
            "nativeSupport": False,
            "scheduler": "super-agents-local-wrapper",
        }

    async def run_due_routines(self, name: str | None = None, force: bool = False) -> JsonObject:
        candidates = await self.reserve_due_routines(name=name, force=force)
        results = []
        for routine in sorted(candidates, key=lambda item: item.name):
            results.append(await self.run_routine(routine, force=force))
        return {
            "count": len(results),
            "results": results,
            "nativeSupport": False,
            "scheduler": "super-agents-local-wrapper",
        }

    async def reserve_due_routines(self, name: str | None = None, force: bool = False) -> list[RoutineRecord]:
        async with self._state_lock:
            def update(state: StateFile) -> list[RoutineRecord]:
                candidates = [
                    routine
                    for routine in state.routines.values()
                    if (not name or routine.name == name) and (force or routine_is_due(routine))
                ]
                if name and not candidates and name not in state.routines:
                    raise ValueError(f"No Super Agents routine found for name {name}.")
                reserved: list[RoutineRecord] = []
                now = iso_now()
                for routine in candidates:
                    run_date = routine_local_date(routine)
                    merged = {
                        **routine.to_json(),
                        "lastRunDate": run_date,
                        "lastStartedAt": now,
                        "lastStatus": "starting",
                        "lastError": None,
                        "updatedAt": now,
                    }
                    reserved_routine = routine_from_patch(merged)
                    state.routines[routine.name] = reserved_routine
                    reserved.append(reserved_routine)
                return reserved

            return update_state_file(self.state_file, update)

    async def run_routine(self, routine: RoutineRecord, *, force: bool = False) -> JsonObject:
        if not routine.enabled and not force:
            return {"name": routine.name, "skipped": True, "reason": "disabled"}
        run_date = routine_local_date(routine)
        try:
            if routine.thread_id:
                target = await self.resolve_queue_target(LabelQueryInput(thread_id=routine.thread_id, cwd=routine.cwd))
                result = await self.start_or_queue_turn(target, routine_turn_input(routine))
            elif routine.target_name:
                target = await self.resolve_queue_target(
                    LabelQueryInput(label=routine.target_name, cwd=routine.cwd, prefer="latest_any")
                )
                result = await self.start_or_queue_turn(target, routine_turn_input(routine))
            else:
                thread_result = await self.start_thread(
                    without_none(
                        {
                            "name": routine.name,
                            "cwd": routine.cwd,
                            "developerInstructions": routine.developer_instructions,
                        }
                    )
                )
                thread_id = extract_thread_id(thread_result)
                if not thread_id:
                    raise RuntimeError(f"Could not start thread for routine {routine.name}.")
                result = await self.start_turn(
                    {**routine_turn_input(routine), "threadId": thread_id, "label": routine.name}
                )
            await self.record_routine_run(
                routine.name,
                {
                    "lastRunDate": run_date,
                    "lastStartedAt": iso_now(),
                    "lastThreadId": get_string(result, "threadId"),
                    "lastTurnId": get_string(result, "turnId"),
                    "lastStatus": "queued" if result.get("queued") else "started",
                    "lastError": None,
                },
            )
            return {"name": routine.name, "ran": True, **result}
        except Exception as exc:
            await self.record_routine_run(
                routine.name,
                {
                    "lastRunDate": run_date,
                    "lastStartedAt": iso_now(),
                    "lastStatus": "failed",
                    "lastError": str(exc),
                },
            )
            logger.exception("Failed to run Super Agents routine name=%s", routine.name)
            return {"name": routine.name, "ran": False, "error": str(exc)}

    async def record_routine_run(self, name: str, patch: JsonObject) -> None:
        async with self._state_lock:
            def update(state: StateFile) -> None:
                routine = state.routines.get(name)
                if routine is None:
                    return
                merged = {**routine.to_json(), **patch, "updatedAt": iso_now()}
                state.routines[name] = routine_from_patch(merged)

            update_state_file(self.state_file, update)

    async def close(self) -> None:
        if self._routine_scheduler_task:
            self._routine_scheduler_task.cancel()
            self._routine_scheduler_task = None
        for task in self._permission_callback_tasks:
            task.cancel()
        self._permission_callback_tasks.clear()
        for task in self._permission_decision_tasks:
            task.cancel()
        self._permission_decision_tasks.clear()
        for task in self._queue_tasks.values():
            task.cancel()
        self._queue_tasks.clear()
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
        started = time.monotonic()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        await self.send({"id": request_id, "method": method, "params": params or {}})
        try:
            result = await asyncio.wait_for(future, timeout=timeout_seconds)
        except TimeoutError:
            self._pending.pop(request_id, None)
            logger.info(
                "dispatch_timing stage=app_server_rpc_timeout request_id=%s method=%s elapsed_ms=%d",
                request_id,
                method,
                int((time.monotonic() - started) * 1000),
            )
            raise TimeoutError(f"Timed out waiting for app-server response to {method}.") from None
        logger.info(
            "dispatch_timing stage=app_server_rpc_response request_id=%s method=%s elapsed_ms=%d",
            request_id,
            method,
            int((time.monotonic() - started) * 1000),
        )
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
        logger.info(
            "dispatch_timing stage=app_server_callback_received request_id=%s method=%s thread_id=%s turn_id=%s",
            request_id,
            method,
            params.get("threadId") or "",
            params.get("turnId") or "",
        )
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
                                "reasoningEffort": turn.reasoning_effort,
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
        self.schedule_permission_callback(pending_request)
        self.record_pending_permission_request(pending_request)

    def record_pending_permission_request(self, pending_request: PendingServerRequest) -> None:
        if not is_permission_request(pending_request.method):
            return
        record_shared_permission_request(pending_request, self.approval_requests_file)
        self.schedule_permission_decision_poll(pending_request)

    def schedule_permission_callback(self, pending_request: PendingServerRequest) -> None:
        if not self._permission_callback or not is_permission_request(pending_request.method):
            return
        try:
            task = asyncio.create_task(self._dispatch_permission_callback(pending_request))
        except RuntimeError:
            logger.exception("Could not schedule Super Agents permission callback without a running event loop.")
            return
        self._permission_callback_tasks.add(task)
        task.add_done_callback(self._permission_callback_tasks.discard)

    async def _dispatch_permission_callback(self, pending_request: PendingServerRequest) -> None:
        callback = self._permission_callback
        if callback is None:
            return
        try:
            result = callback(pending_request)
            if inspect.isawaitable(result):
                result = await result
            if result is None:
                return
            if not isinstance(result, dict):
                logger.warning(
                    "Ignoring Super Agents permission callback result because it is not a JSON object: %r",
                    result,
                )
                return
            if pending_request.id not in self._pending_server_requests:
                return
            await self.answer_request(pending_request.id, result)
        except Exception:
            logger.exception(
                "Super Agents permission callback failed request_id=%s method=%s.",
                pending_request.id,
                pending_request.method,
            )

    def schedule_permission_decision_poll(self, pending_request: PendingServerRequest) -> None:
        try:
            task = asyncio.create_task(self._poll_permission_decision(pending_request))
        except RuntimeError:
            logger.exception("Could not schedule Super Agents permission decision polling without a running event loop.")
            return
        self._permission_decision_tasks.add(task)
        task.add_done_callback(self._permission_decision_tasks.discard)

    async def _poll_permission_decision(self, pending_request: PendingServerRequest) -> None:
        try:
            while pending_request.id in self._pending_server_requests:
                decision = pop_shared_permission_decision(pending_request.id, self.approval_requests_file)
                if decision is not None:
                    await self.answer_request(pending_request.id, decision)
                    return
                await asyncio.sleep(0.5)
        except Exception:
            logger.exception(
                "Failed to apply shared Super Agents approval decision request_id=%s.",
                pending_request.id,
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
        merge_task = asyncio.create_task(
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
                            "reasoningEffort": turn.reasoning_effort,
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
        if turn.status in {"completed", "failed"}:
            merge_task.add_done_callback(
                lambda _task, completed_thread_id=thread_id: self.schedule_queue_drain(completed_thread_id)
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
            session
            for session in sessions
            if not input_data.status or self.session_status(session) == input_data.status
        ]
        return sorted(sessions, key=session_recency, reverse=True)

    async def resolve_session(self, label: str, input_data: LabelQueryInput) -> ResolvedSession:
        prefer = input_data.prefer or "latest_active"
        native_thread = await self.resolve_thread_name(label, input_data)
        native_thread_id = extract_thread_id(native_thread)
        if native_thread_id:
            session = await self.get_session(native_thread_id) or session_from_thread(native_thread, label)
            status = self.session_status(session)
            if prefer == "latest_active" and not is_active_status(status):
                raise ValueError(
                    f"No active Super Agents session found for name {label}. Recent match: {json.dumps(self.session_view(session))}"
                )
            return ResolvedSession(
                session=session,
                turn_id=input_data.turn_id or session.active_turn_id or session.last_turn_id,
                status=status,
            )

        candidates = await self.filtered_sessions(replace(input_data, label=label))
        if not candidates:
            raise ValueError(f"No Super Agents session found for name {label}.")
        active_candidates = [session for session in candidates if is_active_status(self.session_status(session))]
        scoped_candidates = candidates if prefer == "latest_any" else active_candidates
        if not scoped_candidates:
            recent = [self.session_view(session) for session in candidates[:5]]
            raise ValueError(
                f"No active Super Agents session found for name {label}. Recent inactive candidates: {json.dumps(recent)}"
            )
        first = scoped_candidates[0]
        if len(scoped_candidates) > 1 and session_recency(first) == session_recency(scoped_candidates[1]):
            candidates_json = json.dumps([self.session_view(session) for session in scoped_candidates[:5]])
            raise ValueError(f"Ambiguous Super Agents name {label}. Candidates: {candidates_json}")
        return ResolvedSession(
            session=first,
            turn_id=input_data.turn_id or first.active_turn_id or first.last_turn_id,
            status=self.session_status(first),
        )

    async def resolve_queue_target(self, input_data: LabelQueryInput) -> ResolvedSession:
        if input_data.thread_id:
            session = await self.get_session(input_data.thread_id)
            if session is None:
                session = SessionRecord(
                    label=input_data.label,
                    thread_id=input_data.thread_id,
                    cwd=input_data.cwd,
                    updated_at=iso_now(),
                )
            return ResolvedSession(
                session=session,
                turn_id=session.active_turn_id or session.last_turn_id,
                status=self.session_status(session),
            )
        return await self.resolve_session(
            required_label(input_data),
            replace(input_data, prefer=input_data.prefer or "latest_any"),
        )

    def queued_turn_summary(self) -> list[JsonObject]:
        return [
            {
                "threadId": thread_id,
                "queueDepth": len(queue),
                "items": [item.to_json() for item in list(queue)[:5]],
            }
            for thread_id, queue in self._queued_turns.items()
            if queue
        ]

    def schedule_queue_drain(self, thread_id: str) -> None:
        existing = self._queue_tasks.get(thread_id)
        if existing and not existing.done():
            return
        self._queue_tasks[thread_id] = asyncio.create_task(self._drain_queue(thread_id))

    async def _drain_queue(self, thread_id: str) -> None:
        await asyncio.sleep(0)
        while True:
            if self.thread_has_active_turn(thread_id):
                return
            async with self._queue_lock:
                queue = self._queued_turns.get(thread_id)
                queued = queue.popleft() if queue else None
                if queue is not None and not queue:
                    self._queued_turns.pop(thread_id, None)
            if queued is None:
                return
            try:
                await self.start_turn({**queued.input_data, "threadId": thread_id, "label": queued.label})
            except Exception:
                async with self._queue_lock:
                    self._queued_turns.setdefault(thread_id, deque()).appendleft(queued)
                logger.exception("Failed to start queued Super Agents turn for thread_id=%s", thread_id)
                return

    def thread_has_active_turn(self, thread_id: str) -> bool:
        for turn in self._turns.values():
            if turn.thread_id == thread_id and turn.status in {"running", "waiting"}:
                return True
        session = self.session_from_memory(thread_id)
        return bool(session and is_active_status(self.session_status(session)))

    async def resolve_thread_name(self, name: str, input_data: LabelQueryInput) -> JsonObject:
        try:
            response = await self.list_threads(True, name, input_data.cwd, input_data.limit or 50)
        except Exception:
            return {}
        threads = extract_threads(response)
        matches = [thread for thread in threads if extract_thread_name(thread) == name]
        if not matches:
            return {}
        matches.sort(key=thread_recency, reverse=True)
        return matches[0]

    async def recent_items(self, input_data: LabelQueryInput) -> list[JsonObject]:
        if input_data.thread_id:
            session = await self.get_session(input_data.thread_id)
            if session:
                return [self.session_view(session)]
            return []
        try:
            response = await self.list_threads(
                True,
                input_data.label,
                input_data.cwd,
                input_data.limit or 50,
            )
        except Exception:
            response = {}
        threads = extract_threads(response)
        if not threads:
            sessions = await self.filtered_sessions(input_data)
            return [self.session_view(session) for session in sessions]
        items = [
            self.thread_view(thread)
            for thread in threads
            if not input_data.label or extract_thread_name(thread) == input_data.label
        ]
        if input_data.status:
            items = [item for item in items if item.get("status") == input_data.status]
        return sorted(items, key=lambda item: parse_iso_ms(str(item.get("updatedAt") or "")), reverse=True)

    def thread_view(self, thread: JsonObject) -> JsonObject:
        thread_id = extract_thread_id(thread)
        session = self.session_from_memory(thread_id) if thread_id else None
        status = self.session_status(session) if session else normalize_thread_status(thread) or "unknown"
        running_turn_id = (
            session.active_turn_id or session.last_turn_id if session and is_active_status(status) else None
        )
        updated_at = iso_from_thread_time(thread)
        last_event_at = session.last_event_at if session else None
        return without_none(
            {
                "name": extract_thread_name(thread),
                "cwd": extract_thread_cwd(thread),
                "threadId": thread_id,
                "runningTurnId": running_turn_id,
                "lastTurnId": session.last_turn_id if session else None,
                "reasoningEffort": self.session_turn_reasoning_effort(
                    session, running_turn_id or (session.last_turn_id if session else None)
                ),
                "status": status,
                "ageMs": int(time.time() * 1000) - thread_recency(thread),
                "updatedAt": updated_at,
                "lastEventAt": last_event_at,
                "lastEventAgeMs": age_ms(last_event_at),
                "ageSinceUpdateMs": age_ms(updated_at),
                "isLikelyStale": is_likely_stale(status, last_event_at or updated_at),
                "preview": get_string(thread, "preview"),
                "lastUsefulMessage": session.last_useful_message if session else None,
                "pendingRequestCount": self.pending_request_count(thread_id, running_turn_id) if thread_id else None,
            }
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
                "reasoningEffort": self.session_turn_reasoning_effort(session, running_turn_id or session.last_turn_id),
                "status": status,
                "ageMs": int(time.time() * 1000) - parse_iso_ms(started_at),
                "updatedAt": session.updated_at,
                "lastEventAt": session.last_event_at,
                "lastEventAgeMs": age_ms(session.last_event_at),
                "ageSinceUpdateMs": age_ms(session.updated_at),
                "isLikelyStale": is_likely_stale(status, session.last_event_at or session.updated_at),
                "lastUsefulMessage": session.last_useful_message,
                "pendingRequestCount": self.pending_request_count(session.thread_id, running_turn_id),
            }
        )

    def compact_agent_item(self, item: JsonObject, query: LabelQueryInput) -> JsonObject:
        result = dict(item)
        include_preview = query.include_preview if query.include_preview is not None else True
        preview_length = query.preview_length or 160
        if not include_preview:
            result.pop("preview", None)
            result.pop("lastUsefulMessage", None)
        else:
            if isinstance(result.get("preview"), str):
                result["preview"] = preview_text(str(result["preview"]), preview_length)
            if isinstance(result.get("lastUsefulMessage"), str):
                result["lastUsefulMessage"] = preview_text(str(result["lastUsefulMessage"]), preview_length)
        return apply_field_selection(result, query.fields)

    def status_item(self, item: JsonObject) -> JsonObject:
        return without_none(
            {
                "name": item.get("name") or item.get("label"),
                "threadId": item.get("threadId"),
                "turnId": item.get("runningTurnId") or item.get("lastTurnId"),
                "reasoningEffort": item.get("reasoningEffort"),
                "status": item.get("status"),
                "lastEventAt": item.get("lastEventAt"),
                "updatedAt": item.get("updatedAt"),
                "lastEventAgeMs": item.get("lastEventAgeMs"),
                "ageSinceUpdateMs": item.get("ageSinceUpdateMs"),
                "isLikelyStale": item.get("isLikelyStale"),
                "pendingRequestCount": item.get("pendingRequestCount"),
                "cwd": item.get("cwd"),
            }
        )

    def compact_tracked_turn(self, turn: TurnState | None) -> JsonObject:
        if turn is None:
            return {}
        last_event_at = get_string(turn.events[-1], "receivedAt") if turn.events else turn.started_at
        return without_none(
            {
                "threadId": turn.thread_id,
                "turnId": turn.turn_id,
                "status": turn.status,
                "reasoningEffort": turn.reasoning_effort,
                "startedAt": turn.started_at,
                "finishedAt": turn.finished_at,
                "lastEventAt": last_event_at,
                "lastEventAgeMs": age_ms(last_event_at),
                "isLikelyStale": is_likely_stale(turn.status, last_event_at),
                "eventCount": len(turn.events),
                "pendingRequestCount": len(turn.pending_requests),
                "pendingRequestIds": [request.id for request in turn.pending_requests],
            }
        )

    def session_turn_reasoning_effort(self, session: SessionRecord | None, turn_id: str | None) -> str | None:
        if not session or not turn_id:
            return None
        runtime_turn = self._turns.get(turn_key(session.thread_id, turn_id))
        if runtime_turn and runtime_turn.reasoning_effort:
            return runtime_turn.reasoning_effort
        turn_summary = (session.turns or {}).get(turn_id)
        return turn_summary.reasoning_effort if turn_summary else None

    def session_status(self, session: SessionRecord) -> StoredStatus:
        turn_id = session.active_turn_id or session.last_turn_id
        runtime_turn = self._turns.get(turn_key(session.thread_id, turn_id)) if turn_id else None
        return runtime_turn.status if runtime_turn else session.last_status or "unknown"

    def session_from_memory(self, thread_id: str | None) -> SessionRecord | None:
        if not thread_id:
            return None
        return read_state_file_locked(self.state_file).sessions.get(thread_id)

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
        state_session = await self.get_session(thread_id)
        reasoning_effort = (
            tracked_turn.reasoning_effort if tracked_turn else None
        ) or self.session_turn_reasoning_effort(
            state_session,
            turn_id,
        )
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
                        "reasoningEffort": reasoning_effort,
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
        if not is_active_status(tracked_status):
            self.schedule_queue_drain(thread_id)

    async def remember_session(self, thread_id: str, patch: JsonObject) -> None:
        async with self._state_lock:
            def update(state: StateFile) -> None:
                now = iso_now()
                state.sessions[thread_id] = session_from_patch(
                    {"threadId": thread_id, "createdAt": now, **patch, "updatedAt": now}
                )

            update_state_file(self.state_file, update)

    async def merge_session(
        self,
        thread_id: str,
        patch: JsonObject,
        clear_fields: list[str] | None = None,
    ) -> None:
        async with self._state_lock:
            def update(state: StateFile) -> None:
                now = iso_now()
                current = state.sessions.get(thread_id) or SessionRecord(
                    thread_id=thread_id, created_at=now, updated_at=now
                )
                merged_json = {**current.to_json(), **without_none(patch), "threadId": thread_id, "updatedAt": now}
                merged_json["createdAt"] = current.created_at or now
                merged_json["turns"] = merge_turns(current.turns, patch.get("turns"))
                for field_name in clear_fields or []:
                    merged_json.pop(field_name, None)
                state.sessions[thread_id] = session_from_patch(merged_json)

            update_state_file(self.state_file, update)

    async def get_session(self, thread_id: str) -> SessionRecord | None:
        state = await self.read_state()
        return state.sessions.get(thread_id)

    async def read_state(self) -> StateFile:
        async with self._state_lock:
            return read_state_file_locked(self.state_file)

    async def latest_turn_id(self, thread_id: str) -> str | None:
        thread = await self.read_thread(thread_id, True)
        turn = find_latest_turn(thread, active_only=True) or find_latest_turn(thread, active_only=False)
        return get_string(turn, "id") if turn else None

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


async def login_shell_config_override(
    *,
    thread_id: str | None = None,
    label: str | None = None,
) -> JsonObject:
    env = await login_shell_environment()
    set_values = {key: value for key in ["PATH", "SHELL", "HOME", "USER", "LOGNAME"] if (value := env.get(key))}
    set_values[OPENBASE_SUPER_AGENT_THREAD_ID_ENV] = thread_id or ""
    set_values[OPENBASE_SUPER_AGENT_LABEL_ENV] = label or ""
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


def collaboration_mode(
    mode: Mode, model: str, reasoning_effort: str | None, developer_instructions: str | None
) -> JsonObject:
    settings: JsonObject = {"model": model, "developer_instructions": developer_instructions}
    settings["reasoning_effort"] = reasoning_effort or "high"
    return {"mode": mode, "settings": settings}


def with_super_agent_identity_instructions(
    developer_instructions: str | None,
    label: str | None,
) -> str | None:
    normalized_label = super_agent_label(label)
    if not normalized_label:
        return developer_instructions
    identity_line = f"{SUPER_AGENT_IDENTITY_INSTRUCTION_PREFIX} {normalized_label}"
    base = developer_instructions.strip() if developer_instructions else ""
    if not base:
        return identity_line
    if identity_line in base.splitlines():
        return base
    return f"{base}\n\n{identity_line}"


def super_agent_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.split()) or None


def effective_reasoning_effort(input_data: JsonObject) -> str:
    return get_string(input_data, "reasoningEffort") or "high"


def extract_model(value: JsonObject) -> str | None:
    return get_string(value, "model") or get_string(as_object(value.get("thread")), "model")


def extract_thread_id(value: JsonObject) -> str | None:
    return get_string(value, "threadId") or get_string(value, "id") or get_string(as_object(value.get("thread")), "id")


def extract_thread_cwd(value: JsonObject) -> str | None:
    return get_string(value, "cwd") or get_string(as_object(value.get("thread")), "cwd")


def extract_thread_name(value: JsonObject) -> str | None:
    return (
        get_string(value, "name")
        or get_string(value, "threadName")
        or get_string(as_object(value.get("thread")), "name")
        or get_string(as_object(value.get("thread")), "threadName")
    )


def extract_threads(value: JsonObject) -> list[JsonObject]:
    raw_threads = value.get("data") or value.get("threads")
    if not isinstance(raw_threads, list):
        return []
    return [thread for thread in raw_threads if isinstance(thread, dict)]


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


def find_latest_turn(value: Any, active_only: bool) -> JsonObject | None:
    turns = collect_turns(value)
    if active_only:
        turns = [turn for turn in turns if normalize_turn_status(turn) == "running"]
    if not turns:
        return None
    return max(turns, key=turn_recency)


def collect_turns(value: Any) -> list[JsonObject]:
    if isinstance(value, list):
        turns: list[JsonObject] = []
        for item in value:
            turns.extend(collect_turns(item))
        return turns
    if not isinstance(value, dict):
        return []
    turns = [value] if get_string(value, "id") and get_string(value, "status") else []
    for item in value.values():
        turns.extend(collect_turns(item))
    return turns


def normalize_turn_status(turn: JsonObject | None) -> str | None:
    status = get_string(turn, "status") if turn else None
    if not status:
        return None
    if status in {"inProgress", "active"}:
        return "running"
    return status


def normalize_thread_status(thread: JsonObject) -> StoredStatus | None:
    raw_status = thread.get("status")
    if isinstance(raw_status, str):
        if raw_status in {"active", "running", "inProgress"}:
            return "running"
        if raw_status in {"completed", "failed", "cancelled", "waiting", "unknown"}:
            return raw_status  # type: ignore[return-value]
    if isinstance(raw_status, dict):
        kind = get_string(raw_status, "type") or get_string(raw_status, "status")
        if kind in {"active", "running", "inProgress"}:
            return "running"
        if kind in {"completed", "failed", "cancelled", "waiting", "unknown"}:
            return kind  # type: ignore[return-value]
    return None


def to_tracked_turn_status(status: str) -> TrackedStatus:
    if status in {"completed", "failed", "waiting", "cancelled"}:
        return status  # type: ignore[return-value]
    return "running"


def is_active_status(status: str | None) -> bool:
    return status in {"running", "waiting"}


def is_permission_request(method: str) -> bool:
    return "requestApproval" in method


def shared_permission_requests(path: str | Path | None = None) -> list[JsonObject]:
    store = read_permission_store(path)
    raw_requests = as_object(store.get("requests"))
    return [
        item
        for item in raw_requests.values()
        if isinstance(item, dict) and is_permission_request(str(item.get("method") or ""))
    ]


def record_shared_permission_request(
    request: PendingServerRequest,
    path: str | Path | None = None,
) -> None:
    store = read_permission_store(path)
    requests = as_object(store.get("requests"))
    requests[str(request.id)] = request.to_json()
    store["requests"] = requests
    store["decisions"] = as_object(store.get("decisions"))
    write_permission_store(path, store)


def clear_shared_permission_request(request_id: str | int, path: str | Path | None = None) -> None:
    store = read_permission_store(path)
    requests = as_object(store.get("requests"))
    decisions = as_object(store.get("decisions"))
    requests.pop(str(request_id), None)
    decisions.pop(str(request_id), None)
    store["requests"] = requests
    store["decisions"] = decisions
    write_permission_store(path, store)


def write_shared_permission_decision(
    request_id: str | int,
    decision: Literal["accept", "decline", "cancel"],
    path: str | Path | None = None,
) -> bool:
    store = read_permission_store(path)
    requests = as_object(store.get("requests"))
    if str(request_id) not in requests:
        return False
    decisions = as_object(store.get("decisions"))
    decisions[str(request_id)] = {"decision": decision, "decidedAt": iso_now()}
    store["requests"] = requests
    store["decisions"] = decisions
    write_permission_store(path, store)
    return True


def pop_shared_permission_decision(request_id: str | int, path: str | Path | None = None) -> JsonObject | None:
    store = read_permission_store(path)
    decisions = as_object(store.get("decisions"))
    raw_decision = decisions.pop(str(request_id), None)
    if not isinstance(raw_decision, dict):
        return None
    decision = raw_decision.get("decision")
    if decision not in {"accept", "decline", "cancel"}:
        return None
    store["requests"] = as_object(store.get("requests"))
    store["decisions"] = decisions
    write_permission_store(path, store)
    return {"decision": decision}


def read_permission_store(path: str | Path | None = None) -> JsonObject:
    store_path = Path(path or os.environ.get("SUPER_AGENTS_APPROVAL_REQUESTS_FILE") or DEFAULT_APPROVAL_REQUESTS_FILE)
    try:
        raw = json.loads(store_path.read_text(encoding="utf-8"))
    except Exception:
        return {"requests": {}, "decisions": {}}
    if not isinstance(raw, dict):
        return {"requests": {}, "decisions": {}}
    return {"requests": as_object(raw.get("requests")), "decisions": as_object(raw.get("decisions"))}


def write_permission_store(path: str | Path | None, store: JsonObject) -> None:
    store_path = Path(path or os.environ.get("SUPER_AGENTS_APPROVAL_REQUESTS_FILE") or DEFAULT_APPROVAL_REQUESTS_FILE)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {"requests": as_object(store.get("requests")), "decisions": as_object(store.get("decisions"))},
        indent=2,
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=store_path.parent, delete=False) as tmp:
        tmp.write(payload + "\n")
        tmp_name = tmp.name
    os.replace(tmp_name, store_path)


def compact_turn_summary(
    persisted_turn: JsonObject | None,
    tracked_turn: TurnState | None,
    *,
    include_items: bool,
    final_only: bool,
    max_items: int,
    max_output_chars: int,
) -> JsonObject:
    status = normalize_turn_status(persisted_turn) or (tracked_turn.status if tracked_turn else None)
    result = without_none(
        {
            "id": get_string(persisted_turn, "id") if persisted_turn else None,
            "status": status,
            "reasoningEffort": tracked_turn.reasoning_effort if tracked_turn else None,
            "startedAt": scalar_field(persisted_turn, "startedAt"),
            "completedAt": scalar_field(persisted_turn, "completedAt"),
            "lastUsefulMessage": text_preview(persisted_turn),
            "eventCount": len(tracked_turn.events) if tracked_turn else None,
            "pendingRequestCount": len(tracked_turn.pending_requests) if tracked_turn else None,
        }
    )
    if include_items and persisted_turn:
        items = extract_compact_items(
            persisted_turn, final_only=final_only, max_items=max_items, max_output_chars=max_output_chars
        )
        if items:
            result["items"] = items
    return result


def extract_compact_items(
    value: JsonObject, *, final_only: bool, max_items: int, max_output_chars: int
) -> list[JsonObject]:
    raw_items = value.get("items") or value.get("events") or value.get("messages")
    if not isinstance(raw_items, list):
        return []
    if final_only:
        raw_items = raw_items[-1:]
    compacted: list[JsonObject] = []
    for item in raw_items[:max_items]:
        if not isinstance(item, dict):
            continue
        compacted.append(compact_json(item, max_chars=max_output_chars, max_items=max_items, include_diff=False))
    return compacted


def compact_json(value: Any, *, max_chars: int, max_items: int, include_diff: bool) -> JsonObject:
    if not isinstance(value, dict):
        return {}
    result: JsonObject = {}
    used_chars = 0
    for key, item in value.items():
        if not include_diff and is_diff_like_key(str(key)):
            continue
        if isinstance(item, str):
            clipped = preview_text(item, min(max_chars, max(80, max_chars - used_chars)))
            used_chars += len(clipped)
            result[str(key)] = clipped
        elif isinstance(item, int | float | bool) or item is None:
            result[str(key)] = item
        elif isinstance(item, list):
            result[str(key)] = [
                compact_json(child, max_chars=max_chars, max_items=max_items, include_diff=include_diff)
                if isinstance(child, dict)
                else preview_text(str(child), 240)
                for child in item[:max_items]
            ]
        elif isinstance(item, dict):
            result[str(key)] = compact_json(item, max_chars=max_chars, max_items=max_items, include_diff=include_diff)
    return result


def is_diff_like_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in {"diff", "patch"} or lowered.endswith("diff") or lowered.endswith("patch")


def scalar_field(value: JsonObject | None, key: str) -> Any:
    if not isinstance(value, dict):
        return None
    item = value.get(key)
    return item if isinstance(item, str | int | float | bool) else None


def apply_field_selection(value: JsonObject, fields: list[str] | None) -> JsonObject:
    if not fields:
        return value
    allowed = set(fields)
    return {key: item for key, item in value.items() if key in allowed}


def age_ms(iso_value: str | None) -> int | None:
    parsed = parse_iso_ms(iso_value)
    if parsed <= 0:
        return None
    return max(0, int(time.time() * 1000) - parsed)


def is_likely_stale(status: str | None, last_update_at: str | None) -> bool:
    if not is_active_status(status):
        return False
    age = age_ms(last_update_at)
    return bool(age is not None and age > 10 * 60 * 1000)


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
                result[str(turn_id)] = {
                    **as_object(result.get(str(turn_id))),
                    **without_none(summary),
                    "turnId": str(turn_id),
                }
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
                        reasoning_effort=get_string(raw, "reasoningEffort"),
                        started_at=get_string(raw, "startedAt")
                        or get_string(raw, "updatedAt")
                        or "1970-01-01T00:00:00.000Z",
                        updated_at=get_string(raw, "updatedAt") or "1970-01-01T00:00:00.000Z",
                        finished_at=get_string(raw, "finishedAt"),
                        prompt_preview=get_string(raw, "promptPreview"),
                        last_useful_message=get_string(raw, "lastUsefulMessage"),
                        pending_request_ids=[
                            item
                            for item in raw.get("pendingRequestIds", [])
                            if isinstance(item, str | int) and not isinstance(item, bool)
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


def session_from_thread(thread: JsonObject, name: str) -> SessionRecord:
    thread_id = extract_thread_id(thread) or ""
    return SessionRecord(
        label=name,
        thread_id=thread_id,
        cwd=extract_thread_cwd(thread),
        last_status=normalize_thread_status(thread) or "unknown",
        updated_at=iso_from_thread_time(thread),
    )


def routine_from_patch(value: JsonObject) -> RoutineRecord:
    routine = routine_record_from_json(
        value,
        default_timezone=DEFAULT_ROUTINE_TIMEZONE,
        default_updated_at=iso_now(),
    )
    if routine is None:
        raise ValueError("Invalid routine payload.")
    return routine


def routine_turn_input(routine: RoutineRecord) -> JsonObject:
    return without_none(
        {
            "prompt": routine.prompt,
            "cwd": routine.cwd,
            "mode": routine.mode or "default",
            "model": routine.model,
            "reasoningEffort": routine.reasoning_effort,
            "serviceTier": routine.service_tier,
            "developerInstructions": routine.developer_instructions,
            "name": routine.target_name or routine.name,
            "label": routine.target_name or routine.name,
        }
    )


def routine_with_next_run(routine: RoutineRecord) -> JsonObject:
    return {**routine.to_json(), "nextRunAt": safe_routine_next_run_at(routine)}


def routine_next_run_summary(routine: RoutineRecord) -> JsonObject:
    return {
        "name": routine.name,
        "time": routine.time,
        "timezone": routine.timezone or DEFAULT_ROUTINE_TIMEZONE,
        "nextRunAt": safe_routine_next_run_at(routine),
        "lastStatus": routine.last_status,
    }


def routine_next_run_sort_key(routine: RoutineRecord) -> int:
    next_run = safe_routine_next_run_at(routine)
    return parse_iso_ms(next_run) if next_run else 0


def routine_is_due(routine: RoutineRecord) -> bool:
    if not routine.enabled:
        return False
    try:
        now = routine_now(routine)
        hour, minute = parse_routine_time(routine.time)
    except ValueError:
        return False
    run_key = now.date().isoformat()
    if routine.last_run_date == run_key:
        return False
    return (now.hour, now.minute) >= (hour, minute)


def routine_local_date(routine: RoutineRecord) -> str:
    return routine_now(routine).date().isoformat()


def routine_next_run_at(routine: RoutineRecord) -> str:
    now = routine_now(routine)
    hour, minute = parse_routine_time(routine.time)
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if routine.last_run_date == now.date().isoformat() or next_run <= now:
        next_run = next_run + timedelta(days=1)
    return next_run.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def safe_routine_next_run_at(routine: RoutineRecord) -> str | None:
    try:
        return routine_next_run_at(routine)
    except ValueError:
        return None


def routine_now(routine: RoutineRecord) -> datetime:
    timezone_name = routine.timezone or DEFAULT_ROUTINE_TIMEZONE
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo(DEFAULT_ROUTINE_TIMEZONE)
    return datetime.now(tz)


def parse_routine_time(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("routine time must use HH:MM.")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("routine time must use HH:MM in 24-hour time.")
    return hour, minute


def routine_poll_seconds() -> int:
    raw = os.environ.get("SUPER_AGENTS_ROUTINE_POLL_SECONDS")
    try:
        value = int(raw) if raw else DEFAULT_ROUTINE_POLL_SECONDS
    except ValueError:
        return DEFAULT_ROUTINE_POLL_SECONDS
    return max(5, value)


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


def path_basename(value: str) -> str:
    return Path(value).expanduser().name


def thread_recency(thread: JsonObject) -> int:
    for key in ["updatedAtMs", "updated_at_ms"]:
        value = thread.get(key)
        if isinstance(value, int | float):
            return int(value)
    for key in ["updatedAt", "updated_at"]:
        value = thread.get(key)
        if isinstance(value, int | float):
            return int(value * 1000)
        if isinstance(value, str):
            return parse_iso_ms(value)
    return 0


def turn_recency(turn: JsonObject) -> int:
    for key in ["completedAt", "startedAt"]:
        value = turn.get(key)
        if isinstance(value, int | float):
            return int(value * 1000)
    return 0


def iso_from_thread_time(thread: JsonObject) -> str:
    recency = thread_recency(thread)
    if recency <= 0:
        return iso_now()
    return (
        datetime.fromtimestamp(recency / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )


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
