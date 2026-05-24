from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from dataclasses import replace
from pathlib import Path
from typing import Any


from .app_environment import (
    LOGIN_ENV_TIMEOUT_SECONDS,
    OPENBASE_SUPER_AGENT_LABEL_ENV,
    OPENBASE_SUPER_AGENT_THREAD_ID_ENV,
    login_shell_environment,
    parse_null_separated_env,
    read_login_shell_environment,
    websocket_is_open,
)
from .app_formatting import (
    apply_field_selection,
    as_object,
    compact_json,
    compact_turn_summary,
    extract_compact_items,
    find_useful_text,
    is_diff_like_key,
    preview_text,
    scalar_field,
    text_preview,
    without_none,
)
from .app_models import (
    LabelQueryInput,
    LabelResolutionPrefer,
    Mode,
    PendingServerRequest,
    PermissionRequestCallback,
    QueuedTurn,
    ResolvedSession,
    TurnState,
)
from .app_permissions import (
    DEFAULT_APPROVAL_REQUESTS_FILE,
    clear_shared_permission_request,
    is_permission_request,
    read_permission_store,
    record_shared_permission_request,
    shared_permission_requests,
    write_permission_store,
    write_shared_permission_decision,
)
from .app_protocol import (
    SUPER_AGENT_IDENTITY_INSTRUCTION_PREFIX,
    collaboration_mode,
    collect_turns,
    effective_reasoning_effort,
    extract_model,
    extract_notification_thread_id,
    extract_notification_turn_id,
    extract_thread_cwd,
    extract_thread_id,
    extract_thread_name,
    extract_threads,
    extract_turn_id,
    find_latest_turn,
    find_turn,
    is_active_status,
    is_likely_stale,
    normalize_thread_status,
    normalize_turn_status,
    super_agent_label,
    to_tracked_turn_status,
    with_super_agent_identity_instructions,
)
from .app_routines import (
    DEFAULT_ROUTINE_POLL_SECONDS,
    DEFAULT_ROUTINE_TIMEZONE,
    parse_routine_time,
    routine_from_patch,
    routine_is_due,
    routine_local_date,
    routine_next_run_at,
    routine_next_run_sort_key,
    routine_next_run_summary,
    routine_now,
    routine_poll_seconds,
    routine_turn_input,
    routine_with_next_run,
    safe_routine_next_run_at,
)
from .app_sessions import merge_turns, required_label, session_from_patch, session_from_thread, session_recency
from .app_time import (
    age_ms,
    iso_from_thread_time,
    iso_now,
    parse_iso_ms,
    path_basename,
    thread_recency,
    turn_key,
    turn_recency,
)
from .app_client_routines import RoutineClientMixin
from .app_client_sessions import SessionClientMixin
from .app_client_transport import TransportClientMixin
from .state import (
    JsonObject,
    TrackedStatus,
    get_string,
)

DEFAULT_WS_URL = "ws://127.0.0.1:4500"
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_STATE_FILE = Path.home() / ".super-agents" / "state.json"
logger = logging.getLogger(__name__)

__all__ = [
    "CodexAppServerClient",
    "DEFAULT_APPROVAL_REQUESTS_FILE",
    "DEFAULT_MODEL",
    "DEFAULT_ROUTINE_POLL_SECONDS",
    "DEFAULT_ROUTINE_TIMEZONE",
    "DEFAULT_STATE_FILE",
    "DEFAULT_WS_URL",
    "LOGIN_ENV_TIMEOUT_SECONDS",
    "LabelQueryInput",
    "LabelResolutionPrefer",
    "Mode",
    "OPENBASE_SUPER_AGENT_LABEL_ENV",
    "OPENBASE_SUPER_AGENT_THREAD_ID_ENV",
    "PendingServerRequest",
    "PermissionRequestCallback",
    "QueuedTurn",
    "ResolvedSession",
    "SUPER_AGENT_IDENTITY_INSTRUCTION_PREFIX",
    "TurnState",
    "age_ms",
    "apply_field_selection",
    "as_object",
    "clear_shared_permission_request",
    "collect_turns",
    "collaboration_mode",
    "compact_json",
    "compact_turn_summary",
    "effective_reasoning_effort",
    "extract_compact_items",
    "extract_model",
    "extract_notification_thread_id",
    "extract_notification_turn_id",
    "extract_thread_cwd",
    "extract_thread_id",
    "extract_thread_name",
    "extract_threads",
    "extract_turn_id",
    "find_latest_turn",
    "find_turn",
    "find_useful_text",
    "is_active_status",
    "is_diff_like_key",
    "is_likely_stale",
    "is_permission_request",
    "iso_from_thread_time",
    "iso_now",
    "login_shell_config_override",
    "login_shell_environment",
    "merge_turns",
    "normalize_thread_status",
    "normalize_turn_status",
    "parse_iso_ms",
    "parse_null_separated_env",
    "parse_routine_time",
    "path_basename",
    "preview_text",
    "read_login_shell_environment",
    "read_permission_store",
    "record_shared_permission_request",
    "required_label",
    "routine_from_patch",
    "routine_is_due",
    "routine_local_date",
    "routine_next_run_at",
    "routine_next_run_sort_key",
    "routine_next_run_summary",
    "routine_now",
    "routine_poll_seconds",
    "routine_turn_input",
    "routine_with_next_run",
    "safe_routine_next_run_at",
    "scalar_field",
    "session_from_patch",
    "session_from_thread",
    "session_recency",
    "shared_permission_requests",
    "super_agent_label",
    "text_preview",
    "thread_recency",
    "to_tracked_turn_status",
    "turn_key",
    "turn_recency",
    "websocket_is_open",
    "with_super_agent_identity_instructions",
    "without_none",
    "write_permission_store",
    "write_shared_permission_decision",
]


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


class CodexAppServerClient(TransportClientMixin, RoutineClientMixin, SessionClientMixin):
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
