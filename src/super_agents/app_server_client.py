from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

from .app_client_events import EventClientMixin
from .app_client_labels import LabelQueryMixin
from .app_client_permissions import PermissionClientMixin
from .app_client_routines import RoutineClientMixin
from .app_client_sessions import SessionClientMixin
from .app_client_tags import TagsFavoritesMixin
from .app_client_threads import ThreadLifecycleMixin
from .app_client_transport import TransportClientMixin
from .app_client_turns import TurnLifecycleMixin
from .app_endpoint import (
    DEFAULT_WEBSOCKET_ENDPOINT,
    AppServerEndpoint,
    configured_app_server_endpoint,
)
from .app_environment import (
    LOGIN_ENV_TIMEOUT_SECONDS,
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
    extract_queued_id,
    extract_started_turn_id,
    extract_thread_cwd,
    extract_thread_id,
    extract_thread_name,
    extract_threads,
    extract_turn_id,
    find_latest_turn,
    find_turn,
    is_active_status,
    is_likely_stale,
    is_queue_item_id,
    normalize_thread_status,
    normalize_turn_status,
    response_is_queued,
    super_agent_label,
    to_tracked_turn_status,
    with_super_agent_identity_instructions,
)
from .app_routines import (
    DEFAULT_ROUTINE_INTERVAL_SECONDS,
    DEFAULT_ROUTINE_POLL_SECONDS,
    DEFAULT_ROUTINE_TIMEZONE,
    MIN_ROUTINE_INTERVAL_SECONDS,
    parse_routine_interval_seconds,
    parse_routine_time,
    routine_from_patch,
    routine_interval_next_run_at,
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
from .app_sessions import (
    merge_turns,
    required_label,
    session_from_patch,
    session_from_thread,
    session_recency,
)
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
from .backend_config import CODEX_BACKEND, execution_backend, normalize_backend
from .defaults import default_super_agents_model
from .state import JsonObject

DEFAULT_WS_URL = DEFAULT_WEBSOCKET_ENDPOINT
DEFAULT_MODEL = "gpt-5.4"
DEFAULT_STATE_FILE = Path.home() / ".super-agents" / "state.json"
DEFAULT_QUEUE_DIR = Path.home() / ".super-agents" / "queues"
logger = logging.getLogger(__name__)

__all__ = [
    "CodexAppServerClient",
    "DEFAULT_APPROVAL_REQUESTS_FILE",
    "DEFAULT_MODEL",
    "DEFAULT_QUEUE_DIR",
    "DEFAULT_ROUTINE_INTERVAL_SECONDS",
    "DEFAULT_ROUTINE_POLL_SECONDS",
    "DEFAULT_ROUTINE_TIMEZONE",
    "DEFAULT_STATE_FILE",
    "DEFAULT_WS_URL",
    "LOGIN_ENV_TIMEOUT_SECONDS",
    "MIN_ROUTINE_INTERVAL_SECONDS",
    "LabelQueryInput",
    "LabelResolutionPrefer",
    "Mode",
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
    "extract_queued_id",
    "extract_started_turn_id",
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
    "is_queue_item_id",
    "response_is_queued",
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
    "parse_routine_interval_seconds",
    "parse_routine_time",
    "path_basename",
    "preview_text",
    "read_login_shell_environment",
    "read_permission_store",
    "record_shared_permission_request",
    "required_label",
    "routine_from_patch",
    "routine_interval_next_run_at",
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


async def login_shell_config_override() -> JsonObject:
    env = await login_shell_environment()
    set_values = {key: value for key in ["PATH", "SHELL", "HOME", "USER", "LOGNAME"] if (value := env.get(key))}
    return {"shell_environment_policy": {"inherit": "all", "set": set_values}}


def default_model_from_environment() -> str:
    return default_super_agents_model() or DEFAULT_MODEL


class CodexAppServerClient(
    TransportClientMixin,
    PermissionClientMixin,
    ThreadLifecycleMixin,
    TurnLifecycleMixin,
    TagsFavoritesMixin,
    LabelQueryMixin,
    RoutineClientMixin,
    EventClientMixin,
    SessionClientMixin,
):
    def __init__(
        self,
        ws_url: str | None = None,
        state_file: str | Path | None = None,
        default_model: str | None = None,
        permission_callback: PermissionRequestCallback | None = None,
        approval_requests_file: str | Path | None = None,
        queue_dir: str | Path | None = None,
        backend_identity: str | None = None,
        endpoint: str | None = None,
    ) -> None:
        self.backend = normalize_backend(backend_identity or CODEX_BACKEND)
        if execution_backend(self.backend) != CODEX_BACKEND:
            raise ValueError(f"{self.backend} is not a Codex-compatible backend.")
        if ws_url and endpoint:
            raise ValueError("Pass either ws_url or endpoint, not both.")
        self.endpoint: AppServerEndpoint = configured_app_server_endpoint(endpoint or ws_url)
        # Compatibility alias for callers that still inspect ``ws_url``. It
        # may now contain a unix:// endpoint rather than a TCP WebSocket URL.
        self.ws_url = self.endpoint.value
        self.state_file = Path(state_file or os.environ.get("SUPER_AGENTS_STATE_FILE") or DEFAULT_STATE_FILE)
        self.queue_dir = Path(
            queue_dir or os.environ.get("SUPER_AGENTS_QUEUE_DIR") or self.state_file.parent / "queues"
        )
        self.approval_requests_file = Path(
            approval_requests_file
            or os.environ.get("SUPER_AGENTS_APPROVAL_REQUESTS_FILE")
            or self.state_file.with_name("approval-requests.json")
        )
        self.default_model = default_model or default_model_from_environment()
        self._ws: Any | None = None
        self._next_id = 1
        self._pending: dict[str | int, asyncio.Future[Any]] = {}
        self._timed_out_requests: dict[str | int, JsonObject] = {}
        self._pending_server_requests: dict[str | int, PendingServerRequest] = {}
        self._turns: dict[str, TurnState] = {}
        self._connect_lock = asyncio.Lock()
        self._reader_task: asyncio.Task[None] | None = None
        self._state_lock = asyncio.Lock()
        self._queue_tasks: dict[str, asyncio.Task[None]] = {}
        self._routine_scheduler_task: asyncio.Task[None] | None = None
        self._permission_callback = permission_callback
        self._permission_callback_tasks: set[asyncio.Task[None]] = set()
        self._permission_decision_tasks: set[asyncio.Task[None]] = set()
        self._merge_session_tasks: set[asyncio.Task[None]] = set()
        self._initialize_result: JsonObject = {}
        self._last_connection_error: str | None = None

    async def status(self) -> JsonObject:
        ready = await self.check_ready()
        return {
            "backend": self.backend,
            "ready": ready,
            "transport": self.endpoint.transport,
            "endpointSource": self.endpoint.source,
            "appServerEndpoint": self.endpoint.description,
            "websocketUrl": self.ws_url if not self.endpoint.is_unix else None,
            "websocketConnected": websocket_is_open(self._ws),
            "appServerVersion": self._initialize_result.get("userAgent"),
            "lastConnectionError": self._last_connection_error,
            "managedProcess": False,
            "pendingRequests": [request.to_json() for request in self._pending_server_requests.values()],
            "pendingPermissionRequests": [request.to_json() for request in self.pending_permission_requests()],
            "recentTimedOutRpcRequests": self.recent_timed_out_requests(),
            "queuedTurns": self.queued_turn_summary(),
            "activeTurns": [
                self.compact_tracked_turn(turn) for turn in self._turns.values() if self.tracked_turn_is_active(turn)
            ],
            "routines": await self.routine_status_summary(),
        }

    def start_routine_scheduler(self) -> None:
        if self._routine_scheduler_task and not self._routine_scheduler_task.done():
            return
        self._routine_scheduler_task = asyncio.create_task(self._routine_scheduler_loop())

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

    async def _login_shell_config_override(self) -> JsonObject:
        return await login_shell_config_override()
