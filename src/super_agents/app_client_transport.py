from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from typing import Any

import websockets

from .app_environment import _check_ready_sync, websocket_is_open
from .app_formatting import as_object, text_preview
from .app_models import PendingServerRequest, TurnState
from .app_permissions import (
    is_permission_request,
    normalize_permission_response,
    pop_shared_permission_decision,
    record_shared_permission_request,
)
from .app_protocol import (
    extract_notification_thread_id,
    extract_notification_turn_id,
)
from .app_sessions import turn_patch
from .app_time import iso_now, turn_key
from .state import JsonObject

logger = logging.getLogger(__name__)
DEFAULT_WEBSOCKET_MAX_SIZE = 16 * 1024 * 1024
CONTROL_PLANE_DIAGNOSTICS_ENV = "SUPER_AGENTS_CONTROL_PLANE_DIAGNOSTICS"
TIMED_OUT_REQUEST_LIMIT = 50


def websocket_max_size() -> int | None:
    raw = os.environ.get("SUPER_AGENTS_WEBSOCKET_MAX_SIZE")
    if raw is None or raw.strip() == "":
        return DEFAULT_WEBSOCKET_MAX_SIZE
    if raw.lower() in {"none", "unlimited", "0"}:
        return None
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "Ignoring invalid SUPER_AGENTS_WEBSOCKET_MAX_SIZE=%r; using default %s",
            raw,
            DEFAULT_WEBSOCKET_MAX_SIZE,
        )
        return DEFAULT_WEBSOCKET_MAX_SIZE


def control_plane_diagnostics_enabled() -> bool:
    raw = os.environ.get(CONTROL_PLANE_DIAGNOSTICS_ENV, "")
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def rpc_context(params: JsonObject | None) -> tuple[str, str]:
    params = params or {}
    thread_id = str(params.get("threadId") or "")
    turn_id = str(params.get("turnId") or params.get("expectedTurnId") or "")
    return thread_id, turn_id


def rpc_log_context(params: JsonObject | None, context: JsonObject | None = None) -> JsonObject:
    params = params or {}
    context = context or {}
    thread_id = str(context.get("threadId") or params.get("threadId") or "")
    turn_id = str(context.get("turnId") or params.get("turnId") or params.get("expectedTurnId") or "")
    return {
        "dispatch_id": str(context.get("dispatchId") or context.get("mcpCallId") or ""),
        "thread_id": thread_id,
        "turn_id": turn_id,
        "target_name": str(context.get("name") or context.get("label") or ""),
    }


class TransportClientMixin:
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
        for task in self._merge_session_tasks:
            task.cancel()
        self._merge_session_tasks.clear()
        for task in self._queue_tasks.values():
            task.cancel()
        self._queue_tasks.clear()
        if self._reader_task:
            self._reader_task.cancel()
            self._reader_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    async def _connect(self) -> None:
        if not await self.check_ready():
            raise RuntimeError(
                "Codex app-server is not running or not reachable at "
                f"{self.ws_url}. Start the Openbase-managed codex-app-server service."
            )
        self._ws = await asyncio.wait_for(
            websockets.connect(self.ws_url, max_size=websocket_max_size()),
            timeout=5,
        )
        self._reader_task = asyncio.create_task(self._reader_loop())
        await self.request(
            "initialize",
            {
                "clientInfo": {"name": "super-agents-mcp", "title": "Super Agents MCP", "version": "0.1.0"},
                "capabilities": {"experimentalApi": True},
            },
        )
        await self.send({"method": "initialized", "params": {}})

    async def check_ready(self) -> bool:
        ready_url = self.ws_url.replace("ws:", "http:", 1).replace("wss:", "https:", 1).rstrip("/") + "/readyz"
        try:
            return await asyncio.to_thread(_check_ready_sync, ready_url)
        except Exception:
            return False

    async def request(
        self,
        method: str,
        params: JsonObject | None = None,
        timeout_seconds: float = 30,
        *,
        context: JsonObject | None = None,
    ) -> JsonObject:
        request_id = self._next_id
        self._next_id += 1
        started = time.monotonic()
        request_params = params or {}
        log_context = rpc_log_context(request_params, context)
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Any] = loop.create_future()
        self._pending[request_id] = future
        await self.send({"id": request_id, "method": method, "params": request_params})
        try:
            result = await asyncio.wait_for(future, timeout=timeout_seconds)
        except TimeoutError:
            self._pending.pop(request_id, None)
            elapsed_ms = int((time.monotonic() - started) * 1000)
            self.remember_timed_out_request(request_id, method, log_context, started, elapsed_ms)
            logger.info(
                "dispatch_timing stage=app_server_rpc_timeout request_id=%s method=%s "
                "dispatch_id=%s thread_id=%s turn_id=%s target_name=%s elapsed_ms=%d",
                request_id,
                method,
                log_context["dispatch_id"],
                log_context["thread_id"],
                log_context["turn_id"],
                log_context["target_name"],
                elapsed_ms,
            )
            raise TimeoutError(f"Timed out waiting for app-server response to {method}.") from None
        logger.info(
            "dispatch_timing stage=app_server_rpc_response request_id=%s method=%s "
            "dispatch_id=%s thread_id=%s turn_id=%s target_name=%s elapsed_ms=%d",
            request_id,
            method,
            log_context["dispatch_id"],
            log_context["thread_id"],
            log_context["turn_id"],
            log_context["target_name"],
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
            self.log_late_rpc_response(request_id, message)
            return
        if message.get("error"):
            pending.set_exception(RuntimeError(json.dumps(message["error"])))
        else:
            pending.set_result(message.get("result"))

    def remember_timed_out_request(
        self,
        request_id: str | int,
        method: str,
        context: JsonObject,
        started: float,
        elapsed_ms: int,
    ) -> None:
        timed_out = getattr(self, "_timed_out_requests", None)
        if timed_out is None:
            return
        timed_out[request_id] = {
            "method": method,
            "dispatch_id": context.get("dispatch_id") or "",
            "thread_id": context.get("thread_id") or "",
            "turn_id": context.get("turn_id") or "",
            "target_name": context.get("target_name") or "",
            "started": started,
            "timeout_elapsed_ms": elapsed_ms,
        }
        while len(timed_out) > TIMED_OUT_REQUEST_LIMIT:
            timed_out.pop(next(iter(timed_out)))

    def log_late_rpc_response(self, request_id: str | int, message: JsonObject) -> None:
        timed_out = getattr(self, "_timed_out_requests", None)
        if not timed_out:
            return
        metadata = timed_out.get(request_id)
        if not metadata:
            return
        elapsed_ms = int((time.monotonic() - float(metadata.get("started") or time.monotonic())) * 1000)
        metadata["late_response_elapsed_ms"] = elapsed_ms
        metadata["late_response_has_error"] = bool(message.get("error"))
        if not control_plane_diagnostics_enabled():
            return
        logger.info(
            "dispatch_timing stage=app_server_rpc_late_response request_id=%s method=%s "
            "dispatch_id=%s thread_id=%s turn_id=%s target_name=%s "
            "elapsed_ms=%d timeout_elapsed_ms=%s has_error=%s",
            request_id,
            metadata.get("method") or "",
            metadata.get("dispatch_id") or "",
            metadata.get("thread_id") or "",
            metadata.get("turn_id") or "",
            metadata.get("target_name") or "",
            elapsed_ms,
            metadata.get("timeout_elapsed_ms") or "",
            bool(message.get("error")),
        )

    def recent_timed_out_requests(self) -> list[JsonObject]:
        timed_out = getattr(self, "_timed_out_requests", None)
        if not timed_out:
            return []
        now = time.monotonic()
        result: list[JsonObject] = []
        for request_id, metadata in timed_out.items():
            started = float(metadata.get("started") or now)
            result.append(
                {
                    "requestId": request_id,
                    "method": metadata.get("method") or "",
                    "dispatchId": metadata.get("dispatch_id") or "",
                    "threadId": metadata.get("thread_id") or "",
                    "turnId": metadata.get("turn_id") or "",
                    "targetName": metadata.get("target_name") or "",
                    "ageMs": int((now - started) * 1000),
                    "timeoutElapsedMs": metadata.get("timeout_elapsed_ms"),
                    "lateResponseElapsedMs": metadata.get("late_response_elapsed_ms"),
                    "lateResponseHasError": metadata.get("late_response_has_error"),
                }
            )
        return result

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
            last_useful_message = text_preview(params) or method
            self._schedule_merge_session(
                thread_id,
                {
                    "threadId": thread_id,
                    "activeTurnId": turn_id,
                    "lastTurnId": turn_id,
                    "lastStatus": "waiting",
                    "lastEventAt": pending_request.received_at,
                    "lastUsefulMessage": last_useful_message,
                    "turns": {
                        turn_id: turn_patch(
                            turn_id,
                            "waiting",
                            turn=turn,
                            updated_at=pending_request.received_at,
                            last_useful_message=last_useful_message,
                        )
                    },
                },
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
            await self.answer_request(
                pending_request.id,
                normalize_permission_response(pending_request, result),
            )
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
            logger.exception(
                "Could not schedule Super Agents permission decision polling without a running event loop."
            )
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
            if method in {"turn/completed", "turn/failed", "turn/cancelled", "turn/canceled", "turn/interrupted"}:
                logger.warning(
                    "Super Agents terminal notification missing identifiers method=%s thread_id=%s turn_id=%s",
                    method,
                    thread_id or "",
                    turn_id or "",
                )
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
        elif method in {"turn/cancelled", "turn/canceled", "turn/interrupted"}:
            turn.status = "cancelled"
            turn.finished_at = iso_now()
        elif turn.status not in {"waiting", "completed", "failed", "cancelled"}:
            turn.status = "running"
        if turn.status in {"completed", "failed", "cancelled"}:
            self.signal_turn_update(turn)
            logger.info(
                "Super Agents terminal notification received method=%s thread_id=%s turn_id=%s "
                "status=%s received_at=%s",
                method,
                thread_id,
                turn_id,
                turn.status,
                received_at,
            )

        last_useful_message = text_preview(params) or method
        clear_fields = ["activeTurnId"] if turn.status in {"completed", "failed", "cancelled"} else []
        merge_task = self._schedule_merge_session(
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
                    turn_id: turn_patch(
                        turn_id,
                        turn.status,
                        turn=turn,
                        updated_at=received_at,
                        last_useful_message=last_useful_message,
                    )
                },
            },
            clear_fields=clear_fields,
        )
        if turn.status in {"completed", "failed", "cancelled"}:

            def on_terminal_merge_done(
                task: asyncio.Task[None],
                completed_thread_id: str = thread_id,
                completed_turn_id: str = turn_id,
                completed_status: str = turn.status,
            ) -> None:
                self._handle_terminal_merge_done(task, completed_thread_id, completed_turn_id, completed_status)

            merge_task.add_done_callback(on_terminal_merge_done)

    def _handle_terminal_merge_done(
        self,
        task: asyncio.Task[None],
        thread_id: str,
        turn_id: str,
        status: str,
    ) -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            logger.warning(
                "Super Agents terminal state merge cancelled thread_id=%s turn_id=%s status=%s "
                "scheduling_queue_drain=true",
                thread_id,
                turn_id,
                status,
            )
        except Exception:
            logger.exception(
                "Super Agents terminal state merge failed thread_id=%s turn_id=%s status=%s "
                "scheduling_queue_drain=true",
                thread_id,
                turn_id,
                status,
            )
        else:
            logger.info(
                "Super Agents terminal state merge completed thread_id=%s turn_id=%s status=%s "
                "scheduling_queue_drain=true",
                thread_id,
                turn_id,
                status,
            )
        self.schedule_queue_drain(thread_id)

    def _schedule_merge_session(
        self,
        thread_id: str,
        patch: JsonObject,
        clear_fields: list[str] | None = None,
    ) -> asyncio.Task[None]:
        """Schedule a state merge, retaining the task so it cannot be garbage collected."""
        task = asyncio.create_task(self.merge_session(thread_id, patch, clear_fields=clear_fields))
        self._merge_session_tasks.add(task)
        task.add_done_callback(self._merge_session_tasks.discard)
        return task

    def ensure_turn(self, thread_id: str, turn_id: str) -> TurnState:
        key = turn_key(thread_id, turn_id)
        if key not in self._turns:
            self._turns[key] = TurnState(thread_id=thread_id, turn_id=turn_id, status="running", started_at=iso_now())
        return self._turns[key]

    def signal_turn_update(self, turn: TurnState) -> None:
        """Wake any caller awaiting progress on this turn (see wait_for_turn_update)."""
        if not turn.update_event.is_set():
            turn.update_event.set()

    async def wait_for_turn_update(self, thread_id: str, turn_id: str, timeout: float) -> None:
        """Await the next update to a turn instead of polling on a fixed interval.

        Returns as soon as a terminal notification (or other progress signal)
        fires for the turn, or after ``timeout`` seconds as a safety net for the
        rare case a push is missed on an otherwise-healthy connection. Callers
        re-read progress after this returns; the event is a wakeup, not the
        answer. A dropped websocket surfaces through the caller's own progress
        read (which raises), not here.
        """
        turn = self.ensure_turn(thread_id, turn_id)
        event = turn.update_event
        try:
            await asyncio.wait_for(event.wait(), timeout)
        except (TimeoutError, asyncio.TimeoutError):
            pass
        finally:
            # Keep the event latched once terminal so a caller that re-reads
            # progress and still sees a stale non-terminal snapshot wakes again
            # immediately instead of waiting out the fallback timeout.
            if turn.status not in {"completed", "failed", "cancelled"}:
                event.clear()

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
