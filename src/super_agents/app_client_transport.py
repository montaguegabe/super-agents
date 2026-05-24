from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
import time
from typing import Any

import websockets

from .app_environment import _check_ready_sync, login_shell_environment, websocket_is_open
from .app_formatting import as_object, text_preview
from .app_models import PendingServerRequest, TurnState
from .app_permissions import (
    is_permission_request,
    pop_shared_permission_decision,
    record_shared_permission_request,
)
from .app_protocol import (
    extract_notification_thread_id,
    extract_notification_turn_id,
)
from .app_time import iso_now, turn_key
from .state import JsonObject

logger = logging.getLogger(__name__)
DEFAULT_WEBSOCKET_MAX_SIZE = 16 * 1024 * 1024


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
