"""Super Agents backend client for the Claude Agent SDK.

Option building lives in claude_options, prompt templating in claude_prompts,
and log serialization in claude_logs; this module keeps the client itself.
"""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import os
import uuid
from pathlib import Path
from typing import Any, Callable

try:  # POSIX advisory locking; unavailable on Windows.
    import fcntl
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore[assignment]

from super_agents.agent_store import Session, Store, Turn, iso_now
from super_agents.app_formatting import apply_field_selection, without_none
from super_agents.app_models import LabelQueryInput, QueueCancelInput
from super_agents.app_protocol import (
    _without_super_agent_identity_lines,
    is_active_status,
    with_super_agent_identity_instructions,
)
from super_agents.app_sessions import required_label
from super_agents.claude_logs import (
    append_log,
)
from super_agents.claude_logs import (
    message_preview as _message_preview,
)
from super_agents.claude_logs import (
    message_session_id as _message_session_id,
)
from super_agents.claude_logs import (
    message_to_log as _message_to_log,
)
from super_agents.claude_options import (  # noqa: F401  (constants re-exported for compatibility)
    CLAUDE_CONFIG_DIR_ENV,
    CLAUDE_CONFIG_FILENAME,
    CLAUDE_INSTRUCTIONS_FILENAME,
    CLAUDE_PERMISSION_MODE,
    CLAUDE_SDK_ENV_OVERRIDES,
    CLAUDE_SERVICE_TIER_EFFORTS,
    CLAUDE_SETTINGS_FILENAME,
)
from super_agents.claude_options import (
    agent_options as _agent_options,
)
from super_agents.claude_options import (
    claude_effort as _claude_effort,
)
from super_agents.claude_prompts import (
    combine_developer_instructions as _combine_developer_instructions,
)
from super_agents.claude_prompts import (
    with_claude_turn_context as _with_claude_turn_context,
)
from super_agents.defaults import (
    default_super_agents_model,
    default_super_agents_reasoning_effort,
)

JsonObject = dict[str, Any]
SdkLoader = Callable[[], Any]

SDK_IMPORT_ERROR = (
    "Claude Code backend requires the claude-agent-sdk package. "
    "Install super-agents with the claude extra, or install claude-agent-sdk in this environment."
)


class ClaudeAgentSdkClient:
    """Super Agents backend using Claude Code without Anthropic API keys."""

    backend = "claude_code"

    def __init__(self, store: Store | None = None, sdk_loader: SdkLoader | None = None) -> None:
        self.store = store or Store()
        self._sdk_loader = sdk_loader or _load_sdk
        self._sdk_clients: dict[str, Any] = {}
        self._sdk_client_efforts: dict[str, tuple[str | None, str | None]] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._queue_tasks: dict[str, asyncio.Task[None]] = {}
        self._turn_tasks: set[asyncio.Task[None]] = set()
        # Identifies this client instance in the shared store so instances in
        # other processes can tell their cached CLI's conversation leaf is
        # stale (see _sdk_client_for).
        self._instance_id = f"ci_{uuid.uuid4().hex}"
        self._closed = False

    async def status(self) -> JsonObject:
        sessions = self.store.list_sessions(include_inactive=True)
        ready, error = self._sdk_ready()
        return without_none(
            {
                "ready": ready,
                "backend": self.backend,
                "managedProcess": False,
                "sdkPackage": "claude-agent-sdk",
                "sdkError": error,
                "dataStore": str(self.store.path),
                "pendingRequests": [],
                "pendingPermissionRequests": [],
                "queuedTurns": [turn.to_json() for session in sessions for turn in self.store.queued_turns(session.id)],
                "activeTurns": [self._status_item(session) for session in sessions if is_active_status(session.status)],
            }
        )

    async def start_thread(self, input_data: JsonObject) -> JsonObject:
        name = str(input_data.get("name") or input_data.get("label") or "").strip()
        if not name:
            raise ValueError("name is required.")
        agent_name = _optional_str(input_data.get("agentName"))
        developer_instructions = with_super_agent_identity_instructions(
            _optional_str(input_data.get("developerInstructions")),
            name,
            agent_name=agent_name,
        )
        existing = self.store.get_by_name(name)
        if existing is not None and bool(input_data.get("fresh")):
            # Retire the name-holder so the caller gets a brand-new session
            # (and conversation) instead of the reuse-by-name refresh below.
            self.store.rename_session(existing.id, f"{name} (retired {existing.id[-8:]})")
            existing = None
        if existing is not None:
            effective_agent_name = agent_name or existing.agent_name
            effective_developer_instructions = with_super_agent_identity_instructions(
                _optional_str(input_data.get("developerInstructions")) or existing.developer_instructions,
                name,
                agent_name=effective_agent_name,
            )
            session = self.store.update_session(
                existing.id,
                cwd=str(Path(_optional_str(input_data.get("cwd")) or existing.cwd).expanduser()),
                agent_name=effective_agent_name,
                developer_instructions=effective_developer_instructions,
                model=_optional_str(input_data.get("model")) or existing.model or self._default_model(),
                status="waiting",
                active_turn_id=None,
                last_observed_state="Claude Code thread refreshed",
            )
            return {"backend": self.backend, "threadId": session.id, "session": session.to_json()}
        session = self.store.create_session(
            name,
            cwd=_optional_str(input_data.get("cwd")),
            agent_name=agent_name,
            developer_instructions=developer_instructions,
            model=_optional_str(input_data.get("model")) or self._default_model(),
            command=["claude-agent-sdk"],
        )
        return {"backend": self.backend, "threadId": session.id, "session": session.to_json()}

    async def resume_by_label(
        self,
        input_data: LabelQueryInput,
        *,
        developer_instructions: str | None = None,
    ) -> JsonObject:
        session = self._resolve_session(input_data)
        if developer_instructions:
            effective_developer_instructions = with_super_agent_identity_instructions(
                _combine_developer_instructions(
                    _without_super_agent_identity_lines(session.developer_instructions),
                    developer_instructions,
                ),
                session.name,
                session.id,
                session.agent_name,
            )
            session = self.store.update_session(
                session.id,
                developer_instructions=effective_developer_instructions,
            )
        return {
            "backend": self.backend,
            "name": session.name,
            "threadId": session.id,
            "session": session.to_json(),
        }

    async def read_by_label(self, input_data: LabelQueryInput, include_turns: bool = False) -> JsonObject:
        session = self._resolve_session(input_data)
        turns = self.store.list_turns(session.id, limit=input_data.max_items or 20)
        payload: JsonObject = {
            "backend": self.backend,
            "threadId": session.id,
            "name": session.name,
            "session": session.to_json(),
            "logTail": self.store.tail_log(session, lines=80),
        }
        if include_turns:
            payload["turns"] = [self._turn_view(session, turn) for turn in turns]
        else:
            payload["recentTurns"] = [self._turn_view(session, turn) for turn in turns[:5]]
        return payload

    async def rename_by_label(self, input_data: LabelQueryInput, new_name: str) -> JsonObject:
        session = self._resolve_session(input_data)
        renamed = self.store.rename_session(session.id, new_name)
        return {
            "backend": self.backend,
            "renamed": True,
            "name": renamed.name,
            "previousName": session.name,
            "threadId": renamed.id,
        }

    async def answer_request(self, request_id: str | int, result: JsonObject) -> JsonObject:
        return self._unsupported("codex_answer_request", requestId=request_id, result=result)

    async def sessions(self) -> list[JsonObject]:
        return [self._session_view(session) for session in self.store.list_sessions(include_inactive=True)]

    async def active(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        items = [self._agent_item(session, query) for session in self._query_sessions(query, include_inactive=False)]
        return {"backend": self.backend, "count": len(items), "agents": items[: query.limit or 50]}

    async def recent(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        items = [
            self._agent_item(session, query)
            for session in self._query_sessions(query, include_inactive=bool(query.include_inactive))
        ]
        return {"backend": self.backend, "count": len(items), "agents": items[: query.limit or 20]}

    async def compact_status(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        items = [
            self._status_item(session)
            for session in self._query_sessions(query, include_inactive=bool(query.include_inactive))
        ][: query.limit or 50]
        return {
            "backend": self.backend,
            "count": len(items),
            "agents": [apply_field_selection(item, query.fields) for item in items],
        }

    async def resolve_label(self, input_data: LabelQueryInput) -> JsonObject:
        return self._status_item(self._resolve_session(input_data))

    async def progress_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        session = self._resolve_session(input_data)
        payload = self._status_item(session)
        if input_data.turn_id:
            try:
                turn = self.store.get_turn(input_data.turn_id)
            except KeyError as exc:
                raise ValueError(
                    f"No turn {input_data.turn_id} is known for thread {session.id}."
                ) from exc
            if turn.session_id != session.id:
                raise ValueError(
                    f"Turn {input_data.turn_id} does not belong to thread {session.id}."
                )
            payload["turnId"] = turn.id
            payload["status"] = turn.status
            turn_view = self._turn_view(session, turn)
            payload["turn"] = turn_view
            payload["turns"] = [turn_view]
            payload["logTail"] = self.store.tail_log(session, lines=40)
            return payload

        turns = self.store.list_turns(session.id, limit=input_data.max_items or 10)
        payload["turns"] = [self._turn_view(session, turn) for turn in turns]
        payload["logTail"] = self.store.tail_log(session, lines=40)
        return payload

    async def steer_by_label(
        self,
        input_data: LabelQueryInput,
        prompt: str,
        turn_input: JsonObject | None = None,
    ) -> JsonObject:
        session = self._resolve_session(input_data)
        if self._session_is_busy(session):
            return await self._steer_active_turn(
                session,
                prompt,
                turn_input or {},
                requested_turn_id=input_data.turn_id,
            )
        return await self.start_turn_by_label(
            input_data,
            {**(turn_input or {}), "prompt": prompt},
        )

    async def cancel_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        session = self._resolve_session(input_data)
        sdk_client = self._sdk_clients.get(session.id)
        if sdk_client and hasattr(sdk_client, "interrupt"):
            await sdk_client.interrupt()
        if session.active_turn_id:
            self.store.update_turn(session.active_turn_id, status="cancelled", finished_at=iso_now())
        refreshed = self.store.update_session(
            session.id,
            status="cancelled",
            active_turn_id=None,
            last_observed_state="interrupt sent",
        )
        return {"backend": self.backend, "cancelled": True, "threadId": refreshed.id, "name": refreshed.name}

    async def start_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject:
        session = self._resolve_session(input_data)
        if self._session_is_busy(session):
            return await self._steer_active_turn(
                session,
                str(turn_input["prompt"]),
                turn_input,
                requested_turn_id=input_data.turn_id,
            )
        turn = self._launch_turn(
            session,
            turn_input,
            last_observed_state="running via Claude Code",
        )
        return {
            "backend": self.backend,
            "threadId": session.id,
            "name": session.name,
            "turnId": turn.id,
            "turn": turn.to_json(),
            "queued": False,
            "startedImmediately": True,
            "drain": "started_immediately",
        }

    async def queue_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject:
        session = self._resolve_session(input_data)
        if not self._session_is_busy(session):
            return await self.start_turn_by_label(input_data, turn_input)
        turn = self.store.create_turn(
            session.id,
            str(turn_input["prompt"]),
            status="queued",
            mode=_optional_str(turn_input.get("mode")),
            model=_optional_str(turn_input.get("model")) or session.model or self._default_model(),
            reasoning_effort=_optional_str(turn_input.get("reasoningEffort")) or self._default_reasoning_effort(),
            service_tier=_optional_str(turn_input.get("serviceTier")),
        )
        position = len(self.store.queued_turns(session.id))
        self._schedule_queue_drain(session.id)
        return {
            "backend": self.backend,
            "queued": True,
            "threadId": session.id,
            "name": session.name,
            "turnId": turn.id,
            "position": position,
            "queueDepth": position,
            "item": turn.to_json(),
            "drain": "scheduled",
        }

    async def cancel_queued_turn(self, input_data: QueueCancelInput) -> JsonObject:
        session = (
            self._resolve_session(
                LabelQueryInput(label=input_data.label, thread_id=input_data.thread_id, cwd=input_data.cwd)
            )
            if input_data.label or input_data.thread_id
            else None
        )
        if input_data.queue_item_id:
            try:
                turn = self.store.get_turn(input_data.queue_item_id)
            except KeyError as exc:
                raise ValueError(f"No queued Super Agents turn found for queueItemId {input_data.queue_item_id}.") from exc
            if session and turn.session_id != session.id:
                raise ValueError(f"No queued Super Agents turn found for queueItemId {input_data.queue_item_id}.")
            session = session or self.store.get_session(turn.session_id)
            position = self._queued_turn_position(session.id, turn.id)
        elif input_data.position is not None:
            if input_data.position < 1:
                raise ValueError("position must be a positive 1-based queue position.")
            if session is None:
                raise ValueError("threadId or name must be provided when canceling by position.")
            queued = self.store.queued_turns(session.id)
            if len(queued) < input_data.position:
                raise ValueError(f"No queued Super Agents turn found at position {input_data.position}.")
            turn = queued[input_data.position - 1]
            position = input_data.position
        else:
            raise ValueError("queueItemId or position must be provided.")

        if turn.status != "queued":
            raise ValueError(f"Queued Super Agents turn {turn.id} has already started and cannot be cancelled.")
        cancelled = self.store.update_turn(turn.id, status="cancelled", finished_at=iso_now())
        queue_depth = len(self.store.queued_turns(session.id))
        if queue_depth == 0:
            task = self._cancel_queue_drain(session.id)
            if task:
                await asyncio.gather(task, return_exceptions=True)
        return {
            "backend": self.backend,
            "cancelled": True,
            "removed": True,
            "threadId": session.id,
            "name": session.name,
            "queueItemId": turn.id,
            "turnId": turn.id,
            "position": position,
            "queueDepth": queue_depth,
            "item": cancelled.to_json(),
        }

    async def thread_favorite(self, thread_id: str) -> JsonObject:
        return self._unsupported("super_agents_thread_favorite", threadId=thread_id)

    async def tags(self) -> JsonObject:
        return self._unsupported("super_agents_tags")

    async def thread_tags(self, thread_id: str, tags: list[Any] | None = None) -> JsonObject:
        return self._unsupported("super_agents_thread_tags", threadId=thread_id, tags=tags)

    async def report_tags(self, project_path: str, path: str, tags: list[Any] | None = None) -> JsonObject:
        return self._unsupported("super_agents_report_tags", projectPath=project_path, path=path, tags=tags)

    def _default_model(self) -> str | None:
        return default_super_agents_model(backend=self.backend)

    def _default_reasoning_effort(self) -> str:
        return default_super_agents_reasoning_effort()

    def _launch_turn(self, session: Session, turn_input: JsonObject, *, last_observed_state: str) -> Turn:
        sdk = self._require_sdk()
        prompt = str(turn_input["prompt"])
        sdk_prompt = self._prompt_for_session(session, turn_input)
        model = _optional_str(turn_input.get("model")) or session.model or self._default_model()
        reasoning_effort = _optional_str(turn_input.get("reasoningEffort")) or self._default_reasoning_effort()
        service_tier = _optional_str(turn_input.get("serviceTier"))
        turn = self.store.create_turn(
            session.id,
            prompt,
            status="running",
            mode=_optional_str(turn_input.get("mode")),
            model=model,
            reasoning_effort=reasoning_effort,
            service_tier=service_tier,
        )
        self.store.update_session(
            session.id,
            status="running",
            active_turn_id=turn.id,
            last_turn_id=turn.id,
            last_observed_state=last_observed_state,
        )
        self._spawn_turn_task(session.id, turn.id, sdk_prompt, model, reasoning_effort, service_tier, sdk)
        return turn

    def _spawn_turn_task(
        self,
        session_id: str,
        turn_id: str,
        prompt: str,
        model: str | None,
        reasoning_effort: str | None,
        service_tier: str | None,
        sdk: Any,
    ) -> None:
        """Run a turn in the background, retaining the task so it cannot be garbage collected."""
        task = asyncio.create_task(
            self._run_turn(session_id, turn_id, prompt, model, reasoning_effort, service_tier, sdk)
        )
        self._turn_tasks.add(task)
        task.add_done_callback(self._turn_tasks.discard)

    async def _run_turn(
        self,
        session_id: str,
        turn_id: str,
        prompt: str,
        model: str | None,
        reasoning_effort: str | None,
        service_tier: str | None,
        sdk: Any,
    ) -> None:
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock, self._cross_process_session_lock(session_id):
            # Re-read after acquiring the locks: another process may have run
            # turns (advancing the conversation) while this one waited.
            session = self.store.get_session(session_id)
            try:
                if self._turn_was_cancelled(turn_id):
                    self._finish_cancelled_turn(session_id, turn_id)
                    return
                sdk_client = await self._sdk_client_for(
                    session,
                    model,
                    reasoning_effort,
                    service_tier,
                    sdk,
                )
                last_useful_message = ""
                await sdk_client.query(prompt)
                # A freshly resumed CLI can emit a no-op ResultMessage
                # (num_turns == 0) from the resume handshake before this
                # query's response. receive_response() stops at any result,
                # so accepting that no-op would mark the turn completed with
                # no output while the real answer buffers unread — and every
                # later turn would then read the previous prompt's answer
                # (off-by-one responses). Keep reading until a result that
                # actually ran model turns.
                while True:
                    result_message: Any = None
                    async for message in sdk_client.receive_response():
                        append_log(session.log_path, _message_to_log(message))
                        if claude_session_id := _message_session_id(message):
                            self.store.update_session(session_id, backend_session_id=claude_session_id)
                        if useful := _message_preview(message):
                            last_useful_message = useful
                            self.store.update_turn(turn_id, last_useful_message=last_useful_message)
                        if getattr(message, "num_turns", None) is not None:
                            result_message = message
                    if result_message is None or not _is_noop_result(result_message):
                        break
                    if self._turn_was_cancelled(turn_id):
                        break
                if self._turn_was_cancelled(turn_id):
                    self._finish_cancelled_turn(session_id, turn_id)
                else:
                    self.store.update_turn(
                        turn_id,
                        status="completed",
                        finished_at=iso_now(),
                        last_useful_message=last_useful_message or None,
                    )
                    self._finish_session_turn(
                        session_id,
                        turn_id,
                        status="waiting",
                        active_turn_id=None,
                        last_observed_state="Claude Code response completed",
                        last_useful_message=last_useful_message or None,
                    )
            except Exception as exc:
                append_log(session.log_path, f"[{iso_now()}] ERROR {type(exc).__name__}: {exc}\n")
                # The stream may hold unread messages from this turn; a
                # reused client would hand them to the next turn's
                # receive_response(). Drop the client so the next turn
                # resumes from the transcript tip on a clean stream.
                await self._disconnect_sdk_client(session_id)
                if self._turn_was_cancelled(turn_id):
                    self._finish_cancelled_turn(session_id, turn_id)
                else:
                    self.store.update_turn(turn_id, status="failed", finished_at=iso_now(), last_error=str(exc))
                    self._finish_session_turn(
                        session_id,
                        turn_id,
                        status="failed",
                        active_turn_id=None,
                        last_observed_state=str(exc),
                    )
            finally:
                if self._closed:
                    await self._disconnect_sdk_client(session_id)
                self._schedule_queue_drain(session_id)

    async def _steer_active_turn(
        self,
        session: Session,
        prompt: str,
        turn_input: JsonObject,
        *,
        requested_turn_id: str | None = None,
    ) -> JsonObject:
        active_turn_id = session.active_turn_id
        if not active_turn_id:
            return await self.start_turn_by_label(
                LabelQueryInput(thread_id=session.id),
                {**turn_input, "prompt": prompt},
            )

        if requested_turn_id and requested_turn_id != active_turn_id:
            raise RuntimeError(
                f"Expected active turn id `{requested_turn_id}` but found `{active_turn_id}`."
            )

        sdk_client = await self._wait_for_active_sdk_client(session.id)
        refreshed = self.store.get_session(session.id)
        if refreshed.active_turn_id != active_turn_id or not self._session_is_busy(refreshed):
            return await self.start_turn_by_label(
                LabelQueryInput(thread_id=session.id),
                {**turn_input, "prompt": prompt},
            )
        if sdk_client is None:
            raise RuntimeError("Active Claude Code turn cannot accept steering before its SDK client is ready.")

        await sdk_client.query(self._prompt_for_session(refreshed, {**turn_input, "prompt": prompt}))
        self._record_session_leaf_owner(session.id)
        current_turn = self.store.get_turn(active_turn_id)
        self.store.update_session(
            session.id,
            status="running",
            active_turn_id=active_turn_id,
            last_turn_id=active_turn_id,
            last_observed_state="steering active turn via Claude Code",
        )
        return {
            "backend": self.backend,
            "threadId": session.id,
            "name": session.name,
            "turnId": active_turn_id,
            "turn": current_turn.to_json(),
            "queued": False,
            "steered": True,
            "nativeSteer": True,
            "startedImmediately": False,
            "drain": "steered_active_turn",
        }

    async def _wait_for_active_sdk_client(self, session_id: str) -> Any | None:
        for _ in range(100):
            client = self._sdk_clients.get(session_id)
            if client is not None:
                return client
            await asyncio.sleep(0.01)
        return None

    async def _interrupt_active_turn(self, session: Session, *, reason: str) -> str | None:
        turn_id = session.active_turn_id
        if not turn_id:
            return None

        with contextlib.suppress(KeyError):
            self.store.update_turn(turn_id, status="cancelled", finished_at=iso_now())

        sdk_client = self._sdk_clients.get(session.id)
        if sdk_client is not None and hasattr(sdk_client, "interrupt"):
            with contextlib.suppress(Exception):
                await sdk_client.interrupt()
        # The interrupted turn's reader may no longer be consuming (crashed
        # task, other process), leaving its unread response in this client's
        # stream; a reused stream hands that stale response to the next turn
        # and every answer shifts one prompt behind. Drop the client so the
        # steered turn reconnects from the transcript tip on a clean stream.
        await self._disconnect_sdk_client(session.id)

        self.store.update_session(
            session.id,
            status="running",
            active_turn_id=turn_id,
            last_observed_state=f"Claude Code turn interrupted: {reason}",
        )
        return turn_id

    def _finish_session_turn(self, session_id: str, turn_id: str, **fields: object) -> None:
        try:
            current = self.store.get_session(session_id)
        except KeyError:
            return
        if current.active_turn_id == turn_id:
            self.store.update_session(session_id, **fields)

    def _finish_cancelled_turn(self, session_id: str, turn_id: str) -> None:
        self._finish_session_turn(
            session_id,
            turn_id,
            status="cancelled",
            active_turn_id=None,
            last_observed_state="Claude Code response interrupted",
        )

    def _turn_was_cancelled(self, turn_id: str) -> bool:
        try:
            return self.store.get_turn(turn_id).status == "cancelled"
        except KeyError:
            return False

    async def _sdk_client_for(
        self,
        session: Session,
        model: str | None,
        reasoning_effort: str | None,
        service_tier: str | None,
        sdk: Any,
    ) -> Any:
        effective_effort = _claude_effort(reasoning_effort, service_tier)
        existing = self._sdk_clients.get(session.id)
        if existing is not None and not self._owns_session_leaf(session):
            # A client instance in another process ran the last turn, so this
            # cached CLI's in-memory conversation no longer matches the
            # session transcript tip. Reusing it would fork the conversation
            # from a stale leaf and orphan the other process's turns;
            # reconnect so the resume adopts the new tip.
            await self._disconnect_sdk_client(session.id)
            existing = None
        if existing is not None and self._sdk_client_efforts.get(session.id) == (effective_effort, service_tier):
            if model and hasattr(existing, "set_model"):
                await existing.set_model(model)
            self._record_session_leaf_owner(session.id)
            return existing
        await self._disconnect_sdk_client(session.id)
        options = _agent_options(
            sdk,
            session.cwd,
            model,
            effective_effort,
            resume=session.backend_session_id,
        )
        client = sdk.ClaudeSDKClient(options=options)
        await client.connect()
        self._sdk_clients[session.id] = client
        self._sdk_client_efforts[session.id] = (effective_effort, service_tier)
        self._record_session_leaf_owner(session.id)
        return client

    def _owns_session_leaf(self, session: Session) -> bool:
        return session.last_client_instance in (None, self._instance_id)

    def _record_session_leaf_owner(self, session_id: str) -> None:
        self.store.update_session(session_id, last_client_instance=self._instance_id)

    async def _disconnect_sdk_client(self, session_id: str) -> None:
        client = self._sdk_clients.pop(session_id, None)
        self._sdk_client_efforts.pop(session_id, None)
        if client is None:
            return
        disconnect = getattr(client, "disconnect", None)
        if disconnect is None:
            return
        with contextlib.suppress(Exception):
            await disconnect()

    @contextlib.asynccontextmanager
    async def _cross_process_session_lock(self, session_id: str):
        """Serialize a session's turns across processes.

        Several processes can hold client instances for the same store (a
        pool of voice workers plus long-lived MCP servers). Each keeps its
        own Claude CLI subprocess, and every CLI appends conversation entries
        to the one shared session transcript; unserialized turns interleave
        parent chains, and the next resume then forks from a stale leaf and
        silently drops the other writer's turns from the visible
        conversation. flock releases on process death, so a crashed worker
        cannot wedge the session.
        """
        if fcntl is None:
            yield
            return
        lock_dir = self.store.path.parent / "session-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_dir / f"{session_id}.lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    async def close(self) -> None:
        """Disconnect cached Claude CLI clients, flushing their transcripts.

        A live CLI subprocess can buffer transcript entries until it exits;
        a client left connected past its owner's lifetime flushes those
        entries after other processes' turns, corrupting the leaf the next
        resume picks. Sessions with an in-flight turn are left to disconnect
        when that turn finishes (see _run_turn) rather than being killed
        mid-answer here.
        """
        self._closed = True
        queue_tasks = list(self._queue_tasks.values())
        for task in queue_tasks:
            task.cancel()
        self._queue_tasks.clear()
        if queue_tasks:
            await asyncio.gather(*queue_tasks, return_exceptions=True)
        for session_id in list(self._sdk_clients):
            lock = self._session_locks.setdefault(session_id, asyncio.Lock())
            if lock.locked():
                continue
            async with lock:
                await self._disconnect_sdk_client(session_id)

    def _schedule_queue_drain(self, session_id: str) -> None:
        task = self._queue_tasks.get(session_id)
        if task and not task.done():
            return
        self._queue_tasks[session_id] = asyncio.create_task(self._queue_drain_loop(session_id))

    def _cancel_queue_drain(self, session_id: str) -> asyncio.Task[None] | None:
        task = self._queue_tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
            return task
        return None

    async def _queue_drain_loop(self, session_id: str) -> None:
        while True:
            await asyncio.sleep(0.5)
            try:
                session = self.store.get_session(session_id)
            except KeyError:
                return
            if self._session_is_busy(session):
                continue
            queued = self.store.queued_turns(session_id)
            if not queued:
                return
            turn = queued[0]
            sdk = self._require_sdk()
            self.store.update_turn(turn.id, status="running", attempts=turn.attempts + 1)
            self.store.update_session(
                session_id,
                status="running",
                active_turn_id=turn.id,
                last_turn_id=turn.id,
                last_observed_state="running queued turn via Claude Code",
            )
            self._spawn_turn_task(
                session_id,
                turn.id,
                self._prompt_for_session(session, {"prompt": turn.prompt}),
                turn.model,
                turn.reasoning_effort,
                turn.service_tier,
                sdk,
            )
            return

    def _resolve_session(self, input_data: LabelQueryInput) -> Session:
        if input_data.thread_id:
            try:
                return self.store.get_session(input_data.thread_id)
            except KeyError:
                pass
        return self.store.require_by_name(required_label(input_data))

    def _query_sessions(self, query: LabelQueryInput, *, include_inactive: bool) -> list[Session]:
        if query.thread_id or query.label:
            sessions = [self._resolve_session(query)]
        else:
            sessions = self.store.list_sessions(include_inactive=True, status=query.status)
        if query.cwd:
            cwd = str(Path(query.cwd).expanduser())
            sessions = [session for session in sessions if session.cwd == cwd]
        if query.status:
            sessions = [session for session in sessions if session.status == query.status]
        if not include_inactive:
            sessions = [session for session in sessions if is_active_status(session.status)]
        return sessions

    def _session_view(self, session: Session) -> JsonObject:
        return {"backend": self.backend, **session.to_json()}

    def _agent_item(self, session: Session, query: LabelQueryInput) -> JsonObject:
        turns = self.store.list_turns(session.id, limit=1)
        return apply_field_selection(
            without_none(
                {
                    "backend": self.backend,
                    "name": session.name,
                    "agentName": session.agent_name,
                    "threadId": session.id,
                    "turnId": session.active_turn_id or session.last_turn_id,
                    "cwd": session.cwd,
                    "status": session.status,
                    "model": session.model,
                    "updatedAt": session.updated_at,
                    "lastObservedState": session.last_observed_state,
                    "queueDepth": len(self.store.queued_turns(session.id)),
                    "preview": turns[0].to_json().get("promptPreview")
                    if turns and query.include_preview is not False
                    else None,
                }
            ),
            query.fields,
        )

    def _status_item(self, session: Session) -> JsonObject:
        queued = self.store.queued_turns(session.id)
        return without_none(
            {
                "backend": self.backend,
                "name": session.name,
                "agentName": session.agent_name,
                "threadId": session.id,
                "turnId": session.active_turn_id or session.last_turn_id,
                "cwd": session.cwd,
                "status": session.status,
                "model": session.model,
                "activeTurnId": session.active_turn_id,
                "lastTurnId": session.last_turn_id,
                "lastObservedState": session.last_observed_state,
                "lastUsefulMessage": session.last_useful_message,
                "queueDepth": len(queued),
                "updatedAt": session.updated_at,
            }
        )

    def _turn_view(self, session: Session, turn: Any) -> JsonObject:
        data = turn.to_json()
        if "lastUsefulMessage" not in data and turn.id == session.last_turn_id and session.last_useful_message:
            data["lastUsefulMessage"] = session.last_useful_message
        return data

    def _queued_turn_position(self, session_id: str, turn_id: str) -> int:
        for index, turn in enumerate(self.store.queued_turns(session_id), start=1):
            if turn.id == turn_id:
                return index
        return 0

    def _sdk_ready(self) -> tuple[bool, str | None]:
        try:
            self._sdk_loader()
        except Exception as exc:
            return False, f"{SDK_IMPORT_ERROR} ({type(exc).__name__}: {exc})"
        return True, None

    def _session_is_busy(self, session: Session) -> bool:
        return bool(session.active_turn_id or session.status == "running")

    def _prompt_for_session(self, session: Session, turn_input: JsonObject) -> str:
        prompt = str(turn_input["prompt"])
        developer_instructions = _combine_developer_instructions(
            session.developer_instructions,
            _optional_str(turn_input.get("developerInstructions")),
        )
        developer_instructions = with_super_agent_identity_instructions(
            developer_instructions,
            session.name,
            session.id,
            session.agent_name,
        )
        return _with_claude_turn_context(
            prompt,
            cwd=session.cwd,
            developer_instructions=developer_instructions,
        )

    def _require_sdk(self) -> Any:
        try:
            return self._sdk_loader()
        except Exception as exc:
            raise RuntimeError(SDK_IMPORT_ERROR) from exc

    def _unsupported(self, tool: str, **extra: Any) -> JsonObject:
        return {
            "backend": self.backend,
            "supported": False,
            "tool": tool,
            "error": f"{tool} is only available through the Codex app-server backend.",
            **{key: value for key, value in extra.items() if value is not None},
        }


def _is_noop_result(message: Any) -> bool:
    """A ResultMessage that ran no model turns (e.g. a resume handshake)."""
    return getattr(message, "num_turns", None) == 0


def _load_sdk() -> Any:
    return importlib.import_module("claude_agent_sdk")


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None
