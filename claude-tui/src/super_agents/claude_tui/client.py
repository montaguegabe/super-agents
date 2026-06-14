from __future__ import annotations

import asyncio
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

from super_agents.app_formatting import apply_field_selection, without_none
from super_agents.app_models import LabelQueryInput
from super_agents.app_protocol import is_active_status
from super_agents.app_sessions import required_label

from .config import build_claude_command
from .controller import Runtime
from .models import Session
from .storage import Store

JsonObject = dict[str, Any]


class ClaudeTuiClient:
    """MCP client adapter for local Claude Code TUI sessions."""

    backend = "claude-tui"

    def __init__(self, store: Store | None = None, runtime: Runtime | None = None) -> None:
        self.store = store or Store()
        self.runtime = runtime or Runtime(self.store)
        self._queue_tasks: dict[str, asyncio.Task[None]] = {}

    async def status(self) -> JsonObject:
        command = build_claude_command()
        executable = command[0] if command else "claude"
        sessions = self.store.list_sessions(include_inactive=True)
        return {
            "ready": bool(shutil.which(executable) or Path(executable).exists()),
            "backend": self.backend,
            "managedProcess": False,
            "claudeCommand": command,
            "dataStore": str(self.store.path),
            "pendingRequests": [],
            "pendingPermissionRequests": [],
            "queuedTurns": [turn.to_json() for session in sessions for turn in self.store.queued_turns(session.id)],
            "activeTurns": [
                self._status_item(session)
                for session in sessions
                if is_active_status(self._refresh_session(session).status)
            ],
        }

    async def start_thread(self, input_data: JsonObject) -> JsonObject:
        name = str(input_data.get("name") or input_data.get("label") or "").strip()
        if not name:
            raise ValueError("name is required.")
        model = _optional_str(input_data.get("model"))
        session = self.store.create_session(
            name,
            cwd=_optional_str(input_data.get("cwd")),
            agent_name=_optional_str(input_data.get("agentName")),
            model=model,
        )
        await asyncio.to_thread(self.runtime.start, session)
        return {"backend": self.backend, "threadId": session.id, "session": session.to_json()}

    async def resume_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        session = self._resolve_session(replace(input_data, prefer="latest_any"))
        state = await asyncio.to_thread(self.runtime.start, session)
        refreshed = self.store.get_session(session.id)
        return {
            "backend": self.backend,
            "name": refreshed.name,
            "threadId": refreshed.id,
            "observed": state.__dict__,
            "session": refreshed.to_json(),
        }

    async def read_by_label(self, input_data: LabelQueryInput, include_turns: bool = False) -> JsonObject:
        session = self._resolve_session(replace(input_data, prefer="latest_any"))
        turns = self.store.list_turns(session.id, limit=input_data.max_items or 20)
        payload: JsonObject = {
            "backend": self.backend,
            "threadId": session.id,
            "name": session.name,
            "session": session.to_json(),
            "logTail": self.store.tail_log(session, lines=80),
        }
        if include_turns:
            payload["turns"] = [turn.to_json() for turn in turns]
        else:
            payload["recentTurns"] = [turn.to_json() for turn in turns[:5]]
        return payload

    async def rename_by_label(self, input_data: LabelQueryInput, new_name: str) -> JsonObject:
        session = self._resolve_session(replace(input_data, prefer="latest_any"))
        renamed = self.store.rename_session(session.id, new_name)
        return {
            "backend": self.backend,
            "renamed": True,
            "name": renamed.name,
            "previousName": session.name,
            "threadId": renamed.id,
        }

    async def answer_request(self, request_id: str | int, result: JsonObject) -> JsonObject:
        session = self._resolve_session(LabelQueryInput(label=str(request_id), thread_id=str(request_id)))
        decision = _decision_from_result(result)
        await asyncio.to_thread(self.runtime.answer, session, decision)
        return {
            "backend": self.backend,
            "answered": True,
            "threadId": session.id,
            "name": session.name,
            "decision": decision,
        }

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
        session = self._resolve_session(input_data)
        return self._status_item(session)

    async def progress_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        session = self._resolve_session(input_data)
        turns = self.store.list_turns(session.id, limit=input_data.max_items or 10)
        payload = self._status_item(session)
        payload["turns"] = [turn.to_json() for turn in turns]
        payload["logTail"] = self.store.tail_log(session, lines=40)
        return payload

    async def steer_by_label(self, input_data: LabelQueryInput, prompt: str) -> JsonObject:
        session = self._resolve_session(input_data)
        turn = await asyncio.to_thread(self.runtime.send, session, prompt)
        self._schedule_queue_drain(session.id)
        return {
            "backend": self.backend,
            "threadId": session.id,
            "name": session.name,
            "turnId": turn.id,
            "turn": turn.to_json(),
            "queued": False,
            "startedImmediately": True,
        }

    async def cancel_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        session = self._resolve_session(input_data)
        controller = self.runtime.controllers.get(session.id)
        if controller:
            await asyncio.to_thread(self.runtime.cancel, session)
        else:
            self.store.update_session(session.id, status="cancelled", active_turn_id=None)
            if session.active_turn_id:
                self.store.update_turn(session.active_turn_id, status="cancelled")
        refreshed = self.store.get_session(session.id)
        return {"backend": self.backend, "cancelled": True, "threadId": refreshed.id, "name": refreshed.name}

    async def start_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject:
        session = self._resolve_session(replace(input_data, prefer=input_data.prefer or "latest_any"))
        refreshed = self._refresh_session(session)
        if refreshed.active_turn_id or is_active_status(refreshed.status):
            return await self.queue_turn_by_label(input_data, turn_input)
        turn = await asyncio.to_thread(
            self.runtime.send,
            refreshed,
            str(turn_input["prompt"]),
            mode=_optional_str(turn_input.get("mode")),
            model=_optional_str(turn_input.get("model")) or refreshed.model,
        )
        self._schedule_queue_drain(refreshed.id)
        return {
            "backend": self.backend,
            "threadId": refreshed.id,
            "name": refreshed.name,
            "turnId": turn.id,
            "turn": turn.to_json(),
            "queued": False,
            "startedImmediately": True,
            "drain": "started_immediately",
        }

    async def queue_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject:
        session = self._resolve_session(input_data)
        turn = self.store.create_turn(
            session.id,
            str(turn_input["prompt"]),
            status="queued",
            mode=_optional_str(turn_input.get("mode")),
            model=_optional_str(turn_input.get("model")) or session.model,
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

    async def thread_favorite(self, thread_id: str) -> JsonObject:
        return self._unsupported("super_agents_thread_favorite", threadId=thread_id)

    async def tags(self) -> JsonObject:
        return self._unsupported("super_agents_tags")

    async def thread_tags(self, thread_id: str, tags: list[Any] | None = None) -> JsonObject:
        return self._unsupported("super_agents_thread_tags", threadId=thread_id, tags=tags)

    async def report_tags(self, project_path: str, path: str, tags: list[Any] | None = None) -> JsonObject:
        return self._unsupported("super_agents_report_tags", projectPath=project_path, path=path, tags=tags)

    def _resolve_session(self, input_data: LabelQueryInput) -> Session:
        if input_data.thread_id:
            try:
                return self.store.get_session(input_data.thread_id)
            except KeyError:
                pass
        label = required_label(input_data)
        return self.store.require_by_name(label)

    def _query_sessions(self, query: LabelQueryInput, *, include_inactive: bool) -> list[Session]:
        if query.thread_id or query.label:
            sessions = [self._resolve_session(query)]
        else:
            sessions = self.store.list_sessions(include_inactive=True, status=query.status)
        refreshed = [self._refresh_session(session) for session in sessions]
        if query.cwd:
            cwd = str(Path(query.cwd).expanduser())
            refreshed = [session for session in refreshed if session.cwd == cwd]
        if query.status:
            refreshed = [session for session in refreshed if session.status == query.status]
        if not include_inactive:
            refreshed = [session for session in refreshed if is_active_status(session.status)]
        return refreshed

    def _refresh_session(self, session: Session) -> Session:
        controller = self.runtime.controllers.get(session.id)
        if controller:
            controller.observe()
            return self.store.get_session(session.id)
        return session

    def _session_view(self, session: Session) -> JsonObject:
        return {"backend": self.backend, **session.to_json()}

    def _agent_item(self, session: Session, query: LabelQueryInput) -> JsonObject:
        turns = self.store.list_turns(session.id, limit=1)
        item = without_none(
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
        )
        return apply_field_selection(item, query.fields)

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
                "pid": session.pid,
                "activeTurnId": session.active_turn_id,
                "lastTurnId": session.last_turn_id,
                "lastObservedState": session.last_observed_state,
                "queueDepth": len(queued),
                "pendingRequestCount": 1 if session.last_observed_state == "approval prompt detected" else 0,
                "updatedAt": session.updated_at,
            }
        )

    def _schedule_queue_drain(self, session_id: str) -> None:
        task = self._queue_tasks.get(session_id)
        if task and not task.done():
            return
        self._queue_tasks[session_id] = asyncio.create_task(self._queue_drain_loop(session_id))

    async def _queue_drain_loop(self, session_id: str) -> None:
        while True:
            await asyncio.sleep(0.5)
            try:
                session = self.store.get_session(session_id)
            except KeyError:
                return
            if not self.store.queued_turns(session_id):
                return
            controller = self.runtime.controller_for(session)
            started = await asyncio.to_thread(controller.drain_queued)
            if started is None:
                continue

    def _unsupported(self, tool: str, **extra: Any) -> JsonObject:
        return {
            "backend": self.backend,
            "supported": False,
            "tool": tool,
            "error": f"{tool} is only available through the Codex app-server backend.",
            **{key: value for key, value in extra.items() if value is not None},
        }


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _decision_from_result(result: JsonObject) -> str:
    decision = result.get("decision")
    if isinstance(decision, str) and decision:
        return decision
    answers = result.get("answers")
    if isinstance(answers, dict):
        for answer in answers.values():
            if isinstance(answer, dict):
                values = answer.get("answers")
                if isinstance(values, list) and values:
                    return str(values[0])
    raise ValueError("result.decision is required for claude-tui approval answers.")
