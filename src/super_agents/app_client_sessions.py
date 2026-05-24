from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections import deque
from dataclasses import replace

from .app_formatting import (
    apply_field_selection,
    preview_text,
    text_preview,
    without_none,
)
from .app_models import LabelQueryInput, PendingServerRequest, ResolvedSession, TurnState
from .app_protocol import (
    extract_thread_cwd,
    extract_thread_id,
    extract_thread_name,
    extract_threads,
    find_latest_turn,
    is_active_status,
    is_likely_stale,
    normalize_thread_status,
    to_tracked_turn_status,
)
from .app_sessions import merge_turns, required_label, session_from_patch, session_from_thread, session_recency
from .app_time import age_ms, iso_from_thread_time, iso_now, parse_iso_ms, thread_recency, turn_key
from .state import (
    JsonObject,
    SessionRecord,
    StateFile,
    StoredStatus,
    get_string,
    read_state_file_locked,
    update_state_file,
)

logger = logging.getLogger(__name__)


class SessionClientMixin:
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
