from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import replace

from .app_formatting import (
    apply_field_selection,
    preview_text,
    turn_text_preview,
    without_none,
)
from .app_models import LabelQueryInput, PendingServerRequest, QueueCancelInput, ResolvedSession, TurnState
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
from .app_queue import (
    cancel_queued_turn as cancel_queued_turn_item,
    complete_queued_turn,
    queued_turn_summaries,
    release_queued_turn,
    reserve_next_queued_turn,
)
from .app_sessions import (
    merge_turns,
    required_label,
    session_from_patch,
    session_from_thread,
    session_recency,
    turn_patch,
)
from .app_time import age_ms, iso_from_thread_time, iso_now, parse_iso_ms, thread_recency, turn_key
from .item_tags import thread_tags
from .state import (
    JsonObject,
    SessionRecord,
    StateFile,
    StoredStatus,
    get_string,
    read_state_file_locked,
    update_state_file,
)
from .thread_favorites import favorite_status, is_favorite

logger = logging.getLogger(__name__)


def _is_queue_item_id(value: str | None) -> bool:
    return bool(value and value.startswith("q_"))


STALE_ACTIVE_TURN_WARNING = "stale_active_turn"


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
        if input_data.favorite is not None:
            sessions = [session for session in sessions if is_favorite(session.thread_id) is input_data.favorite]
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
        return queued_turn_summaries(self.queue_dir)

    async def cancel_queued_turn(self, input_data: QueueCancelInput) -> JsonObject:
        thread_id = input_data.thread_id
        if not thread_id and input_data.label:
            target = await self.resolve_queue_target(
                LabelQueryInput(label=input_data.label, cwd=input_data.cwd, prefer="latest_any")
            )
            thread_id = target.session.thread_id
        item, position, queue_depth = cancel_queued_turn_item(
            self.queue_dir,
            queue_item_id=input_data.queue_item_id,
            thread_id=thread_id,
            position=input_data.position,
        )
        if queue_depth == 0:
            task = self.cancel_queue_drain_task(item.thread_id)
            if task:
                await asyncio.gather(task, return_exceptions=True)
        return {
            "cancelled": True,
            "removed": True,
            "threadId": item.thread_id,
            "name": item.label,
            "queueItemId": item.id,
            "position": position,
            "queueDepth": queue_depth,
            "item": item.to_json(),
        }

    def cancel_queue_drain_task(self, thread_id: str) -> asyncio.Task[None] | None:
        task = self._queue_tasks.pop(thread_id, None)
        if task and not task.done():
            task.cancel()
            return task
        return None

    def schedule_queue_drain(self, thread_id: str) -> None:
        existing = self._queue_tasks.get(thread_id)
        if existing and not existing.done():
            logger.info(
                "Super Agents queue drain already scheduled thread_id=%s task_done=%s",
                thread_id,
                existing.done(),
            )
            return
        logger.info("Super Agents queue drain scheduled thread_id=%s", thread_id)
        self._queue_tasks[thread_id] = asyncio.create_task(self._drain_queue(thread_id))

    async def _drain_queue(self, thread_id: str) -> None:
        await asyncio.sleep(0)
        while True:
            session = self.session_from_memory(thread_id)
            active_turn_id = session.active_turn_id if session else None
            last_turn_id = session.last_turn_id if session else None
            session_status = self.session_status(session) if session else None
            has_active_turn = self.thread_has_active_turn(thread_id)
            logger.info(
                "Super Agents queue drain check thread_id=%s session_status=%s active_turn_id=%s "
                "last_turn_id=%s has_active_turn=%s",
                thread_id,
                session_status,
                active_turn_id,
                last_turn_id,
                has_active_turn,
            )
            queued = reserve_next_queued_turn(
                self.queue_dir,
                thread_id,
                lambda: has_active_turn,
            )
            if queued is None:
                logger.info("Super Agents queue drain stopped thread_id=%s decision=no_reserved_item", thread_id)
                return
            try:
                logger.info(
                    "Super Agents queue drain starting queued turn thread_id=%s queue_item_id=%s attempts=%d",
                    thread_id,
                    queued.id,
                    queued.attempts,
                )
                await self.start_turn(
                    {
                        **queued.input_data,
                        "threadId": thread_id,
                        "label": queued.label,
                        "agentName": queued.agent_name,
                    }
                )
                complete_queued_turn(self.queue_dir, queued)
                logger.info(
                    "Super Agents queue drain started queued turn thread_id=%s queue_item_id=%s",
                    thread_id,
                    queued.id,
                )
            except Exception as exc:
                release_queued_turn(self.queue_dir, queued, error=exc)
                logger.exception(
                    "Failed to start queued Super Agents turn for thread_id=%s queue_item_id=%s",
                    thread_id,
                    queued.id,
                )
                return

    def thread_has_active_turn(self, thread_id: str) -> bool:
        session = self.session_from_memory(thread_id)
        active_turn_id = session.active_turn_id if session else None
        if _is_queue_item_id(active_turn_id):
            active_turn_id = None
        for turn in self._turns.values():
            if (
                turn.thread_id == thread_id
                and self.tracked_turn_is_active(turn)
                and not _is_queue_item_id(turn.turn_id)
                and active_turn_id == turn.turn_id
            ):
                logger.debug(
                    "Super Agents active-turn gate true from runtime turn thread_id=%s turn_id=%s status=%s "
                    "finished_at=%s",
                    thread_id,
                    turn.turn_id,
                    turn.status,
                    turn.finished_at,
                )
                return True
        session_status = self.session_status(session) if session else None
        result = bool(session and is_active_status(session_status))
        logger.debug(
            "Super Agents active-turn gate from session thread_id=%s result=%s session_status=%s "
            "active_turn_id=%s last_turn_id=%s",
            thread_id,
            result,
            session_status,
            active_turn_id,
            session.last_turn_id if session else None,
        )
        return result

    def tracked_turn_is_active(self, turn: TurnState) -> bool:
        return is_active_status(turn.status) and not turn.finished_at and not _is_queue_item_id(turn.turn_id)

    # The app server's searchTerm is a fuzzy content search whose results can
    # omit a thread whose name matches exactly, so exact-name resolution must
    # never rely on it alone; the paged scan below is the reliable fallback.
    NAME_SCAN_PAGE_SIZE = 50
    NAME_SCAN_MAX_THREADS = 500

    async def resolve_thread_name(self, name: str, input_data: LabelQueryInput) -> JsonObject:
        try:
            response = await self.list_threads(True, name, input_data.cwd, input_data.limit or 50)
        except Exception:
            return {}
        threads = extract_threads(response)
        matches = [thread for thread in threads if extract_thread_name(thread) == name]
        if not matches:
            return await self.scan_threads_for_name(name, input_data.cwd)
        matches.sort(key=thread_recency, reverse=True)
        return matches[0]

    # thread/list pages arrive in thread-creation order, so a "recent" view
    # must fetch a wider window before ranking by update time, or a thread
    # created long ago but touched today never surfaces.
    RECENT_WINDOW_THREADS = 200
    LIST_FETCH_PAGE_SIZE = 100

    async def _recent_thread_window(self, cwd: str | None) -> list[JsonObject]:
        threads: list[JsonObject] = []
        cursor: str | None = None
        while len(threads) < self.RECENT_WINDOW_THREADS:
            response = await self.list_threads(True, None, cwd, self.LIST_FETCH_PAGE_SIZE, cursor=cursor)
            page = extract_threads(response)
            if not page:
                break
            threads.extend(page)
            cursor = response.get("nextCursor") if isinstance(response, dict) else None
            if not isinstance(cursor, str) or not cursor:
                break
        return threads

    async def scan_threads_for_name(self, name: str, cwd: str | None = None) -> JsonObject:
        """Find a thread by exact name by paging through the full listing."""
        cursor: str | None = None
        seen = 0
        while seen < self.NAME_SCAN_MAX_THREADS:
            try:
                response = await self.list_threads(True, None, cwd, self.NAME_SCAN_PAGE_SIZE, cursor=cursor)
            except Exception:
                return {}
            threads = extract_threads(response)
            if not threads:
                return {}
            matches = [thread for thread in threads if extract_thread_name(thread) == name]
            if matches:
                matches.sort(key=thread_recency, reverse=True)
                return matches[0]
            seen += len(threads)
            cursor = response.get("nextCursor") if isinstance(response, dict) else None
            if not isinstance(cursor, str) or not cursor:
                return {}
        return {}

    async def recent_items(self, input_data: LabelQueryInput) -> list[JsonObject]:
        if input_data.thread_id:
            session = await self.get_session(input_data.thread_id)
            if session:
                return [self.session_view(session)]
            return []
        try:
            if input_data.label:
                response = await self.list_threads(
                    True,
                    input_data.label,
                    input_data.cwd,
                    input_data.limit or 50,
                )
                threads = extract_threads(response)
            else:
                threads = await self._recent_thread_window(input_data.cwd)
        except Exception:
            threads = []
        if not threads:
            sessions = await self.filtered_sessions(input_data)
            if sessions or not input_data.label:
                return [self.session_view(session) for session in sessions]
        items = [
            self.thread_view(thread)
            for thread in threads
            if not input_data.label or extract_thread_name(thread) == input_data.label
        ]
        if input_data.label and not items:
            fallback = await self.scan_threads_for_name(input_data.label, input_data.cwd)
            if fallback:
                items = [self.thread_view(fallback)]
        if input_data.status:
            items = [item for item in items if item.get("status") == input_data.status]
        if input_data.favorite is not None:
            items = [item for item in items if item.get("isFavorite") is input_data.favorite]
        return sorted(items, key=lambda item: parse_iso_ms(str(item.get("updatedAt") or "")), reverse=True)

    def thread_view(self, thread: JsonObject) -> JsonObject:
        thread_id = extract_thread_id(thread)
        session = self.session_from_memory(thread_id) if thread_id else None
        native_status = normalize_thread_status(thread)
        status = self.session_status(session) if session else native_status or "unknown"
        if session and native_status and not is_active_status(native_status):
            status = native_status
        status_warning = self.session_status_warning(session)
        running_turn_id = (
            session.active_turn_id or session.last_turn_id if session and is_active_status(status) else None
        )
        if _is_queue_item_id(running_turn_id):
            running_turn_id = None
        updated_at = iso_from_thread_time(thread)
        last_event_at = session.last_event_at if session else None
        favorite = favorite_status(thread_id)
        tags = thread_tags(thread_id)
        return without_none(
            {
                "name": extract_thread_name(thread),
                "agentName": session.agent_name if session else None,
                "cwd": extract_thread_cwd(thread),
                "threadId": thread_id,
                "runningTurnId": running_turn_id,
                "lastTurnId": session.last_turn_id if session else None,
                "reasoningEffort": self.session_turn_reasoning_effort(
                    session, running_turn_id or (session.last_turn_id if session else None)
                ),
                "status": status,
                "isFavorite": favorite["isFavorite"],
                "favoritedAt": favorite["favoritedAt"],
                "tags": tags["tags"],
                "ageMs": int(time.time() * 1000) - thread_recency(thread),
                "updatedAt": updated_at,
                "lastEventAt": last_event_at,
                "lastEventAgeMs": age_ms(last_event_at),
                "ageSinceUpdateMs": age_ms(updated_at),
                "isLikelyStale": bool(status_warning) or is_likely_stale(status, last_event_at or updated_at),
                "statusWarning": status_warning,
                "preview": get_string(thread, "preview"),
                "lastUsefulMessage": session.last_useful_message if session else None,
                "pendingRequestCount": self.pending_request_count(thread_id, running_turn_id) if thread_id else None,
            }
        )

    def session_view(self, session: SessionRecord) -> JsonObject:
        status = self.session_status(session)
        status_warning = self.session_status_warning(session)
        running_turn_id = session.active_turn_id or session.last_turn_id if is_active_status(status) else None
        if _is_queue_item_id(running_turn_id):
            running_turn_id = None
        started_at = session.last_started_at or session.updated_at
        favorite = favorite_status(session.thread_id)
        tags = thread_tags(session.thread_id)
        return without_none(
            {
                "label": session.label,
                "agentName": session.agent_name,
                "group": session.group,
                "cwd": session.cwd,
                "threadId": session.thread_id,
                "runningTurnId": running_turn_id,
                "lastTurnId": session.last_turn_id,
                "reasoningEffort": self.session_turn_reasoning_effort(session, running_turn_id or session.last_turn_id),
                "status": status,
                "isFavorite": favorite["isFavorite"],
                "favoritedAt": favorite["favoritedAt"],
                "tags": tags["tags"],
                "ageMs": int(time.time() * 1000) - parse_iso_ms(started_at),
                "updatedAt": session.updated_at,
                "lastEventAt": session.last_event_at,
                "lastEventAgeMs": age_ms(session.last_event_at),
                "ageSinceUpdateMs": age_ms(session.updated_at),
                "isLikelyStale": bool(status_warning)
                or is_likely_stale(status, session.last_event_at or session.updated_at),
                "statusWarning": status_warning,
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
                "isFavorite": item.get("isFavorite"),
                "favoritedAt": item.get("favoritedAt"),
                "tags": item.get("tags"),
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
        active_turn_id = None if _is_queue_item_id(session.active_turn_id) else session.active_turn_id
        last_turn_id = None if _is_queue_item_id(session.last_turn_id) else session.last_turn_id
        if active_turn_id:
            runtime_turn = self._turns.get(turn_key(session.thread_id, active_turn_id))
            if runtime_turn:
                if is_active_status(runtime_turn.status) and runtime_turn.finished_at:
                    if session.last_status and not is_active_status(session.last_status):
                        return session.last_status
                    return "unknown"
                return runtime_turn.status
            if session.last_status and is_active_status(session.last_status):
                if self.persisted_active_turn_is_stale(session, active_turn_id):
                    return "unknown"
                return session.last_status
            return session.last_status or "unknown"
        if session.last_status and is_active_status(session.last_status):
            if not last_turn_id:
                return "unknown"
            runtime_turn = self._turns.get(turn_key(session.thread_id, last_turn_id))
            if runtime_turn:
                if is_active_status(runtime_turn.status) and runtime_turn.finished_at:
                    return "unknown"
                return runtime_turn.status
            return session.last_status
        return session.last_status or "unknown"

    def session_status_warning(self, session: SessionRecord | None) -> str | None:
        if session is None:
            return None
        active_turn_id = None if _is_queue_item_id(session.active_turn_id) else session.active_turn_id
        if not active_turn_id:
            return None
        if self._turns.get(turn_key(session.thread_id, active_turn_id)) is not None:
            return None
        if (
            session.last_status
            and is_active_status(session.last_status)
            and self.persisted_active_turn_is_stale(session, active_turn_id)
        ):
            return STALE_ACTIVE_TURN_WARNING
        return None

    def persisted_active_turn_is_stale(self, session: SessionRecord, turn_id: str) -> bool:
        turn_summary = (session.turns or {}).get(turn_id)
        last_update_at = turn_summary.updated_at if turn_summary else session.last_event_at or session.updated_at
        return is_likely_stale(session.last_status, last_update_at)

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
        if _is_queue_item_id(turn_id):
            logger.warning(
                "Ignoring queue item id as turn progress thread_id=%s turn_id=%s status=%s",
                thread_id,
                turn_id,
                status,
            )
            self.schedule_queue_drain(thread_id)
            return
        tracked_status: StoredStatus = "unknown" if status == "unknown" else to_tracked_turn_status(status)
        tracked_turn = self._turns.get(turn_key(thread_id, turn_id))
        state_session = await self.get_session(thread_id)
        logger.info(
            "Super Agents turn progress record thread_id=%s turn_id=%s incoming_status=%s "
            "tracked_status=%s previous_active_turn_id=%s previous_last_status=%s pending_requests=%d",
            thread_id,
            turn_id,
            status,
            tracked_status,
            state_session.active_turn_id if state_session else None,
            state_session.last_status if state_session else None,
            len(pending_requests),
        )
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
                "lastUsefulMessage": turn_text_preview(persisted_turn),
                "turns": {
                    turn_id: turn_patch(
                        turn_id,
                        "running" if tracked_status == "unknown" else tracked_status,
                        reasoning_effort=reasoning_effort,
                        started_at=tracked_turn.started_at if tracked_turn else iso_now(),
                        updated_at=iso_now(),
                        finished_at=finished_at,
                        last_useful_message=turn_text_preview(persisted_turn),
                        pending_request_ids=[request.id for request in pending_requests],
                        event_count=len(tracked_turn.events) if tracked_turn else 0,
                    )
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
                old_active_turn_id = current.active_turn_id
                old_last_status = current.last_status
                merged_json = {**current.to_json(), **without_none(patch), "threadId": thread_id, "updatedAt": now}
                merged_json["createdAt"] = current.created_at or now
                merged_json["turns"] = merge_turns(current.turns, patch.get("turns"))
                for field_name in clear_fields or []:
                    merged_json.pop(field_name, None)
                merged = session_from_patch(merged_json)
                logger.info(
                    "Super Agents state merge thread_id=%s old_active_turn_id=%s new_active_turn_id=%s "
                    "old_last_status=%s new_last_status=%s clear_fields=%s state_file=%s",
                    thread_id,
                    old_active_turn_id,
                    merged.active_turn_id,
                    old_last_status,
                    merged.last_status,
                    clear_fields or [],
                    self.state_file,
                )
                state.sessions[thread_id] = merged

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
