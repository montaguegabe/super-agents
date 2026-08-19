from __future__ import annotations

import logging
import time
from dataclasses import replace

from .app_formatting import apply_field_selection, without_none
from .app_models import (
    DEFAULT_ACTIVE_AGENTS_LIMIT,
    DEFAULT_COMPACT_STATUS_LIMIT,
    DEFAULT_RECENT_AGENTS_LIMIT,
    LabelQueryInput,
    ResolvedSession,
)
from .app_protocol import extract_threads, is_active_status, to_tracked_turn_status
from .app_queue import append_queued_turn, new_queued_turn
from .app_sessions import required_label
from .state import JsonObject
from .execution_control import validate_execution

logger = logging.getLogger(__name__)


def _is_thread_not_found_error(exc: RuntimeError) -> bool:
    return "thread not found" in str(exc).lower()


class LabelQueryMixin:
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
            : query.limit or DEFAULT_ACTIVE_AGENTS_LIMIT
        ]
        items = [self.compact_agent_item(item, query) for item in items]
        return {"count": len(items), "agents": items}

    async def recent(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        items = [
            item
            for item in await self.recent_items(query)
            if query.include_inactive or is_active_status(str(item.get("status")))
        ][: query.limit or DEFAULT_RECENT_AGENTS_LIMIT]
        items = [self.compact_agent_item(item, query) for item in items]
        return {"count": len(items), "agents": items}

    async def compact_status(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        items = [
            self.status_item(item)
            for item in await self.recent_items(query)
            if query.include_inactive or is_active_status(str(item.get("status")))
        ][: query.limit or DEFAULT_COMPACT_STATUS_LIMIT]
        return {"count": len(items), "agents": [apply_field_selection(item, query.fields) for item in items]}

    async def resolve_label(self, input_data: LabelQueryInput) -> JsonObject:
        resolved = await self.resolve_session(required_label(input_data), input_data)
        return {
            "label": resolved.session.label,
            "agentName": resolved.session.agent_name,
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

    async def resume_by_label(
        self,
        input_data: LabelQueryInput,
        *,
        developer_instructions: str | None = None,
        replace_developer_instructions: bool = False,
    ) -> JsonObject:
        # Accepted for protocol parity: the app-server resume already applies
        # the provided instructions to the resumed thread outright.
        del replace_developer_instructions
        resolved = await self.resolve_session(required_label(input_data), replace(input_data, prefer="latest_any"))
        result = await self.resume_thread(
            resolved.session.thread_id,
            label=resolved.session.label,
            agent_name=resolved.session.agent_name,
            developer_instructions=developer_instructions,
        )
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

    async def steer_by_label(
        self,
        input_data: LabelQueryInput,
        prompt: str,
        turn_input: JsonObject | None = None,
    ) -> JsonObject:
        dispatch_id = str((turn_input or {}).get("_mcpCallId") or "")
        if input_data.thread_id and input_data.turn_id:
            return await self.steer_turn(
                input_data.thread_id,
                input_data.turn_id,
                prompt,
                dispatch_id=dispatch_id,
            )
        extras = dict(turn_input or {})
        try:
            resolved = await self.resolve_session(required_label(input_data), input_data)
        except ValueError:
            target = await self.resolve_queue_target(input_data)
            return await self._start_or_queue_prompt(target, prompt, extras)
        turn_id = input_data.turn_id or resolved.turn_id
        if not turn_id or not is_active_status(resolved.status):
            return await self._start_or_queue_prompt(resolved, prompt, extras)
        try:
            return await self.steer_turn(resolved.session.thread_id, turn_id, prompt, dispatch_id=dispatch_id)
        except RuntimeError as exc:
            recovered = await self.start_after_terminal_thread_not_found_steer(
                resolved,
                turn_id,
                {"label": resolved.session.label, "agentName": resolved.session.agent_name, **extras, "prompt": prompt},
                exc,
                drain="started_after_terminal_thread_not_found_steer",
            )
            if recovered:
                return recovered
            raise

    async def _start_or_queue_prompt(
        self,
        target: ResolvedSession,
        prompt: str,
        extras: JsonObject,
    ) -> JsonObject:
        return await self.start_or_queue_turn(
            target,
            without_none(
                {
                    "label": target.session.label,
                    "agentName": target.session.agent_name,
                    **extras,
                    "prompt": prompt,
                }
            ),
        )

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
        turn_id = input_data.turn_id or resolved.turn_id
        thread_is_active = is_active_status(resolved.status) or self.thread_has_active_turn(resolved.session.thread_id)
        if turn_id and not turn_id.startswith("q_") and thread_is_active:
            prompt = str(turn_input.get("prompt") or "")
            try:
                result = await self.steer_turn(
                    resolved.session.thread_id,
                    turn_id,
                    prompt,
                    dispatch_id=dispatch_id,
                )
            except RuntimeError as exc:
                if "no active turn to steer" in str(exc):
                    logger.warning(
                        "Resolved active Super Agents thread had no native active turn; "
                        "starting a new turn instead thread_id=%s turn_id=%s status=%s",
                        resolved.session.thread_id,
                        turn_id,
                        resolved.status,
                    )
                    return await self._start_turn_immediately(resolved, turn_input, drain="started_after_stale_steer")
                recovered = await self.start_after_terminal_thread_not_found_steer(
                    resolved,
                    turn_id,
                    turn_input,
                    exc,
                    drain="started_after_terminal_thread_not_found_steer",
                )
                if recovered:
                    return recovered
                raise
            return {
                **result,
                "queued": False,
                "steered": True,
                "startedImmediately": False,
                "drain": "steered_active_turn",
            }
        if thread_is_active:
            logger.warning(
                "Resolved active Super Agents thread without a steerable turn id; "
                "starting a new turn instead of queueing thread_id=%s turn_id=%s status=%s",
                resolved.session.thread_id,
                turn_id,
                resolved.status,
            )
            return await self._start_turn_immediately(resolved, turn_input, drain="started_without_active_turn_id")
        return await self.start_or_queue_turn(resolved, turn_input)

    async def start_after_terminal_thread_not_found_steer(
        self,
        resolved: ResolvedSession,
        turn_id: str,
        turn_input: JsonObject,
        exc: RuntimeError,
        *,
        drain: str,
    ) -> JsonObject | None:
        if not _is_thread_not_found_error(exc):
            return None
        try:
            progress = await self.turn_progress(resolved.session.thread_id, turn_id)
        except RuntimeError:
            return None
        status = str(progress.get("status") or "")
        if is_active_status(status):
            return None
        refreshed_session = await self.get_session(resolved.session.thread_id)
        refreshed = ResolvedSession(
            session=refreshed_session or resolved.session,
            turn_id=None,
            status="unknown" if status == "unknown" else to_tracked_turn_status(status),
        )
        logger.warning(
            "Recovered stale active Super Agents turn after app-server thread-not-found on steer; "
            "starting follow-up thread_id=%s turn_id=%s terminal_status=%s",
            resolved.session.thread_id,
            turn_id,
            status,
        )
        return await self._start_turn_immediately(refreshed, turn_input, drain=drain)

    async def queue_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject:
        target = await self.resolve_queue_target(input_data)
        return await self.start_or_queue_turn(target, turn_input)

    async def start_or_queue_turn(self, target: ResolvedSession, turn_input: JsonObject) -> JsonObject:
        target_status_active = is_active_status(target.status)
        active_gate = self.thread_has_active_turn(target.session.thread_id)
        should_queue = target_status_active or active_gate
        logger.info(
            "Super Agents turn start decision thread_id=%s turn_id=%s status=%s active_turn_id=%s "
            "last_turn_id=%s target_status_active=%s active_gate=%s decision=%s",
            target.session.thread_id,
            target.turn_id,
            target.status,
            target.session.active_turn_id,
            target.session.last_turn_id,
            target_status_active,
            active_gate,
            "queue" if should_queue else "start_immediately",
        )
        if should_queue:
            return await self.enqueue_turn(target, turn_input)
        return await self._start_turn_immediately(target, turn_input, drain="started_immediately")

    async def _start_turn_immediately(
        self,
        target: ResolvedSession,
        turn_input: JsonObject,
        *,
        drain: str,
    ) -> JsonObject:
        result = await self.start_turn({"cwd": target.session.cwd, **turn_input, "threadId": target.session.thread_id})
        return {**result, "queued": False, "startedImmediately": True, "drain": drain}

    async def enqueue_turn(self, target: ResolvedSession, turn_input: JsonObject) -> JsonObject:
        await validate_execution(
            self,
            operation="queue_turn",
            action={
                "method": "turn/queue",
                "params": {"threadId": target.session.thread_id, **turn_input},
            },
            requested_policy=self.permission_overrides(turn_input),
            thread_id=target.session.thread_id,
        )
        queued, position = append_queued_turn(
            self.queue_dir,
            new_queued_turn(
                thread_id=target.session.thread_id,
                label=target.session.label,
                agent_name=target.session.agent_name,
                input_data={"cwd": target.session.cwd, **turn_input},
            ),
        )
        self.schedule_queue_drain(target.session.thread_id)
        drain = "scheduled" if not is_active_status(target.status) else "waiting_for_active_turn"
        logger.info(
            "Super Agents turn queued thread_id=%s queue_item_id=%s position=%d queue_depth=%d "
            "status=%s active_turn_id=%s last_turn_id=%s drain=%s",
            target.session.thread_id,
            queued.id,
            position,
            position,
            target.status,
            target.session.active_turn_id,
            target.session.last_turn_id,
            drain,
        )
        return {
            "queued": True,
            "threadId": target.session.thread_id,
            "name": target.session.label,
            "position": position,
            "queueDepth": position,
            "item": queued.to_json(),
            "drain": drain,
        }
