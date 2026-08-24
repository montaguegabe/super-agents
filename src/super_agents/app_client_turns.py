from __future__ import annotations

import logging
import time
from pathlib import Path

from .app_formatting import (
    apply_field_selection,
    compact_json,
    compact_turn_summary,
    preview_text,
    text_preview,
    turn_text_preview,
)
from .app_models import LabelQueryInput, Mode, TurnState
from .app_protocol import (
    with_base_instructions,
    collaboration_mode,
    effective_reasoning_effort,
    extract_turn_id,
    find_turn,
    is_active_status,
    normalize_turn_status,
    super_agent_label,
    to_tracked_turn_status,
    with_super_agent_identity_instructions,
)
from .app_sessions import turn_patch
from .app_time import iso_now, path_basename, turn_key
from .state import JsonObject, TrackedStatus

logger = logging.getLogger(__name__)


def _is_missing_rollout_error(exc: RuntimeError) -> bool:
    return "no rollout found for thread id" in str(exc)


class TurnLifecycleMixin:
    async def start_turn(self, input_data: JsonObject) -> JsonObject:
        dispatch_id = str(input_data.get("_mcpCallId") or "")
        started = time.monotonic()
        logger.info(
            "dispatch_timing stage=super_agents_turn_start_request dispatch_id=%s "
            "thread_id=%s name=%s agent_name=%s cwd_basename=%s mode=%s",
            dispatch_id,
            input_data.get("threadId"),
            input_data.get("name") or input_data.get("label") or "",
            input_data.get("agentName") or "",
            path_basename(str(input_data.get("cwd") or Path.home())),
            input_data.get("mode") or "default",
        )
        await self.ensure_connected()
        thread_id = str(input_data["threadId"])
        session = await self.get_session(thread_id)
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
        agent_name = super_agent_label(
            input_data.get("agentName")
            if isinstance(input_data.get("agentName"), str)
            else session.agent_name
            if session
            else None
        )
        logger.info(
            "dispatch_timing stage=super_agents_turn_identity_resolved dispatch_id=%s "
            "thread_id=%s label=%s agent_name=%s session_agent_name=%s",
            dispatch_id,
            input_data.get("threadId"),
            label or "",
            agent_name or "",
            session.agent_name if session and session.agent_name else "",
        )
        developer_instructions = with_super_agent_identity_instructions(
            with_base_instructions(
                input_data.get("developerInstructions") if "developerInstructions" in input_data else None
            ),
            label,
            thread_id,
            agent_name,
        )
        params = await self._build_turn_start_params(
            input_data,
            session,
            thread_id,
            model,
            mode,
            reasoning_effort,
            developer_instructions,
        )
        await self._refresh_thread_environment(thread_id, params)
        logger.info(
            "dispatch_timing stage=app_server_turn_start_request dispatch_id=%s "
            "thread_id=%s cwd_basename=%s mode=%s reasoning_effort=%s",
            dispatch_id,
            input_data["threadId"],
            path_basename(str(params["cwd"])),
            mode,
            reasoning_effort,
        )
        result = await self.request(
            "turn/start",
            params,
            context={
                "dispatchId": dispatch_id,
                "threadId": thread_id,
                "name": label,
            },
        )
        turn_id = extract_turn_id(result) or f"{thread_id}:unknown:{int(time.time() * 1000)}"
        logger.info(
            "dispatch_timing stage=app_server_turn_start_response dispatch_id=%s thread_id=%s turn_id=%s elapsed_ms=%d",
            dispatch_id,
            thread_id,
            turn_id,
            int((time.monotonic() - started) * 1000),
        )
        return await self._persist_turn_start(
            input_data,
            params,
            thread_id,
            turn_id,
            result,
            reasoning_effort,
            agent_name,
            model,
            mode,
        )

    async def _build_turn_start_params(
        self,
        input_data: JsonObject,
        session: object | None,
        thread_id: str,
        model: object,
        mode: Mode,
        reasoning_effort: str | None,
        developer_instructions: str | None,
    ) -> JsonObject:
        params: JsonObject = {
            "threadId": thread_id,
            "cwd": input_data.get("cwd") or (session.cwd if session else None) or str(Path.home()),
            "model": model,
            "serviceTier": input_data.get("serviceTier") or "standard",
            **self.permission_overrides(input_data),
            "config": await self._login_shell_config_override(),
            "collaborationMode": collaboration_mode(
                mode,
                str(model),
                reasoning_effort,
                developer_instructions,
            ),
            "input": [{"type": "text", "text": input_data["prompt"]}],
        }
        return params

    async def _refresh_thread_environment(self, thread_id: str, params: JsonObject) -> None:
        try:
            await self.request(
                "thread/resume",
                {
                    "threadId": thread_id,
                    "cwd": params["cwd"],
                    "config": params["config"],
                },
            )
        except RuntimeError as exc:
            if not _is_missing_rollout_error(exc):
                raise
            logger.info(
                "Skipping pre-turn thread environment refresh because no rollout exists yet for thread_id=%s.",
                thread_id,
            )

    async def _persist_turn_start(
        self,
        input_data: JsonObject,
        params: JsonObject,
        thread_id: str,
        turn_id: str,
        result: JsonObject,
        reasoning_effort: str | None,
        agent_name: str | None,
        model: object,
        mode: Mode,
    ) -> JsonObject:
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
        if turn.status in {"completed", "failed", "cancelled"}:
            status: TrackedStatus = turn.status
        else:
            status = "waiting" if turn.pending_requests or turn.status == "waiting" else "running"
        turn.status = status
        clear_fields = [] if is_active_status(status) else ["activeTurnId"]
        await self.merge_session(
            thread_id,
            {
                "label": input_data.get("label"),
                "agentName": agent_name,
                "threadId": thread_id,
                "cwd": str(params["cwd"]),
                "group": input_data.get("group"),
                "model": model,
                "lastTurnId": turn_id,
                "activeTurnId": turn_id if is_active_status(status) else None,
                "lastStartedAt": now,
                "lastFinishedAt": turn.finished_at,
                "lastStatus": status,
                "lastUsefulMessage": text_preview(result),
                "turns": {
                    turn_id: turn_patch(
                        turn_id,
                        status,
                        turn=turn,
                        mode=mode,
                        reasoning_effort=reasoning_effort,
                        updated_at=now,
                        prompt_preview=preview_text(str(input_data["prompt"])),
                        last_useful_message=turn_text_preview(result),
                    )
                },
            },
            clear_fields=clear_fields,
        )
        return {**result, "threadId": thread_id, "turnId": turn_id, "mode": mode, "reasoningEffort": reasoning_effort}

    async def steer_turn(self, thread_id: str, turn_id: str, prompt: str, *, dispatch_id: str = "") -> JsonObject:
        started = time.monotonic()
        logger.info(
            "dispatch_timing stage=app_server_turn_steer_request dispatch_id=%s "
            "thread_id=%s turn_id=%s prompt_chars=%d",
            dispatch_id,
            thread_id,
            turn_id,
            len(prompt),
        )
        await self.ensure_connected()
        result = await self.request(
            "turn/steer",
            {
                "threadId": thread_id,
                "expectedTurnId": turn_id,
                "input": [{"type": "text", "text": prompt}],
            },
            context={
                "dispatchId": dispatch_id,
                "threadId": thread_id,
                "turnId": turn_id,
            },
        )
        logger.info(
            "dispatch_timing stage=app_server_turn_steer_response dispatch_id=%s thread_id=%s turn_id=%s elapsed_ms=%d",
            dispatch_id,
            thread_id,
            turn_id,
            int((time.monotonic() - started) * 1000),
        )
        return result

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
                    turn_id: turn_patch(
                        turn_id,
                        "cancelled",
                        turn=turn,
                        updated_at=turn.finished_at,
                        pending_request_ids=[],
                    )
                },
            },
            clear_fields=["activeTurnId"],
        )
        self.schedule_queue_drain(thread_id)
        return {"cancelled": True, "threadId": thread_id, "turnId": turn_id, "result": result}
