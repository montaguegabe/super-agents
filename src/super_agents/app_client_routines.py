from __future__ import annotations

import logging

from .app_formatting import without_none
from .app_models import LabelQueryInput
from .app_protocol import extract_thread_id
from .app_routines import (
    DEFAULT_ROUTINE_TIMEZONE,
    routine_from_patch,
    routine_fresh_thread_name,
    routine_is_due,
    routine_local_date,
    routine_next_run_sort_key,
    routine_next_run_summary,
    routine_turn_input,
    routine_with_next_run,
)
from .app_time import iso_now
from .state import JsonObject, RoutineRecord, StateFile, get_string, update_state_file

logger = logging.getLogger(__name__)


class RoutineClientMixin:
    async def save_routine(self, input_data: JsonObject) -> JsonObject:
        name = str(input_data["name"])
        async with self._state_lock:

            def update(state: StateFile) -> JsonObject:
                now = iso_now()
                current = state.routines.get(name)
                raw = current.to_json() if current else {"name": name, "createdAt": now}
                for key, value in input_data.items():
                    if value is not None:
                        raw[key] = value
                raw["name"] = name
                raw["updatedAt"] = now
                raw.setdefault("createdAt", now)
                raw.setdefault("enabled", True)
                raw.setdefault("timezone", DEFAULT_ROUTINE_TIMEZONE)
                state.routines[name] = routine_from_patch(raw)
                return state.routines[name].to_json()

            routine = update_state_file(self.state_file, update)
        return {
            "routine": routine,
            "nativeSupport": False,
            "scheduler": "super-agents-local-wrapper",
        }

    async def list_routines(self) -> JsonObject:
        state = await self.read_state()
        routines = sorted(state.routines.values(), key=lambda routine: routine.updated_at, reverse=True)
        return {
            "count": len(routines),
            "routines": [routine_with_next_run(routine) for routine in routines],
            "nativeSupport": False,
            "scheduler": "super-agents-local-wrapper",
        }

    async def read_routine(self, name: str) -> JsonObject:
        state = await self.read_state()
        routine = state.routines.get(name)
        if routine is None:
            raise ValueError(f"No Super Agents routine found for name {name}.")
        return {
            "routine": routine_with_next_run(routine),
            "nativeSupport": False,
            "scheduler": "super-agents-local-wrapper",
        }

    async def delete_routine(self, name: str) -> JsonObject:
        async with self._state_lock:

            def update(state: StateFile) -> JsonObject:
                if name not in state.routines:
                    raise ValueError(f"No Super Agents routine found for name {name}.")
                removed = state.routines.pop(name)
                return removed.to_json()

            removed = update_state_file(self.state_file, update)
        return {"deleted": True, "routine": removed}

    async def routine_status_summary(self) -> JsonObject:
        state = await self.read_state()
        enabled = [routine for routine in state.routines.values() if routine.enabled]
        return {
            "count": len(state.routines),
            "enabledCount": len(enabled),
            "nextRuns": [
                routine_next_run_summary(routine) for routine in sorted(enabled, key=routine_next_run_sort_key)[:5]
            ],
            "nativeSupport": False,
            "scheduler": "super-agents-local-wrapper",
        }

    async def run_due_routines(self, name: str | None = None, force: bool = False) -> JsonObject:
        candidates = await self.reserve_due_routines(name=name, force=force)
        results = []
        for routine in sorted(candidates, key=lambda item: item.name):
            results.append(await self.run_routine(routine, force=force))
        return {
            "count": len(results),
            "results": results,
            "nativeSupport": False,
            "scheduler": "super-agents-local-wrapper",
        }

    async def reserve_due_routines(self, name: str | None = None, force: bool = False) -> list[RoutineRecord]:
        async with self._state_lock:

            def update(state: StateFile) -> list[RoutineRecord]:
                candidates = [
                    routine
                    for routine in state.routines.values()
                    if (not name or routine.name == name) and (force or routine_is_due(routine))
                ]
                if name and not candidates and name not in state.routines:
                    raise ValueError(f"No Super Agents routine found for name {name}.")
                reserved: list[RoutineRecord] = []
                now = iso_now()
                for routine in candidates:
                    run_date = routine_local_date(routine)
                    run_at = now if routine.schedule_type == "interval" else routine.last_run_at
                    merged = {
                        **routine.to_json(),
                        "lastRunDate": run_date,
                        "lastRunAt": run_at,
                        "lastStartedAt": now,
                        "lastStatus": "starting",
                        "lastError": None,
                        "updatedAt": now,
                    }
                    reserved_routine = routine_from_patch(merged)
                    state.routines[routine.name] = reserved_routine
                    reserved.append(reserved_routine)
                return reserved

            return update_state_file(self.state_file, update)

    async def run_routine(self, routine: RoutineRecord, *, force: bool = False) -> JsonObject:
        if not routine.enabled and not force:
            return {"name": routine.name, "skipped": True, "reason": "disabled"}
        run_date = routine_local_date(routine)
        run_at = routine.last_run_at or iso_now()
        try:
            if routine.fresh_thread_per_run:
                thread_name = routine_fresh_thread_name(routine)
                agent_name = await self.routine_fresh_thread_agent_name(routine)
                thread_result = await self.start_thread(
                    without_none(
                        {
                            "name": thread_name,
                            "agentName": agent_name,
                            "cwd": routine.cwd,
                            "approvalPolicy": routine.approval_policy,
                            "sandboxType": routine.sandbox_type,
                            "developerInstructions": routine.developer_instructions,
                        }
                    )
                )
                thread_id = extract_thread_id(thread_result)
                if not thread_id:
                    raise RuntimeError(f"Could not start thread for routine {routine.name}.")
                result = await self.start_turn(
                    {**routine_turn_input(routine), "threadId": thread_id, "name": thread_name, "label": thread_name}
                )
            elif routine.thread_id:
                target = await self.resolve_queue_target(LabelQueryInput(thread_id=routine.thread_id, cwd=routine.cwd))
                result = await self.start_or_queue_turn(target, routine_turn_input(routine))
            elif routine.target_name:
                target = await self.resolve_queue_target(
                    LabelQueryInput(label=routine.target_name, cwd=routine.cwd, prefer="latest_any")
                )
                result = await self.start_or_queue_turn(target, routine_turn_input(routine))
            else:
                thread_result = await self.start_thread(
                    without_none(
                        {
                            "name": routine.name,
                            "cwd": routine.cwd,
                            "approvalPolicy": routine.approval_policy,
                            "sandboxType": routine.sandbox_type,
                            "developerInstructions": routine.developer_instructions,
                        }
                    )
                )
                thread_id = extract_thread_id(thread_result)
                if not thread_id:
                    raise RuntimeError(f"Could not start thread for routine {routine.name}.")
                result = await self.start_turn(
                    {**routine_turn_input(routine), "threadId": thread_id, "label": routine.name}
                )
            await self.record_routine_run(
                routine.name,
                {
                    "lastRunDate": run_date,
                    "lastRunAt": run_at if routine.schedule_type == "interval" else routine.last_run_at,
                    "lastStartedAt": iso_now(),
                    "lastThreadId": get_string(result, "threadId"),
                    "lastTurnId": get_string(result, "turnId"),
                    "lastStatus": "queued" if result.get("queued") else "started",
                    "lastError": None,
                },
            )
            return {"name": routine.name, "ran": True, **result}
        except Exception as exc:
            await self.record_routine_run(
                routine.name,
                {
                    "lastRunDate": run_date,
                    "lastRunAt": run_at if routine.schedule_type == "interval" else routine.last_run_at,
                    "lastStartedAt": iso_now(),
                    "lastStatus": "failed",
                    "lastError": str(exc),
                },
            )
            logger.exception("Failed to run Super Agents routine name=%s", routine.name)
            return {"name": routine.name, "ran": False, "error": str(exc)}

    async def routine_fresh_thread_agent_name(self, routine: RoutineRecord) -> str | None:
        if not routine.target_name:
            return None
        try:
            target = await self.resolve_queue_target(
                LabelQueryInput(label=routine.target_name, cwd=routine.cwd, prefer="latest_any")
            )
        except ValueError:
            return None
        return target.session.agent_name

    async def record_routine_run(self, name: str, patch: JsonObject) -> None:
        async with self._state_lock:

            def update(state: StateFile) -> None:
                routine = state.routines.get(name)
                if routine is None:
                    return
                merged = {**routine.to_json(), **patch, "updatedAt": iso_now()}
                state.routines[name] = routine_from_patch(merged)

            update_state_file(self.state_file, update)
