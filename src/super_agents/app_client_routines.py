from __future__ import annotations

import asyncio
import logging
import os

from .app_formatting import without_none
from .app_models import LabelQueryInput
from .app_protocol import (
    extract_thread_id,
    find_turn,
    is_active_status,
    normalize_turn_status,
)
from .app_routines import (
    DEFAULT_ROUTINE_TIMEZONE,
    parse_routine_command_timeout_seconds,
    routine_active_run_is_stale,
    routine_fresh_thread_name,
    routine_from_patch,
    routine_has_active_run,
    routine_is_due,
    routine_local_date,
    routine_next_run_sort_key,
    routine_next_run_summary,
    routine_turn_input,
    routine_with_next_run,
)
from .app_time import iso_now, turn_key
from .state import (
    JsonObject,
    RoutineRecord,
    StateFile,
    get_string,
    read_state_file,
    update_state_file,
    write_state_file,
)

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
        await self.reconcile_routine_terminal_statuses()
        state = await self.read_state()
        routines = sorted(state.routines.values(), key=lambda routine: routine.updated_at, reverse=True)
        return {
            "count": len(routines),
            "routines": [routine_with_next_run(routine) for routine in routines],
            "nativeSupport": False,
            "scheduler": "super-agents-local-wrapper",
        }

    async def read_routine(self, name: str) -> JsonObject:
        await self.reconcile_routine_terminal_statuses()
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
        await self.reconcile_routine_terminal_statuses()
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
        await self.reconcile_routine_terminal_statuses()
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
                    if (not name or routine.name == name)
                    and (force or (routine_is_due(routine) and not routine_has_active_run(routine)))
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
            result = (
                await self.run_command_routine(routine)
                if routine.kind == "command"
                else await self.run_agent_routine(routine)
            )
            await asyncio.sleep(0)
            launch_status, launch_error = self.routine_launch_status(routine, result)
            await self.record_routine_run(
                routine.name,
                {
                    "lastRunDate": run_date,
                    "lastRunAt": run_at if routine.schedule_type == "interval" else routine.last_run_at,
                    "lastStartedAt": iso_now(),
                    "lastThreadId": get_string(result, "threadId"),
                    "lastTurnId": get_string(result, "turnId"),
                    "lastStatus": launch_status,
                    "lastError": launch_error,
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

    async def run_agent_routine(self, routine: RoutineRecord) -> JsonObject:
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
            return await self.start_turn(
                {**routine_turn_input(routine), "threadId": thread_id, "name": thread_name, "label": thread_name}
            )
        if routine.thread_id:
            target = await self.resolve_queue_target(LabelQueryInput(thread_id=routine.thread_id, cwd=routine.cwd))
            return await self.start_or_queue_turn(target, routine_turn_input(routine))
        if routine.target_name:
            target = await self.resolve_queue_target(
                LabelQueryInput(label=routine.target_name, cwd=routine.cwd, prefer="latest_any")
            )
            return await self.start_or_queue_turn(target, routine_turn_input(routine))
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
        return await self.start_turn({**routine_turn_input(routine), "threadId": thread_id, "label": routine.name})

    async def run_command_routine(self, routine: RoutineRecord) -> JsonObject:
        if not routine.command:
            raise RuntimeError(f"Command routine {routine.name} is missing a command.")
        timeout = parse_routine_command_timeout_seconds(routine.command_timeout_seconds)
        process = await asyncio.create_subprocess_shell(
            routine.command,
            cwd=routine.cwd or None,
            env=os.environ.copy(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except asyncio.TimeoutError as exc:
            process.kill()
            await process.communicate()
            raise RuntimeError(f"Command routine {routine.name} timed out after {timeout} seconds.") from exc
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        return {
            "kind": "command",
            "command": routine.command,
            "exitCode": process.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

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

    def routine_launch_status(self, routine: RoutineRecord, result: JsonObject) -> tuple[str, str | None]:
        if routine.kind == "command":
            exit_code = result.get("exitCode")
            if exit_code == 0:
                return "completed", None
            return "failed", f"Command routine exited with code {exit_code}."
        if result.get("queued"):
            return "queued", None
        thread_id = get_string(result, "threadId")
        turn_id = get_string(result, "turnId")
        if not thread_id or not turn_id:
            return "failed", "Routine turn launch did not return a threadId and turnId."
        turn = self._turns.get(turn_key(thread_id, turn_id))
        if turn and not is_active_status(turn.status):
            return "failed", f"Routine turn became {turn.status} immediately after launch."
        return "started", None

    async def reconcile_routine_terminal_statuses(self) -> None:
        async with self._state_lock:
            initial_state = read_state_file(self.state_file)
            active_routines = [
                routine
                for routine in initial_state.routines.values()
                if routine_has_active_run(routine)
            ]

        resolved_statuses: dict[tuple[str, str | None, str | None], str | None] = {}
        thread_results: dict[str, JsonObject | None] = {}
        for routine in active_routines:
            key = (routine.name, routine.last_thread_id, routine.last_turn_id)
            local_status = self._resolved_routine_turn_status(initial_state, routine)
            if local_status in {"completed", "failed", "cancelled"}:
                resolved_statuses[key] = local_status
                continue
            resolved_statuses[key] = await self._read_routine_turn_status(
                routine,
                thread_results,
            )

        async with self._state_lock:
            state = read_state_file(self.state_file)
            now = iso_now()
            changed = False
            for routine in state.routines.values():
                if not routine_has_active_run(routine):
                    continue
                key = (routine.name, routine.last_thread_id, routine.last_turn_id)
                status = resolved_statuses.get(key) or self._resolved_routine_turn_status(
                    state,
                    routine,
                )
                if status in {"completed", "failed", "cancelled"}:
                    patch: JsonObject = {
                        **routine.to_json(),
                        "lastStatus": "completed" if status == "completed" else "failed",
                        "updatedAt": now,
                    }
                    if status != "completed" and not routine.last_error:
                        patch["lastError"] = f"Routine turn became {status}."
                    state.routines[routine.name] = routine_from_patch(patch)
                    changed = True
                elif routine_active_run_is_stale(routine):
                    state.routines[routine.name] = routine_from_patch(
                        {
                            **routine.to_json(),
                            "lastStatus": "stale",
                            "lastError": (
                                f"Routine run stuck in {routine.last_status} since "
                                f"{routine.last_started_at or 'an unknown time'} with no resolvable turn status; "
                                "marked stale so the routine can run again."
                            ),
                            "updatedAt": now,
                        }
                    )
                    logger.warning(
                        "routine_reconcile marked_stale name=%s lastStartedAt=%s previousStatus=%s",
                        routine.name,
                        routine.last_started_at,
                        routine.last_status,
                    )
                    changed = True
            if changed:
                write_state_file(self.state_file, state)

    async def _read_routine_turn_status(
        self,
        routine: RoutineRecord,
        thread_results: dict[str, JsonObject | None],
    ) -> str | None:
        thread_id = routine.last_thread_id
        turn_id = routine.last_turn_id
        if not thread_id or not turn_id:
            return None
        if thread_id not in thread_results:
            try:
                thread_results[thread_id] = await self.read_thread(thread_id, include_turns=True)
            except Exception:
                logger.debug(
                    "routine_reconcile thread_read_failed name=%s thread_id=%s",
                    routine.name,
                    thread_id,
                    exc_info=True,
                )
                thread_results[thread_id] = None
        result = thread_results[thread_id]
        thread = result.get("thread") if isinstance(result, dict) else None
        if not isinstance(thread, dict):
            return None
        return normalize_turn_status(find_turn(thread, turn_id))

    def _resolved_routine_turn_status(self, state: StateFile, routine: RoutineRecord) -> str | None:
        if not routine.last_thread_id or not routine.last_turn_id:
            return None
        session = state.sessions.get(routine.last_thread_id)
        if session is None:
            return None
        turn = (session.turns or {}).get(routine.last_turn_id)
        if turn is not None:
            return turn.status
        if session.last_turn_id == routine.last_turn_id:
            return session.last_status
        return None

    async def record_routine_run(self, name: str, patch: JsonObject) -> None:
        async with self._state_lock:

            def update(state: StateFile) -> None:
                routine = state.routines.get(name)
                if routine is None:
                    return
                merged = {**routine.to_json(), **patch, "updatedAt": iso_now()}
                state.routines[name] = routine_from_patch(merged)

            update_state_file(self.state_file, update)
