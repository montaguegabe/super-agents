from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .app_formatting import without_none
from .app_time import iso_now, parse_iso_ms
from .state import JsonObject, RoutineRecord, routine_record_from_json

DEFAULT_ROUTINE_TIMEZONE = "America/New_York"
DEFAULT_ROUTINE_POLL_SECONDS = 30


def routine_from_patch(value: JsonObject) -> RoutineRecord:
    routine = routine_record_from_json(
        value,
        default_timezone=DEFAULT_ROUTINE_TIMEZONE,
        default_updated_at=iso_now(),
    )
    if routine is None:
        raise ValueError("Invalid routine payload.")
    return routine


def routine_turn_input(routine: RoutineRecord) -> JsonObject:
    return without_none(
        {
            "prompt": routine.prompt,
            "cwd": routine.cwd,
            "mode": routine.mode or "default",
            "model": routine.model,
            "reasoningEffort": routine.reasoning_effort,
            "serviceTier": routine.service_tier,
            "developerInstructions": routine.developer_instructions,
            "name": routine.target_name or routine.name,
            "label": routine.target_name or routine.name,
        }
    )


def routine_with_next_run(routine: RoutineRecord) -> JsonObject:
    return {**routine.to_json(), "nextRunAt": safe_routine_next_run_at(routine)}


def routine_next_run_summary(routine: RoutineRecord) -> JsonObject:
    return {
        "name": routine.name,
        "time": routine.time,
        "timezone": routine.timezone or DEFAULT_ROUTINE_TIMEZONE,
        "nextRunAt": safe_routine_next_run_at(routine),
        "lastStatus": routine.last_status,
    }


def routine_next_run_sort_key(routine: RoutineRecord) -> int:
    next_run = safe_routine_next_run_at(routine)
    return parse_iso_ms(next_run) if next_run else 0


def routine_is_due(routine: RoutineRecord) -> bool:
    if not routine.enabled:
        return False
    try:
        now = routine_now(routine)
        hour, minute = parse_routine_time(routine.time)
    except ValueError:
        return False
    run_key = now.date().isoformat()
    if routine.last_run_date == run_key:
        return False
    return (now.hour, now.minute) >= (hour, minute)


def routine_local_date(routine: RoutineRecord) -> str:
    return routine_now(routine).date().isoformat()


def routine_next_run_at(routine: RoutineRecord) -> str:
    now = routine_now(routine)
    hour, minute = parse_routine_time(routine.time)
    next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if routine.last_run_date == now.date().isoformat() or next_run <= now:
        next_run = next_run + timedelta(days=1)
    return next_run.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def safe_routine_next_run_at(routine: RoutineRecord) -> str | None:
    try:
        return routine_next_run_at(routine)
    except ValueError:
        return None


def routine_now(routine: RoutineRecord) -> datetime:
    timezone_name = routine.timezone or DEFAULT_ROUTINE_TIMEZONE
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo(DEFAULT_ROUTINE_TIMEZONE)
    return datetime.now(tz)


def parse_routine_time(value: str) -> tuple[int, int]:
    parts = value.split(":")
    if len(parts) != 2:
        raise ValueError("routine time must use HH:MM.")
    hour = int(parts[0])
    minute = int(parts[1])
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("routine time must use HH:MM in 24-hour time.")
    return hour, minute


def routine_poll_seconds() -> int:
    raw = os.environ.get("SUPER_AGENTS_ROUTINE_POLL_SECONDS")
    try:
        value = int(raw) if raw else DEFAULT_ROUTINE_POLL_SECONDS
    except ValueError:
        return DEFAULT_ROUTINE_POLL_SECONDS
    return max(5, value)
