from __future__ import annotations

import os
import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .app_formatting import without_none
from .app_time import iso_now, parse_iso_ms
from .state import JsonObject, RoutineRecord, routine_record_from_json

DEFAULT_ROUTINE_TIMEZONE = "America/New_York"
DEFAULT_ROUTINE_POLL_SECONDS = 30
DEFAULT_ROUTINE_INTERVAL_SECONDS = 60
MIN_ROUTINE_INTERVAL_SECONDS = 5
ACTIVE_ROUTINE_STATUSES = {"starting", "started", "queued"}
DEFAULT_ROUTINE_COMMAND_TIMEOUT_SECONDS = 300


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
            "approvalPolicy": routine.approval_policy,
            "sandboxType": routine.sandbox_type,
            "mode": routine.mode or "default",
            "model": routine.model,
            "reasoningEffort": routine.reasoning_effort,
            "serviceTier": routine.service_tier,
            "developerInstructions": routine.developer_instructions,
            "name": routine.target_name or routine.name,
            "label": routine.target_name or routine.name,
        }
    )


def routine_fresh_thread_name(routine: RoutineRecord) -> str:
    timestamp = routine.last_started_at or routine.last_run_at or iso_now()
    normalized = re.sub(r"[^0-9A-Za-z]+", "-", timestamp).strip("-")
    return f"{routine.name}-{normalized}" if normalized else routine.name


def routine_with_next_run(routine: RoutineRecord) -> JsonObject:
    return {**routine.to_json(), "nextRunAt": safe_routine_next_run_at(routine)}


def routine_next_run_summary(routine: RoutineRecord) -> JsonObject:
    return {
        "name": routine.name,
        "kind": routine.kind,
        "time": routine.time,
        "scheduleType": routine.schedule_type,
        "intervalSeconds": routine.interval_seconds,
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
    if routine.schedule_type == "interval":
        try:
            interval_seconds = parse_routine_interval_seconds(routine.interval_seconds)
        except ValueError:
            return False
        if not routine.last_run_at:
            return True
        last_run_ms = parse_iso_ms(routine.last_run_at)
        if last_run_ms <= 0:
            return True
        now_ms = parse_iso_ms(iso_now())
        return now_ms - last_run_ms >= interval_seconds * 1000
    try:
        now = routine_now(routine)
        hour, minute = parse_routine_time(routine.time)
    except ValueError:
        return False
    run_key = now.date().isoformat()
    if routine.last_run_date == run_key:
        return False
    return (now.hour, now.minute) >= (hour, minute)


def routine_has_active_run(routine: RoutineRecord) -> bool:
    return routine.last_status in ACTIVE_ROUTINE_STATUSES


def routine_local_date(routine: RoutineRecord) -> str:
    return routine_now(routine).date().isoformat()


def routine_next_run_at(routine: RoutineRecord) -> str:
    if routine.schedule_type == "interval":
        return routine_interval_next_run_at(routine)
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


def parse_routine_interval_seconds(value: int | None) -> int:
    interval = value or DEFAULT_ROUTINE_INTERVAL_SECONDS
    if interval < MIN_ROUTINE_INTERVAL_SECONDS:
        raise ValueError("routine interval must be at least 5 seconds.")
    return interval


def routine_interval_next_run_at(routine: RoutineRecord) -> str:
    interval_seconds = parse_routine_interval_seconds(routine.interval_seconds)
    now_ms = parse_iso_ms(iso_now())
    last_run_ms = parse_iso_ms(routine.last_run_at)
    if last_run_ms <= 0:
        next_run_ms = now_ms
    else:
        next_run_ms = max(now_ms, last_run_ms + interval_seconds * 1000)
    return (
        datetime.fromtimestamp(next_run_ms / 1000, timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def routine_poll_seconds() -> int:
    raw = os.environ.get("SUPER_AGENTS_ROUTINE_POLL_SECONDS")
    try:
        value = int(raw) if raw else DEFAULT_ROUTINE_POLL_SECONDS
    except ValueError:
        return DEFAULT_ROUTINE_POLL_SECONDS
    return max(5, value)


def parse_routine_command_timeout_seconds(value: int | None) -> int:
    timeout = value or DEFAULT_ROUTINE_COMMAND_TIMEOUT_SECONDS
    if timeout < 1:
        raise ValueError("routine command timeout must be at least 1 second.")
    return timeout
