from __future__ import annotations

import contextlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Literal, TypeVar

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

JsonObject = dict[str, Any]
TrackedStatus = Literal["running", "completed", "failed", "waiting", "cancelled"]
StoredStatus = TrackedStatus | Literal["unknown"]
Mode = Literal["default", "plan"]
T = TypeVar("T")


@dataclass(slots=True)
class TurnSummary:
    turn_id: str
    status: TrackedStatus
    started_at: str
    updated_at: str
    mode: Mode | None = None
    reasoning_effort: str | None = None
    finished_at: str | None = None
    prompt_preview: str | None = None
    last_useful_message: str | None = None
    pending_request_ids: list[str | int] | None = None
    event_count: int | None = None

    def to_json(self) -> JsonObject:
        return without_none(
            {
                "turnId": self.turn_id,
                "status": self.status,
                "mode": self.mode,
                "reasoningEffort": self.reasoning_effort,
                "startedAt": self.started_at,
                "updatedAt": self.updated_at,
                "finishedAt": self.finished_at,
                "promptPreview": self.prompt_preview,
                "lastUsefulMessage": self.last_useful_message,
                "pendingRequestIds": self.pending_request_ids,
                "eventCount": self.event_count,
            }
        )


@dataclass(slots=True)
class SessionRecord:
    thread_id: str
    updated_at: str
    label: str | None = None
    agent_name: str | None = None
    cwd: str | None = None
    group: str | None = None
    model: str | None = None
    last_turn_id: str | None = None
    active_turn_id: str | None = None
    created_at: str | None = None
    last_started_at: str | None = None
    last_finished_at: str | None = None
    last_status: StoredStatus | None = None
    last_useful_message: str | None = None
    last_event_at: str | None = None
    turns: dict[str, TurnSummary] | None = None

    def to_json(self) -> JsonObject:
        return without_none(
            {
                "label": self.label,
                "agentName": self.agent_name,
                "threadId": self.thread_id,
                "cwd": self.cwd,
                "group": self.group,
                "model": self.model,
                "lastTurnId": self.last_turn_id,
                "activeTurnId": self.active_turn_id,
                "createdAt": self.created_at,
                "lastStartedAt": self.last_started_at,
                "lastFinishedAt": self.last_finished_at,
                "lastStatus": self.last_status,
                "lastUsefulMessage": self.last_useful_message,
                "lastEventAt": self.last_event_at,
                "turns": {key: value.to_json() for key, value in self.turns.items()} if self.turns else None,
                "updatedAt": self.updated_at,
            }
        )


@dataclass(slots=True)
class RoutineRecord:
    name: str
    prompt: str
    time: str
    updated_at: str
    schedule_type: str = "daily"
    interval_seconds: int | None = None
    timezone: str | None = None
    enabled: bool = True
    target_name: str | None = None
    thread_id: str | None = None
    fresh_thread_per_run: bool = False
    cwd: str | None = None
    approval_policy: str | None = None
    sandbox_type: str | None = None
    mode: Mode | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    developer_instructions: str | None = None
    created_at: str | None = None
    last_run_date: str | None = None
    last_run_at: str | None = None
    last_started_at: str | None = None
    last_thread_id: str | None = None
    last_turn_id: str | None = None
    last_status: str | None = None
    last_error: str | None = None

    def to_json(self) -> JsonObject:
        return without_none(
            {
                "name": self.name,
                "prompt": self.prompt,
                "time": self.time,
                "scheduleType": self.schedule_type,
                "intervalSeconds": self.interval_seconds,
                "timezone": self.timezone,
                "enabled": self.enabled,
                "targetName": self.target_name,
                "threadId": self.thread_id,
                "freshThreadPerRun": self.fresh_thread_per_run,
                "cwd": self.cwd,
                "approvalPolicy": self.approval_policy,
                "sandboxType": self.sandbox_type,
                "mode": self.mode,
                "model": self.model,
                "reasoningEffort": self.reasoning_effort,
                "serviceTier": self.service_tier,
                "developerInstructions": self.developer_instructions,
                "createdAt": self.created_at,
                "updatedAt": self.updated_at,
                "lastRunDate": self.last_run_date,
                "lastRunAt": self.last_run_at,
                "lastStartedAt": self.last_started_at,
                "lastThreadId": self.last_thread_id,
                "lastTurnId": self.last_turn_id,
                "lastStatus": self.last_status,
                "lastError": self.last_error,
            }
        )


@dataclass(slots=True)
class StateFile:
    sessions: dict[str, SessionRecord] = field(default_factory=dict)
    routines: dict[str, RoutineRecord] = field(default_factory=dict)

    def to_json(self) -> JsonObject:
        return {
            "sessions": {key: value.to_json() for key, value in self.sessions.items()},
            "routines": {key: value.to_json() for key, value in self.routines.items()},
        }


def read_state_file(path: Path) -> StateFile:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return StateFile()
    sessions = as_session_record_map(raw.get("sessions") if isinstance(raw, dict) else None)
    routines = as_routine_record_map(raw.get("routines") if isinstance(raw, dict) else None)
    return StateFile(sessions=sessions, routines=routines)


def write_state_file(path: Path, state: StateFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.to_json(), indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


@contextlib.contextmanager
def state_file_lock(path: Path) -> Iterator[None]:
    """Exclusive inter-process lock for one Super Agents state file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a+", encoding="utf-8") as handle:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def read_state_file_locked(path: Path) -> StateFile:
    with state_file_lock(path):
        return read_state_file(path)


def update_state_file(path: Path, callback: Callable[[StateFile], T]) -> T:
    with state_file_lock(path):
        state = read_state_file(path)
        result = callback(state)
        write_state_file(path, state)
        return result


def as_session_record_map(value: Any) -> dict[str, SessionRecord]:
    if not isinstance(value, dict):
        return {}
    sessions: dict[str, SessionRecord] = {}
    for thread_id, raw_session in value.items():
        if not isinstance(raw_session, dict):
            continue
        normalized_thread_id = get_string(raw_session, "threadId") or str(thread_id)
        updated_at = get_string(raw_session, "updatedAt") or "1970-01-01T00:00:00.000Z"
        session = SessionRecord(
            label=get_string(raw_session, "label"),
            agent_name=get_string(raw_session, "agentName"),
            thread_id=normalized_thread_id,
            cwd=get_string(raw_session, "cwd"),
            group=get_string(raw_session, "group"),
            model=get_string(raw_session, "model"),
            last_turn_id=get_string(raw_session, "lastTurnId"),
            active_turn_id=get_string(raw_session, "activeTurnId"),
            created_at=get_string(raw_session, "createdAt"),
            last_started_at=get_string(raw_session, "lastStartedAt"),
            last_finished_at=get_string(raw_session, "lastFinishedAt"),
            last_status=as_stored_status(get_string(raw_session, "lastStatus")),
            last_useful_message=get_string(raw_session, "lastUsefulMessage"),
            last_event_at=get_string(raw_session, "lastEventAt"),
            turns=as_turn_summary_map(raw_session.get("turns")),
            updated_at=updated_at,
        )
        sessions[normalized_thread_id] = session
    return sessions


def as_turn_summary_map(value: Any) -> dict[str, TurnSummary] | None:
    if not isinstance(value, dict):
        return None
    turns: dict[str, TurnSummary] = {}
    for turn_id, raw_turn in value.items():
        if not isinstance(raw_turn, dict):
            continue
        normalized_turn_id = get_string(raw_turn, "turnId") or str(turn_id)
        status = as_stored_status(get_string(raw_turn, "status"))
        if status is None or status == "unknown":
            continue
        turn = TurnSummary(
            turn_id=normalized_turn_id,
            status=status,
            mode=as_mode(get_string(raw_turn, "mode")),
            reasoning_effort=get_string(raw_turn, "reasoningEffort"),
            started_at=get_string(raw_turn, "startedAt")
            or get_string(raw_turn, "updatedAt")
            or "1970-01-01T00:00:00.000Z",
            updated_at=get_string(raw_turn, "updatedAt") or "1970-01-01T00:00:00.000Z",
            finished_at=get_string(raw_turn, "finishedAt"),
            prompt_preview=get_string(raw_turn, "promptPreview"),
            last_useful_message=get_string(raw_turn, "lastUsefulMessage"),
            pending_request_ids=as_string_or_number_array(raw_turn.get("pendingRequestIds")),
            event_count=raw_turn.get("eventCount") if isinstance(raw_turn.get("eventCount"), int) else None,
        )
        turns[normalized_turn_id] = turn
    return turns or None


def as_routine_record_map(value: Any) -> dict[str, RoutineRecord]:
    if not isinstance(value, dict):
        return {}
    routines: dict[str, RoutineRecord] = {}
    for name, raw_routine in value.items():
        routine = routine_record_from_json(
            raw_routine,
            name_fallback=str(name),
            require_prompt_time=True,
        )
        if routine is None:
            continue
        routines[routine.name] = routine
    return routines


def routine_record_from_json(
    value: Any,
    *,
    name_fallback: str = "",
    default_time: str = "09:00",
    default_timezone: str | None = None,
    default_updated_at: str = "1970-01-01T00:00:00.000Z",
    require_prompt_time: bool = False,
) -> RoutineRecord | None:
    if not isinstance(value, dict):
        return None

    name = get_string(value, "name") or name_fallback
    prompt = get_string(value, "prompt") or ""
    time_value = get_string(value, "time") or default_time
    schedule_type = as_routine_schedule_type(get_string(value, "scheduleType"))
    if require_prompt_time and not prompt:
        return None
    if require_prompt_time and schedule_type != "interval" and not get_string(value, "time"):
        return None

    return RoutineRecord(
        name=name,
        prompt=prompt,
        time=time_value,
        schedule_type=schedule_type,
        interval_seconds=get_positive_int(value, "intervalSeconds"),
        timezone=get_string(value, "timezone") or default_timezone,
        enabled=value.get("enabled") if isinstance(value.get("enabled"), bool) else True,
        target_name=get_string(value, "targetName"),
        thread_id=get_string(value, "threadId"),
        fresh_thread_per_run=bool(value.get("freshThreadPerRun")) if isinstance(value.get("freshThreadPerRun"), bool) else False,
        cwd=get_string(value, "cwd"),
        approval_policy=get_string(value, "approvalPolicy"),
        sandbox_type=get_string(value, "sandboxType"),
        mode=as_mode(get_string(value, "mode")),
        model=get_string(value, "model"),
        reasoning_effort=get_string(value, "reasoningEffort"),
        service_tier=get_string(value, "serviceTier"),
        developer_instructions=get_string(value, "developerInstructions"),
        created_at=get_string(value, "createdAt"),
        updated_at=get_string(value, "updatedAt") or default_updated_at,
        last_run_date=get_string(value, "lastRunDate"),
        last_run_at=get_string(value, "lastRunAt"),
        last_started_at=get_string(value, "lastStartedAt"),
        last_thread_id=get_string(value, "lastThreadId"),
        last_turn_id=get_string(value, "lastTurnId"),
        last_status=get_string(value, "lastStatus"),
        last_error=get_string(value, "lastError"),
    )


def get_string(value: JsonObject, key: str) -> str | None:
    result = value.get(key)
    return result if isinstance(result, str) else None


def get_positive_int(value: JsonObject, key: str) -> int | None:
    result = value.get(key)
    if isinstance(result, bool):
        return None
    if isinstance(result, int) and result > 0:
        return result
    return None


def as_routine_schedule_type(value: str | None) -> str:
    return value if value in {"daily", "interval"} else "daily"


def as_stored_status(status: str | None) -> StoredStatus | None:
    if status in {"running", "waiting", "completed", "failed", "cancelled", "unknown"}:
        return status  # type: ignore[return-value]
    return None


def as_mode(mode: str | None) -> Mode | None:
    if mode in {"default", "plan"}:
        return mode  # type: ignore[return-value]
    return None


def as_string_or_number_array(value: Any) -> list[str | int] | None:
    if not isinstance(value, list):
        return None
    return [item for item in value if isinstance(item, str | int)]


def without_none(value: JsonObject) -> JsonObject:
    return {key: item for key, item in value.items() if item is not None}
