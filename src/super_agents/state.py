from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

JsonObject = dict[str, Any]
TrackedStatus = Literal["running", "completed", "failed", "waiting", "cancelled"]
StoredStatus = TrackedStatus | Literal["unknown"]
Mode = Literal["default", "plan"]


@dataclass(slots=True)
class TurnSummary:
    turn_id: str
    status: TrackedStatus
    started_at: str
    updated_at: str
    mode: Mode | None = None
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
class StateFile:
    sessions: dict[str, SessionRecord] = field(default_factory=dict)

    def to_json(self) -> JsonObject:
        return {"sessions": {key: value.to_json() for key, value in self.sessions.items()}}


def read_state_file(path: Path) -> StateFile:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return StateFile()
    sessions = as_session_record_map(raw.get("sessions") if isinstance(raw, dict) else None)
    return StateFile(sessions=sessions)


def write_state_file(path: Path, state: StateFile) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(state.to_json(), indent=2) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(payload)
        tmp_name = tmp.name
    os.replace(tmp_name, path)


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


def get_string(value: JsonObject, key: str) -> str | None:
    result = value.get(key)
    return result if isinstance(result, str) else None


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
