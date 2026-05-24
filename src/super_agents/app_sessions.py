from __future__ import annotations

from typing import Any

from .app_formatting import as_object, without_none
from .app_models import LabelQueryInput
from .app_protocol import extract_thread_cwd, extract_thread_id, normalize_thread_status
from .app_time import iso_from_thread_time, iso_now, parse_iso_ms
from .state import JsonObject, SessionRecord, TurnSummary, as_mode, as_stored_status, get_string


def required_label(input_data: LabelQueryInput) -> str:
    if not input_data.label:
        raise ValueError("label must be a non-empty string.")
    return input_data.label


def session_recency(session: SessionRecord) -> int:
    return parse_iso_ms(session.last_event_at or session.updated_at)


def merge_turns(current: dict[str, TurnSummary] | None, patch: Any) -> JsonObject | None:
    if not current and not patch:
        return None
    result: JsonObject = {key: value.to_json() for key, value in (current or {}).items()}
    if isinstance(patch, dict):
        for turn_id, summary in patch.items():
            if isinstance(summary, dict):
                result[str(turn_id)] = {
                    **as_object(result.get(str(turn_id))),
                    **without_none(summary),
                    "turnId": str(turn_id),
                }
    return result


def session_from_patch(value: JsonObject) -> SessionRecord:
    turns = None
    if isinstance(value.get("turns"), dict):
        turns = {}
        for turn_id, raw in value["turns"].items():
            if isinstance(raw, dict):
                status = as_stored_status(get_string(raw, "status"))
                if status and status != "unknown":
                    turns[str(turn_id)] = TurnSummary(
                        turn_id=get_string(raw, "turnId") or str(turn_id),
                        status=status,
                        mode=as_mode(get_string(raw, "mode")),
                        reasoning_effort=get_string(raw, "reasoningEffort"),
                        started_at=get_string(raw, "startedAt")
                        or get_string(raw, "updatedAt")
                        or "1970-01-01T00:00:00.000Z",
                        updated_at=get_string(raw, "updatedAt") or "1970-01-01T00:00:00.000Z",
                        finished_at=get_string(raw, "finishedAt"),
                        prompt_preview=get_string(raw, "promptPreview"),
                        last_useful_message=get_string(raw, "lastUsefulMessage"),
                        pending_request_ids=[
                            item
                            for item in raw.get("pendingRequestIds", [])
                            if isinstance(item, str | int) and not isinstance(item, bool)
                        ]
                        if isinstance(raw.get("pendingRequestIds"), list)
                        else None,
                        event_count=raw.get("eventCount") if isinstance(raw.get("eventCount"), int) else None,
                    )
        if not turns:
            turns = None
    return SessionRecord(
        label=get_string(value, "label"),
        agent_name=get_string(value, "agentName"),
        thread_id=get_string(value, "threadId") or "",
        cwd=get_string(value, "cwd"),
        group=get_string(value, "group"),
        model=get_string(value, "model"),
        last_turn_id=get_string(value, "lastTurnId"),
        active_turn_id=get_string(value, "activeTurnId"),
        created_at=get_string(value, "createdAt"),
        last_started_at=get_string(value, "lastStartedAt"),
        last_finished_at=get_string(value, "lastFinishedAt"),
        last_status=as_stored_status(get_string(value, "lastStatus")),
        last_useful_message=get_string(value, "lastUsefulMessage"),
        last_event_at=get_string(value, "lastEventAt"),
        turns=turns,
        updated_at=get_string(value, "updatedAt") or iso_now(),
    )


def session_from_thread(thread: JsonObject, name: str) -> SessionRecord:
    thread_id = extract_thread_id(thread) or ""
    return SessionRecord(
        label=name,
        agent_name=get_string(thread, "agentName"),
        thread_id=thread_id,
        cwd=extract_thread_cwd(thread),
        last_status=normalize_thread_status(thread) or "unknown",
        updated_at=iso_from_thread_time(thread),
    )
