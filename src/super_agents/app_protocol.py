from __future__ import annotations

from typing import Any

from .app_formatting import as_object
from .app_time import age_ms, turn_recency
from .state import JsonObject, StoredStatus, TrackedStatus, get_string

SUPER_AGENT_IDENTITY_INSTRUCTION_PREFIX = "Super Agent thread name:"
LEGACY_SUPER_AGENT_IDENTITY_INSTRUCTION_PREFIX = "Super Agent name:"
SUPER_AGENT_THREAD_ID_INSTRUCTION_PREFIX = "Super Agent thread id:"
SUPER_AGENT_AGENT_NAME_INSTRUCTION_PREFIX = "Your name is "


def collaboration_mode(
    mode: str, model: str, reasoning_effort: str | None, developer_instructions: str | None
) -> JsonObject:
    settings: JsonObject = {"model": model, "developer_instructions": developer_instructions}
    settings["reasoning_effort"] = reasoning_effort or "high"
    return {"mode": mode, "settings": settings}


def with_super_agent_identity_instructions(
    developer_instructions: str | None,
    label: str | None,
    thread_id: str | None = None,
    agent_name: str | None = None,
) -> str | None:
    normalized_label = super_agent_label(label)
    normalized_thread_id = super_agent_thread_id(thread_id)
    normalized_agent_name = super_agent_label(agent_name)
    if not normalized_label and not normalized_thread_id and not normalized_agent_name:
        return developer_instructions

    identity_lines: list[str] = []
    if normalized_label:
        identity_lines.append(f"{SUPER_AGENT_IDENTITY_INSTRUCTION_PREFIX} {normalized_label}")
    if normalized_thread_id:
        identity_lines.append(f"{SUPER_AGENT_THREAD_ID_INSTRUCTION_PREFIX} {normalized_thread_id}")
    if normalized_agent_name:
        identity_lines.append(f"{SUPER_AGENT_AGENT_NAME_INSTRUCTION_PREFIX}{normalized_agent_name}.")

    base = _without_super_agent_identity_lines(developer_instructions)
    identity_block = "\n".join(identity_lines)
    if not base:
        return identity_block
    return f"{base}\n\n{identity_block}"


def super_agent_label(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return " ".join(value.split()) or None


def super_agent_thread_id(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _without_super_agent_identity_lines(value: str | None) -> str:
    if not value:
        return ""
    lines = [line for line in value.strip().splitlines() if not _is_super_agent_identity_line(line)]
    return "\n".join(lines).strip()


def _is_super_agent_identity_line(line: str) -> bool:
    stripped = line.strip()
    return (
        stripped.startswith(SUPER_AGENT_IDENTITY_INSTRUCTION_PREFIX)
        or stripped.startswith(LEGACY_SUPER_AGENT_IDENTITY_INSTRUCTION_PREFIX)
        or stripped.startswith(SUPER_AGENT_THREAD_ID_INSTRUCTION_PREFIX)
        or (stripped.startswith(SUPER_AGENT_AGENT_NAME_INSTRUCTION_PREFIX) and stripped.endswith("."))
    )


def effective_reasoning_effort(input_data: JsonObject) -> str:
    return get_string(input_data, "reasoningEffort") or "high"


def extract_model(value: JsonObject) -> str | None:
    return get_string(value, "model") or get_string(as_object(value.get("thread")), "model")


def extract_thread_id(value: JsonObject) -> str | None:
    return get_string(value, "threadId") or get_string(value, "id") or get_string(as_object(value.get("thread")), "id")


def extract_thread_cwd(value: JsonObject) -> str | None:
    return get_string(value, "cwd") or get_string(as_object(value.get("thread")), "cwd")


def extract_thread_name(value: JsonObject) -> str | None:
    return (
        get_string(value, "name")
        or get_string(value, "threadName")
        or get_string(as_object(value.get("thread")), "name")
        or get_string(as_object(value.get("thread")), "threadName")
    )


def extract_threads(value: JsonObject) -> list[JsonObject]:
    raw_threads = value.get("data") or value.get("threads")
    if not isinstance(raw_threads, list):
        return []
    return [thread for thread in raw_threads if isinstance(thread, dict)]


def extract_turn_id(value: JsonObject) -> str | None:
    for candidate in (
        get_string(value, "turnId"),
        get_string(value, "id"),
        get_string(as_object(value.get("turn")), "id"),
    ):
        if candidate and not candidate.startswith("q_"):
            return candidate
    return None


def extract_notification_thread_id(value: JsonObject) -> str | None:
    return (
        get_string(value, "threadId")
        or get_string(as_object(value.get("thread")), "id")
        or get_string(as_object(value.get("turn")), "threadId")
        or get_string(as_object(value.get("item")), "threadId")
    )


def extract_notification_turn_id(value: JsonObject) -> str | None:
    return (
        get_string(value, "turnId")
        or get_string(as_object(value.get("turn")), "id")
        or get_string(as_object(value.get("item")), "turnId")
    )


def find_turn(value: Any, turn_id: str) -> JsonObject | None:
    if isinstance(value, list):
        for item in value:
            result = find_turn(item, turn_id)
            if result:
                return result
        return None
    if not isinstance(value, dict):
        return None
    if get_string(value, "id") == turn_id:
        return value
    for item in value.values():
        result = find_turn(item, turn_id)
        if result:
            return result
    return None


def find_latest_turn(value: Any, active_only: bool) -> JsonObject | None:
    turns = collect_turns(value)
    if active_only:
        turns = [turn for turn in turns if normalize_turn_status(turn) == "running"]
    if not turns:
        return None
    return max(turns, key=turn_recency)


def collect_turns(value: Any) -> list[JsonObject]:
    if isinstance(value, list):
        turns: list[JsonObject] = []
        for item in value:
            turns.extend(collect_turns(item))
        return turns
    if not isinstance(value, dict):
        return []
    turns = [value] if get_string(value, "id") and get_string(value, "status") else []
    for item in value.values():
        turns.extend(collect_turns(item))
    return turns


def normalize_turn_status(turn: JsonObject | None) -> str | None:
    status = get_string(turn, "status") if turn else None
    if not status:
        return None
    if status in {"inProgress", "active"}:
        return "running"
    return status


def normalize_thread_status(thread: JsonObject) -> StoredStatus | None:
    raw_status = thread.get("status")
    if isinstance(raw_status, str):
        if raw_status in {"active", "running", "inProgress"}:
            return "running"
        if raw_status in {"completed", "failed", "cancelled", "waiting", "unknown"}:
            return raw_status  # type: ignore[return-value]
    if isinstance(raw_status, dict):
        kind = get_string(raw_status, "type") or get_string(raw_status, "status")
        if kind in {"active", "running", "inProgress"}:
            return "running"
        if kind in {"completed", "failed", "cancelled", "waiting", "unknown"}:
            return kind  # type: ignore[return-value]
    return None


def to_tracked_turn_status(status: str) -> TrackedStatus:
    if status in {"completed", "failed", "waiting", "cancelled"}:
        return status  # type: ignore[return-value]
    return "running"


def is_active_status(status: str | None) -> bool:
    return status in {"running", "waiting"}


def is_likely_stale(status: str | None, last_update_at: str | None) -> bool:
    if not is_active_status(status):
        return False
    age = age_ms(last_update_at)
    return bool(age is not None and age > 10 * 60 * 1000)
