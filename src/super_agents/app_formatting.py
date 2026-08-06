from __future__ import annotations

from typing import Any

from .state import JsonObject, get_string, without_none  # noqa: F401  (without_none re-exported)

USEFUL_TEXT_KEYS = ("text", "message", "content", "summary", "output", "preview")
ASSISTANT_ROLE_VALUES = {"agent", "agentmessage", "assistant", "assistantmessage"}
USER_ROLE_VALUES = {"user", "usermessage"}
METADATA_TEXT_KEYS = {
    "id",
    "itemid",
    "clientid",
    "turnid",
    "threadid",
    "requestid",
    "callid",
    "sessionid",
    "type",
    "role",
    "phase",
    "status",
    "subtype",
    "kind",
    "reasoningeffort",
    "createdat",
    "startedat",
    "completedat",
    "updatedat",
    "finishedat",
    "lasteventat",
}


def preview_text(value: str, max_length: int = 240) -> str:
    normalized = " ".join(value.split())
    return f"{normalized[: max_length - 3]}..." if len(normalized) > max_length else normalized


def text_preview(value: Any) -> str | None:
    text = find_useful_text(value)
    return preview_text(text) if text else None


def turn_text_preview(value: Any) -> str | None:
    text = find_turn_useful_text(value)
    return preview_text(text) if text else None


def find_turn_useful_text(value: Any) -> str | None:
    text = _find_role_text(value, ASSISTANT_ROLE_VALUES)
    if text:
        return text
    return _find_non_user_text(value)


def find_useful_text(value: Any, depth: int = 0) -> str | None:
    if depth > 6 or value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed if len(trimmed) >= 8 else None
    if isinstance(value, list):
        for item in value:
            result = find_useful_text(item, depth + 1)
            if result:
                return result
        return None
    if not isinstance(value, dict):
        return None
    for key in USEFUL_TEXT_KEYS:
        result = find_useful_text(value.get(key), depth + 1)
        if result:
            return result
    for key, item in value.items():
        if _is_metadata_text_key(key):
            continue
        result = find_useful_text(item, depth + 1)
        if result:
            return result
    return None


def _is_metadata_text_key(key: Any) -> bool:
    normalized = "".join(char for char in str(key).lower() if char.isalnum())
    return normalized in METADATA_TEXT_KEYS or normalized.endswith("id")


def _find_role_text(value: Any, role_values: set[str], depth: int = 0) -> str | None:
    if depth > 8 or value is None:
        return None
    if isinstance(value, list):
        for item in reversed(value):
            result = _find_role_text(item, role_values, depth + 1)
            if result:
                return result
        return None
    if not isinstance(value, dict):
        return None
    if _message_role(value) in role_values:
        return _find_message_text(value)
    for key in ("payload", "item", "message", "turn", "items", "events", "messages", "content"):
        result = _find_role_text(value.get(key), role_values, depth + 1)
        if result:
            return result
    return None


def _find_non_user_text(value: Any, depth: int = 0) -> str | None:
    if depth > 8 or value is None:
        return None
    if isinstance(value, list):
        for item in reversed(value):
            result = _find_non_user_text(item, depth + 1)
            if result:
                return result
        return None
    if not isinstance(value, dict):
        return None
    role = _message_role(value)
    if role in USER_ROLE_VALUES:
        return None
    if role:
        return _find_message_text(value)
    for key in ("payload", "item", "message", "turn", "items", "events", "messages", "content"):
        result = _find_non_user_text(value.get(key), depth + 1)
        if result:
            return result
    if _find_role_text(value, USER_ROLE_VALUES):
        return None
    if text := find_useful_text(value):
        return text
    return None


def _message_role(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    role = _role_value(value.get("role"))
    if role in ASSISTANT_ROLE_VALUES | USER_ROLE_VALUES:
        return role
    item_type = _role_value(value.get("type"))
    if item_type in ASSISTANT_ROLE_VALUES | USER_ROLE_VALUES:
        return item_type
    payload = value.get("payload")
    if isinstance(payload, dict):
        return _message_role(payload)
    return role or item_type


def _find_message_text(value: Any, depth: int = 0) -> str | None:
    if depth > 6 or value is None:
        return None
    if isinstance(value, str):
        return value.strip() or None
    if isinstance(value, list):
        for item in value:
            result = _find_message_text(item, depth + 1)
            if result:
                return result
        return None
    if not isinstance(value, dict):
        return None
    for key in USEFUL_TEXT_KEYS:
        result = _find_message_text(value.get(key), depth + 1)
        if result:
            return result
    for key, item in value.items():
        if _is_metadata_text_key(key):
            continue
        result = _find_message_text(item, depth + 1)
        if result:
            return result
    return None


def _role_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return "".join(char for char in value.lower() if char.isalnum()) or None


def as_object(value: Any) -> JsonObject:
    return value if isinstance(value, dict) else {}


def compact_turn_summary(
    persisted_turn: JsonObject | None,
    tracked_turn: Any | None,
    *,
    include_items: bool,
    final_only: bool,
    max_items: int,
    max_output_chars: int,
) -> JsonObject:
    from .app_protocol import normalize_turn_status

    status = normalize_turn_status(persisted_turn) or (tracked_turn.status if tracked_turn else None)
    result = without_none(
        {
            "id": get_string(persisted_turn, "id") if persisted_turn else None,
            "status": status,
            "reasoningEffort": tracked_turn.reasoning_effort if tracked_turn else None,
            "startedAt": scalar_field(persisted_turn, "startedAt"),
            "completedAt": scalar_field(persisted_turn, "completedAt"),
            "lastUsefulMessage": turn_text_preview(persisted_turn),
            "eventCount": len(tracked_turn.events) if tracked_turn else None,
            "pendingRequestCount": len(tracked_turn.pending_requests) if tracked_turn else None,
        }
    )
    if include_items and persisted_turn:
        items = extract_compact_items(
            persisted_turn, final_only=final_only, max_items=max_items, max_output_chars=max_output_chars
        )
        if items:
            result["items"] = items
    return result


def extract_compact_items(
    value: JsonObject, *, final_only: bool, max_items: int, max_output_chars: int
) -> list[JsonObject]:
    raw_items = value.get("items") or value.get("events") or value.get("messages")
    if not isinstance(raw_items, list):
        return []
    if final_only:
        raw_items = raw_items[-1:]
    compacted: list[JsonObject] = []
    for item in raw_items[:max_items]:
        if not isinstance(item, dict):
            continue
        compacted.append(compact_json(item, max_chars=max_output_chars, max_items=max_items, include_diff=False))
    return compacted


def compact_json(value: Any, *, max_chars: int, max_items: int, include_diff: bool) -> JsonObject:
    if not isinstance(value, dict):
        return {}
    result: JsonObject = {}
    used_chars = 0
    for key, item in value.items():
        if not include_diff and is_diff_like_key(str(key)):
            continue
        if isinstance(item, str):
            clipped = preview_text(item, min(max_chars, max(80, max_chars - used_chars)))
            used_chars += len(clipped)
            result[str(key)] = clipped
        elif isinstance(item, int | float | bool) or item is None:
            result[str(key)] = item
        elif isinstance(item, list):
            result[str(key)] = [
                compact_json(child, max_chars=max_chars, max_items=max_items, include_diff=include_diff)
                if isinstance(child, dict)
                else preview_text(str(child), 240)
                for child in item[:max_items]
            ]
        elif isinstance(item, dict):
            result[str(key)] = compact_json(item, max_chars=max_chars, max_items=max_items, include_diff=include_diff)
    return result


def is_diff_like_key(key: str) -> bool:
    lowered = key.lower()
    return lowered in {"diff", "patch"} or lowered.endswith("diff") or lowered.endswith("patch")


def scalar_field(value: JsonObject | None, key: str) -> Any:
    if not isinstance(value, dict):
        return None
    item = value.get(key)
    return item if isinstance(item, str | int | float | bool) else None


def apply_field_selection(value: JsonObject, fields: list[str] | None) -> JsonObject:
    if not fields:
        return value
    allowed = set(fields)
    return {key: item for key, item in value.items() if key in allowed}
