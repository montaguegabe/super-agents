from __future__ import annotations

from typing import Any

from .state import JsonObject, get_string, without_none  # noqa: F401  (without_none re-exported)


def preview_text(value: str, max_length: int = 240) -> str:
    normalized = " ".join(value.split())
    return f"{normalized[: max_length - 3]}..." if len(normalized) > max_length else normalized


def text_preview(value: Any) -> str | None:
    text = find_useful_text(value)
    return preview_text(text) if text else None


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
    for key in ["text", "message", "content", "summary", "output", "preview"]:
        result = find_useful_text(value.get(key), depth + 1)
        if result:
            return result
    for item in value.values():
        result = find_useful_text(item, depth + 1)
        if result:
            return result
    return None


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
            "lastUsefulMessage": text_preview(persisted_turn),
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
