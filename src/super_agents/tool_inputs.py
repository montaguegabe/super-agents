"""Input-coercion helpers for MCP tool handlers.

These small helpers read and validate raw JSON tool arguments into typed
Python values. They intentionally contain no product/domain logic so they can
be reused by any tool handler.
"""

from __future__ import annotations

from typing import Any

JsonObject = dict[str, Any]


def required_string(value: JsonObject, key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string.")
    return result


def required_object(value: JsonObject, key: str) -> JsonObject:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{key} must be an object.")
    return result


def required_request_id(value: JsonObject) -> str | int:
    result = value.get("requestId")
    if isinstance(result, str | int) and not isinstance(result, bool):
        return result
    raise ValueError("requestId must be a string or number.")


def optional_string(value: JsonObject, key: str) -> str | None:
    result = value.get(key)
    return result if isinstance(result, str) and result else None


def optional_boolean_or_none(value: JsonObject, key: str) -> bool | None:
    result = value.get(key)
    return result if isinstance(result, bool) else None


def optional_boolean(value: JsonObject, key: str, default: bool) -> bool:
    result = value.get(key)
    return result if isinstance(result, bool) else default


def optional_number(value: JsonObject, key: str) -> int | None:
    result = value.get(key)
    if isinstance(result, int | float) and not isinstance(result, bool) and result > 0:
        return int(result)
    return None


def optional_mode(value: JsonObject, key: str) -> str | None:
    result = optional_string(value, key)
    return result if result in {"default", "plan"} else None


def optional_prefer(value: JsonObject, key: str) -> str | None:
    result = optional_string(value, key)
    return result if result in {"latest_active", "latest_any"} else None


def optional_string_array(value: JsonObject, key: str) -> list[str] | None:
    result = value.get(key)
    if not isinstance(result, list):
        return None
    return [item for item in result if isinstance(item, str) and item]
