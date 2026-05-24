from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from .state import JsonObject


def age_ms(iso_value: str | None) -> int | None:
    parsed = parse_iso_ms(iso_value)
    if parsed <= 0:
        return None
    return max(0, int(time.time() * 1000) - parsed)


def turn_key(thread_id: str, turn_id: str | None) -> str:
    return f"{thread_id}:{turn_id}"


def path_basename(value: str) -> str:
    return Path(value).expanduser().name


def thread_recency(thread: JsonObject) -> int:
    for key in ["updatedAtMs", "updated_at_ms"]:
        value = thread.get(key)
        if isinstance(value, int | float):
            return int(value)
    for key in ["updatedAt", "updated_at"]:
        value = thread.get(key)
        if isinstance(value, int | float):
            return int(value * 1000)
        if isinstance(value, str):
            return parse_iso_ms(value)
    return 0


def turn_recency(turn: JsonObject) -> int:
    for key in ["completedAt", "startedAt"]:
        value = turn.get(key)
        if isinstance(value, int | float):
            return int(value * 1000)
    return 0


def iso_from_thread_time(thread: JsonObject) -> str:
    recency = thread_recency(thread)
    if recency <= 0:
        return iso_now()
    return (
        datetime.fromtimestamp(recency / 1000, timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_iso_ms(value: str | None) -> int:
    if not value:
        return 0
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return 0
    return int(parsed.timestamp() * 1000)
