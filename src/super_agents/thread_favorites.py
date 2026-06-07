"""Read Openbase Coder's local thread favorite metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

FAVORITES_FILE = "thread-favorites.json"


def favorite_status(thread_id: str | None) -> dict[str, str | bool | None]:
    normalized = _normalize_thread_id(thread_id)
    entry = read_favorites().get(normalized) if normalized else None
    favorited_at = entry.get("favorited_at") if isinstance(entry, dict) else None
    return {
        "threadId": normalized,
        "isFavorite": bool(entry),
        "favoritedAt": favorited_at if isinstance(favorited_at, str) else None,
    }


def is_favorite(thread_id: str | None) -> bool:
    return bool(favorite_status(thread_id)["isFavorite"])


def read_favorites() -> dict[str, dict[str, Any]]:
    path = favorites_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    threads = payload.get("threads") if isinstance(payload, dict) else None
    if not isinstance(threads, dict):
        return {}
    favorites: dict[str, dict[str, Any]] = {}
    for raw_thread_id, raw_entry in threads.items():
        thread_id = _normalize_thread_id(raw_thread_id)
        if not thread_id or not isinstance(raw_entry, dict):
            continue
        favorites[thread_id] = {
            "thread_id": thread_id,
            "favorited_at": raw_entry.get("favorited_at")
            if isinstance(raw_entry.get("favorited_at"), str)
            else None,
        }
    return favorites


def favorites_path() -> Path:
    data_dir = os.environ.get("OPENBASE_CODER_CLI_DATA_DIR")
    return Path(data_dir).expanduser() / FAVORITES_FILE if data_dir else Path.home() / ".openbase" / FAVORITES_FILE


def _normalize_thread_id(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""
