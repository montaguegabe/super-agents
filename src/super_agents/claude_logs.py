"""Log serialization for Claude Agent SDK messages."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

from super_agents.agent_store import iso_now


def append_log(path_value: str | None, text: str) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def message_to_log(message: Any) -> str:
    payload = _jsonable_message(message)
    return f"[{iso_now()}] {json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)}\n"


def _jsonable_message(message: Any) -> Any:
    if dataclasses.is_dataclass(message):
        return dataclasses.asdict(message)
    if isinstance(message, dict):
        return message
    data = {
        key: value
        for key in ("type", "subtype", "result", "session_id", "content")
        if (value := getattr(message, key, None)) is not None
    }
    return data or repr(message)


def message_preview(message: Any) -> str:
    result = getattr(message, "result", None)
    if isinstance(result, str) and result.strip():
        return result.strip()
    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts).strip()
    return ""


def message_session_id(message: Any) -> str | None:
    session_id = getattr(message, "session_id", None)
    return session_id if isinstance(session_id, str) and session_id else None
