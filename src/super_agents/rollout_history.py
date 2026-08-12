"""Fallback turn reconstruction from Codex rollout files.

Codex CLIs that write threads in ``paginated`` history mode persist turn
items only to the rollout JSONL; the app-server serves ``thread/read`` turns
for those threads from a projection database that is populated when the
thread is loaded. Until something resumes the thread, ``thread/read`` and
``thread/turns/list`` both return zero turns even though the conversation is
fully on disk. This module rebuilds app-server-shaped turns directly from
the rollout file so reads of not-loaded paginated threads still have a body.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .state import JsonObject

logger = logging.getLogger(__name__)


def needs_rollout_turn_fallback(thread: Any, include_turns: bool) -> bool:
    """True when a thread/read result has no readable turns to serve.

    Only paginated-history threads suffer this: legacy threads get their
    turns inlined by the app-server even while not loaded.
    """
    if not include_turns or not isinstance(thread, dict):
        return False
    if thread.get("turns"):
        return False
    if thread.get("historyMode") != "paginated":
        return False
    status = thread.get("status")
    status_type = status.get("type") if isinstance(status, dict) else status
    if status_type != "notLoaded":
        return False
    path = thread.get("path")
    return isinstance(path, str) and bool(path)


def rollout_fallback_turns(path: str | Path) -> list[JsonObject]:
    """Reconstruct app-server-shaped turns from a rollout JSONL file.

    Returns an empty list when the file is missing or holds no turns, so
    callers can fall back to the original (empty) payload unchanged.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        logger.debug("Unable to read rollout file %s", path, exc_info=True)
        return []

    turns_by_id: dict[str, JsonObject] = {}
    ordered_ids: list[str] = []

    def turn_for(turn_id: str, record: JsonObject, payload: JsonObject) -> JsonObject:
        turn = turns_by_id.get(turn_id)
        if turn is None:
            turn = {
                "id": turn_id,
                "items": [],
                "itemsView": "full",
                "status": "inProgress",
                "error": None,
                "startedAt": payload.get("started_at") or _record_epoch_seconds(record),
                "completedAt": None,
            }
            turns_by_id[turn_id] = turn
            ordered_ids.append(turn_id)
        return turn

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if not isinstance(record, dict) or record.get("type") != "event_msg":
            continue
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        turn_id = payload.get("turn_id") or payload.get("turnId")
        if not isinstance(turn_id, str) or not turn_id:
            continue
        event_type = payload.get("type")
        if event_type == "task_started":
            turn_for(turn_id, record, payload)
        elif event_type == "item_completed":
            item = payload.get("item")
            if isinstance(item, dict):
                turn_for(turn_id, record, payload)["items"].append(_normalize_rollout_item(item))
        elif event_type == "task_complete":
            turn = turn_for(turn_id, record, payload)
            turn["status"] = "completed"
            turn["completedAt"] = payload.get("completed_at") or _record_epoch_seconds(record)
        elif event_type == "turn_aborted":
            turn = turn_for(turn_id, record, payload)
            turn["status"] = "interrupted"
            turn["completedAt"] = payload.get("completed_at") or _record_epoch_seconds(record)

    return [turns_by_id[turn_id] for turn_id in ordered_ids]


def _normalize_rollout_item(item: JsonObject) -> JsonObject:
    """Map a rollout item onto the shape thread/read inlines for legacy threads.

    Rollout items use PascalCase type tags and keep agent text in a content
    list; app-server items use camelCase tags and a flat ``text`` field.
    """
    normalized = dict(item)
    item_type = _lower_first(item.get("type"))
    if item_type:
        normalized["type"] = item_type
    content = normalized.get("content")
    if isinstance(content, list):
        normalized["content"] = [
            {**part, "type": _lower_first(part.get("type")) or part.get("type")} if isinstance(part, dict) else part
            for part in content
        ]
    if item_type == "agentMessage" and not isinstance(normalized.get("text"), str):
        normalized["text"] = "\n\n".join(
            part.get("text", "")
            for part in normalized.get("content") or []
            if isinstance(part, dict) and part.get("type") == "text" and part.get("text")
        )
        normalized.pop("content", None)
    if item_type == "fileChange" and isinstance(normalized.get("changes"), dict):
        normalized["changes"] = [
            {
                "path": change_path,
                "kind": {"type": change.get("type")},
                "diff": change.get("diff") or change.get("content"),
            }
            for change_path, change in normalized["changes"].items()
            if isinstance(change, dict)
        ]
    return normalized


def _lower_first(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value[:1].lower() + value[1:]


def _record_epoch_seconds(record: JsonObject) -> int | None:
    timestamp = record.get("timestamp")
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None
