from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from dataclasses import replace
from pathlib import Path
from typing import Callable

from .app_models import QueuedTurn
from .app_time import iso_now, parse_iso_ms
from .state import JsonObject, get_string, state_file_lock

DEFAULT_QUEUE_STARTING_STALE_SECONDS = 600


def queue_file_for_thread(queue_dir: Path, thread_id: str) -> Path:
    digest = hashlib.sha256(thread_id.encode("utf-8")).hexdigest()
    return queue_dir / f"{digest}.json"


def new_queued_turn(
    *,
    thread_id: str,
    label: str | None,
    agent_name: str | None,
    input_data: JsonObject,
) -> QueuedTurn:
    return QueuedTurn(
        id=f"q_{uuid.uuid4().hex}",
        thread_id=thread_id,
        label=label,
        agent_name=agent_name,
        input_data=input_data,
        queued_at=iso_now(),
    )


def append_queued_turn(queue_dir: Path, item: QueuedTurn) -> tuple[QueuedTurn, int]:
    path = queue_file_for_thread(queue_dir, item.thread_id)
    with state_file_lock(path):
        items = read_thread_queue_unlocked(path, item.thread_id)
        items.append(item)
        write_thread_queue_unlocked(path, item.thread_id, items)
        return item, visible_queue_depth(items)


def reserve_next_queued_turn(
    queue_dir: Path,
    thread_id: str,
    has_active_turn: Callable[[], bool],
) -> QueuedTurn | None:
    path = queue_file_for_thread(queue_dir, thread_id)
    with state_file_lock(path):
        items = read_thread_queue_unlocked(path, thread_id)
        if has_unexpired_starting_item(items):
            return None
        if has_active_turn():
            return None
        for index, item in enumerate(items):
            if item.status != "queued" and not starting_item_is_stale(item):
                continue
            reserved = replace(
                item,
                status="starting",
                started_at=iso_now(),
                attempts=item.attempts + 1,
                last_error=None,
            )
            items[index] = reserved
            write_thread_queue_unlocked(path, thread_id, items)
            return reserved
        return None


def complete_queued_turn(queue_dir: Path, item: QueuedTurn) -> None:
    path = queue_file_for_thread(queue_dir, item.thread_id)
    with state_file_lock(path):
        items = [queued for queued in read_thread_queue_unlocked(path, item.thread_id) if queued.id != item.id]
        write_thread_queue_unlocked(path, item.thread_id, items)


def release_queued_turn(queue_dir: Path, item: QueuedTurn, error: Exception) -> None:
    path = queue_file_for_thread(queue_dir, item.thread_id)
    replacement = replace(item, status="queued", started_at=None, last_error=str(error))
    with state_file_lock(path):
        items = read_thread_queue_unlocked(path, item.thread_id)
        for index, queued in enumerate(items):
            if queued.id == item.id:
                items[index] = replacement
                break
        else:
            items.insert(0, replacement)
        write_thread_queue_unlocked(path, item.thread_id, items)


def queued_turn_summaries(queue_dir: Path) -> list[JsonObject]:
    if not queue_dir.is_dir():
        return []
    summaries: list[JsonObject] = []
    for path in sorted(queue_dir.glob("*.json")):
        with state_file_lock(path):
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            thread_id = get_string(raw, "threadId")
            if not thread_id:
                continue
            items = [item for item in read_items(raw.get("items"), thread_id) if item.status in {"queued", "starting"}]
            if not items:
                continue
            summaries.append(
                {
                    "threadId": thread_id,
                    "queueDepth": visible_queue_depth(items),
                    "items": [item.to_json() for item in items[:5]],
                }
            )
    return summaries


def read_thread_queue_unlocked(path: Path, thread_id: str) -> list[QueuedTurn]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    return read_items(raw.get("items") if isinstance(raw, dict) else None, thread_id)


def write_thread_queue_unlocked(path: Path, thread_id: str, items: list[QueuedTurn]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not items:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        return
    payload = json.dumps(
        {
            "threadId": thread_id,
            "items": [item.to_json() for item in items],
        },
        indent=2,
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        tmp.write(payload + "\n")
        tmp_name = tmp.name
    os.replace(tmp_name, path)


def read_items(value: object, thread_id: str) -> list[QueuedTurn]:
    if not isinstance(value, list):
        return []
    items: list[QueuedTurn] = []
    for raw in value:
        if not isinstance(raw, dict):
            continue
        input_data = raw.get("inputData") or raw.get("input")
        if not isinstance(input_data, dict):
            continue
        status = get_string(raw, "status") or "queued"
        if status not in {"queued", "starting"}:
            status = "queued"
        items.append(
            QueuedTurn(
                id=get_string(raw, "id") or f"q_{uuid.uuid4().hex}",
                thread_id=get_string(raw, "threadId") or thread_id,
                label=get_string(raw, "label"),
                agent_name=get_string(raw, "agentName"),
                input_data=input_data,
                queued_at=get_string(raw, "queuedAt") or iso_now(),
                status=status,
                started_at=get_string(raw, "startedAt"),
                attempts=raw.get("attempts") if isinstance(raw.get("attempts"), int) else 0,
                last_error=get_string(raw, "lastError"),
            )
        )
    return items


def visible_queue_depth(items: list[QueuedTurn]) -> int:
    return len([item for item in items if item.status in {"queued", "starting"}])


def has_unexpired_starting_item(items: list[QueuedTurn]) -> bool:
    return any(item.status == "starting" and not starting_item_is_stale(item) for item in items)


def starting_item_is_stale(item: QueuedTurn) -> bool:
    if item.status != "starting":
        return False
    if not item.started_at:
        return True
    return int(time.time() * 1000) - parse_iso_ms(item.started_at) > DEFAULT_QUEUE_STARTING_STALE_SECONDS * 1000
