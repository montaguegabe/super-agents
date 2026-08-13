"""Synthesized turn views from Claude Code session transcript JSONL files.

Sessions imported from an existing Claude Code home (for example by a
thread-sync integration) carry their conversation history only in the
transcript JSONL under ``<config dir>/projects/``; nothing backfills the
Super Agents turns table for them. These helpers parse that transcript into
read-only turn views so read paths can display imported history. They are a
fallback: sessions whose turns ran through Super Agents keep their store
turns, which remain authoritative for status and errors.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from super_agents.agent_store import Session, preview
from super_agents.claude_options import CLAUDE_CONFIG_DIR_ENV

JsonObject = dict[str, Any]

CLAUDE_PROJECTS_DIR_NAME = "projects"
# Parsed transcripts keyed by path, invalidated on (mtime_ns, size) change so
# steady-state polling of a detail page does not re-parse multi-MB files.
_TRANSCRIPT_CACHE: dict[str, tuple[int, int, list[JsonObject]]] = {}
_TRANSCRIPT_CACHE_MAX_ENTRIES = 32


def transcript_turn_views(session: Session, limit: int = 20) -> list[JsonObject]:
    """Newest-first turn views parsed from the session's transcript JSONL.

    Empty when the session has no backend session id or no transcript file.
    """
    if not session.backend_session_id:
        return []
    path = transcript_path(session)
    if path is None:
        return []
    turns = _cached_transcript_turns(path, session.id)
    recent = turns[-limit:] if limit and limit > 0 else turns
    return list(reversed(recent))


def transcript_path(session: Session) -> Path | None:
    """Locate the transcript JSONL for the session's backend session id."""
    projects_dir = claude_config_dir() / CLAUDE_PROJECTS_DIR_NAME
    file_name = f"{session.backend_session_id}.jsonl"
    candidate = projects_dir / _project_dir_name(session.cwd or "") / file_name
    if candidate.is_file():
        return candidate
    # The project-dir mangle tracks the Claude Code CLI; fall back to a scan
    # in case the algorithm drifts or the session moved directories.
    try:
        return next(projects_dir.glob(f"*/{file_name}"))
    except (StopIteration, OSError):
        return None


def claude_config_dir() -> Path:
    configured = os.environ.get(CLAUDE_CONFIG_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".claude"


def _project_dir_name(cwd: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "-", cwd)


def _cached_transcript_turns(path: Path, session_id: str) -> list[JsonObject]:
    try:
        stat = path.stat()
    except OSError:
        return []
    key = str(path)
    cached = _TRANSCRIPT_CACHE.get(key)
    if cached is not None and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
        return cached[2]
    turns = _parse_transcript_turns(path, session_id)
    while len(_TRANSCRIPT_CACHE) >= _TRANSCRIPT_CACHE_MAX_ENTRIES:
        _TRANSCRIPT_CACHE.pop(next(iter(_TRANSCRIPT_CACHE)))
    _TRANSCRIPT_CACHE[key] = (stat.st_mtime_ns, stat.st_size, turns)
    return turns


def _parse_transcript_turns(path: Path, session_id: str) -> list[JsonObject]:
    turns: list[JsonObject] = []
    current: JsonObject | None = None
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                entry = _json_object(line)
                if entry is None or entry.get("isSidechain"):
                    continue
                entry_type = entry.get("type")
                if entry_type == "user" and not entry.get("isMeta"):
                    text = _message_text(entry)
                    if not text:
                        continue
                    current = _new_turn(entry, text, session_id, index=len(turns))
                    turns.append(current)
                elif entry_type == "assistant":
                    text = _message_text(entry)
                    if not text:
                        continue
                    if current is None:
                        # History continued from a prior session: replies may
                        # precede the first prompt captured in this file.
                        current = _new_turn(entry, "", session_id, index=len(turns))
                        turns.append(current)
                    _append_reply(current, entry, text)
    except OSError:
        return []
    return turns


def _new_turn(entry: JsonObject, prompt: str, session_id: str, *, index: int) -> JsonObject:
    timestamp = _timestamp(entry)
    items: list[JsonObject] = []
    if prompt:
        items.append({"type": "userMessage", "content": [{"type": "text", "text": prompt}]})
    turn: JsonObject = {
        "turnId": str(entry.get("uuid") or f"transcript-{index}"),
        "sessionId": session_id,
        "status": "completed",
        "source": "transcript",
        "items": items,
    }
    if prompt:
        turn["promptPreview"] = preview(prompt)
    if timestamp:
        turn["createdAt"] = timestamp
        turn["updatedAt"] = timestamp
    return turn


def _append_reply(turn: JsonObject, entry: JsonObject, text: str) -> None:
    turn["items"].append({"type": "agentMessage", "text": text})
    turn["lastUsefulMessage"] = text
    model = _message_model(entry)
    if model:
        turn["model"] = model
    timestamp = _timestamp(entry)
    if timestamp:
        turn.setdefault("createdAt", timestamp)
        turn["updatedAt"] = timestamp
        turn["finishedAt"] = timestamp


def _message_model(entry: JsonObject) -> str | None:
    message = entry.get("message")
    if not isinstance(message, dict):
        return None
    model = message.get("model")
    return model if isinstance(model, str) and model else None


def _json_object(line: str) -> JsonObject | None:
    stripped = line.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _message_text(entry: JsonObject) -> str:
    message = entry.get("message")
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _timestamp(entry: JsonObject) -> str | None:
    value = entry.get("timestamp")
    return value if isinstance(value, str) and value else None
