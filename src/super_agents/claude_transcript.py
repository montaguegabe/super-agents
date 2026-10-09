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
import threading
from pathlib import Path
from typing import Any

from super_agents.agent_store import Session, preview
from super_agents.claude_options import CLAUDE_CONFIG_DIR_ENV
from super_agents.claude_prompts import user_prompt_for_display

JsonObject = dict[str, Any]

CLAUDE_PROJECTS_DIR_NAME = "projects"
# Parsed transcripts keyed by path. An unchanged file (same mtime_ns and
# size) is served from cache; a grown file is parsed incrementally from the
# previously consumed byte offset, since Claude Code only appends to a
# transcript. Only a shrunk file forces a full re-parse. Without this, every
# detail poll of an active terminal session re-parsed the whole multi-MB
# file on each new line.
_TRANSCRIPT_CACHE: dict[str, "_ParsedTranscript"] = {}
_TRANSCRIPT_CACHE_MAX_ENTRIES = 32
# Reads run on worker threads; one parser state per file must not be fed by
# two threads at once.
_TRANSCRIPT_CACHE_LOCK = threading.Lock()


class _ParsedTranscript:
    __slots__ = ("mtime_ns", "size", "consumed", "turns", "current", "session_id")

    def __init__(self, session_id: str) -> None:
        self.mtime_ns = -1
        self.size = 0
        # Bytes parsed so far; always ends on a newline boundary.
        self.consumed = 0
        self.turns: list[JsonObject] = []
        # The turn still collecting assistant replies at the consumed offset.
        self.current: JsonObject | None = None
        self.session_id = session_id


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
    with _TRANSCRIPT_CACHE_LOCK:
        return _cached_transcript_turns_locked(path, session_id)


def _cached_transcript_turns_locked(path: Path, session_id: str) -> list[JsonObject]:
    try:
        stat = path.stat()
    except OSError:
        return []
    key = str(path)
    parsed = _TRANSCRIPT_CACHE.get(key)
    if parsed is not None and parsed.session_id != session_id:
        parsed = None
    if parsed is not None and parsed.mtime_ns == stat.st_mtime_ns and parsed.size == stat.st_size:
        return parsed.turns
    if parsed is None or stat.st_size < parsed.consumed:
        parsed = _ParsedTranscript(session_id)
    try:
        _parse_transcript_append(path, parsed)
    except OSError:
        return []
    parsed.mtime_ns = stat.st_mtime_ns
    parsed.size = stat.st_size
    if key not in _TRANSCRIPT_CACHE:
        while len(_TRANSCRIPT_CACHE) >= _TRANSCRIPT_CACHE_MAX_ENTRIES:
            _TRANSCRIPT_CACHE.pop(next(iter(_TRANSCRIPT_CACHE)))
    _TRANSCRIPT_CACHE[key] = parsed
    return parsed.turns


def _parse_transcript_turns(path: Path, session_id: str) -> list[JsonObject]:
    parsed = _ParsedTranscript(session_id)
    try:
        _parse_transcript_append(path, parsed)
    except OSError:
        return []
    return parsed.turns


def _parse_transcript_append(path: Path, parsed: _ParsedTranscript) -> None:
    """Feed the bytes appended since ``parsed.consumed`` into the turn list.

    Only whole lines are consumed; a partially written trailing line waits
    for the next read. Raises OSError for the caller to handle.
    """
    with path.open("rb") as handle:
        handle.seek(parsed.consumed)
        chunk = handle.read()
    end = chunk.rfind(b"\n") + 1
    if end == 0:
        return
    text_chunk = chunk[:end].decode("utf-8", errors="replace")
    turns = parsed.turns
    current = parsed.current
    session_id = parsed.session_id
    for line in text_chunk.splitlines():
        entry = _json_object(line)
        if entry is None or entry.get("isSidechain"):
            continue
        entry_type = entry.get("type")
        if entry_type == "user" and not entry.get("isMeta"):
            text = _message_text(entry)
            if not text:
                continue
            current = _new_turn(entry, user_prompt_for_display(text), session_id, index=len(turns))
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
    parsed.current = current
    parsed.consumed += end


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
