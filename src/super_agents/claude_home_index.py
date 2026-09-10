"""Last-interaction index over the local Claude Code home.

Session listings previously showed only sessions already registered in the
agent store, so a Claude Code session the user was actively talking to —
started outside Super Agents (CLI, IDE, harness) — never appeared, and
registered rows went stale at whatever ``updated_at`` their last import
wrote. This module indexes the Claude home's transcript JSONL files by
last-interaction time (file mtime): newly interacted-with sessions are
registered into the store, and already-registered sessions get their
``updated_at`` refreshed, so listings sorted by update time reflect what the
user actually touched most recently.

Session names honor user renames. Claude Code's /rename appends a
``custom-title`` entry to the transcript; the latest one names the session,
falling back to a preview of the first user prompt. Refresh sweeps re-read the
title so renames made after registration propagate, tracking the last synced
title in ``transcript_title`` so a store-side rename (``super_agents_rename``)
is only overridden by a *newer* transcript rename, never re-clobbered by an
unchanged one. ``transcript_title`` is NULL only for rows written before title
sync existed; those get a one-time backfill scan even when idle, recording ""
when no title exists so the scan is not repeated.
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from super_agents.agent_store import Session, Store, preview
from super_agents.claude_transcript import CLAUDE_PROJECTS_DIR_NAME, claude_config_dir

logger = logging.getLogger(__name__)

# How many newest transcripts get registered/refreshed per sweep. Sorting is
# by mtime over the full scan; only writes are bounded.
MAX_SESSIONS_PER_SWEEP = 50
# Throttle between sweeps so steady-state listing polls do not rescan.
SWEEP_INTERVAL_SECONDS = 15.0
# Lines examined for the first meaningful user message before giving up.
_NAME_SCAN_MAX_LINES = 200

_last_sweep_by_store: dict[str, float] = {}


def refresh_last_interaction_index(store: Store, *, now: float | None = None) -> int:
    """Sweep the Claude home and sync last-interaction times into the store.

    Returns the number of sessions registered or refreshed. Throttled per
    store path; safe to call before every listing.
    """
    key = str(store.path)
    current = now if now is not None else time.monotonic()
    last = _last_sweep_by_store.get(key)
    if last is not None and current - last < SWEEP_INTERVAL_SECONDS:
        return 0
    _last_sweep_by_store[key] = current
    try:
        return _sweep(store)
    except Exception:
        logger.warning("Claude home last-interaction sweep failed", exc_info=True)
        return 0


def _sweep(store: Store) -> int:
    projects_dir = claude_config_dir() / CLAUDE_PROJECTS_DIR_NAME
    transcripts = _transcripts_by_recency(projects_dir)
    if not transcripts:
        return 0
    changed = 0
    for path, mtime in transcripts[:MAX_SESSIONS_PER_SWEEP]:
        backend_session_id = path.stem
        interacted_at = _iso_from_epoch(mtime)
        existing = _session_for_backend_id(store, backend_session_id)
        if existing is not None:
            session, synced_title = existing
            fresh = not session.updated_at or session.updated_at < interacted_at
            if not fresh and synced_title is not None:
                continue
            # Idle rows only reach here for the one-time title backfill; keep
            # their real last-activity time instead of bumping it to "now".
            updates: dict[str, object] = {"updated_at": interacted_at if fresh else session.updated_at}
            title = _latest_custom_title(path) or ""
            if title != synced_title:
                updates["transcript_title"] = title
                if title and title != session.name:
                    with store.connect() as conn:
                        updates["name"] = _unique_session_name(conn, title, exclude_id=session.id)
            store.update_session(session.id, **updates)
            changed += 1
            continue
        registered = _register_transcript_session(store, path, backend_session_id, interacted_at)
        if registered:
            changed += 1
    return changed


def _transcripts_by_recency(projects_dir: Path) -> list[tuple[Path, float]]:
    transcripts: list[tuple[Path, float]] = []
    try:
        project_dirs = [entry for entry in projects_dir.iterdir() if entry.is_dir()]
    except OSError:
        return []
    for project_dir in project_dirs:
        with contextlib.suppress(OSError):
            for path in project_dir.iterdir():
                if path.suffix != ".jsonl" or not path.is_file():
                    continue
                with contextlib.suppress(OSError):
                    transcripts.append((path, path.stat().st_mtime))
    transcripts.sort(key=lambda item: item[1], reverse=True)
    return transcripts


def _session_for_backend_id(store: Store, backend_session_id: str) -> tuple[Session, str | None] | None:
    """(session, last synced transcript title) or None when unregistered."""
    with store.connect() as conn:
        row = conn.execute(
            "select * from sessions where backend_session_id = ?",
            (backend_session_id,),
        ).fetchone()
    if row is None:
        return None
    from super_agents.agent_store import row_to_session

    return row_to_session(row), row["transcript_title"]


def _register_transcript_session(
    store: Store,
    path: Path,
    backend_session_id: str,
    interacted_at: str,
) -> bool:
    parsed = _parse_transcript_head(path)
    if parsed is None:
        return False
    name, cwd, created_at = parsed
    custom_title = _latest_custom_title(path)
    session_id = f"claude_{backend_session_id.replace('-', '')}"
    with store.connect() as conn:
        if conn.execute("select 1 from sessions where id = ?", (session_id,)).fetchone():
            return False
        unique_name = _unique_session_name(conn, custom_title or name)
        conn.execute(
            """
            insert into sessions (
                id, name, transcript_title, cwd, command_json, status, backend, last_observed_state,
                backend_session_id, created_at, updated_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,
                unique_name,
                custom_title or "",
                cwd or str(Path.home()),
                json.dumps(["claude", "--resume", backend_session_id]),
                # Idle local sessions read as completed; activity is conveyed
                # by updated_at (last interaction), not a live status.
                "completed",
                store.backend,
                "Claude Code session indexed from local transcripts",
                backend_session_id,
                created_at or interacted_at,
                interacted_at,
            ),
        )
    return True


def _parse_transcript_head(path: Path) -> tuple[str, str | None, str | None] | None:
    """(name, cwd, created_at) from the transcript's first meaningful entries.

    Returns None when no user message exists (e.g. sidechain-only files), so
    tool-agent transcripts never surface as user sessions.
    """
    cwd: str | None = None
    created_at: str | None = None
    try:
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index >= _NAME_SCAN_MAX_LINES:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    entry = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict) or entry.get("isSidechain"):
                    continue
                if cwd is None and isinstance(entry.get("cwd"), str) and entry["cwd"]:
                    cwd = entry["cwd"]
                if created_at is None and isinstance(entry.get("timestamp"), str):
                    created_at = entry["timestamp"]
                if entry.get("type") != "user" or entry.get("isMeta"):
                    continue
                text = _user_text(entry)
                if text:
                    return preview(text, limit=80) or "Claude Code session", cwd, created_at
    except OSError:
        return None
    return None


def _user_text(entry: dict) -> str | None:
    message = entry.get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    parts: list[str] = []
    if isinstance(content, str):
        parts = [content]
    elif isinstance(content, list):
        parts = [
            block.get("text", "")
            for block in content
            if isinstance(block, dict) and block.get("type") == "text" and isinstance(block.get("text"), str)
        ]
    text = "\n".join(part for part in parts if part).strip()
    if not text or text.startswith(("<command-name>", "<local-command")):
        return None
    return text


def _latest_custom_title(path: Path) -> str | None:
    """Last user-set title from the transcript's ``custom-title`` entries.

    /rename appends these, so the newest one wins. The substring pre-filter
    keeps the scan cheap on large transcripts that were never renamed.
    """
    title: str | None = None
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if '"custom-title"' not in line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict) or entry.get("type") != "custom-title":
                    continue
                value = entry.get("customTitle")
                if isinstance(value, str) and value.strip():
                    title = value.strip()
    except OSError:
        return None
    return preview(title, limit=80)


def _unique_session_name(conn, base_name: str, exclude_id: str | None = None) -> str:
    candidate = base_name
    suffix = 2
    while True:
        row = conn.execute("select id from sessions where name = ?", (candidate,)).fetchone()
        if row is None or row["id"] == exclude_id:
            return candidate
        suffix_text = f" ({suffix})"
        candidate = f"{base_name[: 80 - len(suffix_text)]}{suffix_text}"
        suffix += 1


def _iso_from_epoch(epoch_seconds: float) -> str:
    return (
        datetime.fromtimestamp(epoch_seconds, tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    )
