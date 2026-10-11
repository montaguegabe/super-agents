from __future__ import annotations

import json
import logging
import os
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .backend_config import BACKENDS, execution_backend
from .claude_prompts import user_prompt_for_title
from .state import state_file_lock

logger = logging.getLogger(__name__)

JsonObject = dict[str, Any]
APP_DIR_ENV = "SUPER_AGENTS_CLAUDE_CODE_HOME"


@dataclass(frozen=True)
class Session:
    id: str
    name: str
    cwd: str
    command: list[str]
    status: str
    created_at: str
    updated_at: str
    agent_name: str | None = None
    developer_instructions: str | None = None
    model: str | None = None
    pid: int | None = None
    active_turn_id: str | None = None
    last_turn_id: str | None = None
    last_observed_state: str | None = None
    last_useful_message: str | None = None
    backend_session_id: str | None = None
    backend: str | None = None
    last_client_instance: str | None = None
    last_exit_code: int | None = None
    log_path: str | None = None
    raw_log_path: str | None = None
    title: str | None = None
    auto_title: bool = False

    def to_json(self, include_paths: bool = True) -> JsonObject:
        data: JsonObject = {
            "id": self.id,
            "name": self.name,
            "title": self.title,
            "agentName": self.agent_name,
            "developerInstructions": self.developer_instructions,
            "cwd": self.cwd,
            "command": self.command,
            "model": self.model,
            "status": self.status,
            "pid": self.pid,
            "activeTurnId": self.active_turn_id,
            "lastTurnId": self.last_turn_id,
            "lastObservedState": self.last_observed_state,
            "lastUsefulMessage": self.last_useful_message,
            "backendSessionId": self.backend_session_id,
            "backend": self.backend,
            "lastClientInstance": self.last_client_instance,
            "lastExitCode": self.last_exit_code,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
        if include_paths:
            data["logPath"] = self.log_path
            data["rawLogPath"] = self.raw_log_path
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class Turn:
    id: str
    session_id: str
    prompt: str
    status: str
    created_at: str
    updated_at: str
    mode: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    service_tier: str | None = None
    finished_at: str | None = None
    attempts: int = 0
    last_error: str | None = None
    # The backend's structured reason for the failure, when it gave one
    # (Claude Agent SDK AssistantMessage.error, e.g. "authentication_failed").
    error_kind: str | None = None
    last_useful_message: str | None = None
    # Set once the model's response for this turn has completed, even if the
    # turn stays "running" to wait on background tasks it spawned (a dev
    # server, a watch). Lets a later orphan sweep tell "finished, abandoned
    # its background tasks" apart from "died mid-response".
    response_finished_at: str | None = None
    # Steering messages delivered into the turn while it ran, in order, as
    # {"text": ..., "createdAt": ...} objects.
    steers: tuple[JsonObject, ...] = ()

    def to_json(self) -> JsonObject:
        return {
            key: value
            for key, value in {
                "turnId": self.id,
                "sessionId": self.session_id,
                "promptPreview": preview(self.prompt),
                "status": self.status,
                "mode": self.mode,
                "model": self.model,
                "reasoningEffort": self.reasoning_effort,
                "serviceTier": self.service_tier,
                "createdAt": self.created_at,
                "updatedAt": self.updated_at,
                "finishedAt": self.finished_at,
                "attempts": self.attempts,
                "lastError": self.last_error,
                "errorKind": self.error_kind,
                "lastUsefulMessage": self.last_useful_message,
                "responseFinishedAt": self.response_finished_at,
                "steers": list(self.steers) or None,
            }.items()
            if value is not None
        }


class Store:
    def __init__(self, path: Path | None = None, *, backend: str | None = None) -> None:
        self.path = path or database_path()
        self.backend = backend
        self.path.parent.mkdir(parents=True, exist_ok=True)
        logs_dir().mkdir(parents=True, exist_ok=True)
        self._init()

    def scope_backend(self, backend: str) -> None:
        """Claim legacy rows and constrain this store instance to one backend."""
        if self.backend and self.backend != backend:
            raise ValueError(f"Store is already scoped to backend {self.backend}.")
        self.backend = backend
        with self.connect() as conn:
            self._claim_legacy_rows(conn)

    def _claim_legacy_rows(self, conn: sqlite3.Connection) -> None:
        """Rows written before backends were recorded belong to this store."""
        if self.backend:
            conn.execute("update sessions set backend = ? where backend is null", (self.backend,))

    def _scope(self) -> tuple[str, ...]:
        """Every identity this store's rows may carry.

        A store is scoped by execution backend, not by identity: claude_code
        and openbase_cloud sessions run on the same Claude Code engine on
        one machine, and several processes there may hold different
        identities (the API services, the Super Agents MCP server inside a
        session, a thread-sync import). Each process keeps writing its own
        identity on the rows it creates, and every one of them sees all the
        machine's Claude Code sessions. Rows of other execution backends are
        never visible.
        """
        if not self.backend:
            return ()
        mine = execution_backend(self.backend)
        return tuple(sorted(candidate for candidate in BACKENDS if execution_backend(candidate) == mine))

    def _scope_sql(self) -> tuple[str, list[str]]:
        scope = self._scope()
        return f"backend in ({', '.join('?' for _ in scope)})", list(scope)

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with state_file_lock(self.path):
            try:
                self._init_schema()
            except sqlite3.DatabaseError as exc:
                if not self._is_corrupt_database_error(exc):
                    raise
                quarantined_path = self._quarantine_corrupt_database()
                logger.error(
                    "Super Agents database %s is corrupt; preserved it at %s and initialized a fresh store",
                    self.path,
                    quarantined_path,
                )
                self._init_schema()

    def _init_schema(self) -> None:
        from .claude_turn_output import initialize_turn_output

        with self.connect() as conn:
            initialize_turn_output(conn)
            conn.executescript(
                """
                create table if not exists sessions (
                    id text primary key,
                    name text not null unique,
                    agent_name text,
                    developer_instructions text,
                    cwd text not null,
                    command_json text not null,
                    model text,
                    status text not null,
                    pid integer,
                    active_turn_id text,
                    last_turn_id text,
                    last_observed_state text,
                    last_useful_message text,
                    backend_session_id text,
                    backend text,
                    transcript_title text,
                    last_client_instance text,
                    last_exit_code integer,
                    log_path text,
                    raw_log_path text,
                    created_at text not null,
                    updated_at text not null
                );
                create table if not exists turns (
                    id text primary key,
                    session_id text not null references sessions(id) on delete cascade,
                    prompt text not null,
                    mode text,
                    model text,
                    reasoning_effort text,
                    service_tier text,
                    status text not null,
                    attempts integer not null default 0,
                    last_error text,
                    last_useful_message text,
                    created_at text not null,
                    updated_at text not null,
                    finished_at text
                );
                create index if not exists turns_session_idx on turns(session_id, created_at);
                """
            )
            columns = {row["name"] for row in conn.execute("pragma table_info(turns)").fetchall()}
            if "reasoning_effort" not in columns:
                conn.execute("alter table turns add column reasoning_effort text")
            if "service_tier" not in columns:
                conn.execute("alter table turns add column service_tier text")
            if "last_useful_message" not in columns:
                conn.execute("alter table turns add column last_useful_message text")
            if "steers_json" not in columns:
                conn.execute("alter table turns add column steers_json text")
            if "response_finished_at" not in columns:
                conn.execute("alter table turns add column response_finished_at text")
            if "error_kind" not in columns:
                conn.execute("alter table turns add column error_kind text")
            session_columns = {row["name"] for row in conn.execute("pragma table_info(sessions)").fetchall()}
            if "developer_instructions" not in session_columns:
                conn.execute("alter table sessions add column developer_instructions text")
            if "backend_session_id" not in session_columns:
                conn.execute("alter table sessions add column backend_session_id text")
            if "last_client_instance" not in session_columns:
                conn.execute("alter table sessions add column last_client_instance text")
            if "backend" not in session_columns:
                conn.execute("alter table sessions add column backend text")
            if "transcript_title" not in session_columns:
                conn.execute("alter table sessions add column transcript_title text")
            if "title" not in session_columns:
                conn.execute("alter table sessions add column title text")
            if "auto_title" not in session_columns:
                conn.execute("alter table sessions add column auto_title integer not null default 0")
            if conn.execute("pragma user_version").fetchone()[0] < 1:
                _strip_session_id_suffixes(conn)
                conn.execute("pragma user_version = 1")
            self._claim_legacy_rows(conn)

    @staticmethod
    def _is_corrupt_database_error(exc: sqlite3.DatabaseError) -> bool:
        error_code = getattr(exc, "sqlite_errorcode", None)
        if error_code in {sqlite3.SQLITE_CORRUPT, sqlite3.SQLITE_NOTADB}:
            return True
        message = str(exc).lower()
        return "database disk image is malformed" in message or "file is not a database" in message

    def _quarantine_corrupt_database(self) -> Path:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        quarantined_path = self.path.with_name(f"{self.path.name}.corrupt-{timestamp}")
        for suffix in ("", "-wal", "-shm"):
            source = Path(f"{self.path}{suffix}")
            destination = Path(f"{quarantined_path}{suffix}")
            try:
                source.replace(destination)
            except FileNotFoundError:
                continue
        return quarantined_path

    def create_session(
        self,
        name: str,
        cwd: str | None = None,
        *,
        agent_name: str | None = None,
        developer_instructions: str | None = None,
        model: str | None = None,
        command: list[str] | None = None,
        auto_title: bool = False,
    ) -> Session:
        now = iso_now()
        session_id = f"s_{uuid.uuid4().hex}"
        logs = logs_dir()
        log_path = str(logs / f"{session_id}.log")
        raw_log_path = str(logs / f"{session_id}.raw.log")
        resolved_cwd = str(Path(cwd or default_cwd()).expanduser())
        resolved_command = command or []
        with self.connect() as conn:
            conn.execute(
                """
                insert into sessions (
                    id, name, agent_name, developer_instructions, cwd, command_json, model, status,
                    backend, log_path, raw_log_path, created_at, updated_at, title, auto_title
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    name,
                    agent_name,
                    developer_instructions,
                    resolved_cwd,
                    command_to_json(resolved_command),
                    model,
                    "unknown",
                    self.backend,
                    log_path,
                    raw_log_path,
                    now,
                    now,
                    conversation_title(Path(resolved_cwd).name or "Thread") if auto_title else None,
                    auto_title,
                ),
            )
        return self.get_session(session_id)

    def rename_session(self, session_id: str, new_name: str) -> Session:
        with self.connect() as conn:
            conn.execute(
                "update sessions set name = ?, updated_at = ? where id = ?",
                (new_name, iso_now(), session_id),
            )
        return self.get_session(session_id)

    def get_session(self, session_id: str) -> Session:
        with self.connect() as conn:
            if self.backend:
                clause, params = self._scope_sql()
                row = conn.execute(
                    f"select * from sessions where id = ? and {clause}",
                    (session_id, *params),
                ).fetchone()
            else:
                row = conn.execute("select * from sessions where id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"No session with id {session_id}")
        return row_to_session(row)

    def get_by_name(self, name: str) -> Session | None:
        with self.connect() as conn:
            if self.backend:
                clause, params = self._scope_sql()
                row = conn.execute(
                    f"select * from sessions where name = ? and {clause}",
                    (name, *params),
                ).fetchone()
            else:
                row = conn.execute("select * from sessions where name = ?", (name,)).fetchone()
        return row_to_session(row) if row else None

    def get_name_holder(self, name: str) -> Session | None:
        """The session holding this name regardless of backend.

        ``name`` is unique across the whole table, so a session created under
        another backend still blocks the name; callers that are about to
        create a session need to see that holder even though ``get_by_name``
        scopes to this store's backend.
        """
        with self.connect() as conn:
            row = conn.execute("select * from sessions where name = ?", (name,)).fetchone()
        return row_to_session(row) if row else None

    def retire_name_holder(self, name: str) -> Session | None:
        """Free a name by renaming whichever session holds it, any backend.

        Unlike ``rename_session`` this never refetches through the store's
        backend scope, so it also works on a holder another backend owns.
        Returns the previous holder, or None when the name was free.
        """
        holder = self.get_name_holder(name)
        if holder is None:
            return None
        with self.connect() as conn:
            conn.execute(
                "update sessions set name = ?, updated_at = ? where id = ?",
                (unique_session_name(conn, f"{name} (retired)", exclude_id=holder.id), iso_now(), holder.id),
            )
        return holder

    def get_by_agent_name(self, agent_name: str) -> Session | None:
        """The session an agent name refers to, when no thread carries it as a name.

        People and the Dispatcher address Super Agents by their agent name
        ("tell Marian to…"), while thread names are generated labels. Matching
        is case-insensitive; a session with a running turn wins, then the most
        recently updated one.
        """
        wanted = agent_name.strip().lower()
        if not wanted:
            return None
        with self.connect() as conn:
            if self.backend:
                clause, params = self._scope_sql()
                rows = conn.execute(
                    f"select * from sessions where lower(agent_name) = ? and {clause}",
                    (wanted, *params),
                ).fetchall()
            else:
                rows = conn.execute("select * from sessions where lower(agent_name) = ?", (wanted,)).fetchall()
        sessions = [row_to_session(row) for row in rows]
        if not sessions:
            return None
        sessions.sort(key=lambda session: (bool(session.active_turn_id), session.updated_at or ""), reverse=True)
        return sessions[0]

    def require_by_name(self, name: str) -> Session:
        session = self.get_by_name(name) or self.get_by_agent_name(name)
        if session is None:
            raise KeyError(f"No session named {name}")
        return session

    def list_sessions(self, include_inactive: bool = True, status: str | None = None) -> list[Session]:
        query = "select * from sessions"
        params: list[object] = []
        clauses: list[str] = []
        if not include_inactive:
            clauses.append("status in ('running', 'waiting')")
        if status:
            clauses.append("status = ?")
            params.append(status)
        if self.backend:
            clause, scope = self._scope_sql()
            clauses.append(clause)
            params.extend(scope)
        if clauses:
            query += " where " + " and ".join(clauses)
        query += " order by updated_at desc"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [row_to_session(row) for row in rows]

    def update_session(self, session_id: str, **fields: object) -> Session:
        allowed = {
            "title",
            "auto_title",
            "agent_name",
            "developer_instructions",
            "cwd",
            "model",
            "status",
            "pid",
            "active_turn_id",
            "last_turn_id",
            "last_observed_state",
            "last_useful_message",
            "backend_session_id",
            "last_client_instance",
            "last_exit_code",
            # A rename recorded in the backend's own transcript (for example
            # Claude Code's /rename). `name` follows it; `transcript_title`
            # remembers what was synced so store-side renames are not
            # re-clobbered by unchanged transcript titles.
            "name",
            "transcript_title",
            # Explicit updated_at lets administrative writes (for example
            # orphan reconciliation) preserve the session's real last-activity
            # time instead of bumping it to "now".
            "updated_at",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates.setdefault("updated_at", iso_now())
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [session_id]
        with self.connect() as conn:
            conn.execute(f"update sessions set {assignments} where id = ?", values)
        return self.get_session(session_id)

    def create_turn(
        self,
        session_id: str,
        prompt: str,
        *,
        status: str = "queued",
        mode: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        service_tier: str | None = None,
    ) -> Turn:
        now = iso_now()
        turn_id = f"t_{uuid.uuid4().hex}"
        with self.connect() as conn:
            conn.execute(
                """
                insert into turns (
                    id, session_id, prompt, mode, model, reasoning_effort, service_tier, status, created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    prompt,
                    mode,
                    model,
                    reasoning_effort,
                    service_tier,
                    status,
                    now,
                    now,
                ),
            )
            # The first accepted prompt owns the automatic title, even when
            # execution later fails. The guarded update shares the turn insert
            # transaction, so concurrent writers cannot replace the first title.
            title = conversation_title(prompt)
            conn.execute(
                "update sessions set title = coalesce(?, title), auto_title = 0 where id = ? and auto_title = 1",
                (title, session_id),
            )
            active_turn_id = (
                turn_id
                if status in {"running", "waiting"}
                else conn.execute(
                    "select active_turn_id from sessions where id = ?",
                    (session_id,),
                ).fetchone()["active_turn_id"]
            )
            conn.execute(
                "update sessions set last_turn_id = ?, active_turn_id = ?, updated_at = ? where id = ?",
                (turn_id, active_turn_id, now, session_id),
            )
        return self.get_turn(turn_id)

    def claim_queued_turn(self, session_id: str) -> Turn | None:
        """Atomically claim the first queued turn across owner/caller drainers."""
        self.get_session(session_id)  # Enforce this store's backend scope.
        now = iso_now()
        with self.connect() as conn:
            conn.execute("begin immediate")
            session = conn.execute("select * from sessions where id = ?", (session_id,)).fetchone()
            if session["active_turn_id"] or session["status"] in {"running", "waiting"}:
                return None
            row = conn.execute(
                "select * from turns where session_id = ? and status = 'queued' order by created_at, rowid limit 1",
                (session_id,),
            ).fetchone()
            if row is None:
                return None
            conn.execute(
                "update turns set status = 'running', attempts = attempts + 1, updated_at = ? where id = ?",
                (now, row["id"]),
            )
            conn.execute(
                "update sessions set status = 'running', active_turn_id = ?, last_turn_id = ?, "
                "updated_at = ?, last_observed_state = 'running queued turn via Claude Code' where id = ?",
                (row["id"], row["id"], now, session_id),
            )
        return self.get_turn(row["id"])

    def update_turn(self, turn_id: str, *, only_if_active: bool = False, **fields: object) -> Turn:
        """Update a turn, optionally rejecting writes after it becomes terminal.

        The status predicate runs in the same SQL statement as the write so a
        concurrent cancellation wins over late stream output or completion.
        Return the current row even when the conditional write is skipped.
        """
        allowed = {
            "status",
            "attempts",
            "last_error",
            "error_kind",
            "finished_at",
            "last_useful_message",
            "response_finished_at",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = iso_now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [turn_id]
        with self.connect() as conn:
            condition = " and status in ('running', 'waiting')" if only_if_active else ""
            conn.execute(f"update turns set {assignments} where id = ?{condition}", values)
        return self.get_turn(turn_id)

    def get_turn(self, turn_id: str) -> Turn:
        with self.connect() as conn:
            row = conn.execute("select * from turns where id = ?", (turn_id,)).fetchone()
        if row is None:
            raise KeyError(f"No turn with id {turn_id}")
        return row_to_turn(row)

    def append_turn_steer(self, turn_id: str, text: str) -> Turn:
        """Record a steering message delivered into a running turn."""
        now = iso_now()
        with self.connect() as conn:
            conn.execute("begin immediate")
            row = conn.execute("select steers_json from turns where id = ?", (turn_id,)).fetchone()
            if row is None:
                raise KeyError(f"No turn with id {turn_id}")
            steers = list(steers_from_json(row["steers_json"]))
            output_count = conn.execute("select count(*) from turn_output where turn_id = ?", (turn_id,)).fetchone()[0]
            steers.append({"text": text, "createdAt": now, "outputCount": output_count})
            conn.execute(
                "update turns set steers_json = ?, updated_at = ? where id = ?",
                (json.dumps(steers), now, turn_id),
            )
        return self.get_turn(turn_id)

    def list_turns(self, session_id: str, limit: int = 20) -> list[Turn]:
        with self.connect() as conn:
            rows = conn.execute(
                # rowid breaks same-millisecond ties so the newest insert is
                # the newest turn (and listings agree with this ordering).
                "select * from turns where session_id = ? order by created_at desc, rowid desc limit ?",
                (session_id, limit),
            ).fetchall()
        return [row_to_turn(row) for row in rows]

    def latest_turns_by_session(self) -> dict[str, Turn]:
        """Newest turn for every session in one query.

        Session listings need each session's latest turn (model, reasoning
        effort); one query beats one ``list_turns`` round-trip per session.
        """
        with self.connect() as conn:
            rows = conn.execute(
                """
                select t.* from turns t
                join (
                    select session_id, max(created_at) as created_at
                    from turns group by session_id
                ) latest
                on latest.session_id = t.session_id and latest.created_at = t.created_at
                order by t.rowid desc
                """
            ).fetchall()
        latest: dict[str, Turn] = {}
        for row in rows:
            turn = row_to_turn(row)
            # Ties on created_at (same millisecond): the newest insert wins,
            # matching list_turns' ordering.
            latest.setdefault(turn.session_id, turn)
        return latest

    def queued_turns(self, session_id: str | None = None) -> list[Turn]:
        params: list[object] = []
        query = "select * from turns where status = 'queued'"
        if session_id:
            query += " and session_id = ?"
            params.append(session_id)
        query += " order by created_at asc"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [row_to_turn(row) for row in rows]

    def tail_log(self, session: Session, lines: int = 80) -> list[str]:
        if not session.log_path:
            return []
        path = Path(session.log_path)
        if not path.exists():
            return []
        return tail_file(path, lines)


def app_dir() -> Path:
    configured = os.environ.get(APP_DIR_ENV)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "super-agents-claude-code"


def database_path() -> Path:
    return app_dir() / "state.sqlite3"


def logs_dir() -> Path:
    return app_dir() / "logs"


def default_cwd() -> str:
    return os.getcwd()


def iso_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def command_to_json(command: list[str]) -> str:
    return json.dumps(command)


def command_from_json(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(part) for part in raw] if isinstance(raw, list) else []


def preview(text: str | None, limit: int = 180) -> str | None:
    if text is None:
        return None
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "..."


def conversation_title(text: str) -> str | None:
    """A compact display title. Titles need not be unique (the session name is
    the unique lookup key), so they never carry the session id."""
    compact = " ".join(user_prompt_for_title(text).split())
    if not compact:
        return None
    if len(compact) > 80:
        compact = compact[:77].rstrip() + "..."
    return compact


def unique_session_name(conn: sqlite3.Connection, base_name: str, exclude_id: str | None = None) -> str:
    """``base_name``, or ``base_name (2)``, ``(3)``… when another session holds it."""
    candidate = base_name
    suffix = 2
    while True:
        row = conn.execute("select id from sessions where name = ?", (candidate,)).fetchone()
        if row is None or row["id"] == exclude_id:
            return candidate
        suffix_text = f" ({suffix})"
        candidate = f"{base_name[: 80 - len(suffix_text)]}{suffix_text}"
        suffix += 1


def _strip_session_id_suffixes(conn: sqlite3.Connection) -> None:
    """One-time cleanup of the session-id suffix older versions stored in
    display titles (``Hi (1a2b3c4d)``) and in imported or retired session
    names (``project (1a2b3c4d)``, ``dispatcher (retired 1a2b3c4d)``)."""
    conn.execute(
        """
        update sessions set title = substr(title, 1, length(title) - 11)
        where length(title) > 11 and substr(title, -11) = ' (' || substr(id, -8) || ')'
        """
    )
    rows = conn.execute(
        """
        select id, name from sessions
        where substr(name, -11) = ' (' || substr(id, -8) || ')'
           or substr(name, -19) = ' (retired ' || substr(id, -8) || ')'
        """
    ).fetchall()
    for row in rows:
        name, suffix = row["name"], row["id"][-8:]
        if name.endswith(f" (retired {suffix})"):
            base = name[: -len(f" (retired {suffix})")] + " (retired)"
        else:
            base = name[: -len(f" ({suffix})")] or "Thread"
        conn.execute(
            "update sessions set name = ? where id = ?",
            (unique_session_name(conn, base, exclude_id=row["id"]), row["id"]),
        )


def row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        name=row["name"],
        title=row["title"],
        auto_title=bool(row["auto_title"]),
        agent_name=row["agent_name"],
        developer_instructions=row["developer_instructions"],
        cwd=row["cwd"],
        command=command_from_json(row["command_json"]),
        model=row["model"],
        status=row["status"],
        pid=row["pid"],
        active_turn_id=row["active_turn_id"],
        last_turn_id=row["last_turn_id"],
        last_observed_state=row["last_observed_state"],
        last_useful_message=row["last_useful_message"],
        backend_session_id=row["backend_session_id"],
        backend=row["backend"],
        last_client_instance=row["last_client_instance"],
        last_exit_code=row["last_exit_code"],
        log_path=row["log_path"],
        raw_log_path=row["raw_log_path"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def row_to_turn(row: sqlite3.Row) -> Turn:
    return Turn(
        id=row["id"],
        session_id=row["session_id"],
        prompt=row["prompt"],
        mode=row["mode"],
        model=row["model"],
        reasoning_effort=row["reasoning_effort"],
        service_tier=row["service_tier"],
        status=row["status"],
        attempts=row["attempts"],
        last_error=row["last_error"],
        error_kind=row["error_kind"],
        last_useful_message=row["last_useful_message"],
        response_finished_at=row["response_finished_at"],
        steers=steers_from_json(row["steers_json"]),
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        finished_at=row["finished_at"],
    )


def steers_from_json(value: object) -> tuple[JsonObject, ...]:
    if not isinstance(value, str) or not value:
        return ()
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return ()
    if not isinstance(raw, list):
        return ()
    return tuple(item for item in raw if isinstance(item, dict))


def tail_file(path: Path, lines: int) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read().splitlines()[-lines:]


def sessions_to_json(sessions: Iterable[Session]) -> list[dict[str, object]]:
    return [session.to_json() for session in sessions]
