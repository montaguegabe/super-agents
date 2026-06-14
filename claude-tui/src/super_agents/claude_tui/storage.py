from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Iterable

from .config import build_claude_command, default_cwd
from .models import Session, Turn, command_from_json, command_to_json
from .paths import database_path, logs_dir
from .timeutil import iso_now


class Store:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or database_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        logs_dir().mkdir(parents=True, exist_ok=True)
        self._init()

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self.connect() as conn:
            conn.executescript(
                """
                create table if not exists sessions (
                    id text primary key,
                    name text not null unique,
                    agent_name text,
                    cwd text not null,
                    command_json text not null,
                    model text,
                    status text not null,
                    pid integer,
                    active_turn_id text,
                    last_turn_id text,
                    last_observed_state text,
                    last_useful_message text,
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
                    status text not null,
                    attempts integer not null default 0,
                    last_error text,
                    created_at text not null,
                    updated_at text not null,
                    finished_at text
                );
                create index if not exists turns_session_idx on turns(session_id, created_at);
                """
            )
            columns = {
                row["name"]
                for row in conn.execute("pragma table_info(turns)").fetchall()
            }
            if "reasoning_effort" not in columns:
                conn.execute("alter table turns add column reasoning_effort text")

    def create_session(
        self,
        name: str,
        cwd: str | None = None,
        *,
        agent_name: str | None = None,
        model: str | None = None,
        command: list[str] | None = None,
    ) -> Session:
        now = iso_now()
        session_id = f"s_{uuid.uuid4().hex}"
        logs = logs_dir()
        log_path = str(logs / f"{session_id}.log")
        raw_log_path = str(logs / f"{session_id}.raw.log")
        resolved_cwd = str(Path(cwd or default_cwd()).expanduser())
        resolved_command = command or build_claude_command(model)
        with self.connect() as conn:
            conn.execute(
                """
                insert into sessions (
                    id, name, agent_name, cwd, command_json, model, status,
                    log_path, raw_log_path, created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    name,
                    agent_name,
                    resolved_cwd,
                    command_to_json(resolved_command),
                    model,
                    "unknown",
                    log_path,
                    raw_log_path,
                    now,
                    now,
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
            row = conn.execute("select * from sessions where id = ?", (session_id,)).fetchone()
        if row is None:
            raise KeyError(f"No session with id {session_id}")
        return row_to_session(row)

    def get_by_name(self, name: str) -> Session | None:
        with self.connect() as conn:
            row = conn.execute("select * from sessions where name = ?", (name,)).fetchone()
        return row_to_session(row) if row else None

    def require_by_name(self, name: str) -> Session:
        session = self.get_by_name(name)
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
        if clauses:
            query += " where " + " and ".join(clauses)
        query += " order by updated_at desc"
        with self.connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [row_to_session(row) for row in rows]

    def update_session(self, session_id: str, **fields: object) -> Session:
        allowed = {
            "status",
            "pid",
            "active_turn_id",
            "last_turn_id",
            "last_observed_state",
            "last_useful_message",
            "last_exit_code",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = iso_now()
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
    ) -> Turn:
        now = iso_now()
        turn_id = f"t_{uuid.uuid4().hex}"
        with self.connect() as conn:
            conn.execute(
                """
                insert into turns (
                    id, session_id, prompt, mode, model, reasoning_effort, status, created_at, updated_at
                ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    turn_id,
                    session_id,
                    prompt,
                    mode,
                    model,
                    reasoning_effort,
                    status,
                    now,
                    now,
                ),
            )
            conn.execute(
                "update sessions set last_turn_id = ?, active_turn_id = ?, updated_at = ? where id = ?",
                (turn_id, turn_id if status in {"running", "waiting"} else None, now, session_id),
            )
        return self.get_turn(turn_id)

    def update_turn(self, turn_id: str, **fields: object) -> Turn:
        allowed = {"status", "attempts", "last_error", "finished_at"}
        updates = {key: value for key, value in fields.items() if key in allowed}
        updates["updated_at"] = iso_now()
        assignments = ", ".join(f"{key} = ?" for key in updates)
        values = list(updates.values()) + [turn_id]
        with self.connect() as conn:
            conn.execute(f"update turns set {assignments} where id = ?", values)
        return self.get_turn(turn_id)

    def get_turn(self, turn_id: str) -> Turn:
        with self.connect() as conn:
            row = conn.execute("select * from turns where id = ?", (turn_id,)).fetchone()
        if row is None:
            raise KeyError(f"No turn with id {turn_id}")
        return row_to_turn(row)

    def list_turns(self, session_id: str, limit: int = 20) -> list[Turn]:
        with self.connect() as conn:
            rows = conn.execute(
                "select * from turns where session_id = ? order by created_at desc limit ?",
                (session_id, limit),
            ).fetchall()
        return [row_to_turn(row) for row in rows]

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


def row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        id=row["id"],
        name=row["name"],
        agent_name=row["agent_name"],
        cwd=row["cwd"],
        command=command_from_json(row["command_json"]),
        model=row["model"],
        status=row["status"],
        pid=row["pid"],
        active_turn_id=row["active_turn_id"],
        last_turn_id=row["last_turn_id"],
        last_observed_state=row["last_observed_state"],
        last_useful_message=row["last_useful_message"],
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
        status=row["status"],
        attempts=row["attempts"],
        last_error=row["last_error"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        finished_at=row["finished_at"],
    )


def tail_file(path: Path, lines: int) -> list[str]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return handle.read().splitlines()[-lines:]


def sessions_to_json(sessions: Iterable[Session]) -> list[dict[str, object]]:
    return [session.to_json() for session in sessions]
