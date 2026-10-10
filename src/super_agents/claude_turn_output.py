from __future__ import annotations

import sqlite3
import uuid
from typing import Any


def initialize_turn_output(connection: sqlite3.Connection) -> None:
    connection.execute(
        "create table if not exists turn_output (id text primary key, turn_id text not null, text text not null)"
    )
    connection.execute("create index if not exists turn_output_turn on turn_output(turn_id)")


def append_turn_output(store: Any, turn_id: str, text: str) -> str | None:
    item_id = uuid.uuid4().hex
    with store.connect() as connection:
        inserted = connection.execute(
            "insert into turn_output (id, turn_id, text) "
            "select ?, id, ? from turns where id = ? and status in ('running', 'waiting')",
            (item_id, text, turn_id),
        ).rowcount
    return item_id if inserted else None


def read_turn_output(store: Any, turn_id: str) -> list[dict[str, str]]:
    with store.connect() as connection:
        rows = connection.execute(
            "select id, text from turn_output where turn_id = ? order by rowid", (turn_id,)
        ).fetchall()
    return [{"type": "agentMessage", "id": row["id"], "text": row["text"]} for row in rows]
