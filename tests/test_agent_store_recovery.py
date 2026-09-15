from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from super_agents.agent_store import Store


def test_corrupt_database_is_preserved_and_replaced(tmp_path, caplog) -> None:
    database = tmp_path / "state.sqlite3"
    corrupt_contents = b"not a sqlite database"
    database.write_bytes(corrupt_contents)
    wal = tmp_path / "state.sqlite3-wal"
    shm = tmp_path / "state.sqlite3-shm"
    wal.write_bytes(b"wal evidence")
    shm.write_bytes(b"shm evidence")

    with caplog.at_level(logging.ERROR):
        store = Store(database)

    assert store.list_sessions() == []
    with sqlite3.connect(database) as conn:
        assert conn.execute("pragma quick_check").fetchone() == ("ok",)

    quarantined = list(tmp_path.glob("state.sqlite3.corrupt-*"))
    quarantined_main = [path for path in quarantined if not path.name.endswith(("-wal", "-shm"))]
    assert len(quarantined_main) == 1
    assert quarantined_main[0].read_bytes() == corrupt_contents
    assert Path(f"{quarantined_main[0]}-wal").read_bytes() == b"wal evidence"
    assert Path(f"{quarantined_main[0]}-shm").exists()
    assert "initialized a fresh store" in caplog.text


def test_non_corruption_database_error_is_not_swallowed(tmp_path, monkeypatch) -> None:
    database = tmp_path / "state.sqlite3"

    def fail_schema(_self) -> None:
        raise sqlite3.DatabaseError("unrelated database failure")

    monkeypatch.setattr(Store, "_init_schema", fail_schema)

    try:
        Store(database)
    except sqlite3.DatabaseError as exc:
        assert str(exc) == "unrelated database failure"
    else:
        raise AssertionError("expected unrelated database failure to propagate")
