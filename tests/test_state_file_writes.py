"""The state file is rewritten only when its content changes.

Every update path rewrites the whole file, including no-op updates, and a
rewrite that only bumps the mtime misleads anything keyed on it: a container
image upgrade copies ``~/.super-agents`` onto the data volume and the Cloud
refuses the redeploy when the live store looks newer than the copy, which
happened on staging on 2026-10-09 with byte-identical rewrites.
"""

from __future__ import annotations

from pathlib import Path

from super_agents.state import (
    SessionRecord,
    StateFile,
    read_state_file,
    update_state_file,
    update_state_file_when,
    write_state_file,
)


def _identity(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_ino, stat.st_mtime_ns


def test_identical_content_leaves_the_file_untouched(tmp_path: Path) -> None:
    path = tmp_path / ".super-agents" / "state.json"
    state = StateFile()
    write_state_file(path, state)
    before = _identity(path)

    write_state_file(path, read_state_file(path))
    update_state_file(path, lambda _state: None)
    update_state_file_when(path, lambda _state: True)

    assert _identity(path) == before


def test_changed_content_is_written_atomically(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    write_state_file(path, StateFile())
    before = _identity(path)

    def add_session(state: StateFile) -> None:
        state.sessions["t1"] = SessionRecord(thread_id="t1", updated_at="2026-10-09T00:00:00.000Z", label="dispatcher")

    update_state_file(path, add_session)

    assert _identity(path) != before
    assert read_state_file(path).sessions["t1"].label == "dispatcher"
    assert not list(tmp_path.glob("tmp*"))


def test_first_write_creates_the_file(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "state.json"
    write_state_file(path, StateFile())
    assert read_state_file(path).to_json() == StateFile().to_json()
