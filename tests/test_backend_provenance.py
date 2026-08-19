from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from super_agents.agent_store import Store
from super_agents.backend_provenance import BackendProvenanceStore


def test_provenance_persists_configured_identity_and_rejects_reassignment(
    tmp_path: Path,
) -> None:
    path = tmp_path / "provenance.json"
    store = BackendProvenanceStore(path)
    store.remember(
        "openbase_cloud",
        thread_ids={"thread-1"},
        turn_ids={"turn-1"},
        request_ids={"request-1"},
    )

    restarted = BackendProvenanceStore(path)

    assert restarted.backend_for_thread("thread-1") == "openbase_cloud"
    assert restarted.backend_for_turn("turn-1") == "openbase_cloud"
    assert restarted.backend_for_request("request-1") == "openbase_cloud"
    assert restarted.backends_for_request("request-1") == {"openbase_cloud"}
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(ValueError, match="refusing to reassign"):
        restarted.remember("claude_code", thread_ids={"thread-1"})


def test_provenance_reads_legacy_file_without_backend_maps(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({"version": 0}), encoding="utf-8")

    state = BackendProvenanceStore(path).read()

    assert state.threads == {}
    assert state.turns == {}
    assert state.requests == {}


def test_legacy_single_request_owner_migrates_and_collisions_are_preserved(tmp_path: Path) -> None:
    path = tmp_path / "provenance.json"
    path.write_text(json.dumps({"version": 1, "requests": {"request-1": "codex"}}), encoding="utf-8")
    store = BackendProvenanceStore(path)

    store.remember("claude_code", request_ids={"request-1"})

    assert store.backend_for_request("request-1") is None
    assert store.backends_for_request("request-1") == {"codex", "claude_code"}


def test_claude_store_claims_legacy_rows_for_configured_backend(tmp_path: Path) -> None:
    path = tmp_path / "claude.sqlite3"
    legacy = Store(path)
    session = legacy.create_session("legacy")

    cloud = Store(path, backend="openbase_cloud")

    claimed = cloud.get_session(session.id)
    assert claimed.backend == "openbase_cloud"
    assert cloud.get_by_name("legacy") == claimed
