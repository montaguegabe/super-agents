"""A thread started without a cwd lands in the runtime's configured default.

An Openbase cloud workspace keeps only /data across an image upgrade, and
Super Agents started with no project landed in $HOME (Codex) or the hosting
process's directory (Claude), both on the image layer; work created there was
lost on the next upgrade (production, 2026-10-16). The container entrypoint
now names its durable projects dir in SUPER_AGENTS_DEFAULT_CWD; installs
without it keep each backend's own fallback.
"""

from __future__ import annotations

import os
from pathlib import Path

from super_agents.agent_store import DEFAULT_CWD_ENV, Store, default_cwd


def test_configured_default_wins_over_the_backend_fallback(monkeypatch, tmp_path) -> None:
    projects = tmp_path / "workspace"
    projects.mkdir()
    monkeypatch.setenv(DEFAULT_CWD_ENV, str(projects))
    assert default_cwd(str(Path.home())) == str(projects)
    assert default_cwd() == str(projects)


def test_unset_keeps_the_backend_fallback_and_the_process_cwd(monkeypatch) -> None:
    monkeypatch.delenv(DEFAULT_CWD_ENV, raising=False)
    assert default_cwd(str(Path.home())) == str(Path.home())
    assert default_cwd() == os.getcwd()


def test_a_missing_configured_directory_is_ignored(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(DEFAULT_CWD_ENV, str(tmp_path / "gone"))
    assert default_cwd("/fallback") == "/fallback"
    monkeypatch.setenv(DEFAULT_CWD_ENV, "   ")
    assert default_cwd("/fallback") == "/fallback"


def test_claude_sessions_without_a_cwd_use_the_configured_default(monkeypatch, tmp_path) -> None:
    projects = tmp_path / "workspace"
    projects.mkdir()
    monkeypatch.setenv(DEFAULT_CWD_ENV, str(projects))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir()
    monkeypatch.delenv("SUPER_AGENTS_CLAUDE_CODE_HOME", raising=False)
    session = Store().create_session("no-project", cwd=None)
    assert session.cwd == str(projects)
