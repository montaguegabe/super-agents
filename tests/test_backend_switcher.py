from __future__ import annotations

from pathlib import Path

import pytest

from super_agents.backend import current_backend, main as backend_main, set_backend


def test_backend_switch_updates_env_file_for_claude_sdk(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("KEEP_ME=1\nOPENBASE_CODEX_BACKEND=codex\n", encoding="utf-8")

    status = set_backend("claude-code", env_file)

    content = env_file.read_text(encoding="utf-8")
    assert status.backend == "claude-agent-sdk"
    assert "KEEP_ME=1" in content
    assert "OPENBASE_CODING_BACKEND=claude-agent-sdk" in content
    assert "OPENBASE_CODEX_BACKEND=codex" in content
    assert "CODEX_CLAUDE_" not in content

    status = set_backend("codex", env_file)

    assert status.backend == "codex"
    assert "OPENBASE_CODING_BACKEND=codex" in env_file.read_text(encoding="utf-8")


def test_backend_cli_status_and_use(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env_file = tmp_path / ".env"

    assert backend_main(["--env-file", str(env_file), "use", "claude-code"]) == 0
    assert current_backend(env_file).backend == "claude-agent-sdk"

    assert backend_main(["--env-file", str(env_file), "status"]) == 0
    output = capsys.readouterr().out
    assert "Backend set to claude-agent-sdk" in output
    assert "Backend: claude-agent-sdk" in output


def test_backend_switch_supports_claude_tui(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"

    status = set_backend("claude-tui", env_file)

    assert status.backend == "claude-tui"
    assert "OPENBASE_CODING_BACKEND=claude-tui" in env_file.read_text(encoding="utf-8")


def test_backend_switch_reads_legacy_env_key(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENBASE_CODEX_BACKEND=claude-agent-sdk\n", encoding="utf-8")

    assert current_backend(env_file).backend == "claude-agent-sdk"
