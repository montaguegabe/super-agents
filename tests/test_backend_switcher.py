from __future__ import annotations

from pathlib import Path

import pytest

from super_agents.backend import current_backend, main as backend_main, set_backend


def test_backend_switch_updates_env_file_for_claude_sdk(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("KEEP_ME=1\nOPENBASE_CODEX_BACKEND=codex\n", encoding="utf-8")

    status = set_backend("claude-code", env_file)

    content = env_file.read_text(encoding="utf-8")
    assert status.backend == "claude_code"
    assert "KEEP_ME=1" in content
    assert "OPENBASE_CODING_BACKEND=claude_code" in content
    assert "OPENBASE_CODEX_BACKEND=codex" in content
    assert "CODEX_CLAUDE_" not in content

    status = set_backend("codex", env_file)

    assert status.backend == "codex"
    assert "OPENBASE_CODING_BACKEND=codex" in env_file.read_text(encoding="utf-8")


def test_backend_cli_status_and_use(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env_file = tmp_path / ".env"

    assert backend_main(["--env-file", str(env_file), "use", "claude-code"]) == 0
    assert current_backend(env_file).backend == "claude_code"

    assert backend_main(["--env-file", str(env_file), "status"]) == 0
    output = capsys.readouterr().out
    assert "Backend set to claude_code" in output
    assert "Backend: claude_code" in output


def test_backend_switch_supports_openbase_cloud(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"

    status = set_backend("openbase-cloud", env_file)

    assert status.backend == "openbase_cloud"
    assert "OPENBASE_CODING_BACKEND=openbase_cloud" in env_file.read_text(encoding="utf-8")
