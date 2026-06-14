from __future__ import annotations

from pathlib import Path

import pytest

from super_agents import backend_clients
from super_agents.backend_clients import (
    CLAUDE_CODE_BACKEND,
    CODEX_BACKEND,
    OPENBASE_CLOUD_BACKEND,
    backend_from_environment,
    configured_backend_from_environment,
    client_from_environment,
    normalize_backend,
)
from super_agents.claude_sdk import ClaudeAgentSdkClient


def test_backend_normalization_supports_three_canonical_modes() -> None:
    assert normalize_backend("codex") == CODEX_BACKEND
    assert normalize_backend("openbase-cloud") == OPENBASE_CLOUD_BACKEND
    assert normalize_backend("claude-code") == CLAUDE_CODE_BACKEND
    assert normalize_backend("claude-agent-sdk") == CLAUDE_CODE_BACKEND
    assert normalize_backend("claude-tui") == CLAUDE_CODE_BACKEND


def test_client_factory_uses_claude_agent_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude-code")

    assert backend_from_environment() == "claude_code"
    assert isinstance(client_from_environment(), ClaudeAgentSdkClient)


def test_client_factory_maps_legacy_claude_tui_to_claude_code(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude-tui")

    assert backend_from_environment() == "claude_code"
    assert isinstance(client_from_environment(), ClaudeAgentSdkClient)


def test_client_factory_reads_legacy_backend_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENBASE_CODING_BACKEND", raising=False)
    monkeypatch.setenv("OPENBASE_CODEX_BACKEND", "claude-code")

    assert backend_from_environment() == "claude_code"
    assert isinstance(client_from_environment(), ClaudeAgentSdkClient)


def test_client_factory_reads_backend_from_env_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENBASE_CODING_BACKEND=openbase-cloud\n", encoding="utf-8")
    monkeypatch.delenv("OPENBASE_CODING_BACKEND", raising=False)
    monkeypatch.delenv("OPENBASE_CODEX_BACKEND", raising=False)
    monkeypatch.setattr(backend_clients, "DEFAULT_ENV_FILE", env_file)

    assert configured_backend_from_environment() == "openbase_cloud"
    assert backend_from_environment() == "codex"
