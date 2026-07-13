from __future__ import annotations

from pathlib import Path

import pytest

from super_agents import backend_config
from super_agents.backend_clients import (
    CLAUDE_CODE_BACKEND,
    CODEX_BACKEND,
    OPENBASE_CLOUD_BACKEND,
    backend_from_environment,
    client_from_environment,
    configured_backend_from_environment,
    default_backend_from_environment,
    normalize_backend,
)
from super_agents.claude_sdk import ClaudeAgentSdkClient
from super_agents.multi_backend import MultiBackendClient


def test_backend_normalization_supports_three_canonical_modes() -> None:
    assert normalize_backend("codex") == CODEX_BACKEND
    assert normalize_backend("openbase-cloud") == OPENBASE_CLOUD_BACKEND
    assert normalize_backend("claude-code") == CLAUDE_CODE_BACKEND
    assert normalize_backend("claude_code") == CLAUDE_CODE_BACKEND


def test_client_factory_uses_claude_agent_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude-code")

    assert backend_from_environment() == "claude_code"
    assert isinstance(client_from_environment(), ClaudeAgentSdkClient)


def test_client_factory_reads_backend_from_env_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENBASE_CODING_BACKEND=openbase-cloud\n", encoding="utf-8")
    monkeypatch.delenv("OPENBASE_CODING_BACKEND", raising=False)
    monkeypatch.delenv("OPENBASE_CODEX_BACKEND", raising=False)
    monkeypatch.setattr(backend_config, "DEFAULT_ENV_FILE", env_file)

    assert configured_backend_from_environment() == "openbase_cloud"
    assert backend_from_environment() == "codex"


def test_default_backend_override_wins_over_configured_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_BACKEND", "claude-code")

    assert default_backend_from_environment() == CLAUDE_CODE_BACKEND


def test_default_backend_falls_back_to_configured_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")
    monkeypatch.delenv("SUPER_AGENTS_DEFAULT_BACKEND", raising=False)

    assert default_backend_from_environment() == CLAUDE_CODE_BACKEND


def test_multi_backend_client_default_follows_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_BACKEND", "claude_code")

    client = MultiBackendClient(clients={"codex": object(), "claude_code": object()})

    assert client.backend == CLAUDE_CODE_BACKEND


def test_multi_backend_client_constructor_default_wins_over_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_BACKEND", "claude_code")

    client = MultiBackendClient(default_backend="codex", clients={"codex": object(), "claude_code": object()})

    assert client.backend == CODEX_BACKEND


def test_multi_backend_client_collapses_openbase_cloud_override_to_codex(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_BACKEND", "openbase-cloud")

    client = MultiBackendClient(clients={"codex": object(), "claude_code": object()})

    assert client.backend == CODEX_BACKEND
