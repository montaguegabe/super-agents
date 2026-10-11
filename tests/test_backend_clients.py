from __future__ import annotations

from pathlib import Path

import pytest

from super_agents import backend_config
from super_agents.backend_clients import (
    CLAUDE_CODE_BACKEND,
    CODEX_BACKEND,
    OPENBASE_CLOUD_BACKEND,
    OPENBASE_CLOUD_CODEX_BACKEND,
    backend_from_environment,
    client_for_backend,
    client_from_environment,
    configured_backend_from_environment,
    default_backend_from_environment,
    normalize_backend,
)
from super_agents.claude_sdk import ClaudeAgentSdkClient
from super_agents.app_server_client import CodexAppServerClient


def test_backend_normalization_supports_three_canonical_modes() -> None:
    assert normalize_backend("codex") == CODEX_BACKEND
    assert normalize_backend("openbase-cloud") == OPENBASE_CLOUD_BACKEND
    assert normalize_backend("claude-code") == CLAUDE_CODE_BACKEND
    assert normalize_backend("claude_code") == CLAUDE_CODE_BACKEND
    assert normalize_backend("openbase-cloud-codex") == OPENBASE_CLOUD_CODEX_BACKEND


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
    assert backend_from_environment() == "claude_code"
    assert isinstance(client_from_environment(), ClaudeAgentSdkClient)


def test_internal_openbase_cloud_codex_still_uses_codex(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("OPENBASE_CODING_BACKEND=openbase-cloud-codex\n", encoding="utf-8")
    monkeypatch.delenv("OPENBASE_CODING_BACKEND", raising=False)
    monkeypatch.setattr(backend_config, "DEFAULT_ENV_FILE", env_file)

    assert configured_backend_from_environment() == "openbase_cloud_codex"
    assert backend_from_environment() == "codex"


def test_default_backend_override_preserves_configured_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_BACKEND", "openbase_cloud")

    assert default_backend_from_environment() == "openbase_cloud"


def test_explicit_client_factories_preserve_cloud_identities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_CODE_HOME", str(tmp_path / "claude"))

    cloud = client_for_backend("openbase_cloud")
    cloud_codex = client_for_backend("openbase_cloud_codex")

    assert isinstance(cloud, ClaudeAgentSdkClient)
    assert cloud.backend == "openbase_cloud"
    assert cloud._permission_gate.backend == "openbase_cloud"
    assert isinstance(cloud_codex, CodexAppServerClient)
    assert cloud_codex.backend == "openbase_cloud_codex"
