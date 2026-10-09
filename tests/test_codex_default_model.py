from __future__ import annotations

import json

import pytest

from super_agents.app_server_client import DEFAULT_MODEL, CodexAppServerClient


@pytest.fixture
def model_config(monkeypatch, tmp_path):
    config_path = tmp_path / "dispatcher-config.json"
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("SUPER_AGENTS_CALLER_MODEL", "opus")
    monkeypatch.setenv("AGENT_MODEL", "opus")
    monkeypatch.delenv("SUPER_AGENTS_CODEX_PROFILE_PATH", raising=False)
    monkeypatch.delenv("SUPER_AGENTS_OPENBASE_CLOUD_CODEX_PROFILE_PATH", raising=False)
    return config_path


@pytest.mark.parametrize("global_backend", ["openbase_cloud", "claude_code", "codex"])
@pytest.mark.parametrize("client_backend", ["codex", "openbase_cloud_codex"])
def test_codex_client_uses_its_own_backend_model(monkeypatch, tmp_path, model_config, global_backend, client_backend):
    model_config.write_text(
        json.dumps(
            {
                "backend_models": {
                    "claude_code": {"super_agents": "opus"},
                    "codex": {"super_agents": "gpt-5.5"},
                    "openbase_cloud_codex": {"super_agents": "gpt-5.4"},
                }
            }
        )
    )
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", global_backend)

    client = CodexAppServerClient(
        endpoint="ws://unused",
        state_file=tmp_path / "state.json",
        backend_identity=client_backend,
    )

    assert client.default_model == ("gpt-5.5" if client_backend == "codex" else "gpt-5.4")


@pytest.mark.parametrize("configured_codex_model", [None, "opus"])
def test_codex_client_falls_back_to_codex_model(monkeypatch, tmp_path, model_config, configured_codex_model):
    model_config.write_text(
        json.dumps(
            {
                "backend_models": {
                    "claude_code": {"super_agents": "opus"},
                    "codex": {"super_agents": configured_codex_model},
                }
            }
        )
    )
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "openbase_cloud")

    client = CodexAppServerClient(endpoint="ws://unused", state_file=tmp_path / "state.json")

    assert client.default_model == DEFAULT_MODEL


def test_codex_client_preserves_explicit_model(monkeypatch, tmp_path, model_config):
    model_config.write_text(json.dumps({"backend_models": {"claude_code": {"super_agents": "opus"}}}))
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "openbase_cloud")

    client = CodexAppServerClient(
        endpoint="ws://unused",
        state_file=tmp_path / "state.json",
        default_model="gpt-5.5",
    )

    assert client.default_model == "gpt-5.5"
