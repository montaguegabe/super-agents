from __future__ import annotations

import json

import pytest

from super_agents.config_profiles import codex_profile_config, merge_config
from super_agents.claude_options import managed_claude_config_options


def test_profile_is_opt_in_and_backend_specific(tmp_path, monkeypatch):
    monkeypatch.delenv("SUPER_AGENTS_CODEX_PROFILE_PATH", raising=False)
    monkeypatch.delenv("SUPER_AGENTS_OPENBASE_CLOUD_CODEX_PROFILE_PATH", raising=False)
    assert codex_profile_config() == {}
    direct = tmp_path / "openbase.config.toml"
    cloud = tmp_path / "cloud.config.toml"
    direct.write_text('model = "direct-model"\nmodel_reasoning_effort = "high"\n')
    cloud.write_text('model_provider = "test-proxy"\n')
    monkeypatch.setenv("SUPER_AGENTS_CODEX_PROFILE_PATH", str(direct))
    monkeypatch.setenv("SUPER_AGENTS_OPENBASE_CLOUD_CODEX_PROFILE_PATH", str(cloud))
    assert codex_profile_config()["model"] == "direct-model"
    assert codex_profile_config("openbase_cloud_codex") == {"model_provider": "test-proxy"}
    direct.unlink()
    with pytest.raises(FileNotFoundError):
        codex_profile_config()


def test_profile_merge_preserves_nested_shell_values():
    profile = {"model": "profile-model", "shell_environment_policy": {"set": {"PROFILE_VALUE": "one"}}}
    merged = merge_config(profile, {"shell_environment_policy": {"set": {"AGENT_SESSION_ID": "thread-id"}}})
    assert merged["shell_environment_policy"]["set"] == {"PROFILE_VALUE": "one", "AGENT_SESSION_ID": "thread-id"}
    assert profile["shell_environment_policy"]["set"] == {"PROFILE_VALUE": "one"}


def test_profile_model_is_fallback_below_explicit_role(tmp_path, monkeypatch):
    from super_agents.defaults import default_super_agents_model

    profile = tmp_path / "profile.toml"
    profile.write_text('model = "profile-model"\n')
    roles = tmp_path / "roles.json"
    monkeypatch.setenv("SUPER_AGENTS_CODEX_PROFILE_PATH", str(profile))
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(roles))
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    monkeypatch.delenv("SUPER_AGENTS_CALLER_MODEL", raising=False)
    assert default_super_agents_model(backend="codex") == "profile-model"
    roles.write_text(json.dumps({"backend_models": {"codex": {"super_agents": "role-model"}}}))
    assert default_super_agents_model(backend="codex") == "role-model"


def test_claude_settings_and_mcp_are_session_scoped(tmp_path, monkeypatch):
    config_dir = tmp_path / "claude"
    config_dir.mkdir()
    state = config_dir / ".claude.json"
    state.write_text(json.dumps({"mcpServers": {"personal": {"command": "personal-mcp"}}}))
    settings = tmp_path / "profile-settings.json"
    settings.write_text('{"model":"haiku"}')
    mcp = tmp_path / "profile-mcp.json"
    mcp.write_text('{"mcpServers":{"managed":{"command":"managed-mcp"}}}')
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_SETTINGS_PATH", str(settings))
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_MCP_CONFIG_PATH", str(mcp))
    before = state.read_bytes()
    options = managed_claude_config_options()
    assert options["settings"] == str(settings)
    assert set(options["mcp_servers"]) == {"personal", "managed"}
    assert state.read_bytes() == before
    settings.write_text("invalid JSON")
    with pytest.raises(ValueError):
        managed_claude_config_options()
