from __future__ import annotations

import subprocess
from types import SimpleNamespace

import pytest

from super_agents.claude_options import (
    CLAUDE_EXTRA_ARGS_ENV,
    OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN_ENV,
    OPENBASE_CLOUD_ANTHROPIC_BASE_URL_ENV,
    _openbase_cloud_anthropic_auth_token,
    agent_options,
    claude_extra_args,
)


class _FakeOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


_FAKE_SDK = SimpleNamespace(ClaudeAgentOptions=_FakeOptions)


def test_extra_args_absent_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OPENBASE_CODING_BACKEND", raising=False)
    monkeypatch.delenv(CLAUDE_EXTRA_ARGS_ENV, raising=False)

    options = agent_options(_FAKE_SDK, "/tmp", None, None, resume=None)

    assert "extra_args" not in options.kwargs
    assert claude_extra_args() is None


def test_extra_args_passed_through(monkeypatch) -> None:
    monkeypatch.delenv("OPENBASE_CODING_BACKEND", raising=False)
    monkeypatch.setenv(CLAUDE_EXTRA_ARGS_ENV, '{"chrome": null, "max-turns": 5}')

    options = agent_options(_FAKE_SDK, "/tmp", None, None, resume=None)

    assert options.kwargs["extra_args"] == {"chrome": None, "max-turns": "5"}


def test_extra_args_ignores_invalid_payloads(monkeypatch) -> None:
    monkeypatch.delenv("OPENBASE_CODING_BACKEND", raising=False)
    for raw in ("not json", '"a string"', "[]", "{}", "   "):
        monkeypatch.setenv(CLAUDE_EXTRA_ARGS_ENV, raw)
        assert claude_extra_args() is None


def test_openbase_cloud_backend_sets_anthropic_proxy_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "openbase_cloud")
    monkeypatch.setenv("OPENBASE_CODER_CLI_WEB_BACKEND_URL", "http://localhost:8000")
    monkeypatch.setenv(OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN_ENV, "machine-token")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv(OPENBASE_CLOUD_ANTHROPIC_BASE_URL_ENV, raising=False)

    options = agent_options(_FAKE_SDK, "/tmp", None, None, resume=None)

    assert options.kwargs["env"] == {
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_BASE_URL": "http://localhost:8000/api/openbase/llm/anthropic",
        "ANTHROPIC_AUTH_TOKEN": "machine-token",
    }


def test_explicit_cloud_identity_overrides_process_backend(monkeypatch) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")
    monkeypatch.setenv("OPENBASE_CODER_CLI_WEB_BACKEND_URL", "http://localhost:8000")
    monkeypatch.setenv(OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN_ENV, "machine-token")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    options = agent_options(
        _FAKE_SDK,
        "/tmp",
        None,
        None,
        resume=None,
        backend="openbase_cloud",
    )

    assert options.kwargs["model"] == "claude-haiku-4-5-20251001"
    assert options.kwargs["env"]["ANTHROPIC_AUTH_TOKEN"] == "machine-token"


def test_openbase_cloud_backend_pins_claude_aliases(monkeypatch) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "openbase_cloud")
    monkeypatch.setenv(OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN_ENV, "machine-token")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    options = agent_options(_FAKE_SDK, "/tmp", "fable", None, resume=None)

    assert options.kwargs["model"] == "claude-fable-5-1"


def test_openbase_cloud_backend_defaults_unset_model_to_haiku(monkeypatch) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "openbase_cloud")
    monkeypatch.setenv(OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN_ENV, "machine-token")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    options = agent_options(_FAKE_SDK, "/tmp", None, None, resume=None)

    assert options.kwargs["model"] == "claude-haiku-4-5-20251001"


def test_local_claude_backend_leaves_unset_model_to_sdk(monkeypatch) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    options = agent_options(_FAKE_SDK, "/tmp", None, None, resume=None)

    assert "model" not in options.kwargs


def test_openbase_cloud_backend_passes_public_models_through(monkeypatch) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "openbase_cloud")
    monkeypatch.setenv(OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN_ENV, "machine-token")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    options = agent_options(_FAKE_SDK, "/tmp", "openbase-claude", None, resume=None)

    assert options.kwargs["model"] == "openbase-claude"


def test_local_claude_backend_keeps_aliases(monkeypatch) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    options = agent_options(_FAKE_SDK, "/tmp", "fable", None, resume=None)

    assert options.kwargs["model"] == "fable"


def test_openbase_cloud_anthropic_base_url_strips_v1(monkeypatch) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "openbase_cloud")
    monkeypatch.setenv(
        OPENBASE_CLOUD_ANTHROPIC_BASE_URL_ENV,
        "https://example.test/api/openbase/llm/anthropic/v1",
    )
    monkeypatch.setenv(OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN_ENV, "machine-token")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)

    options = agent_options(_FAKE_SDK, "/tmp", None, None, resume=None)

    assert options.kwargs["env"]["ANTHROPIC_BASE_URL"] == "https://example.test/api/openbase/llm/anthropic"


def test_base_instructions_env_appends_to_preset_system_prompt(monkeypatch, tmp_path) -> None:
    instructions = tmp_path / "AGENTS.md"
    instructions.write_text("Base instructions.\n", encoding="utf-8")
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("SUPER_AGENTS_BASE_INSTRUCTIONS_PATH", str(instructions))
    monkeypatch.setattr(
        "super_agents.claude_options.claude_state_path",
        lambda: tmp_path / "no-state.json",
    )

    options = agent_options(_FAKE_SDK, "/tmp", None, None, resume=None)

    assert options.kwargs["system_prompt"] == {
        "type": "preset",
        "preset": "claude_code",
        "append": "Base instructions.\n",
    }
    assert options.kwargs["setting_sources"] == ["user", "project"]
    assert "settings" not in options.kwargs


def test_missing_instructions_still_selects_preset_system_prompt(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("SUPER_AGENTS_BASE_INSTRUCTIONS_PATH", raising=False)
    monkeypatch.setattr(
        "super_agents.claude_options.claude_state_path",
        lambda: tmp_path / "no-state.json",
    )

    options = agent_options(_FAKE_SDK, "/tmp", None, None, resume=None)

    # The SDK's None default yields an *empty* system prompt; the bare preset
    # keeps the stock claude_code prompt when no instructions file exists.
    assert options.kwargs["system_prompt"] == {"type": "preset", "preset": "claude_code"}


def test_replace_mode_substitutes_system_prompt_file(monkeypatch, tmp_path) -> None:
    instructions = tmp_path / "AGENTS.md"
    instructions.write_text("Base instructions.\n", encoding="utf-8")
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("SUPER_AGENTS_BASE_INSTRUCTIONS_PATH", str(instructions))
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_SYSTEM_PROMPT_MODE", "replace")
    monkeypatch.setattr(
        "super_agents.claude_options.claude_state_path",
        lambda: tmp_path / "no-state.json",
    )

    options = agent_options(_FAKE_SDK, "/tmp", None, None, resume=None)

    assert options.kwargs["system_prompt"] == {
        "type": "file",
        "path": str(instructions),
    }


def test_invalid_system_prompt_mode_raises(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_SYSTEM_PROMPT_MODE", "prepend")
    monkeypatch.setattr(
        "super_agents.claude_options.claude_state_path",
        lambda: tmp_path / "no-state.json",
    )

    with pytest.raises(ValueError, match="SUPER_AGENTS_CLAUDE_SYSTEM_PROMPT_MODE"):
        agent_options(_FAKE_SDK, "/tmp", None, None, resume=None)


def test_cloud_family_aliases_pin_latest_but_explicit_old_ids_still_resolve():
    from super_agents.claude_options import openbase_cloud_claude_model

    for alias, model in {
        "haiku": "claude-haiku-4-5-20251001",
        "sonnet": "claude-sonnet-5",
        "opus": "claude-opus-5-5",
        "fable": "claude-fable-5-1",
    }.items():
        assert openbase_cloud_claude_model(alias, "openbase_cloud") == model
        assert openbase_cloud_claude_model(model, "openbase_cloud") == model
    for old in ("claude-fable-5", "claude-opus-4-8", "claude-haiku-4-5"):
        assert openbase_cloud_claude_model(old, "openbase_cloud") == old


def _token_runs(monkeypatch, outcomes):
    """Patch subprocess.run to play ``outcomes`` in order; returns the timeouts used."""
    monkeypatch.delenv(OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN_ENV, raising=False)
    timeouts: list[float] = []
    queue = list(outcomes)

    def fake_run(argv, **kwargs):
        timeouts.append(kwargs["timeout"])
        outcome = queue.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr("super_agents.claude_options.subprocess.run", fake_run)
    return timeouts


def test_machine_token_retries_a_slow_cli_with_more_time(monkeypatch) -> None:
    timeouts = _token_runs(
        monkeypatch,
        [
            subprocess.TimeoutExpired(["openbase-coder"], 30),
            SimpleNamespace(returncode=0, stdout="obmt_token\n"),
        ],
    )

    assert _openbase_cloud_anthropic_auth_token() == "obmt_token"
    assert timeouts == [30, 60]


def test_machine_token_timeout_does_not_blame_sign_in(monkeypatch) -> None:
    _token_runs(
        monkeypatch,
        [
            subprocess.TimeoutExpired(["openbase-coder"], 30),
            subprocess.TimeoutExpired(["openbase-coder"], 60),
        ],
    )

    with pytest.raises(RuntimeError) as exc_info:
        _openbase_cloud_anthropic_auth_token()

    message = str(exc_info.value)
    assert "Timed out" in message
    assert "login" not in message


@pytest.mark.parametrize(
    "outcome",
    [
        OSError("CLI unavailable"),
        SimpleNamespace(returncode=1, stdout=""),
        SimpleNamespace(returncode=0, stdout=" \n"),
    ],
)
@pytest.mark.parametrize("after_timeout", [False, True])
def test_machine_token_missing_login_keeps_the_login_hint(monkeypatch, outcome, after_timeout) -> None:
    outcomes = [subprocess.TimeoutExpired(["openbase-coder"], 30)] if after_timeout else []
    timeouts = _token_runs(monkeypatch, [*outcomes, outcome])

    with pytest.raises(RuntimeError, match="openbase-coder login"):
        _openbase_cloud_anthropic_auth_token()
    assert timeouts == ([30, 60] if after_timeout else [30])
