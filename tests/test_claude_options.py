from __future__ import annotations

from types import SimpleNamespace

from super_agents.claude_options import (
    CLAUDE_EXTRA_ARGS_ENV,
    OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN_ENV,
    OPENBASE_CLOUD_ANTHROPIC_BASE_URL_ENV,
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
