from __future__ import annotations

from types import SimpleNamespace

from super_agents.claude_options import (
    CLAUDE_EXTRA_ARGS_ENV,
    agent_options,
    claude_extra_args,
)


class _FakeOptions:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


_FAKE_SDK = SimpleNamespace(ClaudeAgentOptions=_FakeOptions)


def test_extra_args_absent_by_default(monkeypatch) -> None:
    monkeypatch.delenv(CLAUDE_EXTRA_ARGS_ENV, raising=False)

    options = agent_options(_FAKE_SDK, "/tmp", None, None, resume=None)

    assert "extra_args" not in options.kwargs
    assert claude_extra_args() is None


def test_extra_args_passed_through(monkeypatch) -> None:
    monkeypatch.setenv(CLAUDE_EXTRA_ARGS_ENV, '{"chrome": null, "max-turns": 5}')

    options = agent_options(_FAKE_SDK, "/tmp", None, None, resume=None)

    assert options.kwargs["extra_args"] == {"chrome": None, "max-turns": "5"}


def test_extra_args_ignores_invalid_payloads(monkeypatch) -> None:
    for raw in ("not json", '"a string"', "[]", "{}", "   "):
        monkeypatch.setenv(CLAUDE_EXTRA_ARGS_ENV, raw)
        assert claude_extra_args() is None
