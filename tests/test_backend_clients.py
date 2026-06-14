from __future__ import annotations

import os
from pathlib import Path

import pytest

from super_agents.app_models import LabelQueryInput
from super_agents.backend_clients import (
    CLAUDE_AGENT_SDK_BACKEND,
    CLAUDE_TUI_BACKEND,
    CODEX_BACKEND,
    backend_from_environment,
    client_from_environment,
    normalize_backend,
)
from super_agents.claude_sdk import ClaudeAgentSdkClient
from super_agents.claude_tui.client import ClaudeTuiClient
from super_agents.claude_tui.storage import Store


FAKE_CLAUDE = """\
import sys
print("Fake Claude ready >", flush=True)
for line in sys.stdin:
    text = line.strip()
    print("echo:" + text, flush=True)
    print(">", flush=True)
"""


def test_backend_normalization_supports_three_canonical_modes() -> None:
    assert normalize_backend("codex") == CODEX_BACKEND
    assert normalize_backend("claude-code") == CLAUDE_AGENT_SDK_BACKEND
    assert normalize_backend("claude-agent-sdk") == CLAUDE_AGENT_SDK_BACKEND
    assert normalize_backend("claude-tui") == CLAUDE_TUI_BACKEND


def test_client_factory_uses_claude_agent_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude-code")
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_TUI_HOME", str(tmp_path))

    assert backend_from_environment() == "claude-agent-sdk"
    assert isinstance(client_from_environment(), ClaudeAgentSdkClient)


def test_client_factory_uses_claude_tui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude-tui")
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_TUI_HOME", str(tmp_path))

    assert backend_from_environment() == "claude-tui"
    assert isinstance(client_from_environment(), ClaudeTuiClient)


def test_client_factory_reads_legacy_backend_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("OPENBASE_CODING_BACKEND", raising=False)
    monkeypatch.setenv("OPENBASE_CODEX_BACKEND", "claude-code")
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_TUI_HOME", str(tmp_path))

    assert backend_from_environment() == "claude-agent-sdk"
    assert isinstance(client_from_environment(), ClaudeAgentSdkClient)


@pytest.mark.asyncio
async def test_claude_tui_client_start_turn_read_and_status(tmp_path: Path) -> None:
    previous_home = os.environ.get("SUPER_AGENTS_CLAUDE_TUI_HOME")
    os.environ["SUPER_AGENTS_CLAUDE_TUI_HOME"] = str(tmp_path)
    try:
        script = tmp_path / "fake_claude.py"
        script.write_text(FAKE_CLAUDE, encoding="utf-8")
        store = Store(tmp_path / "state.sqlite3")
        session = store.create_session("fake", cwd=str(tmp_path), command=["python3", str(script)])
        client = ClaudeTuiClient(store=store)

        result = await client.start_turn_by_label(LabelQueryInput(label="fake"), {"prompt": "hello"})
        status = await client.compact_status(LabelQueryInput(include_inactive=True))
        read = await client.read_by_label(LabelQueryInput(label="fake"), include_turns=True)

        assert result["backend"] == "claude-tui"
        assert result["queued"] is False
        assert status["agents"][0]["threadId"] == session.id
        assert read["turns"][0]["promptPreview"] == "hello"
    finally:
        if previous_home is None:
            os.environ.pop("SUPER_AGENTS_CLAUDE_TUI_HOME", None)
        else:
            os.environ["SUPER_AGENTS_CLAUDE_TUI_HOME"] = previous_home
