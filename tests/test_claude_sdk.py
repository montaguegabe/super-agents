from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient
from super_agents.claude_tui.storage import Store


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeAssistantMessage:
    content: list[FakeTextBlock]


@dataclass
class FakeResultMessage:
    result: str


class FakeClaudeAgentOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeClaudeSDKClient:
    prompts: list[str] = []
    options_seen: list[FakeClaudeAgentOptions] = []
    env_seen: list[tuple[str, bool]] = []

    def __init__(self, options: FakeClaudeAgentOptions) -> None:
        self.options = options
        self.connected = False
        self.prompt = ""
        FakeClaudeSDKClient.options_seen.append(options)
        FakeClaudeSDKClient.env_seen.append(("init", "ANTHROPIC_API_KEY" in os.environ))

    async def connect(self) -> None:
        FakeClaudeSDKClient.env_seen.append(("connect", "ANTHROPIC_API_KEY" in os.environ))
        self.connected = True

    async def query(self, prompt: str) -> None:
        FakeClaudeSDKClient.env_seen.append(("query", "ANTHROPIC_API_KEY" in os.environ))
        self.prompt = prompt
        FakeClaudeSDKClient.prompts.append(prompt)

    async def receive_response(self):
        FakeClaudeSDKClient.env_seen.append(("receive", "ANTHROPIC_API_KEY" in os.environ))
        yield FakeAssistantMessage([FakeTextBlock(f"reply:{self.prompt}")])
        yield FakeResultMessage(f"done:{self.prompt}")

    async def set_model(self, model: str | None = None) -> None:
        self.options.kwargs["model"] = model

    async def interrupt(self) -> None:
        self.interrupted = True


class FakeSdk:
    ClaudeAgentOptions = FakeClaudeAgentOptions
    ClaudeSDKClient = FakeClaudeSDKClient


def fake_sdk_loader() -> FakeSdk:
    return FakeSdk()


@pytest.mark.asyncio
async def test_claude_sdk_client_runs_turn_through_agent_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_TUI_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "must-not-reach-sdk")
    FakeClaudeSDKClient.prompts = []
    FakeClaudeSDKClient.options_seen = []
    FakeClaudeSDKClient.env_seen = []
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    started = await client.start_thread({"name": "sdk", "cwd": str(tmp_path), "model": "sonnet"})
    result = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "hello"})

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")
    session = store.get_session(started["threadId"])
    log = "\n".join(store.tail_log(session, lines=20))

    assert result["backend"] == "claude_code"
    assert FakeClaudeSDKClient.prompts == ["hello"]
    assert FakeClaudeSDKClient.options_seen[-1].kwargs == {"cwd": str(tmp_path), "model": "sonnet"}
    assert FakeClaudeSDKClient.env_seen
    assert all(seen is False for _stage, seen in FakeClaudeSDKClient.env_seen)
    assert os.environ["ANTHROPIC_API_KEY"] == "must-not-reach-sdk"
    assert store.get_turn(result["turnId"]).status == "completed"
    assert session.status == "waiting"
    assert "reply:hello" in log
    assert "done:hello" in log


@pytest.mark.asyncio
async def test_claude_sdk_client_passes_reasoning_effort_to_agent_sdk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_TUI_HOME", str(tmp_path))
    FakeClaudeSDKClient.prompts = []
    FakeClaudeSDKClient.options_seen = []
    FakeClaudeSDKClient.env_seen = []
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    await client.start_thread({"name": "sdk", "cwd": str(tmp_path), "model": "sonnet"})
    result = await client.start_turn_by_label(
        LabelQueryInput(label="sdk"),
        {"prompt": "hello", "reasoningEffort": "xhigh"},
    )

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")

    assert FakeClaudeSDKClient.options_seen[-1].kwargs == {
        "cwd": str(tmp_path),
        "model": "sonnet",
        "effort": "xhigh",
    }
    assert store.get_turn(result["turnId"]).reasoning_effort == "xhigh"


@pytest.mark.asyncio
async def test_claude_sdk_status_reports_missing_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_TUI_HOME", str(tmp_path))
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: (_ for _ in ()).throw(ModuleNotFoundError("no sdk")))

    status = await client.status()

    assert status["ready"] is False
    assert status["backend"] == "claude_code"
    assert "claude-agent-sdk" in status["sdkError"]


@pytest.mark.asyncio
async def test_claude_sdk_waiting_session_accepts_follow_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_TUI_HOME", str(tmp_path))
    FakeClaudeSDKClient.prompts = []
    FakeClaudeSDKClient.options_seen = []
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    first = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "first"})
    await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")
    second = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "second"})

    await wait_for(lambda: store.get_turn(second["turnId"]).status == "completed")

    assert second["queued"] is False
    assert FakeClaudeSDKClient.prompts == ["first", "second"]


@pytest.mark.asyncio
async def test_claude_sdk_busy_session_start_steers_instead_of_queueing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_TUI_HOME", str(tmp_path))
    FakeClaudeSDKClient.prompts = []
    FakeClaudeSDKClient.options_seen = []
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    first = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "first"})
    second = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "second"})

    await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")
    await wait_for(lambda: store.get_turn(second["turnId"]).status == "completed")

    assert second["queued"] is False
    assert second["steered"] is True
    assert second["drain"] == "steered_active_turn"
    assert FakeClaudeSDKClient.prompts == ["first", "second"]
    assert store.queued_turns(second["threadId"]) == []


@pytest.mark.asyncio
async def test_claude_sdk_queued_turn_preserves_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_TUI_HOME", str(tmp_path))
    FakeClaudeSDKClient.prompts = []
    FakeClaudeSDKClient.options_seen = []
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path), "model": "sonnet"})

    first = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "first"})
    queued = await client.queue_turn_by_label(
        LabelQueryInput(label="sdk"),
        {"prompt": "second", "reasoningEffort": "low"},
    )

    await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")
    await wait_for(lambda: store.get_turn(queued["turnId"]).status == "completed")

    assert queued["queued"] is True
    assert store.get_turn(queued["turnId"]).reasoning_effort == "low"
    assert FakeClaudeSDKClient.options_seen[-1].kwargs == {
        "cwd": str(tmp_path),
        "model": "sonnet",
        "effort": "low",
    }


@pytest.mark.asyncio
async def test_claude_sdk_queue_on_idle_session_starts_immediately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_TUI_HOME", str(tmp_path))
    FakeClaudeSDKClient.prompts = []
    FakeClaudeSDKClient.options_seen = []
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    result = await client.queue_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "queued while idle"})

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")

    assert result["queued"] is False
    assert result["startedImmediately"] is True
    assert FakeClaudeSDKClient.prompts == ["queued while idle"]
    assert store.queued_turns(result["threadId"]) == []


async def wait_for(predicate, timeout: float = 2.0) -> None:  # type: ignore[no-untyped-def]
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("timed out waiting for predicate")
