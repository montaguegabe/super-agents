from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from super_agents.app_models import LabelQueryInput
from super_agents.agent_store import Store
from super_agents.claude_sdk import ClaudeAgentSdkClient


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeAssistantMessage:
    content: list[FakeTextBlock]


@dataclass
class FakeResultMessage:
    result: str
    session_id: str = "b01bd0f7-f1b0-485e-a47c-d831645174b9"


class FakeClaudeAgentOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeClaudeSDKClient:
    prompts: list[str] = []
    options_seen: list[FakeClaudeAgentOptions] = []
    env_seen: list[tuple[str, bool]] = []
    blocked_prompts: set[str] = set()
    interrupted_prompts: list[str] = []

    def __init__(self, options: FakeClaudeAgentOptions) -> None:
        self.options = options
        self.connected = False
        self.prompt = ""
        self.interrupted = False
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
        while _user_prompt(self.prompt) in FakeClaudeSDKClient.blocked_prompts and not self.interrupted:
            await asyncio.sleep(0.01)
        yield FakeAssistantMessage([FakeTextBlock(f"reply:{self.prompt}")])
        yield FakeResultMessage(f"done:{self.prompt}")

    async def set_model(self, model: str | None = None) -> None:
        self.options.kwargs["model"] = model

    async def interrupt(self) -> None:
        self.interrupted = True
        FakeClaudeSDKClient.interrupted_prompts.append(_user_prompt(self.prompt))


class FakeSdk:
    ClaudeAgentOptions = FakeClaudeAgentOptions
    ClaudeSDKClient = FakeClaudeSDKClient


def fake_sdk_loader() -> FakeSdk:
    return FakeSdk()


def _user_prompt(prompt: str) -> str:
    return prompt.rsplit("\n\n", 1)[-1]


def assert_claude_prompt(prompt: str, user_prompt: str, cwd: Path, developer_instructions: str | None = None) -> None:
    assert prompt.endswith(f"\n\n{user_prompt}")
    assert f"Current working directory: {cwd}" in prompt
    if developer_instructions:
        assert developer_instructions in prompt


@pytest.fixture(autouse=True)
def reset_fake_claude_sdk() -> None:
    FakeClaudeSDKClient.prompts = []
    FakeClaudeSDKClient.options_seen = []
    FakeClaudeSDKClient.env_seen = []
    FakeClaudeSDKClient.blocked_prompts = set()
    FakeClaudeSDKClient.interrupted_prompts = []


@pytest.mark.asyncio
async def test_claude_sdk_client_runs_turn_through_agent_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    assert len(FakeClaudeSDKClient.prompts) == 1
    assert_claude_prompt(FakeClaudeSDKClient.prompts[0], "hello", tmp_path)
    assert FakeClaudeSDKClient.options_seen[-1].kwargs == {
        "cwd": str(tmp_path),
        "model": "sonnet",
        "permission_mode": "bypassPermissions",
    }
    assert FakeClaudeSDKClient.env_seen
    assert all(seen is False for _stage, seen in FakeClaudeSDKClient.env_seen)
    assert os.environ["ANTHROPIC_API_KEY"] == "must-not-reach-sdk"
    assert store.get_turn(result["turnId"]).status == "completed"
    assert store.get_turn(result["turnId"]).last_useful_message.endswith("\n\nhello")
    assert store.get_session(started["threadId"]).backend_session_id == "b01bd0f7-f1b0-485e-a47c-d831645174b9"
    assert session.status == "waiting"
    assert "reply:" in log
    assert "done:" in log

    readback = await client.read_by_label(LabelQueryInput(label="sdk"), include_turns=True)
    assert readback["turns"][0]["lastUsefulMessage"].endswith("\n\nhello")


@pytest.mark.asyncio
async def test_claude_sdk_client_passes_reasoning_effort_to_agent_sdk(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
        "permission_mode": "bypassPermissions",
        "effort": "xhigh",
    }
    assert store.get_turn(result["turnId"]).reasoning_effort == "xhigh"


@pytest.mark.asyncio
async def test_claude_sdk_persists_thread_developer_instructions(tmp_path: Path) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    started = await client.start_thread(
        {
            "name": "dispatcher",
            "cwd": str(tmp_path),
            "agentName": "Grace",
            "developerInstructions": "The random fruit is persimmon.",
        }
    )
    result = await client.start_turn_by_label(LabelQueryInput(label="dispatcher"), {"prompt": "what fruit?"})

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")

    session = store.get_session(started["threadId"])
    assert session.developer_instructions == (
        "The random fruit is persimmon.\n\nSuper Agent thread name: dispatcher\nYour name is Grace."
    )
    assert_claude_prompt(
        FakeClaudeSDKClient.prompts[-1],
        "what fruit?",
        tmp_path,
        "The random fruit is persimmon.",
    )
    assert "Super Agent thread name: dispatcher" in FakeClaudeSDKClient.prompts[-1]
    assert f"Super Agent thread id: {started['threadId']}" in FakeClaudeSDKClient.prompts[-1]
    assert "Your name is Grace." in FakeClaudeSDKClient.prompts[-1]


@pytest.mark.asyncio
async def test_claude_sdk_resume_by_label_adds_voice_instructions_to_session(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    started = await client.start_thread(
        {
            "name": "dispatcher",
            "cwd": str(tmp_path),
            "agentName": "Grace",
            "developerInstructions": "Super agent instructions.",
        }
    )

    await client.resume_by_label(
        LabelQueryInput(thread_id=started["threadId"], cwd=str(tmp_path)),
        developer_instructions="Direct voice instructions.",
    )
    await client.resume_by_label(
        LabelQueryInput(thread_id=started["threadId"], cwd=str(tmp_path)),
        developer_instructions="Direct voice instructions.",
    )
    result = await client.start_turn_by_label(
        LabelQueryInput(thread_id=started["threadId"], cwd=str(tmp_path)),
        {"prompt": "what instructions?"},
    )

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")

    session = store.get_session(started["threadId"])
    assert session.developer_instructions == (
        "Super agent instructions.\n\nDirect voice instructions.\n\n"
        "Super Agent thread name: dispatcher\n"
        f"Super Agent thread id: {started['threadId']}\n"
        "Your name is Grace."
    )
    assert "Super agent instructions." in FakeClaudeSDKClient.prompts[-1]
    assert "Direct voice instructions." in FakeClaudeSDKClient.prompts[-1]
    assert session.developer_instructions.count("Direct voice instructions.") == 1


@pytest.mark.asyncio
async def test_claude_sdk_turn_developer_instructions_compose_with_thread_instructions(tmp_path: Path) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    await client.start_thread(
        {
            "name": "dispatcher",
            "cwd": str(tmp_path),
            "agentName": "Grace",
            "developerInstructions": "The random fruit is persimmon.",
        }
    )
    result = await client.start_turn_by_label(
        LabelQueryInput(label="dispatcher"),
        {"prompt": "what fruit?", "developerInstructions": "The random fruit is quince."},
    )

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")

    assert_claude_prompt(
        FakeClaudeSDKClient.prompts[-1],
        "what fruit?",
        tmp_path,
        "The random fruit is quince.",
    )
    assert "The random fruit is persimmon." in FakeClaudeSDKClient.prompts[-1]
    assert "Super Agent thread name: dispatcher" in FakeClaudeSDKClient.prompts[-1]
    assert "Super Agent thread id:" in FakeClaudeSDKClient.prompts[-1]
    assert "Your name is Grace." in FakeClaudeSDKClient.prompts[-1]


@pytest.mark.asyncio
async def test_claude_sdk_start_thread_refreshes_existing_named_session(tmp_path: Path) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    first = await client.start_thread(
        {
            "name": "dispatcher",
            "cwd": str(tmp_path / "old"),
            "agentName": "Grace",
            "developerInstructions": "old fruit",
        }
    )
    refreshed = await client.start_thread(
        {
            "name": "dispatcher",
            "cwd": str(tmp_path),
            "agentName": "Dottie",
            "developerInstructions": "new fruit",
        }
    )

    assert refreshed["threadId"] == first["threadId"]
    session = store.get_session(refreshed["threadId"])
    assert session.cwd == str(tmp_path)
    assert session.agent_name == "Dottie"
    assert session.developer_instructions == (
        "new fruit\n\nSuper Agent thread name: dispatcher\nYour name is Dottie."
    )
    assert session.status == "waiting"


@pytest.mark.asyncio
async def test_claude_sdk_start_thread_refresh_preserves_existing_agent_name(tmp_path: Path) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    first = await client.start_thread(
        {
            "name": "report-agent",
            "cwd": str(tmp_path),
            "agentName": "Grace",
            "developerInstructions": "base instructions",
        }
    )
    refreshed = await client.start_thread(
        {
            "name": "report-agent",
            "cwd": str(tmp_path),
        }
    )

    assert refreshed["threadId"] == first["threadId"]
    session = store.get_session(refreshed["threadId"])
    assert session.agent_name == "Grace"
    assert session.developer_instructions == (
        "base instructions\n\nSuper Agent thread name: report-agent\nYour name is Grace."
    )


@pytest.mark.asyncio
async def test_claude_sdk_uses_managed_claude_config_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeClaudeSDKClient.prompts = []
    FakeClaudeSDKClient.options_seen = []
    config_dir = tmp_path / "claude_config"
    config_dir.mkdir()
    settings_path = config_dir / "settings.json"
    config_path = config_dir / ".claude.json"
    instructions_path = config_dir / "CLAUDE.md"
    settings_path.write_text('{"model": "sonnet"}\n', encoding="utf-8")
    mcp_servers = {
        "super-agents": {
            "type": "stdio",
            "command": "super-agents-mcp",
        },
        "other-server": {
            "type": "stdio",
            "command": "other-mcp",
        },
    }
    config_path.write_text(json.dumps({"mcpServers": mcp_servers}) + "\n", encoding="utf-8")
    instructions_path.write_text("Openbase instructions\n", encoding="utf-8")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(config_dir))
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})
    result = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "hello"})

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")

    assert FakeClaudeSDKClient.options_seen[-1].kwargs == {
        "cwd": str(tmp_path),
        "permission_mode": "bypassPermissions",
        "env": {"CLAUDE_CONFIG_DIR": str(config_dir)},
        "settings": str(settings_path),
        "setting_sources": ["project"],
        "system_prompt": {"type": "file", "path": str(instructions_path)},
        "mcp_servers": mcp_servers,
    }


@pytest.mark.asyncio
async def test_claude_sdk_status_reports_missing_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: (_ for _ in ()).throw(ModuleNotFoundError("no sdk")))

    status = await client.status()

    assert status["ready"] is False
    assert status["backend"] == "claude_code"
    assert "claude-agent-sdk" in status["sdkError"]


@pytest.mark.asyncio
async def test_claude_sdk_waiting_session_accepts_follow_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
    assert [_user_prompt(prompt) for prompt in FakeClaudeSDKClient.prompts] == ["first", "second"]


@pytest.mark.asyncio
async def test_claude_sdk_follow_up_resumes_native_claude_session_after_sdk_client_loss(tmp_path: Path) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    first = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "build a calculator"})
    await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")
    store.update_turn(first["turnId"], last_useful_message="Done. I wrote index.html.")
    client._sdk_clients = {}
    client._sdk_client_efforts = {}

    second = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "what did you just work on?"})

    await wait_for(lambda: store.get_turn(second["turnId"]).status == "completed")

    prompt = FakeClaudeSDKClient.prompts[-1]
    assert prompt.endswith("\n\nwhat did you just work on?")
    assert "Previous turns in this Openbase thread:" not in prompt
    assert "build a calculator" not in prompt
    assert FakeClaudeSDKClient.options_seen[-1].kwargs["resume"] == "b01bd0f7-f1b0-485e-a47c-d831645174b9"


@pytest.mark.asyncio
async def test_claude_sdk_read_backfills_latest_turn_message_from_session(tmp_path: Path) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    started = await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})
    turn = store.create_turn(
        started["threadId"],
        "older turn without per-turn output",
        status="completed",
    )
    store.update_session(
        started["threadId"],
        status="waiting",
        last_turn_id=turn.id,
        last_useful_message="Recovered latest Claude Code response.",
    )

    readback = await client.read_by_label(LabelQueryInput(label="sdk"), include_turns=True)

    assert readback["turns"][0]["lastUsefulMessage"] == "Recovered latest Claude Code response."


@pytest.mark.asyncio
async def test_claude_sdk_busy_session_start_steers_instead_of_queueing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeClaudeSDKClient.prompts = []
    FakeClaudeSDKClient.options_seen = []
    FakeClaudeSDKClient.blocked_prompts = {"first"}
    FakeClaudeSDKClient.interrupted_prompts = []
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    first = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "first"})
    await wait_for(lambda: [_user_prompt(prompt) for prompt in FakeClaudeSDKClient.prompts] == ["first"])
    second = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "second"})

    await wait_for(lambda: store.get_turn(first["turnId"]).status == "cancelled")
    await wait_for(lambda: store.get_turn(second["turnId"]).status == "completed")

    assert second["queued"] is False
    assert second["steered"] is True
    assert second["drain"] == "steered_active_turn"
    assert second["interruptedTurnId"] == first["turnId"]
    assert FakeClaudeSDKClient.interrupted_prompts == ["first"]
    assert [_user_prompt(prompt) for prompt in FakeClaudeSDKClient.prompts] == ["first", "second"]
    assert store.queued_turns(second["threadId"]) == []


@pytest.mark.asyncio
async def test_claude_sdk_steer_by_label_preserves_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeClaudeSDKClient.prompts = []
    FakeClaudeSDKClient.options_seen = []
    FakeClaudeSDKClient.blocked_prompts = {"first"}
    FakeClaudeSDKClient.interrupted_prompts = []
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path), "model": "sonnet"})

    first = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "first"})
    await wait_for(lambda: [_user_prompt(prompt) for prompt in FakeClaudeSDKClient.prompts] == ["first"])
    steered = await client.steer_by_label(
        LabelQueryInput(label="sdk"),
        "second",
        {"reasoningEffort": "low"},
    )

    await wait_for(lambda: store.get_turn(first["turnId"]).status == "cancelled")
    await wait_for(lambda: store.get_turn(steered["turnId"]).status == "completed")

    assert steered["steered"] is True
    assert steered["interruptedTurnId"] == first["turnId"]
    assert FakeClaudeSDKClient.interrupted_prompts == ["first"]
    assert store.get_turn(steered["turnId"]).reasoning_effort == "low"
    assert FakeClaudeSDKClient.options_seen[-1].kwargs == {
        "cwd": str(tmp_path),
        "model": "sonnet",
        "permission_mode": "bypassPermissions",
        "effort": "low",
        "resume": "b01bd0f7-f1b0-485e-a47c-d831645174b9",
    }


@pytest.mark.asyncio
async def test_claude_sdk_queued_turn_preserves_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
        "permission_mode": "bypassPermissions",
        "effort": "low",
        "resume": "b01bd0f7-f1b0-485e-a47c-d831645174b9",
    }


@pytest.mark.asyncio
async def test_claude_sdk_queue_on_idle_session_starts_immediately(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeClaudeSDKClient.prompts = []
    FakeClaudeSDKClient.options_seen = []
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    result = await client.queue_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "queued while idle"})

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")

    assert result["queued"] is False
    assert result["startedImmediately"] is True
    assert [_user_prompt(prompt) for prompt in FakeClaudeSDKClient.prompts] == ["queued while idle"]
    assert store.queued_turns(result["threadId"]) == []


async def wait_for(predicate, timeout: float = 2.0) -> None:  # type: ignore[no-untyped-def]
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("timed out waiting for predicate")
