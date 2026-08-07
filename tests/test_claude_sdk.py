from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput, QueueCancelInput
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
    num_turns: int = 1


class FakeClaudeAgentOptions:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeClaudeSDKClient:
    prompts: list[str] = []
    options_seen: list[FakeClaudeAgentOptions] = []
    env_seen: list[tuple[str, bool]] = []
    blocked_prompts: set[str] = set()
    interrupted_prompts: list[str] = []
    disconnect_count: int = 0
    # Each new client yields this many no-op (num_turns == 0) results before
    # the real response, mimicking the CLI's resume handshake result.
    handshake_results: int = 0

    def __init__(self, options: FakeClaudeAgentOptions) -> None:
        self.options = options
        self.connected = False
        self.prompt = ""
        self.interrupted = False
        self.pending_handshake_results = FakeClaudeSDKClient.handshake_results
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
        if self.pending_handshake_results > 0:
            self.pending_handshake_results -= 1
            yield FakeResultMessage("", num_turns=0)
            return
        while _user_prompt(self.prompt) in FakeClaudeSDKClient.blocked_prompts and not self.interrupted:
            await asyncio.sleep(0.01)
        yield FakeAssistantMessage([FakeTextBlock(f"reply:{self.prompt}")])
        yield FakeResultMessage(f"done:{self.prompt}")

    async def set_model(self, model: str | None = None) -> None:
        self.options.kwargs["model"] = model

    async def interrupt(self) -> None:
        self.interrupted = True
        FakeClaudeSDKClient.interrupted_prompts.append(_user_prompt(self.prompt))

    async def disconnect(self) -> None:
        self.connected = False
        FakeClaudeSDKClient.disconnect_count += 1


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
def reset_fake_claude_sdk(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "SUPER_AGENTS_DEFAULT_CONFIG_PATH",
        str(tmp_path / "missing-dispatcher-config.json"),
    )
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")
    monkeypatch.delenv("CLAUDE_CONFIG_DIR", raising=False)
    monkeypatch.delenv("SUPER_AGENTS_CLAUDE_EXTRA_ARGS", raising=False)
    monkeypatch.delenv("OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.delenv("OPENBASE_CLOUD_ANTHROPIC_BASE_URL", raising=False)
    FakeClaudeSDKClient.prompts = []
    FakeClaudeSDKClient.options_seen = []
    FakeClaudeSDKClient.env_seen = []
    FakeClaudeSDKClient.blocked_prompts = set()
    FakeClaudeSDKClient.interrupted_prompts = []
    FakeClaudeSDKClient.disconnect_count = 0
    FakeClaudeSDKClient.handshake_results = 0


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
        "effort": "high",
        "env": {"ANTHROPIC_API_KEY": ""},
    }
    # The process environment is never mutated; the key is blanked for the
    # spawned CLI through the SDK env option instead.
    assert FakeClaudeSDKClient.env_seen
    assert all(seen is True for _stage, seen in FakeClaudeSDKClient.env_seen)
    assert FakeClaudeSDKClient.options_seen[-1].kwargs["env"] == {"ANTHROPIC_API_KEY": ""}
    assert os.environ["ANTHROPIC_API_KEY"] == "must-not-reach-sdk"
    assert store.get_turn(result["turnId"]).status == "completed"
    assert store.get_turn(result["turnId"]).last_useful_message.endswith("\n\nhello")
    assert store.get_session(started["threadId"]).backend_session_id == "b01bd0f7-f1b0-485e-a47c-d831645174b9"
    assert session.status == "completed"
    assert "reply:" in log
    assert "done:" in log

    readback = await client.read_by_label(LabelQueryInput(label="sdk"), include_turns=True)
    assert readback["turns"][0]["lastUsefulMessage"].endswith("\n\nhello")


@pytest.mark.asyncio
async def test_claude_sdk_uses_super_agents_model_and_reasoning_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "dispatcher-config.json"
    config_path.write_text(
        json.dumps(
            {
                "super_agents_reasoning_effort": "low",
                "backend_models": {"claude_code": {"super_agents": "sonnet"}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPER_AGENTS_DEFAULT_CONFIG_PATH", str(config_path))
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    started = await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})
    result = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "hello"})

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")

    assert store.get_session(started["threadId"]).model == "sonnet"
    assert FakeClaudeSDKClient.options_seen[-1].kwargs == {
        "cwd": str(tmp_path),
        "model": "sonnet",
        "permission_mode": "bypassPermissions",
        "effort": "low",
        "env": {"ANTHROPIC_API_KEY": ""},
    }
    assert store.get_turn(result["turnId"]).reasoning_effort == "low"


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
        "env": {"ANTHROPIC_API_KEY": ""},
    }
    assert store.get_turn(result["turnId"]).reasoning_effort == "xhigh"


@pytest.mark.asyncio
async def test_claude_sdk_maps_fast_service_tier_to_low_effort(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    await client.start_thread({"name": "sdk", "cwd": str(tmp_path), "model": "sonnet"})
    result = await client.start_turn_by_label(
        LabelQueryInput(label="sdk"),
        {"prompt": "hello", "reasoningEffort": "high", "serviceTier": "fast"},
    )

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")

    assert FakeClaudeSDKClient.options_seen[-1].kwargs == {
        "cwd": str(tmp_path),
        "model": "sonnet",
        "permission_mode": "bypassPermissions",
        "effort": "low",
        "env": {"ANTHROPIC_API_KEY": ""},
    }
    assert store.get_turn(result["turnId"]).reasoning_effort == "high"
    assert store.get_turn(result["turnId"]).service_tier == "fast"
    assert store.get_turn(result["turnId"]).to_json()["serviceTier"] == "fast"


@pytest.mark.asyncio
async def test_claude_sdk_maps_standard_service_tier_to_high_effort(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    await client.start_thread({"name": "sdk", "cwd": str(tmp_path), "model": "sonnet"})
    result = await client.start_turn_by_label(
        LabelQueryInput(label="sdk"),
        {"prompt": "hello", "serviceTier": "standard"},
    )

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")

    assert FakeClaudeSDKClient.options_seen[-1].kwargs == {
        "cwd": str(tmp_path),
        "model": "sonnet",
        "permission_mode": "bypassPermissions",
        "effort": "high",
        "env": {"ANTHROPIC_API_KEY": ""},
    }
    assert store.get_turn(result["turnId"]).service_tier == "standard"


@pytest.mark.asyncio
async def test_claude_sdk_explicit_non_high_effort_overrides_service_tier(
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    await client.start_thread({"name": "sdk", "cwd": str(tmp_path), "model": "sonnet"})
    result = await client.start_turn_by_label(
        LabelQueryInput(label="sdk"),
        {"prompt": "hello", "reasoningEffort": "xhigh", "serviceTier": "fast"},
    )

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")

    assert FakeClaudeSDKClient.options_seen[-1].kwargs == {
        "cwd": str(tmp_path),
        "model": "sonnet",
        "permission_mode": "bypassPermissions",
        "effort": "xhigh",
        "env": {"ANTHROPIC_API_KEY": ""},
    }


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
    assert session.developer_instructions == ("new fruit\n\nSuper Agent thread name: dispatcher\nYour name is Dottie.")
    assert session.status == "completed"


@pytest.mark.asyncio
async def test_claude_sdk_start_thread_fresh_retires_existing_named_session(tmp_path: Path) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)

    first = await client.start_thread({"name": "dispatcher", "cwd": str(tmp_path)})
    fresh = await client.start_thread({"name": "dispatcher", "cwd": str(tmp_path), "fresh": True})

    assert fresh["threadId"] != first["threadId"]
    assert store.get_by_name("dispatcher").id == fresh["threadId"]
    retired = store.get_session(first["threadId"])
    assert retired.name == f"dispatcher (retired {first['threadId'][-8:]})"


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
        "effort": "high",
        "env": {"CLAUDE_CONFIG_DIR": str(config_dir), "ANTHROPIC_API_KEY": ""},
        "settings": str(settings_path),
        "setting_sources": ["user", "project"],
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
async def test_claude_sdk_progress_scopes_to_requested_turn(tmp_path: Path) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    started = await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})
    older = store.create_turn(
        started["threadId"],
        "older cancelled fragment",
        status="cancelled",
    )
    target = store.create_turn(
        started["threadId"],
        "current follow-up",
        status="completed",
    )
    store.update_turn(target.id, last_useful_message="The scoped turn answer.")
    store.update_session(
        started["threadId"],
        status="waiting",
        last_turn_id=target.id,
        last_useful_message="The session preview answer.",
    )

    progress = await client.progress_by_label(
        LabelQueryInput(label="sdk", turn_id=target.id, max_items=5)
    )

    assert progress["turnId"] == target.id
    assert progress["status"] == "completed"
    assert progress["turn"]["lastUsefulMessage"] == "The scoped turn answer."
    assert progress["turns"] == [progress["turn"]]
    assert older.id not in json.dumps(progress)


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

    await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")

    assert second["queued"] is False
    assert second["steered"] is True
    assert second["nativeSteer"] is True
    assert second["drain"] == "steered_active_turn"
    assert second["turnId"] == first["turnId"]
    assert FakeClaudeSDKClient.interrupted_prompts == []
    assert [_user_prompt(prompt) for prompt in FakeClaudeSDKClient.prompts] == ["first", "second"]
    assert store.queued_turns(second["threadId"]) == []
    assert "second" in (store.get_turn(first["turnId"]).last_useful_message or "")


@pytest.mark.asyncio
async def test_claude_sdk_steer_by_label_uses_native_active_turn_steering(
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

    await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")

    assert steered["steered"] is True
    assert steered["nativeSteer"] is True
    assert steered["turnId"] == first["turnId"]
    assert FakeClaudeSDKClient.interrupted_prompts == []
    assert store.get_turn(steered["turnId"]).reasoning_effort == "high"
    assert FakeClaudeSDKClient.options_seen[-1].kwargs == {
        "cwd": str(tmp_path),
        "model": "sonnet",
        "permission_mode": "bypassPermissions",
        "effort": "high",
        "env": {"ANTHROPIC_API_KEY": ""},
    }


@pytest.mark.asyncio
async def test_claude_sdk_turn_skips_noop_handshake_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A resume-handshake result (num_turns == 0) must not complete the turn.

    Accepting it records an empty turn while the real answer buffers unread,
    after which every turn returns the previous prompt's answer.
    """
    FakeClaudeSDKClient.handshake_results = 1
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    result = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "hello"})

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")
    turn = store.get_turn(result["turnId"])
    assert turn.last_useful_message is not None
    assert "hello" in turn.last_useful_message


@pytest.mark.asyncio
async def test_claude_sdk_native_steer_does_not_reconnect_or_read_stale_stream(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Native steering keeps the active Claude Code stream and updates the
    in-flight turn instead of creating a replacement turn."""
    FakeClaudeSDKClient.blocked_prompts = {"first"}
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    first = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "first"})
    await wait_for(lambda: [_user_prompt(prompt) for prompt in FakeClaudeSDKClient.prompts] == ["first"])
    clients_before_steer = len(FakeClaudeSDKClient.options_seen)
    second = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "second"})

    await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")

    assert second["turnId"] == first["turnId"]
    assert FakeClaudeSDKClient.disconnect_count == 0
    assert len(FakeClaudeSDKClient.options_seen) == clients_before_steer
    turn = store.get_turn(first["turnId"])
    assert turn.last_useful_message is not None
    assert "second" in turn.last_useful_message


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
        "env": {"ANTHROPIC_API_KEY": ""},
        "resume": "b01bd0f7-f1b0-485e-a47c-d831645174b9",
    }


@pytest.mark.asyncio
async def test_claude_sdk_cancel_queued_turn_by_id_and_position(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    FakeClaudeSDKClient.blocked_prompts = {"first"}
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    first = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "first"})
    await wait_for(lambda: [_user_prompt(prompt) for prompt in FakeClaudeSDKClient.prompts] == ["first"])
    keep = await client.queue_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "keep"})
    remove = await client.queue_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "remove"})

    removed = await client.cancel_queued_turn(QueueCancelInput(queue_item_id=remove["turnId"]))

    assert removed["cancelled"] is True
    assert removed["queueItemId"] == remove["turnId"]
    assert removed["position"] == 2
    assert removed["queueDepth"] == 1
    assert [turn.id for turn in store.queued_turns(first["threadId"])] == [keep["turnId"]]

    removed_by_position = await client.cancel_queued_turn(QueueCancelInput(label="sdk", position=1))

    assert removed_by_position["queueItemId"] == keep["turnId"]
    assert removed_by_position["queueDepth"] == 0
    assert store.queued_turns(first["threadId"]) == []

    await client.cancel_by_label(LabelQueryInput(label="sdk"))
    await wait_for(lambda: store.get_turn(first["turnId"]).status == "cancelled")
    await client.close()


@pytest.mark.asyncio
async def test_claude_sdk_cancel_queued_turn_rejects_missing_and_started_items(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})
    active = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "running"})

    with pytest.raises(ValueError, match="No queued Super Agents turn found"):
        await client.cancel_queued_turn(QueueCancelInput(queue_item_id="t_missing"))
    with pytest.raises(ValueError, match="already started"):
        await client.cancel_queued_turn(QueueCancelInput(queue_item_id=active["turnId"]))

    await wait_for(lambda: store.get_turn(active["turnId"]).status == "completed")
    await client.close()


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


@pytest.mark.asyncio
async def test_claude_sdk_reuses_cached_client_for_consecutive_turns(tmp_path: Path) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    first = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "one"})
    await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")
    second = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "two"})
    await wait_for(lambda: store.get_turn(second["turnId"]).status == "completed")

    assert len(FakeClaudeSDKClient.options_seen) == 1
    assert FakeClaudeSDKClient.disconnect_count == 0


@pytest.mark.asyncio
async def test_claude_sdk_close_disconnects_cached_clients(tmp_path: Path) -> None:
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})
    result = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "hello"})
    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")

    await client.close()

    assert FakeClaudeSDKClient.disconnect_count == 1
    assert client._sdk_clients == {}


@pytest.mark.asyncio
async def test_claude_sdk_reconnects_when_another_instance_ran_last_turn(tmp_path: Path) -> None:
    """Two client instances sharing a store must not fork the conversation.

    Reusing a cached CLI whose in-memory conversation predates another
    instance's turns resumes from a stale leaf and orphans those turns —
    the dispatcher "I never started Thandi" amnesia. The second instance's
    write must force the first to disconnect and re-resume the session.
    """
    store = Store(tmp_path / "state.sqlite3")
    first_instance = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    await first_instance.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    first = await first_instance.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "one"})
    await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")
    assert len(FakeClaudeSDKClient.options_seen) == 1

    second_instance = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    second = await second_instance.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "two"})
    await wait_for(lambda: store.get_turn(second["turnId"]).status == "completed")
    assert len(FakeClaudeSDKClient.options_seen) == 2
    assert FakeClaudeSDKClient.options_seen[-1].kwargs["resume"] == "b01bd0f7-f1b0-485e-a47c-d831645174b9"

    third = await first_instance.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "three"})
    await wait_for(lambda: store.get_turn(third["turnId"]).status == "completed")

    # The first instance saw the second instance's turn, dropped its stale
    # cached client, and resumed the native session instead of replying on
    # its outdated in-memory conversation.
    assert FakeClaudeSDKClient.disconnect_count == 1
    assert len(FakeClaudeSDKClient.options_seen) == 3
    assert FakeClaudeSDKClient.options_seen[-1].kwargs["resume"] == "b01bd0f7-f1b0-485e-a47c-d831645174b9"
    assert store.get_session(first["threadId"]).last_client_instance == first_instance._instance_id


async def wait_for(predicate, timeout: float = 2.0) -> None:  # type: ignore[no-untyped-def]
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.05)
    raise AssertionError("timed out waiting for predicate")


class FifoStreamClaudeSDKClient(FakeClaudeSDKClient):
    """Models the real SDK stream: responses return in query order (FIFO)."""

    queue: list[str] = []
    hold: bool = False

    async def query(self, prompt: str) -> None:
        await super().query(prompt)
        FifoStreamClaudeSDKClient.queue.append(prompt)

    async def receive_response(self):
        while FifoStreamClaudeSDKClient.hold or not FifoStreamClaudeSDKClient.queue:
            await asyncio.sleep(0.01)
        prompt = _user_prompt(FifoStreamClaudeSDKClient.queue.pop(0))
        yield FakeAssistantMessage([FakeTextBlock(f"reply:{prompt}")])
        yield FakeResultMessage(f"done:{prompt}")


class FifoStreamSdk:
    ClaudeAgentOptions = FakeClaudeAgentOptions
    ClaudeSDKClient = FifoStreamClaudeSDKClient


@pytest.mark.asyncio
async def test_claude_sdk_steered_response_does_not_shift_next_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A steer's response must be consumed by the steered turn.

    Regression for the voice off-by-one: a follow-up steered into a finishing
    turn produced an extra response on the shared stream; the next turn's
    reader consumed that stale response as its own answer, and every later
    turn answered the previous prompt.
    """
    FifoStreamClaudeSDKClient.queue = []
    FifoStreamClaudeSDKClient.hold = True
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: FifoStreamSdk())
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    first = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "hello"})
    await wait_for(
        lambda: [_user_prompt(prompt) for prompt in FakeClaudeSDKClient.prompts] == ["hello"]
    )
    steered = await client.steer_by_label(LabelQueryInput(label="sdk"), "how are you", {})
    assert steered["turnId"] == first["turnId"]

    FifoStreamClaudeSDKClient.hold = False
    await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")

    second = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "summarize"})
    await wait_for(lambda: store.get_turn(second["turnId"]).status == "completed")

    first_turn = store.get_turn(first["turnId"])
    second_turn = store.get_turn(second["turnId"])
    assert "how are you" in (first_turn.last_useful_message or "")
    assert "summarize" in (second_turn.last_useful_message or "")
    assert "how are you" not in (second_turn.last_useful_message or "")


class CoalescingClaudeSDKClient(FakeClaudeSDKClient):
    """Models CLI coalescing: all queued queries produce a single response."""

    queue: list[str] = []
    hold: bool = False

    async def query(self, prompt: str) -> None:
        await super().query(prompt)
        CoalescingClaudeSDKClient.queue.append(prompt)

    async def receive_response(self):
        while CoalescingClaudeSDKClient.hold or not CoalescingClaudeSDKClient.queue:
            await asyncio.sleep(0.01)
        prompts = [_user_prompt(item) for item in CoalescingClaudeSDKClient.queue]
        CoalescingClaudeSDKClient.queue.clear()
        yield FakeAssistantMessage([FakeTextBlock(f"reply:{prompts[-1]}")])
        yield FakeResultMessage(f"done:{prompts[-1]}")


class CoalescingSdk:
    ClaudeAgentOptions = FakeClaudeAgentOptions
    ClaudeSDKClient = CoalescingClaudeSDKClient


@pytest.mark.asyncio
async def test_claude_sdk_coalesced_steers_do_not_hang_the_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The CLI can merge rapid queries into one response; the turn must not
    wait forever for responses that were coalesced away."""
    CoalescingClaudeSDKClient.queue = []
    CoalescingClaudeSDKClient.hold = True
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: CoalescingSdk())
    client._steer_drain_timeout_seconds = 0.05
    await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})

    first = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "hello"})
    await wait_for(
        lambda: [_user_prompt(prompt) for prompt in FakeClaudeSDKClient.prompts] == ["hello"]
    )
    steered = await client.steer_by_label(LabelQueryInput(label="sdk"), "and also this", {})
    assert steered["turnId"] == first["turnId"]

    CoalescingClaudeSDKClient.hold = False
    await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")

    first_turn = store.get_turn(first["turnId"])
    assert "and also this" in (first_turn.last_useful_message or "")

    second = await client.start_turn_by_label(LabelQueryInput(label="sdk"), {"prompt": "next question"})
    await wait_for(lambda: store.get_turn(second["turnId"]).status == "completed")
    second_turn = store.get_turn(second["turnId"])
    assert "next question" in (second_turn.last_useful_message or "")


@pytest.mark.asyncio
async def test_claude_sdk_resume_can_replace_developer_instructions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """replace_developer_instructions swaps stale instructions for current
    ones instead of appending, so template updates propagate on resume."""
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    started = await client.start_thread(
        {"name": "sdk", "cwd": str(tmp_path), "developerInstructions": "Old stale rules."}
    )

    resumed = await client.resume_by_label(
        LabelQueryInput(label="sdk"),
        developer_instructions="New current rules.",
        replace_developer_instructions=True,
    )

    session = store.get_session(started["threadId"])
    assert resumed["threadId"] == started["threadId"]
    assert "New current rules." in (session.developer_instructions or "")
    assert "Old stale rules." not in (session.developer_instructions or "")
    assert "Super Agent thread name: sdk" in (session.developer_instructions or "")

    overlaid = await client.resume_by_label(
        LabelQueryInput(label="sdk"),
        developer_instructions="Additional overlay.",
    )
    session = store.get_session(overlaid["threadId"])
    assert "New current rules." in (session.developer_instructions or "")
    assert "Additional overlay." in (session.developer_instructions or "")


def _insert_ghost_turn(store: Store, thread_id: str, *, updated_at: str) -> str:
    import sqlite3 as _sqlite3

    turn_id = "t_ghost"
    with _sqlite3.connect(store.path) as conn:
        conn.execute(
            "insert into turns (id, session_id, prompt, status, created_at, updated_at) "
            "values (?, ?, 'ghost prompt', 'running', ?, ?)",
            (turn_id, thread_id, updated_at, updated_at),
        )
        conn.execute(
            "update sessions set status='running', active_turn_id=? where id=?",
            (turn_id, thread_id),
        )
    return turn_id


@pytest.mark.asyncio
async def test_claude_sdk_startup_sweep_fails_orphaned_running_turns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A service restart kills turn tasks without touching the store; the
    sweep must terminalize those ghost rows so they stop looking active."""
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    started = await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})
    turn_id = _insert_ghost_turn(store, started["threadId"], updated_at="2026-01-01T00:00:00.000Z")

    reconciled = client.reconcile_orphaned_turns()

    assert reconciled == 1
    turn = store.get_turn(turn_id)
    assert turn.status == "failed"
    assert "process exited" in (turn.last_error or "")
    session = store.get_session(started["threadId"])
    assert session.active_turn_id is None
    assert session.status == "failed"


@pytest.mark.asyncio
async def test_claude_sdk_startup_sweep_leaves_fresh_turns_alone(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from super_agents.agent_store import iso_now as _iso_now

    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    started = await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})
    turn_id = _insert_ghost_turn(store, started["threadId"], updated_at=_iso_now())

    reconciled = client.reconcile_orphaned_turns()

    assert reconciled == 0
    assert store.get_turn(turn_id).status == "running"


@pytest.mark.asyncio
async def test_claude_sdk_steer_reclaims_orphaned_turn_into_fresh_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A message steered at a dead turn must become a fresh turn instead of
    black-holing against a turn whose owning process is gone."""
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    started = await client.start_thread({"name": "sdk", "cwd": str(tmp_path)})
    ghost_turn_id = _insert_ghost_turn(
        store, started["threadId"], updated_at="2026-01-01T00:00:00.000Z"
    )

    steered = await client.steer_by_label(LabelQueryInput(label="sdk"), "hello again", {})

    assert steered["turnId"] != ghost_turn_id
    await wait_for(lambda: store.get_turn(steered["turnId"]).status == "completed")
    assert "hello again" in (store.get_turn(steered["turnId"]).last_useful_message or "")
    assert store.get_turn(ghost_turn_id).status == "failed"
