from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from test_claude_sdk import (  # noqa: F401  (autouse fixture)
    FakeClaudeAgentOptions,
    FakeClaudeSDKClient,
    FakeResultMessage,
    reset_fake_claude_sdk,
    wait_for,
)

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_compaction import context_tokens_from_transcript
from super_agents.claude_sdk import ClaudeAgentSdkClient


def _assistant(cache_read: int, created: int, *, sidechain: bool = False) -> str:
    return json.dumps(
        {
            "type": "assistant",
            "isSidechain": sidechain,
            "message": {
                "usage": {
                    "input_tokens": 10,
                    "cache_read_input_tokens": cache_read,
                    "cache_creation_input_tokens": created,
                    "output_tokens": 99,
                }
            },
        }
    )


def test_context_tokens_reads_the_latest_main_thread_call(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text("\n".join([_assistant(1000, 5), _assistant(150_000, 30_000), _assistant(9, 9, sidechain=True)]))
    assert context_tokens_from_transcript(path) == 180_010
    with path.open("a") as handle:
        handle.write("\n" + json.dumps({"type": "system", "subtype": "compact_boundary"}))
    assert context_tokens_from_transcript(path) == 0
    assert context_tokens_from_transcript(tmp_path / "missing.jsonl") is None
    assert context_tokens_from_transcript(None) is None


class CompactingClient(FakeClaudeSDKClient):
    """Answers /compact like Claude Code: a num_turns == 0 result after a delay."""

    release: asyncio.Event = asyncio.Event()
    compacts: list[str] = []

    async def query(self, prompt: str) -> None:
        if prompt == "/compact":
            CompactingClient.compacts.append(prompt)
            self.prompt = prompt
            return
        await super().query(prompt)

    async def receive_response(self):
        if getattr(self, "prompt", None) == "/compact":
            await CompactingClient.release.wait()
            self.prompt = None
            yield FakeResultMessage("", num_turns=0)
            return
        async for message in super().receive_response():
            yield message


class CompactingSdk:
    ClaudeAgentOptions = FakeClaudeAgentOptions
    ClaudeSDKClient = CompactingClient


@pytest.fixture
def client_with_big_context(tmp_path, monkeypatch):
    CompactingClient.release = asyncio.Event()
    CompactingClient.compacts = []
    monkeypatch.setattr(ClaudeAgentSdkClient, "session_context_tokens", lambda self, session: 150_000, raising=False)
    store = Store(tmp_path / "compact.sqlite3")
    return store, ClaudeAgentSdkClient(store=store, sdk_loader=CompactingSdk)


@pytest.mark.asyncio
async def test_idle_session_is_compacted_and_a_new_request_waits_behind_it(client_with_big_context, tmp_path):
    """VM2, 2026-10-11 08:01Z: the Dispatcher auto-compacted at the start of a
    relayed steer and stalled it 34 s. Compaction now runs between requests;
    a request arriving meanwhile waits for it and is never dropped."""
    store, client = client_with_big_context
    thread_id = (await client.start_thread({"name": "dispatcher", "cwd": str(tmp_path)}))["threadId"]
    compaction = asyncio.create_task(
        client.compact_session_by_label(LabelQueryInput(thread_id=thread_id), min_context_tokens=120_000)
    )
    await wait_for(lambda: CompactingClient.compacts == ["/compact"])
    assert "Compacting" in (store.get_session(thread_id).last_observed_state or "")

    turn = await client.start_turn_by_label(LabelQueryInput(thread_id=thread_id), {"prompt": "hello"})
    await asyncio.sleep(0.1)
    assert store.get_turn(turn["turnId"]).status == "running"
    assert FakeClaudeSDKClient.prompts == []  # still queued behind the compaction

    CompactingClient.release.set()
    result = await compaction
    assert result["compacted"] is True and result["contextTokens"] == 150_000
    await wait_for(lambda: store.get_turn(turn["turnId"]).status == "completed")
    assert "hello" in (store.get_turn(turn["turnId"]).last_useful_message or "")
    await client.close()


@pytest.mark.asyncio
async def test_compaction_skips_busy_and_small_sessions(client_with_big_context, tmp_path, monkeypatch):
    store, client = client_with_big_context
    thread_id = (await client.start_thread({"name": "dispatcher", "cwd": str(tmp_path)}))["threadId"]
    small = await client.compact_session_by_label(LabelQueryInput(thread_id=thread_id), min_context_tokens=180_000)
    assert small["compacted"] is False and small["reason"] == "below_threshold"

    FakeClaudeSDKClient.blocked_prompts = {"long task"}
    turn = await client.start_turn_by_label(LabelQueryInput(thread_id=thread_id), {"prompt": "long task"})
    await wait_for(lambda: bool(FakeClaudeSDKClient.prompts))
    busy = await client.compact_session_by_label(LabelQueryInput(thread_id=thread_id), min_context_tokens=1)
    assert busy["compacted"] is False and busy["reason"] == "busy"
    assert CompactingClient.compacts == []
    FakeClaudeSDKClient.blocked_prompts = set()
    await wait_for(lambda: store.get_turn(turn["turnId"]).status == "completed")
    await client.close()


@pytest.mark.asyncio
async def test_a_compaction_that_never_finishes_times_out_and_drops_the_client(client_with_big_context, tmp_path):
    store, client = client_with_big_context
    thread_id = (await client.start_thread({"name": "dispatcher", "cwd": str(tmp_path)}))["threadId"]
    result = await client.compact_session_by_label(
        LabelQueryInput(thread_id=thread_id), min_context_tokens=1, timeout_seconds=0.05
    )
    assert result["compacted"] is False and result["reason"] == "timeout"
    assert thread_id not in client._sdk_clients
    assert "Compacting" not in (store.get_session(thread_id).last_observed_state or "")
    await client.close()


@pytest.mark.asyncio
async def test_recent_lists_finished_agents_by_default(tmp_path):
    """VM2, 2026-10-11 08:41Z: after a compaction the Dispatcher asked
    super_agents_recent which agents exist and got only its own running turn,
    because recent meant "active unless includeInactive"; it then told the user
    no other Super Agents existed while four finished ones did."""
    store = Store(tmp_path / "recent.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=CompactingSdk)
    dispatcher = await client.start_thread({"name": "dispatcher", "cwd": str(tmp_path)})
    workers = [
        await client.start_thread({"name": f"worker-{n}", "agentName": name, "cwd": str(tmp_path)})
        for n, name in enumerate(["Cory", "Lucy", "Daisy", "Renee"])
    ]
    for worker in workers:
        store.update_session(worker["threadId"], status="completed")
    live = store.create_turn(dispatcher["threadId"], "which agents exist?", status="running")
    store.update_session(dispatcher["threadId"], status="running", active_turn_id=live.id)

    everyone = await client.recent(LabelQueryInput())
    names = {agent.get("agentName") for agent in everyone["agents"]}
    assert {"Cory", "Lucy", "Daisy", "Renee"} <= names
    running_only = await client.recent(LabelQueryInput(include_inactive=False))
    assert [agent["threadId"] for agent in running_only["agents"]] == [dispatcher["threadId"]]
    active = await client.active(LabelQueryInput())
    assert [agent["threadId"] for agent in active["agents"]] == [dispatcher["threadId"]]
    await client.close()
