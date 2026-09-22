"""Blocked-tool steering and truthful terminal-result regression tests."""
import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient


class Options:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class BlockedToolClient:
    instances = []
    missing_terminal = False
    terminal_error = False
    suppress_followup = False

    def __init__(self, options):
        self.requests = asyncio.Queue()
        self.interrupted = asyncio.Event()
        self.started = asyncio.Event()
        self.operations = []
        self.instances.append(self)

    async def connect(self):
        pass

    async def query(self, prompt):
        self.operations.append(("query", prompt))
        await self.requests.put(prompt)

    async def interrupt(self):
        self.operations.append(("interrupt", None))
        self.interrupted.set()

    async def disconnect(self):
        pass

    async def receive_response(self):
        prompt = await self.requests.get()
        if prompt.endswith("blocked tool"):
            self.started.set()
            await self.interrupted.wait()
            # Interrupts before a model finishes can return zero-turn errors;
            # these are real results, not resume-handshake no-ops.
            yield SimpleNamespace(result="tool interrupted", num_turns=0, is_error=True)
            return
        if self.suppress_followup:
            await asyncio.Event().wait()
        await asyncio.sleep(.04)
        yield SimpleNamespace(content=[SimpleNamespace(text=prompt.split("\n\n")[-1])])
        if not self.missing_terminal:
            yield SimpleNamespace(result=prompt.split("\n\n")[-1], num_turns=1, is_error=self.terminal_error)


@pytest.fixture
def sdk():
    BlockedToolClient.instances = []
    BlockedToolClient.missing_terminal = False
    BlockedToolClient.terminal_error = False
    BlockedToolClient.suppress_followup = False
    return SimpleNamespace(ClaudeSDKClient=BlockedToolClient, ClaudeAgentOptions=Options)


async def terminal(store, turn_id):
    async with asyncio.timeout(2):
        while store.get_turn(turn_id).status == "running":
            await asyncio.sleep(.005)
    return store.get_turn(turn_id)


@pytest.mark.asyncio
async def test_explicit_interrupt_stops_tool_and_preserves_delayed_correction(tmp_path: Path, sdk):
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: sdk)
    client._steer_drain_timeout_seconds = .01
    client._interrupted_steer_start_timeout_seconds = .3
    await client.start_thread({"name": "elm", "cwd": str(tmp_path)})
    initial = await client.start_turn_by_label(LabelQueryInput(label="elm"), {"prompt": "blocked tool"})
    while not BlockedToolClient.instances:
        await asyncio.sleep(.005)
    transport = BlockedToolClient.instances[0]
    await asyncio.wait_for(transport.started.wait(), 1)
    result = await client.steer_by_label(LabelQueryInput(label="elm"), "corrected result", {"interruptCurrentWork": True})
    completed = await terminal(store, initial["turnId"])
    assert result["turnId"] == initial["turnId"]
    assert result["interruptedCurrentWork"] is True
    assert [op[0] for op in transport.operations] == ["query", "interrupt", "query"]
    assert completed.status == "completed"
    assert completed.last_useful_message == "corrected result"
    assert len(BlockedToolClient.instances) == 1
    assert store.queued_turns(initial["threadId"]) == []
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["error", "missing_terminal"])
async def test_error_or_missing_terminal_is_not_success(tmp_path: Path, sdk, mode):
    BlockedToolClient.terminal_error = mode == "error"
    BlockedToolClient.missing_terminal = mode == "missing_terminal"
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: sdk)
    await client.start_thread({"name": "elm", "cwd": str(tmp_path)})
    started = await client.start_turn_by_label(LabelQueryInput(label="elm"), {"prompt": "partial answer"})
    turn = await terminal(store, started["turnId"])
    assert turn.status == "failed"
    assert "unverified" in turn.last_error
    await client.close()


@pytest.mark.asyncio
async def test_interrupted_followup_timeout_is_not_coalesced_success(tmp_path: Path, sdk):
    BlockedToolClient.suppress_followup = True
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: sdk)
    client._interrupted_steer_start_timeout_seconds = .03
    await client.start_thread({"name": "elm", "cwd": str(tmp_path)})
    initial = await client.start_turn_by_label(LabelQueryInput(label="elm"), {"prompt": "blocked tool"})
    while not BlockedToolClient.instances:
        await asyncio.sleep(.005)
    await asyncio.wait_for(BlockedToolClient.instances[0].started.wait(), 1)
    await client.steer_by_label(LabelQueryInput(label="elm"), "corrected result", {"interruptCurrentWork": True})
    failed = await terminal(store, initial["turnId"])
    assert failed.status == "failed"
    assert "Interrupted steer" in failed.last_error
    await client.close()


@pytest.mark.asyncio
async def test_steer_records_turn_steers_for_thread_reads(tmp_path: Path, sdk):
    """Steering text persists on the turn row and appears in thread reads.

    The voice pipeline steers from a different process than the one serving
    history reads, so in-memory bookkeeping alone leaves steers invisible to
    clients (iOS showed only the turn's first message).
    """
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: sdk)
    client._steer_drain_timeout_seconds = .01
    client._interrupted_steer_start_timeout_seconds = .3
    await client.start_thread({"name": "elm", "cwd": str(tmp_path)})
    initial = await client.start_turn_by_label(LabelQueryInput(label="elm"), {"prompt": "blocked tool"})
    while not BlockedToolClient.instances:
        await asyncio.sleep(.005)
    transport = BlockedToolClient.instances[0]
    await asyncio.wait_for(transport.started.wait(), 1)

    await client.steer_by_label(
        LabelQueryInput(label="elm"),
        "<voice>also update the docs</voice>",
        {"interruptCurrentWork": True},
    )
    await terminal(store, initial["turnId"])

    steers = store.get_turn(initial["turnId"]).steers
    assert [steer["text"] for steer in steers] == ["<voice>also update the docs</voice>"]
    assert all(steer["createdAt"] for steer in steers)

    readback = await client.read_by_label(LabelQueryInput(label="elm"), include_turns=True)
    turn_view = next(t for t in readback["turns"] if t["turnId"] == initial["turnId"])
    assert [steer["text"] for steer in turn_view["steers"]] == ["<voice>also update the docs</voice>"]
    await client.close()
