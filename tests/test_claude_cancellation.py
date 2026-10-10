"""Cancellation must stop the SDK owner and preserve per-turn output."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient


@pytest.mark.parametrize("cancel", [False, True])
async def test_machine_token_wait_keeps_event_loop_responsive(tmp_path, monkeypatch, cancel):
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_CODE_HOME", str(tmp_path))
    monkeypatch.delenv("OPENBASE_CLOUD_ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr("super_agents.claude_options.claude_state_path", lambda: tmp_path / "claude.json")
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()
    finished = asyncio.Event()
    connected = []

    def read_token(*args, **kwargs):
        loop.call_soon_threadsafe(entered.set)
        try:
            assert release.wait(2), "Token read blocked the event loop"
            return SimpleNamespace(returncode=0, stdout="obmt_test")
        finally:
            loop.call_soon_threadsafe(finished.set)

    monkeypatch.setattr("super_agents.claude_options.subprocess.run", read_token)

    class SDK:
        def __init__(self, **kwargs):
            pass

        async def connect(self):
            connected.append(self)

        async def disconnect(self):
            pass

    sdk = SimpleNamespace(ClaudeSDKClient=SDK, ClaudeAgentOptions=lambda **kwargs: SimpleNamespace(**kwargs))
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: sdk, backend_identity="openbase_cloud")
    session = store.create_session(name="token", cwd=str(tmp_path))
    task = asyncio.create_task(client._sdk_client_for(session, None, None, None, sdk))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert not task.done()
        if cancel:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        release.set()
        await asyncio.wait_for(finished.wait(), 1)
        if not cancel:
            await task
        assert len(connected) == (0 if cancel else 1)
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        await client.close()


def assistant(text):
    return SimpleNamespace(content=[SimpleNamespace(text=text)])


async def until(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0.005)


@pytest.mark.parametrize("foreign", [False, True])
@pytest.mark.parametrize("partial", ["", "own partial"])
async def test_cancel_stops_silent_owner_and_never_inherits_previous_answer(tmp_path, monkeypatch, foreign, partial):
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_CODE_HOME", str(tmp_path))
    monkeypatch.setattr("super_agents.claude_options.claude_state_path", lambda: tmp_path / "claude.json")
    waiting = asyncio.Event()
    release = asyncio.Event()
    stopped = asyncio.Event()
    interrupted = asyncio.Event()
    instances = []

    class SDK:
        def __init__(self, **kwargs):
            self.prompt = ""
            instances.append(self)

        async def connect(self):
            pass

        async def query(self, prompt):
            self.prompt = prompt.rsplit("\n\n", 1)[-1]

        async def receive_response(self):
            if self.prompt == "cancel this":
                if partial:
                    yield assistant(partial)
                waiting.set()
                # Model a stream that does not acknowledge interrupt and would
                # deliver a full answer later unless its owner stops it.
                await release.wait()
                yield assistant("late answer must never be persisted")
            else:
                yield assistant(f"answer:{self.prompt}")
            yield SimpleNamespace(result=f"answer:{self.prompt}", num_turns=1)

        async def interrupt(self):
            interrupted.set()
            await asyncio.Event().wait()  # Broken acknowledgement must be bounded.

        async def disconnect(self):
            stopped.set()

    events = []

    class Client(ClaudeAgentSdkClient):
        def handle_notification(self, method, params):
            events.append((method, params))

    def loader():
        return SimpleNamespace(ClaudeSDKClient=SDK, ClaudeAgentOptions=lambda **kw: kw)
    store = Store(tmp_path / "state.sqlite3")
    owner = Client(store=store, sdk_loader=loader)
    canceller = Client(store=Store(store.path), sdk_loader=loader) if foreign else owner
    thread_id = (await owner.start_thread({"name": "test", "cwd": str(tmp_path)}))["threadId"]
    query = LabelQueryInput(thread_id=thread_id)
    try:
        first = await owner.start_turn_by_label(query, {"prompt": "previous"})
        await until(lambda: store.get_turn(first["turnId"]).status == "completed")
        turn_id = (await owner.start_turn_by_label(query, {"prompt": "cancel this"}))["turnId"]
        await asyncio.wait_for(waiting.wait(), 1)
        # The user stops approximately one second after turn_started.
        await asyncio.sleep(1)
        async with asyncio.timeout(1):
            assert (await canceller.cancel_by_label(query))["cancelled"]
            await until(lambda: any(m == "turn/completed" and p["turnId"] == turn_id for m, p in events))
            await stopped.wait()
        assert interrupted.is_set()
        frozen = store.get_turn(turn_id)
        assert frozen.status == "cancelled"
        assert frozen.last_useful_message == (partial or None)
        read = await canceller.read_by_label(query, include_turns=True)
        assert read["turns"][0].get("lastUsefulMessage", "") == partial
        release.set()
        recovery = await owner.start_turn_by_label(query, {"prompt": "recovery"})
        await until(lambda: store.get_turn(recovery["turnId"]).status == "completed")
        assert len(instances) == 2  # Never reuse the interrupted stream.
        assert store.get_turn(recovery["turnId"]).last_useful_message == "answer:recovery"
        assert store.get_turn(turn_id) == frozen
        assert store.get_turn(first["turnId"]).last_useful_message == "answer:previous"
    finally:
        release.set()
        await asyncio.gather(*owner._turn_tasks, return_exceptions=True)
        await owner.close()
        if foreign:
            await canceller.close()


def test_cancelled_turn_rejects_late_inflight_writes(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_CODE_HOME", str(tmp_path))
    store = Store(tmp_path / "state.sqlite3")
    session = store.create_session(name="atomic", cwd=str(tmp_path))
    turn = store.create_turn(session.id, "cancel", status="running")
    frozen = store.update_turn(turn.id, status="cancelled", finished_at="cancel-time", last_useful_message="partial")
    for update in ({"last_useful_message": "late output"}, {"status": "completed", "finished_at": "late-time"}):
        assert store.update_turn(turn.id, only_if_active=True, **update) == frozen


@pytest.mark.parametrize("phase", ["connect", "query", "background"])
async def test_cancel_during_startup_closes_only_the_owned_sdk(tmp_path, monkeypatch, phase):
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_CODE_HOME", str(tmp_path))
    monkeypatch.setattr("super_agents.claude_options.claude_state_path", lambda: tmp_path / "claude.json")
    entered, stopped = asyncio.Event(), asyncio.Event()
    queries = []
    instances = []

    class SDK:
        def __init__(self, **kwargs):
            instances.append(self)

        async def connect(self):
            if phase == "connect":
                entered.set()
                await asyncio.Event().wait()

        async def query(self, prompt):
            queries.append(prompt)
            if phase == "query":
                entered.set()
                await asyncio.Event().wait()

        async def receive_response(self):
            if not hasattr(self, "responded"):
                self.responded = True
                yield SimpleNamespace(subtype="task_started", task_id="task-test")
                yield SimpleNamespace(num_turns=1, result="own response")
            else:
                entered.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    self.reader_closed = True

        async def interrupt(self):
            pass

        async def disconnect(self):
            stopped.set()

    client = ClaudeAgentSdkClient(
        store=Store(tmp_path / "state.sqlite3"),
        sdk_loader=lambda: SimpleNamespace(ClaudeSDKClient=SDK, ClaudeAgentOptions=lambda **kw: kw),
    )
    thread_id = (await client.start_thread({"name": "startup", "cwd": str(tmp_path)}))["threadId"]
    query = LabelQueryInput(thread_id=thread_id)
    turn_id = (await client.start_turn_by_label(query, {"prompt": "stop"}))["turnId"]
    await asyncio.wait_for(entered.wait(), 1)
    assert (await client.cancel_by_label(query))["cancelled"]
    await asyncio.wait_for(stopped.wait(), 1)
    await asyncio.gather(*client._turn_tasks)
    assert client.store.get_turn(turn_id).status == "cancelled"
    assert not client._sdk_clients
    if phase == "connect":
        assert queries == []
    if phase == "background":
        assert instances[0].reader_closed
    assert not (await client.cancel_by_label(query))["cancelled"]
    await client.close()
