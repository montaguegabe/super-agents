"""Voice review findings 12-14 (2026-10-11), reproductions by the reviewer."""

import asyncio

import pytest
from test_claude_sdk import FakeClaudeSDKClient, reset_fake_claude_sdk, wait_for  # noqa: F401  (autouse reset)
from test_claude_compaction import CompactingClient, CompactingSdk
from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient


@pytest.mark.asyncio
async def test_steer_cannot_overtake_initial_request_waiting_for_compaction(tmp_path, monkeypatch):
    CompactingClient.release = asyncio.Event()
    CompactingClient.compacts = []
    monkeypatch.setattr(ClaudeAgentSdkClient, "session_context_tokens", lambda *_: 150000)
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=CompactingSdk)
    compaction = None
    try:
        thread = await client.start_thread({"name": "dispatcher", "cwd": str(tmp_path)})
        query = LabelQueryInput(thread_id=thread["threadId"])
        compaction = asyncio.create_task(client.compact_session_by_label(query, min_context_tokens=1))
        await wait_for(lambda: CompactingClient.compacts == ["/compact"])
        turn = await client.start_turn_by_label(query, {"prompt": "Create the draft in folder A"})
        await asyncio.sleep(0)
        correction = await client.start_turn_by_label(query, {"prompt": "Use folder B instead"})
        assert correction["confirmed"] and correction["steered"]
        CompactingClient.release.set()
        await asyncio.wait_for(compaction, 3)
        await wait_for(lambda: store.get_turn(turn["turnId"]).status == "completed")
        observed = [p.rsplit("\n\n", 1)[-1] for p in FakeClaudeSDKClient.prompts]
        assert observed == ["Create the draft in folder A", "Use folder B instead"], observed
    finally:
        CompactingClient.release.set()
        if compaction:
            await asyncio.wait_for(compaction, 3)
        await client.close()


@pytest.mark.asyncio
async def test_close_during_compaction_releases_the_sdk_transport(tmp_path, monkeypatch):
    CompactingClient.release = asyncio.Event()
    CompactingClient.compacts = []
    monkeypatch.setattr(ClaudeAgentSdkClient, "session_context_tokens", lambda *_: 150000)
    client = ClaudeAgentSdkClient(store=Store(tmp_path / "close.sqlite3"), sdk_loader=CompactingSdk)
    compaction = None
    sdk_client = None
    try:
        thread = await client.start_thread({"name": "dispatcher", "cwd": str(tmp_path)})
        compaction = asyncio.create_task(
            client.compact_session_by_label(LabelQueryInput(thread_id=thread["threadId"]), min_context_tokens=1)
        )
        await wait_for(lambda: CompactingClient.compacts == ["/compact"])
        sdk_client = client._sdk_clients[thread["threadId"]]
        await client.close()
        CompactingClient.release.set()
        await asyncio.wait_for(compaction, 3)
        assert not sdk_client.connected, "Compaction completed after owner close but left SDK transport connected"
    finally:
        CompactingClient.release.set()
        if compaction:
            await asyncio.wait_for(compaction, 3)
        await client.close()


@pytest.mark.asyncio
async def test_request_waiting_for_compaction_rechecks_instruction_presence(tmp_path, monkeypatch):
    import json
    import super_agents.claude_sdk as sdk_module

    transcript = tmp_path / "transcript.jsonl"
    CompactingClient.release = asyncio.Event()
    CompactingClient.compacts = []

    class BoundaryClient(CompactingClient):
        async def receive_response(self):
            compacting = getattr(self, "prompt", None) == "/compact"
            async for msg in super().receive_response():
                if compacting:
                    with transcript.open("a") as f:
                        f.write(json.dumps({"type": "system", "subtype": "compact_boundary"}) + "\n")
                yield msg

    class Sdk(CompactingSdk):
        ClaudeSDKClient = BoundaryClient

    monkeypatch.setattr(ClaudeAgentSdkClient, "session_context_tokens", lambda *_: 150000)
    monkeypatch.setattr(sdk_module, "transcript_path", lambda *_: transcript)
    store = Store(tmp_path / "rules.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=Sdk)
    compaction = None
    rules = "Route all requested work through the required dispatch tools."
    try:
        thread = await client.start_thread({"name": "dispatcher", "cwd": str(tmp_path), "developerInstructions": rules})
        query = LabelQueryInput(thread_id=thread["threadId"])
        session = store.update_session(thread["threadId"], backend_session_id="resumed-session")
        full = client._prompt_for_session(session, {"prompt": "earlier request"})
        transcript.write_text(json.dumps({"type": "user", "message": {"content": full}}) + "\n")
        assert rules not in client._prompt_for_session(session, {"prompt": "recent follow-up"})
        compaction = asyncio.create_task(client.compact_session_by_label(query, min_context_tokens=1))
        await wait_for(lambda: CompactingClient.compacts == ["/compact"])
        turn = await client.start_turn_by_label(query, {"prompt": "Please handle my next task"})
        CompactingClient.release.set()
        await asyncio.wait_for(compaction, 3)
        await wait_for(lambda: store.get_turn(turn["turnId"]).status == "completed")
        assert "compact_boundary" in transcript.read_text()
        assert rules in FakeClaudeSDKClient.prompts[-1], FakeClaudeSDKClient.prompts[-1]
    finally:
        CompactingClient.release.set()
        if compaction:
            await asyncio.wait_for(compaction, 3)
        await client.close()


@pytest.mark.asyncio
async def test_live_steer_cannot_overtake_second_buffered_steer(tmp_path, monkeypatch):
    """Finding 12 follow-up: a correction arriving while older buffered ones
    drain must not overtake them (observed A/B/D/C instead of A/B/C/D)."""
    CompactingClient.release = asyncio.Event()
    CompactingClient.compacts = []
    first_written = asyncio.Event()
    release_first_write = asyncio.Event()
    newest_waiting = asyncio.Event()

    class OrderedTransport(CompactingClient):
        def __init__(self, options):
            super().__init__(options)
            self.write_lock = asyncio.Lock()

        async def query(self, prompt):
            user = prompt.rsplit("\n\n", 1)[-1]
            if user == "Use folder D instead":
                newest_waiting.set()
            # The installed SDK serializes transport.write with an anyio Lock.
            async with self.write_lock:
                await super().query(prompt)
                if user == "Use folder B instead":
                    first_written.set()
                    await release_first_write.wait()

    class Sdk(CompactingSdk):
        ClaudeSDKClient = OrderedTransport

    monkeypatch.setattr(ClaudeAgentSdkClient, "session_context_tokens", lambda *_: 150000)
    store = Store(tmp_path / "ordered.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=Sdk)
    compaction = newest = None
    try:
        thread = await client.start_thread({"name": "dispatcher", "cwd": str(tmp_path)})
        query = LabelQueryInput(thread_id=thread["threadId"])
        compaction = asyncio.create_task(client.compact_session_by_label(query, min_context_tokens=1))
        await wait_for(lambda: CompactingClient.compacts == ["/compact"])
        turn = await client.start_turn_by_label(query, {"prompt": "Create in folder A"})
        for prompt in ["Use folder B instead", "Use folder C instead"]:
            accepted = await client.start_turn_by_label(query, {"prompt": prompt})
            assert accepted["confirmed"] and accepted["buffered"]
        CompactingClient.release.set()
        await asyncio.wait_for(first_written.wait(), 3)
        newest = asyncio.create_task(client.start_turn_by_label(query, {"prompt": "Use folder D instead"}))
        # Fixed behaviour: D joins the ordered buffer behind C instead of
        # reaching the transport's write lock (the reviewer's original gate
        # waited for D's write to start, which the fix prevents).
        done, _ = await asyncio.wait(
            {newest, asyncio.ensure_future(newest_waiting.wait())}, timeout=3, return_when=asyncio.FIRST_COMPLETED
        )
        assert done
        release_first_write.set()
        assert (await newest)["confirmed"]
        await wait_for(lambda: store.get_turn(turn["turnId"]).status == "completed")
        sent = [p.rsplit("\n\n", 1)[-1] for p in FakeClaudeSDKClient.prompts]
        assert sent == ["Create in folder A", "Use folder B instead", "Use folder C instead", "Use folder D instead"], (
            sent
        )
    finally:
        CompactingClient.release.set()
        release_first_write.set()
        if compaction:
            await asyncio.wait_for(compaction, 3)
        if newest:
            await asyncio.wait_for(newest, 3)
        await client.close()
