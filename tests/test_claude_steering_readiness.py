"""BUG 20: readiness belongs to the active owner, not the steering caller."""

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from super_agents.agent_store import Store, iso_now
from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient
from super_agents.mcp_server import _tool_super_agents_steer
from test_claude_sdk import (
    BackgroundTaskClaudeSDKClient,
    BackgroundTaskSdk,
    FakeClaudeSDKClient,
    fake_sdk_loader,
    reset_fake_claude_sdk,  # noqa: F401 -- fixture isolates SDK configuration
    wait_for,
)


@pytest.mark.parametrize("backend", ["claude_code", "openbase_cloud"])
async def test_second_client_steers_owner_during_background_drain(tmp_path, backend):
    BackgroundTaskClaudeSDKClient.tasks_done = asyncio.Event()
    BackgroundTaskClaudeSDKClient.followup = True
    store = Store(tmp_path / "state.sqlite3")
    owner = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: BackgroundTaskSdk(), backend_identity=backend)
    caller = ClaudeAgentSdkClient(store=Store(store.path), sdk_loader=fake_sdk_loader, backend_identity=backend)
    thread = await owner.start_thread({"name": "birch", "cwd": str(tmp_path)})
    first = await owner.start_turn_by_label(LabelQueryInput(label="birch"), {"prompt": "list test files"})
    await wait_for(
        lambda: "waiting on 2 background" in (store.get_session(thread["threadId"]).last_observed_state or "")
    )
    assert caller._sdk_clients == {}
    try:
        result = await _tool_super_agents_steer(caller).handler(
            {"name": "birch", "prompt": "include byte sizes; birch-902"}
        )
        assert result["steered"] is True
        assert result["queued"] is False
        assert result["turnId"] == first["turnId"]
        assert len(FakeClaudeSDKClient.options_seen) == 1
        assert FakeClaudeSDKClient.prompts[-1].endswith("include byte sizes; birch-902")
        assert [x["text"] for x in store.get_turn(first["turnId"]).steers] == ["include byte sizes; birch-902"]
        assert not store.queued_turns(thread["threadId"])
    finally:
        BackgroundTaskClaudeSDKClient.tasks_done.set()
        await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")
        await asyncio.gather(*owner._turn_tasks)
        assert owner._active_owner(thread["threadId"]) is None
        await caller.close()
        await owner.close()


async def test_unavailable_owner_queues_durable_followup_and_drains_once(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    owner = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    caller = ClaudeAgentSdkClient(store=Store(store.path), sdk_loader=fake_sdk_loader)
    thread = await owner.start_thread({"name": "birch", "cwd": str(tmp_path)})
    # A live foreign owner holds its session lock but has no transport in the
    # caller's process. Use the real lock and persistent store, no readiness mock.
    first = store.create_turn(thread["threadId"], "working", status="running")
    store.update_session(thread["threadId"], status="running")
    async with owner._cross_process_session_lock(thread["threadId"]):
        result = await caller.steer_by_label(
            LabelQueryInput(label="birch"), "byte sizes; birch-902", {"interruptCurrentWork": True}
        )
        assert result["queued"] is True
        assert result["steered"] is False
        assert result["interruptedCurrentWork"] is False
        assert "not interrupted" in result["message"]
        assert result["turnId"] != first.id
        assert Store(store.path).get_turn(result["turnId"]).prompt == "byte sizes; birch-902"
        assert store.get_turn(first.id).steers == ()
        await caller.close()  # Queue survives the submitting client's lifetime.
        store.update_turn(first.id, status="completed", finished_at=iso_now())
        store.update_session(thread["threadId"], active_turn_id=None, status="completed")
    owner._schedule_queue_drain(thread["threadId"])
    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")
    assert len(FakeClaudeSDKClient.prompts) == 1
    assert FakeClaudeSDKClient.prompts[0].endswith("byte sizes; birch-902")
    await asyncio.gather(*owner._turn_tasks)
    await owner.close()


async def test_rejected_steer_plainly_reports_no_delivery_or_queue(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    thread = await client.start_thread({"name": "birch", "cwd": str(tmp_path)})
    first = store.create_turn(thread["threadId"], "working", status="running")
    store.update_session(thread["threadId"], status="running")
    with pytest.raises(RuntimeError, match="Nothing was delivered or queued"):
        await client.steer_by_label(LabelQueryInput(label="birch", turn_id="wrong"), "lost instruction")
    assert not store.get_turn(first.id).steers
    assert not store.queued_turns(thread["threadId"])
    await client.close()


def test_queue_claim_is_atomic_across_callers(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    session = store.create_session(name="birch", cwd=str(tmp_path), command=["claude-agent-sdk"])
    turn = store.create_turn(session.id, "followup", status="queued")
    # Multiple independent SQLite connections may drain after owner completion.
    with ThreadPoolExecutor(max_workers=4) as pool:
        claimed = list(pool.map(lambda _: Store(store.path).claim_queued_turn(session.id), range(4)))
    assert [item.id for item in claimed if item] == [turn.id]
    assert store.get_turn(turn.id).attempts == 1
    assert store.get_session(session.id).active_turn_id == turn.id


async def test_connecting_owner_queues_without_losing_instruction(tmp_path):
    from test_claude_sdk import FakeClaudeAgentOptions

    connected = asyncio.Event()

    class SlowClient(FakeClaudeSDKClient):
        async def connect(self):
            await connected.wait()
            await super().connect()

    class SlowSdk:
        ClaudeAgentOptions = FakeClaudeAgentOptions
        ClaudeSDKClient = SlowClient

    store = Store(tmp_path / "state.sqlite3")
    owner = ClaudeAgentSdkClient(store=store, sdk_loader=SlowSdk)
    caller = ClaudeAgentSdkClient(store=Store(store.path), sdk_loader=SlowSdk)
    thread = await owner.start_thread({"name": "birch", "cwd": str(tmp_path)})
    first = await owner.start_turn_by_label(LabelQueryInput(label="birch"), {"prompt": "initial"})
    try:
        result = await caller.steer_by_label(LabelQueryInput(label="birch"), "followup")
        assert result["queued"] is True
        assert store.get_turn(result["turnId"]).status == "queued"
        assert store.get_turn(first["turnId"]).status == "running"
    finally:
        connected.set()
    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")
    assert [p.rsplit("\n\n", 1)[-1] for p in FakeClaudeSDKClient.prompts] == ["initial", "followup"]
    await asyncio.gather(*owner._turn_tasks, *caller._turn_tasks)
    assert owner._active_owner(thread["threadId"]) is None
    await caller.close()
    await owner.close()


async def test_mcp_failed_steer_never_promises_delivery(tmp_path):
    import json

    from mcp.shared.memory import create_connected_server_and_client_session

    from super_agents.mcp_server import create_server

    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    thread = await client.start_thread({"name": "birch", "cwd": str(tmp_path)})
    store.create_turn(thread["threadId"], "working", status="running")
    store.update_session(thread["threadId"], status="running")
    async with create_connected_server_and_client_session(create_server(client)) as session:
        result = await session.call_tool(
            "super_agents_steer", {"name": "birch", "turnId": "wrong", "prompt": "new instruction"}
        )
    assert result.isError
    payload = json.loads(result.content[0].text)
    assert "Nothing was delivered or queued" in payload["error"]
    assert "Do not claim" in payload["message"]
    assert not store.queued_turns(thread["threadId"])
    await client.close()
