"""Accepted steers invalidate completion and return to the response drain."""

import asyncio

import pytest

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient
from test_claude_sdk import (
    FakeClaudeSDKClient,
    FifoStreamClaudeSDKClient,
    FifoStreamSdk,
    reset_fake_claude_sdk,  # noqa: F401 -- isolate managed configuration
    wait_for,
)


@pytest.mark.parametrize("interrupt", [False, True])
async def test_accepted_steer_during_completion_validation_is_consumed(tmp_path, interrupt):
    """Voice review finding 9: the confirmed correction must reach completion."""
    entered, release = asyncio.Event(), asyncio.Event()
    validated, published = [], []
    FifoStreamClaudeSDKClient.queue = []
    FifoStreamClaudeSDKClient.hold = False

    async def validate(session, turn, result):
        validated.append(result.result)
        entered.set()
        await release.wait()
        published.append(result.result)

    store = Store(tmp_path / "validation.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=FifoStreamSdk, validate_turn_result=validate)
    try:
        await client.start_thread({"name": "worker", "cwd": str(tmp_path)})
        query = LabelQueryInput(label="worker")
        start = await client.start_turn_by_label(query, {"prompt": "Inspect the first file"})
        await asyncio.wait_for(entered.wait(), 3)
        steer = await client.start_turn_by_label(
            query, {"prompt": "Also inspect the second file", "interruptCurrentWork": interrupt}
        )
        assert steer["confirmed"] and steer["steered"]
        assert steer["turnId"] == start["turnId"]
        release.set()
        await asyncio.wait_for(asyncio.gather(*client._turn_tasks), 3)
        turn = store.get_turn(start["turnId"])
        assert turn.status == "completed", (turn.status, turn.last_error)
        assert not FifoStreamClaudeSDKClient.queue
        assert validated == ["done:Inspect the first file", "done:Also inspect the second file"]
        assert published == ["done:Also inspect the second file"]
        assert turn.response_finished_at is not None
        assert FakeClaudeSDKClient.disconnect_count == 0
        assert not client._session_interrupted_steer_followups

        # The next turn must not inherit the accepted correction's response.
        next_turn = await client.start_turn_by_label(query, {"prompt": "Summarize"})
        await asyncio.wait_for(asyncio.gather(*client._turn_tasks), 3)
        assert store.get_turn(next_turn["turnId"]).last_useful_message == "done:Summarize"
    finally:
        release.set()
        await client.close()


@pytest.mark.parametrize("stale_outcome", ["cancel", "return", "error"])
async def test_repeated_steers_preserve_validation_cleanup_and_wait_for_fresh_result(tmp_path, stale_outcome):
    entered, cleaning, release_cleanup = asyncio.Event(), asyncio.Event(), asyncio.Event()
    cleaned = []
    validated, published = [], []
    FifoStreamClaudeSDKClient.queue = []
    FifoStreamClaudeSDKClient.hold = False
    store = Store(tmp_path / "validation.sqlite3")

    async def validate(session, turn, result):
        validated.append(result.result)
        if len(validated) == 1:
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                await release_cleanup.wait()
                cleaned.append(True)
                if stale_outcome == "cancel":
                    raise
                if stale_outcome == "error":
                    raise ValueError("Obsolete publication failed while being revoked")
                return
        published.append(result.result)

    client = ClaudeAgentSdkClient(store=store, sdk_loader=FifoStreamSdk, validate_turn_result=validate)
    client._steer_drain_timeout_seconds = 0.01
    try:
        await client.start_thread({"name": "worker", "cwd": str(tmp_path)})
        query = LabelQueryInput(label="worker")
        turn_id = (await client.start_turn_by_label(query, {"prompt": "inspect"}))["turnId"]
        await asyncio.wait_for(entered.wait(), 3)
        FifoStreamClaudeSDKClient.hold = True
        assert (await client.steer_by_label(query, "include sizes", {}))["confirmed"]
        await asyncio.wait_for(cleaning.wait(), 3)
        assert (await client.steer_by_label(query, "include dates too", {}))["confirmed"]
        release_cleanup.set()
        await wait_for(lambda: bool(cleaned))
        # New work arrived after draining, so it cannot have been coalesced
        # into the original result, even if it takes a while to start.
        await asyncio.sleep(0.05)
        assert store.get_turn(turn_id).status == "running"
        assert store.get_turn(turn_id).response_finished_at is None
        assert published == []
        assert len(FifoStreamClaudeSDKClient.queue) == 2
        assert FakeClaudeSDKClient.disconnect_count == 0
        FifoStreamClaudeSDKClient.hold = False
        await asyncio.wait_for(asyncio.gather(*client._turn_tasks), 3)
        assert store.get_turn(turn_id).status == "completed"
        assert validated == ["done:inspect", "done:include dates too"]
        assert published == ["done:include dates too"]
        assert not client._session_pending_results
        assert not client._result_validation._active
    finally:
        release_cleanup.set()
        FifoStreamClaudeSDKClient.hold = False
        await client.close()


async def test_missing_post_validation_response_cannot_reuse_old_success(tmp_path):
    entered = asyncio.Event()
    FifoStreamClaudeSDKClient.queue = []
    FifoStreamClaudeSDKClient.hold = False
    store = Store(tmp_path / "validation.sqlite3")

    class EndsAfterOriginal(FifoStreamClaudeSDKClient):
        async def receive_response(self):
            if entered.is_set():
                return
            async for message in super().receive_response():
                yield message

    class Sdk(FifoStreamSdk):
        ClaudeSDKClient = EndsAfterOriginal

    async def validate(*args):
        entered.set()
        await asyncio.Event().wait()

    client = ClaudeAgentSdkClient(store=store, sdk_loader=Sdk, validate_turn_result=validate)
    try:
        await client.start_thread({"name": "worker", "cwd": str(tmp_path)})
        query = LabelQueryInput(label="worker")
        turn_id = (await client.start_turn_by_label(query, {"prompt": "inspect"}))["turnId"]
        await asyncio.wait_for(entered.wait(), 3)
        assert (await client.steer_by_label(query, "include sizes", {}))["confirmed"]
        await asyncio.wait_for(asyncio.gather(*client._turn_tasks), 3)
        turn = store.get_turn(turn_id)
        assert turn.status == "failed"
        assert "without a terminal result" in turn.last_error
        assert turn.response_finished_at is None
    finally:
        await client.close()
