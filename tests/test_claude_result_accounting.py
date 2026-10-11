"""Result ownership survives new queries during logging, writes and cleanup."""

import asyncio
import threading

import pytest

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient
from test_claude_sdk import (
    FakeAssistantMessage,
    FakeClaudeAgentOptions,
    FakeClaudeSDKClient,
    FakeResultMessage,
    FakeToolUseBlock,
    _user_prompt,
    reset_fake_claude_sdk,  # noqa: F401 -- isolate managed configuration
)


@pytest.mark.parametrize("with_validator", [False, True])
@pytest.mark.parametrize("wakeup", [False, True])
@pytest.mark.parametrize("pause_at", ["result_write", "log", "cleanup"])
async def test_steer_arriving_during_wakeup_result_write_is_not_consumed_by_that_result(
    tmp_path, monkeypatch, with_validator, wakeup, pause_at
):
    """Finding 10 reproduction, extended to every await after result receipt."""
    waiting, fire_result = asyncio.Event(), asyncio.Event()
    write_entered, release_write, arm_write = threading.Event(), threading.Event(), threading.Event()
    correction_waiting, release_correction = asyncio.Event(), asyncio.Event()
    responses, validated = [], []

    def pause():
        arm_write.clear()
        write_entered.set()
        assert release_write.wait(5)

    class Transport(FakeClaudeSDKClient):
        cycles = 0

        def __init__(self, options):
            super().__init__(options)
            self.queries = asyncio.Queue()

        async def query(self, prompt):
            await super().query(prompt)
            self.queries.put_nowait(_user_prompt(prompt))

        async def receive_response(self):
            self.cycles += 1
            if self.cycles == 1 and wakeup:
                await self.queries.get()
                yield FakeAssistantMessage([FakeToolUseBlock("ScheduleWakeup", {"delaySeconds": 600})])
                yield FakeResultMessage("Scheduled wakeup.")
            elif self.cycles == (2 if wakeup else 1):
                if not wakeup:
                    await self.queries.get()
                waiting.set()
                await fire_result.wait()
                responses.append("old result")
                arm_write.set()
                try:
                    yield FakeResultMessage("Old work done.")
                finally:
                    if pause_at == "cleanup":
                        await asyncio.to_thread(pause)
            else:
                correction_waiting.set()
                await release_correction.wait()
                prompt = await self.queries.get()
                responses.append(prompt)
                yield FakeResultMessage(f"done:{prompt}")

    class Sdk:
        ClaudeAgentOptions = FakeClaudeAgentOptions
        ClaudeSDKClient = Transport

    store = Store(tmp_path / "accounting.sqlite3")
    original = store.update_turn

    def delayed_result_write(turn_id, **kwargs):
        if (
            pause_at == "result_write"
            and arm_write.is_set()
            and threading.current_thread() is not threading.main_thread()
            and set(kwargs) == {"only_if_active", "response_finished_at"}
        ):
            pause()
        return original(turn_id, **kwargs)

    monkeypatch.setattr(store, "update_turn", delayed_result_write)
    from super_agents import claude_sdk

    original_log = claude_sdk.append_log

    def delayed_result_log(path, text):
        if pause_at == "log" and arm_write.is_set() and '"result": "Old work done."' in text:
            pause()
        original_log(path, text)

    monkeypatch.setattr(claude_sdk, "append_log", delayed_result_log)

    async def validate(session, turn, result):
        validated.append(result.result)

    client = ClaudeAgentSdkClient(store=store, sdk_loader=Sdk, validate_turn_result=validate if with_validator else None)
    client._steer_drain_timeout_seconds = 0.01
    try:
        await client.start_thread({"name": "wakeup", "cwd": str(tmp_path)})
        query = LabelQueryInput(label="wakeup")
        start = await client.start_turn_by_label(query, {"prompt": "Inspect the project"})
        await asyncio.wait_for(waiting.wait(), 3)
        fire_result.set()
        assert await asyncio.to_thread(write_entered.wait, 3)
        steer = await client.start_turn_by_label(query, {"prompt": "correction"})
        assert steer["confirmed"] and steer["steered"]
        assert steer["turnId"] == start["turnId"]
        release_write.set()
        await asyncio.wait_for(correction_waiting.wait(), 3)
        # The accepted query postdates the result; even a slow follow-up must
        # not be dropped by the short timeout for possibly coalesced queries.
        await asyncio.sleep(0.05)
        turn = store.get_turn(start["turnId"])
        assert turn.status == "running"
        assert turn.response_finished_at is None
        assert FakeClaudeSDKClient.disconnect_count == 0
        assert validated == []
        release_correction.set()
        await asyncio.wait_for(asyncio.gather(*client._turn_tasks), 3)
        assert responses == ["old result", "correction"]
        turn = store.get_turn(start["turnId"])
        assert turn.status == "completed", turn.last_error
        assert turn.last_useful_message == "done:correction"
        assert validated == (["done:correction"] if with_validator else [])
        assert not client._session_pending_results

        next_turn = await client.start_turn_by_label(query, {"prompt": "next"})
        await asyncio.wait_for(asyncio.gather(*client._turn_tasks), 3)
        assert store.get_turn(next_turn["turnId"]).last_useful_message == "done:next"
        assert responses == ["old result", "correction", "next"]
    finally:
        release_write.set()
        release_correction.set()
        await client.close()


@pytest.mark.parametrize("missing_result", [False, True])
async def test_query_accepted_during_coalesced_timeout_still_needs_a_result(tmp_path, missing_result):
    """A new query cannot inherit the timeout eligibility of an older one."""
    first_waiting, release_first = asyncio.Event(), asyncio.Event()
    drain_waiting, fresh_waiting, release_fresh = asyncio.Event(), asyncio.Event(), asyncio.Event()

    class Transport(FakeClaudeSDKClient):
        cycles = 0

        async def receive_response(self):
            self.cycles += 1
            if self.cycles == 1:
                first_waiting.set()
                await release_first.wait()
                yield FakeResultMessage("Original and early steer coalesced.")
            elif self.cycles == 2:
                drain_waiting.set()
                await asyncio.Event().wait()
            else:
                fresh_waiting.set()
                await release_fresh.wait()
                if not missing_result:
                    yield FakeResultMessage("Fresh correction handled.")

    class Sdk:
        ClaudeAgentOptions = FakeClaudeAgentOptions
        ClaudeSDKClient = Transport

    store = Store(tmp_path / "coalescing.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=Sdk)
    client._steer_drain_timeout_seconds = 0.1
    try:
        await client.start_thread({"name": "worker", "cwd": str(tmp_path)})
        query = LabelQueryInput(label="worker")
        start = await client.start_turn_by_label(query, {"prompt": "inspect"})
        await asyncio.wait_for(first_waiting.wait(), 3)
        assert (await client.steer_by_label(query, "early correction", {}))["confirmed"]
        release_first.set()
        await asyncio.wait_for(drain_waiting.wait(), 3)
        assert (await client.steer_by_label(query, "fresh correction", {}))["confirmed"]
        await asyncio.wait_for(fresh_waiting.wait(), 3)
        assert store.get_turn(start["turnId"]).status == "running"
        assert FakeClaudeSDKClient.disconnect_count == 0
        release_fresh.set()
        await asyncio.wait_for(asyncio.gather(*client._turn_tasks), 3)
        turn = store.get_turn(start["turnId"])
        assert turn.status == ("failed" if missing_result else "completed"), turn.last_error
        if missing_result:
            assert "without a terminal result" in turn.last_error
            assert turn.response_finished_at is None
        else:
            assert turn.last_useful_message == "Fresh correction handled."
    finally:
        release_first.set()
        release_fresh.set()
        await client.close()
