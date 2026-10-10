"""Application callbacks run at SDK connection and verified turn boundaries."""

import asyncio
import json
from types import SimpleNamespace

import pytest

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient
from test_claude_sdk import (
    BackgroundTaskClaudeSDKClient,
    BackgroundTaskSdk,
    FakeClaudeAgentOptions,
    FakeClaudeSDKClient,
    FifoStreamClaudeSDKClient,
    FifoStreamSdk,
    fake_sdk_loader,
    reset_fake_claude_sdk,  # noqa: F401 -- isolate managed configuration
    wait_for,
)


@pytest.mark.parametrize("permission_mode", ["default", "bypassPermissions"])
async def test_configure_preserves_options_and_runs_before_fresh_and_resumed_connect(
    tmp_path, monkeypatch, permission_mode,
):
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_PERMISSION_MODE", permission_mode)
    existing_server = {"command": "unrelated-server"}
    (tmp_path / "isolated-claude-state.json").write_text(json.dumps({"mcpServers": {
        "existing": existing_server, "super-agents": {"command": "super-agents-mcp"},
    }}))
    existing_hook = object()
    added_hook = object()
    calls = []

    class Options(FakeClaudeAgentOptions):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.__dict__.update(kwargs)
            self.hooks = {"Existing": [existing_hook]}

    class SDKClient(FakeClaudeSDKClient):
        def __init__(self, options):
            assert options.hooks == {"Existing": [existing_hook], "Stop": [added_hook]}
            assert options.mcp_servers["existing"] == existing_server
            assert options.mcp_servers["super-agents"]["type"] == "sdk"
            assert options.mcp_servers["extension"] == {"type": "sdk", "name": "extension"}
            assert options.cwd == str(tmp_path)
            assert options.model == "model-a"
            assert options.disallowed_tools == ["Task"]
            assert options.setting_sources == ["user", "project"]
            assert options.permission_mode == permission_mode
            assert callable(getattr(options, "can_use_tool", None)) == (permission_mode != "bypassPermissions")
            super().__init__(options)

    sdk = SimpleNamespace(ClaudeAgentOptions=Options, ClaudeSDKClient=SDKClient)

    def configure(session, options, actual_sdk):
        assert actual_sdk is sdk
        calls.append((session.id, options.kwargs.get("resume")))
        options.hooks.setdefault("Stop", []).append(added_hook)
        options.mcp_servers["extension"] = {"type": "sdk", "name": "extension"}

    store = Store(tmp_path / "store.sqlite3")
    client = ClaudeAgentSdkClient(
        store=store, sdk_loader=lambda: sdk,
        disallowed_tools_for_session=lambda _: ("Task",), configure_session=configure,
    )
    thread = await client.start_thread({"name": "worker", "cwd": str(tmp_path)})
    session = store.get_session(thread["threadId"])
    try:
        first = await client._sdk_client_for(session, "model-a", None, None, sdk)
        assert first.connected
        assert await client._sdk_client_for(session, "model-a", None, None, sdk) is first
        assert calls == [(session.id, None)]
        await client._disconnect_sdk_client(session.id)
        session = store.update_session(session.id, backend_session_id="saved-conversation")
        resumed = await client._sdk_client_for(session, "model-a", None, None, sdk)
        assert resumed is not first and resumed.connected
        assert calls == [(session.id, None), (session.id, "saved-conversation")]
    finally:
        await client.close()


async def test_validator_runs_after_background_work_and_before_completion(tmp_path):
    BackgroundTaskClaudeSDKClient.tasks_done = asyncio.Event()
    BackgroundTaskClaudeSDKClient.followup = True
    entered, release = asyncio.Event(), asyncio.Event()
    observed = []
    store = Store(tmp_path / "store.sqlite3")

    async def validate(session, turn, result):
        assert BackgroundTaskClaudeSDKClient.tasks_done.is_set()
        assert not client._session_pending_results
        assert turn.status == "running"
        assert turn == store.get_turn(turn.id)
        assert session == store.get_session(session.id)
        observed.append(result.result)
        entered.set()
        await release.wait()

    client = ClaudeAgentSdkClient(
        store=store, sdk_loader=lambda: BackgroundTaskSdk(), validate_turn_result=validate,
    )
    client._background_task_poll_seconds = 0.01
    thread = await client.start_thread({"name": "worker", "cwd": str(tmp_path)})
    turn_id = (await client.start_turn_by_label(LabelQueryInput(label="worker"), {"prompt": "inspect"}))["turnId"]
    try:
        await wait_for(lambda: "waiting on 2 background" in (store.get_session(thread["threadId"]).last_observed_state or ""))
        assert not entered.is_set()
        BackgroundTaskClaudeSDKClient.tasks_done.set()
        await asyncio.wait_for(entered.wait(), 2)
        assert store.get_turn(turn_id).status == "running"
        release.set()
        await asyncio.gather(*client._turn_tasks)
        assert store.get_turn(turn_id).status == "completed"
        assert observed == ["synthesized findings"]
    finally:
        release.set()
        BackgroundTaskClaudeSDKClient.tasks_done.set()
        await client.close()


@pytest.mark.parametrize("outcome", ["error", "missing", "cancelled", "unfinished_background"])
async def test_unverified_or_cancelled_results_skip_validation(tmp_path, outcome):
    calls = []
    store = Store(tmp_path / "store.sqlite3")

    class SDKClient(FakeClaudeSDKClient):
        async def receive_response(self):
            if outcome == "cancelled":
                store.update_turn(store.get_session(thread_id).active_turn_id, status="cancelled")
            if outcome == "unfinished_background":
                if getattr(self, "responded", False):
                    return
                self.responded = True
                yield SimpleNamespace(subtype="task_started", task_id="still-running", description="work")
            if outcome != "missing":
                yield SimpleNamespace(num_turns=1, result="result", is_error=outcome == "error")

    async def validate(*args):
        calls.append(args)

    sdk = SimpleNamespace(ClaudeAgentOptions=FakeClaudeAgentOptions, ClaudeSDKClient=SDKClient)
    client = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: sdk, validate_turn_result=validate)
    thread_id = (await client.start_thread({"name": "worker", "cwd": str(tmp_path)}))["threadId"]
    turn_id = (await client.start_turn_by_label(LabelQueryInput(label="worker"), {"prompt": "inspect"}))["turnId"]
    await asyncio.gather(*client._turn_tasks)
    assert calls == []
    assert store.get_turn(turn_id).status == ("cancelled" if outcome == "cancelled" else "failed")
    assert not client._sdk_clients
    assert store.get_session(thread_id).active_turn_id is None
    await client.close()


@pytest.mark.parametrize("outcome", ["failure", "cancelled", "new_response"])
async def test_validator_cannot_commit_success_after_failure_cancel_or_new_work(tmp_path, outcome):
    store = Store(tmp_path / "store.sqlite3")
    seen = []

    async def validate(session, turn, result):
        seen.append(turn.id)
        assert turn.status == "running"
        await asyncio.sleep(0)
        if outcome == "failure":
            raise ValueError("Application validation failed")
        if outcome == "cancelled":
            store.update_turn(turn.id, status="cancelled")
        else:
            client._register_pending_result(session.id)

    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader, validate_turn_result=validate)
    thread = await client.start_thread({"name": "worker", "cwd": str(tmp_path)})
    turn_id = (await client.start_turn_by_label(LabelQueryInput(label="worker"), {"prompt": "inspect"}))["turnId"]
    await asyncio.gather(*client._turn_tasks)
    turn = store.get_turn(turn_id)
    assert seen == [turn_id]
    assert turn.status == ("cancelled" if outcome == "cancelled" else "failed")
    if outcome == "failure":
        assert turn.last_error == "Application validation failed"
    assert not client._sdk_clients
    assert not client._session_pending_results
    assert store.get_session(thread["threadId"]).active_turn_id is None
    await client.close()


async def test_configuration_failure_uses_failed_turn_cleanup_without_constructing_sdk(tmp_path):
    store = Store(tmp_path / "store.sqlite3")

    def configure(*args):
        raise ValueError("Application configuration failed")

    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader, configure_session=configure)
    thread = await client.start_thread({"name": "worker", "cwd": str(tmp_path)})
    turn_id = (await client.start_turn_by_label(LabelQueryInput(label="worker"), {"prompt": "inspect"}))["turnId"]
    await asyncio.gather(*client._turn_tasks)
    turn = store.get_turn(turn_id)
    assert turn.status == "failed"
    assert turn.last_error == "Application configuration failed"
    assert not FakeClaudeSDKClient.options_seen
    assert not client._sdk_clients
    assert store.get_session(thread["threadId"]).active_turn_id is None
    await client.close()


async def test_validator_receives_current_turn_after_all_steered_responses(tmp_path):
    FifoStreamClaudeSDKClient.queue = []
    FifoStreamClaudeSDKClient.hold = True
    store = Store(tmp_path / "store.sqlite3")
    validated = []

    async def validate(session, turn, result):
        assert not FifoStreamClaudeSDKClient.queue
        assert not client._session_pending_results
        assert [item["text"] for item in turn.steers] == ["include sizes"]
        validated.append(result.result)

    client = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: FifoStreamSdk(), validate_turn_result=validate)
    await client.start_thread({"name": "worker", "cwd": str(tmp_path)})
    query = LabelQueryInput(label="worker")
    turn_id = (await client.start_turn_by_label(query, {"prompt": "inspect"}))["turnId"]
    await wait_for(lambda: bool(FifoStreamClaudeSDKClient.queue))
    try:
        assert (await client.steer_by_label(query, "include sizes", {}))["turnId"] == turn_id
        FifoStreamClaudeSDKClient.hold = False
        await asyncio.gather(*client._turn_tasks)
        assert validated == ["done:include sizes"]
        assert store.get_turn(turn_id).status == "completed"
    finally:
        FifoStreamClaudeSDKClient.hold = False
        await client.close()


async def test_cancellation_interrupts_awaiting_validator(tmp_path):
    store = Store(tmp_path / "store.sqlite3")
    entered, exited = asyncio.Event(), asyncio.Event()

    async def validate(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            exited.set()

    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader, validate_turn_result=validate)
    thread = await client.start_thread({"name": "worker", "cwd": str(tmp_path)})
    query = LabelQueryInput(label="worker")
    turn_id = (await client.start_turn_by_label(query, {"prompt": "inspect"}))["turnId"]
    await asyncio.wait_for(entered.wait(), 2)
    assert (await client.cancel_by_label(query))["cancelled"]
    await asyncio.wait_for(exited.wait(), 2)
    await asyncio.gather(*client._turn_tasks)
    assert store.get_turn(turn_id).status == "cancelled"
    assert store.get_session(thread["threadId"]).active_turn_id is None
    assert not client._sdk_clients
    await client.close()
