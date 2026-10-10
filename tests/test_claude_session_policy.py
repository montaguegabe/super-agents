from types import SimpleNamespace

import pytest

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient
from super_agents.claude_system_prompt import compose_system_prompt, supports_refreshable_system_prompt
from test_claude_sdk import (
    FakeClaudeSDKClient,
    FakeSdk,
    reset_fake_claude_sdk as reset_fake_claude_sdk,
    wait_for,
)


class SnapshotPrompt:
    snapshot: bool


class CurrentSdk(FakeSdk):
    types = SimpleNamespace(SystemPromptPreset=SnapshotPrompt, SystemPromptCustom=SnapshotPrompt)


@pytest.mark.parametrize("base", [None, {"type": "preset", "preset": "claude_code"}])
def test_older_sdk_keeps_legacy_path(base):
    assert not supports_refreshable_system_prompt(FakeSdk())
    assert compose_system_prompt(base, "Session rules", FakeSdk()) == base
    assert compose_system_prompt(base, None, CurrentSdk()) == base


def test_composition_preserves_stock_base_and_explicit_quiet():
    base = {"type": "preset", "preset": "claude_code", "append": "Global rules."}
    quiet = "Announce work by default, except when the user requests silence.\nThis task is text-only: no speech."
    result = compose_system_prompt(base, quiet, CurrentSdk())
    assert result == {**base, "append": "Global rules.\n\n" + quiet, "snapshot": False}
    assert "snapshot" not in base


def test_replace_mode_keeps_custom_base(tmp_path):
    path = tmp_path / "base.md"
    path.write_text("Custom replacement base.")
    assert compose_system_prompt({"type": "file", "path": str(path)}, "Session policy", CurrentSdk()) == {
        "type": "custom", "prompt": "Custom replacement base.\n\nSession policy", "snapshot": False,
    }


async def _turn(client, store, thread_id, prompt="Read the existing file.", **extra):
    result = await client.start_turn_by_label(LabelQueryInput(thread_id=thread_id), {"prompt": prompt, **extra})
    await wait_for(lambda: store.get_turn(result["turnId"]).status == "completed")
    return result


@pytest.mark.asyncio
async def test_fresh_worker_policy_and_identity_are_system_not_user(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPER_AGENTS_THREAD_INTRO_COMMAND", "")
    base = tmp_path / "base.md"
    base.write_text("Global base.")
    monkeypatch.setenv("SUPER_AGENTS_BASE_INSTRUCTIONS_PATH", str(base))
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=CurrentSdk)
    thread = await client.start_thread({
        "name": "reader", "agentName": "Cooper", "cwd": str(tmp_path),
        "developerInstructions": "Report completion. Respect explicit quiet requests.",
    })
    await _turn(client, store, thread["threadId"], "Read the file silently. No speech.")
    options = FakeClaudeSDKClient.options_seen[-1].kwargs
    system = options["system_prompt"]
    assert system["type"] == "preset" and system["preset"] == "claude_code"
    assert system["snapshot"] is False
    assert system["append"].startswith("Global base.\n\nReport completion.")
    assert "Respect explicit quiet requests." in system["append"]
    assert "Your name is Cooper." in system["append"]
    assert thread["threadId"] in system["append"]
    query = FakeClaudeSDKClient.prompts[-1]
    assert query.endswith("Read the file silently. No speech.")
    assert "Your name is Cooper." not in query
    assert "Report completion." not in query
    assert f"Current working directory: {tmp_path}" in query
    await client.close()


@pytest.mark.asyncio
async def test_resumed_policy_identity_changes_refresh_without_new_session(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=CurrentSdk)
    thread = await client.start_thread({
        "name": "reader", "agentName": "Old Name", "cwd": str(tmp_path),
        "developerInstructions": "Old rules.",
    })
    sid = thread["threadId"]
    first = await _turn(client, store, sid)
    backend_id = store.get_session(sid).backend_session_id
    old_client = client._sdk_clients[sid]
    await _turn(client, store, sid, "Follow up.")
    assert client._sdk_clients[sid] is old_client
    store.update_session(sid, agent_name="Current Name", developer_instructions="Current rules. No speech.")
    await _turn(client, store, sid, "Continue quietly.")
    new_client = client._sdk_clients[sid]
    assert new_client is not old_client and not old_client.connected
    assert new_client.options.kwargs["resume"] == backend_id
    assert store.get_session(sid).backend_session_id == backend_id
    system = new_client.options.kwargs["system_prompt"]
    assert system["snapshot"] is False
    assert "Current rules. No speech." in system["append"]
    assert "Your name is Current Name." in system["append"]
    assert "Old Name" not in system["append"] and "Old rules" not in system["append"]
    assert store.get_turn(first["turnId"]).status == "completed"
    await client.close()


@pytest.mark.asyncio
async def test_active_quiet_steer_preserves_client_and_pending_context(tmp_path):
    FakeClaudeSDKClient.blocked_prompts = {"first"}
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=CurrentSdk)
    sid = (await client.start_thread({
        "name": "reader", "cwd": str(tmp_path),
        "developerInstructions": "Respect explicit requests for silence.",
    }))["threadId"]
    first = await client.start_turn_by_label(LabelQueryInput(thread_id=sid), {"prompt": "first"})
    await wait_for(lambda: bool(FakeClaudeSDKClient.prompts))
    existing = client._sdk_clients[sid]
    followup = await client.start_turn_by_label(LabelQueryInput(thread_id=sid), {
        "prompt": "Finish silently.", "developerInstructions": "No speech for this update.",
    })
    await wait_for(lambda: store.get_turn(first["turnId"]).status == "completed")
    assert followup["turnId"] == first["turnId"]
    assert client._sdk_clients[sid] is existing
    assert FakeClaudeSDKClient.disconnect_count == 0
    assert "No speech for this update." in FakeClaudeSDKClient.prompts[-1]
    assert "Finish silently." in store.get_turn(first["turnId"]).last_useful_message
    await client.close()


@pytest.mark.asyncio
async def test_new_process_resumes_same_history_with_latest_stored_policy(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=CurrentSdk)
    sid = (await client.start_thread({"name": "reader", "cwd": str(tmp_path)}))["threadId"]
    await _turn(client, store, sid)
    backend_id = store.get_session(sid).backend_session_id
    await client.close()
    store.update_session(sid, developer_instructions="Updated instructions.", agent_name="Robin")
    resumed = ClaudeAgentSdkClient(store=store, sdk_loader=CurrentSdk)
    await _turn(resumed, store, sid, "Remember our earlier work.")
    options = FakeClaudeSDKClient.options_seen[-1].kwargs
    assert options["resume"] == backend_id
    assert options["system_prompt"]["snapshot"] is False
    assert "Updated instructions." in options["system_prompt"]["append"]
    assert "Your name is Robin." in options["system_prompt"]["append"]
    assert store.get_session(sid).backend_session_id == backend_id
    await resumed.close()


@pytest.mark.asyncio
async def test_turn_overlay_is_scoped_and_warm_options_match(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=CurrentSdk)
    thread = await client.start_thread({
        "name": "reader", "cwd": str(tmp_path), "developerInstructions": "Persistent policy.",
    })
    sid = thread["threadId"]
    turn_options = {"developerInstructions": "Only this turn: stay silent."}
    await client.warm_session_by_label(LabelQueryInput(thread_id=sid), turn_options)
    warmed = client._sdk_clients[sid]
    await _turn(client, store, sid, **turn_options)
    assert client._sdk_clients[sid] is warmed
    assert "Only this turn: stay silent." in warmed.options.kwargs["system_prompt"]["append"]
    assert "Only this turn" not in store.get_session(sid).developer_instructions
    await _turn(client, store, sid, "Next task.")
    assert client._sdk_clients[sid] is not warmed
    assert "Only this turn" not in client._sdk_clients[sid].options.kwargs["system_prompt"]["append"]
    await client.close()


@pytest.mark.asyncio
async def test_base_change_reconnects_cached_client(tmp_path, monkeypatch):
    base = tmp_path / "base.md"
    base.write_text("Base version one.")
    monkeypatch.setenv("SUPER_AGENTS_BASE_INSTRUCTIONS_PATH", str(base))
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=CurrentSdk)
    sid = (await client.start_thread({"name": "reader", "cwd": str(tmp_path)}))["threadId"]
    await _turn(client, store, sid)
    old = client._sdk_clients[sid]
    base.write_text("Base version two.")
    await _turn(client, store, sid)
    assert client._sdk_clients[sid] is not old
    assert "Base version two." in client._sdk_clients[sid].options.kwargs["system_prompt"]["append"]
    await client.close()
