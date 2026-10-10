"""Managed idle continuation and non-forking foreign submission receipts."""

import asyncio
import json
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event

import pytest

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient
from super_agents.mcp_server import _tool_super_agents_steer
from test_claude_sdk import (
    FakeClaudeSDKClient,
    _InboxProbeServer,
    fake_sdk_loader,
    reset_fake_claude_sdk,  # noqa: F401
)


@pytest.mark.parametrize("backend", ["claude_code", "openbase_cloud"])
@pytest.mark.parametrize("second_wrapper", [False, True])
async def test_idle_followup_uses_retained_owner_with_live_inbox(tmp_path, monkeypatch, backend, second_wrapper):
    store = Store(tmp_path / "state.sqlite3")
    owner = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader, backend_identity=backend)
    caller = ClaudeAgentSdkClient(store=Store(store.path), sdk_loader=fake_sdk_loader, backend_identity=backend)
    thread = await owner.start_thread({"name": "birch", "cwd": str(tmp_path)})
    first = await owner.start_turn_by_label(LabelQueryInput(label="birch"), {"prompt": "inspect files"})
    await asyncio.gather(*owner._turn_tasks)
    session = store.get_session(thread["threadId"])
    retained = owner._sdk_clients[session.id]
    assert owner._active_owner(session.id) is None
    assert caller._transport_owner(session) is owner
    monkeypatch.setenv("CLAUDE_INBOX_REGISTRY_DIR", str(tmp_path))
    with tempfile.TemporaryDirectory(dir="/tmp") as sockets:
        sock = Path(sockets) / "s.sock"
        inbox = _InboxProbeServer(sock, session.backend_session_id)
        await inbox.start()
        (tmp_path / f"{session.backend_session_id}.json").write_text(json.dumps({"socket": str(sock)}))
        try:
            result = await _tool_super_agents_steer(caller if second_wrapper else owner).handler(
                {"name": "birch", "prompt": "explain validation"}
            )
            assert result["startedImmediately"] is True
            assert result["queued"] is False
            assert result["turnId"] != first["turnId"]
            assert result["turn"]["turnId"] == result["turnId"]
            assert Store(store.path).get_turn(result["turnId"]).prompt == "explain validation"
            await asyncio.gather(*owner._turn_tasks)
            assert store.get_turn(result["turnId"]).status == "completed"
            assert owner._sdk_clients[session.id] is retained
            assert caller._sdk_clients == {}
            assert len(FakeClaudeSDKClient.options_seen) == 1
            assert len(FakeClaudeSDKClient.prompts) == 2
            assert inbox.frames == []
        finally:
            await inbox.stop()
            await caller.close()
            await owner.close()
    assert caller._transport_owner(store.get_session(session.id)) is None


async def test_stale_leaf_and_shutdown_do_not_route_to_old_transport(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    owner = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    caller = ClaudeAgentSdkClient(store=Store(store.path), sdk_loader=fake_sdk_loader)
    thread = await owner.start_thread({"name": "birch", "cwd": str(tmp_path)})
    await owner.start_turn_by_label(LabelQueryInput(label="birch"), {"prompt": "one"})
    await asyncio.gather(*owner._turn_tasks)
    store.update_session(thread["threadId"], last_client_instance="another-owner")
    assert caller._transport_owner(store.get_session(thread["threadId"])) is None
    await owner.close()
    result = await caller.steer_by_label(LabelQueryInput(label="birch"), "two")
    await asyncio.gather(*caller._turn_tasks)
    assert result["startedImmediately"] is True
    assert len(FakeClaudeSDKClient.options_seen) == 2
    assert store.get_session(thread["threadId"]).last_client_instance == caller._instance_id
    await caller.close()
    assert caller._transport_owner(store.get_session(thread["threadId"])) is None


@pytest.mark.parametrize("same_wrapper", [False, True])
async def test_other_event_loop_owner_is_unavailable_without_inbox(tmp_path, monkeypatch, same_wrapper):
    monkeypatch.setenv("CLAUDE_INBOX_REGISTRY_DIR", str(tmp_path / "empty"))
    ready, finish = Event(), Event()
    owners = []
    path = tmp_path / "state.sqlite3"

    def run_owner():
        async def run():
            owner = ClaudeAgentSdkClient(store=Store(path), sdk_loader=fake_sdk_loader)
            await owner.start_thread({"name": "birch", "cwd": str(tmp_path)})
            await owner.start_turn_by_label(LabelQueryInput(label="birch"), {"prompt": "one"})
            await asyncio.gather(*owner._turn_tasks)
            owners.append(owner)
            ready.set()
            await asyncio.to_thread(finish.wait)
            await owner.close()
        asyncio.run(run())

    with ThreadPoolExecutor(max_workers=1) as pool:
        worker = pool.submit(run_owner)
        assert await asyncio.to_thread(ready.wait, 5)
        caller = owners[0] if same_wrapper else ClaudeAgentSdkClient(store=Store(path), sdk_loader=fake_sdk_loader)
        try:
            result = await caller.steer_by_label(LabelQueryInput(label="birch"), "two")
            assert result["delivery"] == "unavailable"
            assert result["reason"] == "owner_outside_execution_context"
            assert result["turnId"] is None
            assert not result["startedImmediately"]
            assert not result["queued"]
            direct = await caller.start_turn_by_label(LabelQueryInput(label="birch"), {"prompt": "two"})
            assert direct["delivery"] == "unavailable"
            assert len(caller.store.list_turns(caller.store.require_by_name("birch").id)) == 1
            assert len(FakeClaudeSDKClient.prompts) == 1
            if same_wrapper:
                with pytest.raises(RuntimeError, match="owning process and event loop"):
                    await caller.close()
                assert not caller._closed
            if not same_wrapper:
                assert caller._sdk_clients == {}
        finally:
            finish.set()
            if not same_wrapper:
                await caller.close()
            await asyncio.wrap_future(worker)


@pytest.mark.parametrize("busy", [False, True])
@pytest.mark.parametrize("drain_error", [False, True])
async def test_peer_consumes_then_closes_never_duplicates(tmp_path, monkeypatch, busy, drain_error):
    monkeypatch.setenv("CLAUDE_INBOX_REGISTRY_DIR", str(tmp_path))
    store = Store(tmp_path / "state.sqlite3")
    caller = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    thread = await caller.start_thread({"name": "birch", "cwd": str(tmp_path)})
    store.update_session(thread["threadId"], backend_session_id="external")
    active = store.create_turn(thread["threadId"], "working", status="running") if busy else None
    received = []
    if drain_error:
        drain = asyncio.StreamWriter.drain

        async def fail_after_drain(writer):
            await drain(writer)
            await asyncio.sleep(0.02)  # Let the real peer consume the written frame.
            raise TimeoutError("ambiguous drain acknowledgement")

        monkeypatch.setattr(asyncio.StreamWriter, "drain", fail_after_drain)

    async def consume(reader, writer):
        received.append(json.loads(await reader.readline()))
        writer.close()
        await writer.wait_closed()

    with tempfile.TemporaryDirectory(dir="/tmp") as sockets:
        sock = Path(sockets) / "s.sock"
        server = await asyncio.start_unix_server(consume, path=str(sock))
        (tmp_path / "external.json").write_text(json.dumps({"socket": str(sock)}))
        async with server:
            result = await caller.steer_by_label(LabelQueryInput(label="birch"), "followup")
        assert result["written"] is (not drain_error)
        assert result["mayHaveBeenWritten"] is True
        assert result["reason"] == ("write_failed" if drain_error else "peer_closed_without_ack")
        assert result["messageId"] == received[0]["msg_id"]
        assert result["turnId"] is None
        assert result["activeTurnId"] == (active.id if active else None)
        assert not result["steered"] and not result["confirmed"] and not result["queued"]
        assert len(store.list_turns(thread["threadId"])) == int(busy)
        assert FakeClaudeSDKClient.prompts == []
        assert (tmp_path / "external.json").exists()
    await caller.close()


async def test_real_foreign_process_inbox_is_not_managed_continuation(tmp_path, monkeypatch):
    """A completed retained SDK in a different process cannot be borrowed."""
    monkeypatch.setenv("CLAUDE_INBOX_REGISTRY_DIR", str(tmp_path))
    script = '''
import asyncio,json,sys
from pathlib import Path
from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient
from test_claude_sdk import fake_sdk_loader, FakeClaudeSDKClient
import super_agents.claude_options as options
options.claude_state_path = lambda: Path(sys.argv[1]) / "isolated-claude-state.json"
async def main():
    root, sock = Path(sys.argv[1]), sys.argv[2]
    owner = ClaudeAgentSdkClient(store=Store(root / "state.sqlite3"), sdk_loader=fake_sdk_loader)
    thread = await owner.start_thread({"name":"birch", "cwd":str(root)})
    first = await owner.start_turn_by_label(LabelQueryInput(label="birch"), {"prompt":"one"})
    await asyncio.gather(*owner._turn_tasks)
    native = owner.store.get_session(thread["threadId"]).backend_session_id
    frames = []
    async def consume(reader, writer):
        frames.append(json.loads(await reader.readline()))
        writer.close()
        await writer.wait_closed()
    server = await asyncio.start_unix_server(consume, path=sock)
    (root / f"{native}.json").write_text(json.dumps({"socket":sock}))
    print(json.dumps(first), flush=True)
    await asyncio.to_thread(sys.stdin.readline)
    server.close()
    await server.wait_closed()
    await owner.close()
    print(json.dumps({"frames":frames,"queries":len(FakeClaudeSDKClient.prompts)}), flush=True)
asyncio.run(main())
'''
    # Supply only the isolated source/tests to the fixture, never the real SDK.
    import os
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(Path(__file__).parent.parent / "src"), str(Path(__file__).parent)])}
    with tempfile.TemporaryDirectory(dir="/tmp") as sockets:
        process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path), f"{sockets}/s.sock"],
                                   stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        try:
            first = json.loads(await asyncio.wait_for(asyncio.to_thread(process.stdout.readline), 10))
            caller = ClaudeAgentSdkClient(store=Store(tmp_path / "state.sqlite3"), sdk_loader=fake_sdk_loader)
            result = await caller.steer_by_label(LabelQueryInput(label="birch"), "two")
            assert result["delivery"] == "inbox" and result["confirmed"] is False
            assert result["turnId"] is None and not result["startedImmediately"]
            assert not result["steered"] and not result["queued"]
            assert len(caller.store.list_turns(first["threadId"])) == 1
            assert caller._sdk_clients == {}
            await caller.close()
            process.stdin.write("done\n")
            process.stdin.flush()
            output, errors = await asyncio.to_thread(process.communicate, timeout=10)
            assert process.returncode == 0, errors
            receipt = json.loads(output)
            assert receipt["queries"] == 1
            assert len(receipt["frames"]) == 1
            assert receipt["frames"][0]["msg_id"] == result["messageId"]
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
