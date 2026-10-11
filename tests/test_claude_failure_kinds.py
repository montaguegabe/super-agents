from __future__ import annotations

import pytest

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_failure_kinds import BACKEND_UNAVAILABLE, START_TIMEOUT, classify_turn_failure
from super_agents.claude_sdk import ClaudeAgentSdkClient
from test_claude_sdk import FakeClaudeSDKClient, FakeSdk, reset_fake_claude_sdk, wait_for  # noqa: F401


def test_classify_known_failures():
    assert (
        classify_turn_failure(RuntimeError("Claude Code did not start in /x: Control request timeout: initialize"))
        == START_TIMEOUT
    )

    class CLIConnectionError(Exception):
        pass

    wrapped = RuntimeError("Claude Code did not start in /x: boom")
    wrapped.__cause__ = CLIConnectionError("boom")
    assert classify_turn_failure(wrapped) == BACKEND_UNAVAILABLE
    assert (
        classify_turn_failure(RuntimeError("Codex app-server is not running or not reachable at /s"))
        == BACKEND_UNAVAILABLE
    )
    # The CLI killed mid-turn (steer harness injection, 2026-10-11).
    assert (
        classify_turn_failure(RuntimeError("Command failed with exit code -9 (exit code: -9)")) == BACKEND_UNAVAILABLE
    )
    assert classify_turn_failure(RuntimeError("stream ended without a terminal result")) is None


@pytest.mark.asyncio
async def test_failed_start_records_a_kind_on_the_turn(tmp_path):
    class HangingStartClient(FakeClaudeSDKClient):
        async def connect(self) -> None:
            raise Exception("Control request timeout: initialize")

    class HangingSdk(FakeSdk):
        ClaudeSDKClient = HangingStartClient

    store = Store(tmp_path / "kind.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=HangingSdk)
    thread_id = (await client.start_thread({"name": "kind", "cwd": str(tmp_path)}))["threadId"]
    result = await client.start_turn_by_label(LabelQueryInput(thread_id=thread_id), {"prompt": "hello"})
    await wait_for(lambda: store.get_turn(result["turnId"]).status == "failed")
    turn = store.get_turn(result["turnId"])
    assert turn.error_kind == START_TIMEOUT
    assert "did not start" in (turn.last_error or "")
    await client.close()
