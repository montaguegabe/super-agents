from types import SimpleNamespace

import pytest

from super_agents.agent_store import Store
from super_agents.claude_sdk import ClaudeAgentSdkClient
from super_agents.claude_turn_output import read_turn_output


@pytest.mark.asyncio
async def test_intermediate_code_survives_final_reply_reconnect_and_cancellation(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    session = store.create_session(name="auth", cwd=str(tmp_path), command=[])
    turn = store.create_turn(session.id, "Sign in", status="running")
    client = ClaudeAgentSdkClient(store=store)
    for text in ["Device code: EXAMPLE", "Approve on your phone."]:
        await client._consume_stream_message(
            session, session.id, turn.id, SimpleNamespace(content=[SimpleNamespace(text=text)]), {}
        )
    await client._consume_stream_message(
        session, session.id, turn.id, SimpleNamespace(result="Approve on your phone.", num_turns=1), {}
    )
    await client._consume_stream_message(
        session,
        session.id,
        turn.id,
        SimpleNamespace(content=[SimpleNamespace(text="private tool output")], tool_use_result={}),
        {},
    )
    store.update_turn(turn.id, status="cancelled")
    await client.close()
    reopened = Store(store.path)
    assert [item["text"] for item in read_turn_output(reopened, turn.id)] == [
        "Device code: EXAMPLE",
        "Approve on your phone.",
    ]
    assert reopened.get_turn(turn.id).last_useful_message == "Approve on your phone."


@pytest.mark.asyncio
async def test_new_tool_work_clears_response_finished_marker(tmp_path):
    store = Store(tmp_path / "state.sqlite3")
    session = store.create_session(name="auth", cwd=str(tmp_path), command=[])
    turn = store.create_turn(session.id, "Sign in", status="running")
    store.update_turn(turn.id, response_finished_at="2026-10-10T04:16:10Z")
    client = ClaudeAgentSdkClient(store=store)
    await client._consume_stream_message(
        session,
        session.id,
        turn.id,
        SimpleNamespace(content=[SimpleNamespace(name="Bash", input={"command": "work"})]),
        {},
    )
    assert store.get_turn(turn.id).response_finished_at is None
    assert read_turn_output(store, turn.id) == []
    await client.close()


def test_steers_remain_between_replies_after_reopen(tmp_path):
    from super_agents.claude_turn_output import append_turn_output, ordered_turn_items

    store = Store(tmp_path / "state.sqlite3")
    session = store.create_session(name="chat", cwd=str(tmp_path), command=[])
    turn = store.create_turn(session.id, "Hello", status="running")
    append_turn_output(store, turn.id, "First answer")
    store.append_turn_steer(turn.id, "Follow up")
    store.append_turn_steer(turn.id, "Clarification")
    append_turn_output(store, turn.id, "Second answer")
    store.update_turn(turn.id, status="cancelled")
    reopened = Store(store.path)
    items = ordered_turn_items(reopened, reopened.get_turn(turn.id))
    assert [item.get("text") or item["content"][0]["text"] for item in items] == [
        "Hello",
        "First answer",
        "Follow up",
        "Clarification",
        "Second answer",
    ]
    assert len({item["id"] for item in items}) == 5
