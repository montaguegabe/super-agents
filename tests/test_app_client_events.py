from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
from pathlib import Path

import pytest

from super_agents.app_client_events import EventClientMixin
from super_agents.app_client_routines import RoutineClientMixin
from super_agents.state import read_state_file


class EventClientStub(RoutineClientMixin, EventClientMixin):
    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self._state_lock = asyncio.Lock()

    async def read_state(self):
        return read_state_file(self.state_file)


async def make_command_loop(client: EventClientStub, name: str = "echo-loop", **extra) -> None:
    await client.save_routine(
        {
            "name": name,
            "kind": "command",
            "command": "printenv SUPER_AGENTS_EVENT_JSON",
            "scheduleType": "interval",
            "intervalSeconds": 3600,
            **extra,
        }
    )


@pytest.mark.asyncio
async def test_agent_loop_webhook_trigger_requires_sender_allowlist(tmp_path: Path) -> None:
    client = EventClientStub(tmp_path / "state.json")
    await client.save_routine({"name": "pr-loop", "prompt": "Handle PR feedback.", "scheduleType": "interval"})

    with pytest.raises(ValueError, match="senderAllowlist"):
        await client.add_routine_trigger("pr-loop", {"description": "GitHub PR comments"})

    result = await client.add_routine_trigger(
        "pr-loop",
        {"senderPath": "sender.id", "senderAllowlist": ["12345"]},
    )
    assert result["trigger"]["token"]
    assert result["trigger"]["id"].startswith("trg-")


@pytest.mark.asyncio
async def test_webhook_delivery_runs_command_loop_with_event_context(tmp_path: Path) -> None:
    client = EventClientStub(tmp_path / "state.json")
    await make_command_loop(client)
    created = await client.add_routine_trigger(
        "echo-loop",
        {"filters": [{"path": "action", "op": "equals", "value": "created"}]},
    )
    token = created["trigger"]["token"]

    body = json.dumps({"action": "created", "comment": {"body": "/openbase fix tests"}}).encode("utf-8")
    result = await client.deliver_webhook_event(token, headers={"X-GitHub-Delivery": "guid-1"}, body=body)

    assert result["status"] == "delivered"
    assert result["eventId"] == "guid-1"
    event = json.loads(result["run"]["stdout"].strip())
    assert event["payload"]["action"] == "created"

    state = read_state_file(tmp_path / "state.json")
    routine = state.routines["echo-loop"]
    trigger = (routine.triggers or [])[0]
    assert trigger.event_count == 1
    assert trigger.last_event_id == "guid-1"
    # Event runs must not consume the schedule: lastRunAt/lastRunDate untouched.
    assert routine.last_run_at is None
    assert routine.last_run_date is None
    assert routine.last_status == "completed"


@pytest.mark.asyncio
async def test_webhook_delivery_dedupes_filters_and_rejects_unknown_tokens(tmp_path: Path) -> None:
    client = EventClientStub(tmp_path / "state.json")
    await make_command_loop(client)
    created = await client.add_routine_trigger(
        "echo-loop",
        {"filters": [{"path": "action", "op": "equals", "value": "created"}]},
    )
    token = created["trigger"]["token"]
    body = json.dumps({"action": "created"}).encode("utf-8")

    assert (await client.deliver_webhook_event("0" * 32, body=body))["status"] == "unknown_token"

    first = await client.deliver_webhook_event(token, headers={"X-GitHub-Delivery": "guid-2"}, body=body)
    duplicate = await client.deliver_webhook_event(token, headers={"X-GitHub-Delivery": "guid-2"}, body=body)
    assert first["status"] == "delivered"
    assert duplicate["status"] == "duplicate"

    filtered = await client.deliver_webhook_event(
        token,
        headers={"X-GitHub-Delivery": "guid-3"},
        body=json.dumps({"action": "deleted"}).encode("utf-8"),
    )
    assert filtered["status"] == "filtered"


@pytest.mark.asyncio
async def test_webhook_delivery_verifies_hmac_signature(tmp_path: Path) -> None:
    client = EventClientStub(tmp_path / "state.json")
    await make_command_loop(client)
    created = await client.add_routine_trigger("echo-loop", {"hmacSecret": "shhh"})
    token = created["trigger"]["token"]
    body = json.dumps({"action": "created"}).encode("utf-8")
    signature = "sha256=" + hmac.new(b"shhh", body, hashlib.sha256).hexdigest()

    bad = await client.deliver_webhook_event(token, headers={"X-Hub-Signature-256": "sha256=" + "0" * 64}, body=body)
    missing = await client.deliver_webhook_event(token, body=body)
    good = await client.deliver_webhook_event(
        token, headers={"X-Hub-Signature-256": signature, "X-GitHub-Delivery": "guid-4"}, body=body
    )

    assert bad["status"] == "rejected"
    assert missing["status"] == "rejected"
    assert good["status"] == "delivered"


@pytest.mark.asyncio
async def test_sender_allowlist_is_enforced_when_configured(tmp_path: Path) -> None:
    client = EventClientStub(tmp_path / "state.json")
    await make_command_loop(client)
    created = await client.add_routine_trigger(
        "echo-loop",
        {"senderPath": "sender.id", "senderAllowlist": ["12345"]},
    )
    token = created["trigger"]["token"]

    denied = await client.deliver_webhook_event(
        token,
        headers={"X-GitHub-Delivery": "guid-5"},
        body=json.dumps({"sender": {"id": 999}}).encode("utf-8"),
    )
    allowed = await client.deliver_webhook_event(
        token,
        headers={"X-GitHub-Delivery": "guid-6"},
        body=json.dumps({"sender": {"id": 12345}}).encode("utf-8"),
    )

    assert denied["status"] == "unauthorized_sender"
    assert allowed["status"] == "delivered"

    state = read_state_file(tmp_path / "state.json")
    routine = state.routines["echo-loop"]
    assert routine.last_status == "completed"


@pytest.mark.asyncio
async def test_external_events_never_start_agent_loops_without_allowlist(tmp_path: Path) -> None:
    client = EventClientStub(tmp_path / "state.json")
    await client.save_routine({"name": "pr-loop", "prompt": "Handle PR feedback.", "scheduleType": "interval"})
    created = await client.add_routine_trigger(
        "pr-loop",
        {"senderPath": "sender.id", "senderAllowlist": ["12345"]},
    )
    token = created["trigger"]["token"]

    # Simulate an allowlist wiped by a raw state edit: delivery must still deny.
    async with client._state_lock:
        state = read_state_file(client.state_file)
        trigger = (state.routines["pr-loop"].triggers or [])[0]
        trigger.sender_allowlist = None
        trigger.sender_path = None
        from super_agents.state import write_state_file

        write_state_file(client.state_file, state)

    result = await client.deliver_webhook_event(
        token,
        headers={"X-GitHub-Delivery": "guid-7"},
        body=json.dumps({"sender": {"id": 12345}}).encode("utf-8"),
    )
    assert result["status"] == "unauthorized_sender"


@pytest.mark.asyncio
async def test_emit_and_remove_trigger(tmp_path: Path) -> None:
    client = EventClientStub(tmp_path / "state.json")
    await make_command_loop(client)

    emitted = await client.emit_routine_event("echo-loop", {"note": "manual run"})
    assert emitted["status"] == "delivered"
    event = json.loads(emitted["run"]["stdout"].strip())
    assert event["origin"] == "local"
    assert event["payload"] == {"note": "manual run"}

    created = await client.add_routine_trigger("echo-loop", {})
    trigger_id = created["trigger"]["id"]
    removed = await client.remove_routine_trigger("echo-loop", trigger_id)
    assert removed["deleted"] is True
    assert not read_state_file(tmp_path / "state.json").routines["echo-loop"].triggers

    with pytest.raises(ValueError):
        await client.remove_routine_trigger("echo-loop", trigger_id)
