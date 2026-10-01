from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
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


def _touch(path: Path, text: str, mtime_ns: int) -> None:
    path.write_text(text)
    os.utime(path, ns=(mtime_ns, mtime_ns))


async def _deliveries(client: EventClientStub, name: str | None = None) -> list[dict]:
    return [item for item in await client.sweep_file_triggers(name=name) if item["status"] == "delivered"]


@pytest.mark.asyncio
async def test_file_trigger_validates_watch_path_and_needs_no_allowlist(tmp_path: Path) -> None:
    client = EventClientStub(tmp_path / "state.json")
    await client.save_routine({"name": "flag-loop", "prompt": "Handle the flag.", "scheduleType": "interval"})

    with pytest.raises(ValueError, match="watchPath"):
        await client.add_routine_trigger("flag-loop", {"type": "file"})
    with pytest.raises(ValueError, match="absolute"):
        await client.add_routine_trigger("flag-loop", {"type": "file", "watchPath": "relative/*.md"})
    with pytest.raises(ValueError, match="type"):
        await client.add_routine_trigger("flag-loop", {"type": "cron"})

    created = await client.add_routine_trigger("flag-loop", {"type": "file", "watchPath": "~/inbox/*.md"})
    trigger = created["trigger"]
    assert trigger["type"] == "file"
    assert trigger["watchPath"] == str(Path.home() / "inbox" / "*.md")
    assert trigger.get("token") is None
    assert trigger.get("seenFiles") is None


@pytest.mark.asyncio
async def test_file_trigger_baselines_then_fires_on_create_modify_and_recreate(tmp_path: Path) -> None:
    client = EventClientStub(tmp_path / "state.json")
    await make_command_loop(client)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox / "old.md", "already there", 1_000_000_000_000_000_000)
    await client.add_routine_trigger("echo-loop", {"type": "file", "watchPath": str(inbox / "*.md")})

    # First sweep records the existing file without firing.
    assert await client.sweep_file_triggers() == []
    seen = read_state_file(tmp_path / "state.json").routines["echo-loop"].triggers[0].seen_files
    assert seen == {str(inbox / "old.md"): 1_000_000_000_000_000_000}

    _touch(inbox / "review-request.md", "Please review.", 1_000_000_001_000_000_000)
    delivered = await _deliveries(client)
    assert len(delivered) == 1
    event = json.loads(delivered[0]["run"]["stdout"].strip())
    assert event["origin"] == "local"
    assert event["id"] == f"file:{inbox / 'review-request.md'}@1000000001000000000"
    assert event["payload"] == {
        "path": str(inbox / "review-request.md"),
        "name": "review-request.md",
        "dir": str(inbox),
        "mtime": 1_000_000_001,
        "change": "created",
        "contents": "Please review.",
    }

    # Unchanged files stay quiet; a touch fires again as "modified".
    assert await _deliveries(client) == []
    _touch(inbox / "review-request.md", "Please review again.", 1_000_000_002_000_000_000)
    delivered = await _deliveries(client)
    assert len(delivered) == 1
    assert json.loads(delivered[0]["run"]["stdout"].strip())["payload"]["change"] == "modified"

    # Deleting forgets the file, so recreating it fires as "created" again.
    (inbox / "review-request.md").unlink()
    assert await client.sweep_file_triggers() == []
    _touch(inbox / "review-request.md", "Third time.", 1_000_000_003_000_000_000)
    delivered = await _deliveries(client)
    assert len(delivered) == 1
    assert json.loads(delivered[0]["run"]["stdout"].strip())["payload"]["change"] == "created"

    routine = read_state_file(tmp_path / "state.json").routines["echo-loop"]
    assert routine.triggers[0].event_count == 3
    # Event runs never consume the schedule.
    assert routine.last_run_at is None


@pytest.mark.asyncio
async def test_file_trigger_fire_existing_filters_and_disabled_loops(tmp_path: Path) -> None:
    client = EventClientStub(tmp_path / "state.json")
    await make_command_loop(client)
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    _touch(inbox / "qa-request.md", "qa", 1_000_000_000_000_000_000)
    _touch(inbox / "notes.md", "notes", 1_000_000_000_000_000_000)
    await client.add_routine_trigger(
        "echo-loop",
        {
            "type": "file",
            "watchPath": str(inbox / "*.md"),
            "fireExisting": True,
            "filters": [{"path": "name", "op": "endsWith", "value": "-request.md"}],
        },
    )

    results = await client.sweep_file_triggers()
    assert [(item["status"], Path(item["eventId"].split("@")[0][5:]).name) for item in results] == [
        ("filtered", "notes.md"),
        ("delivered", "qa-request.md"),
    ]

    await client.save_routine({"name": "echo-loop", "enabled": False})
    _touch(inbox / "merge-request.md", "merge", 1_000_000_001_000_000_000)
    # A disabled loop still tracks files (so nothing fires later for stale changes) but runs nothing.
    assert await client.sweep_file_triggers() == []
    seen = read_state_file(tmp_path / "state.json").routines["echo-loop"].triggers[0].seen_files
    assert str(inbox / "merge-request.md") in seen


@pytest.mark.asyncio
async def test_run_due_routines_sweeps_file_triggers_except_forced_runs(tmp_path: Path) -> None:
    client = EventClientStub(tmp_path / "state.json")
    await make_command_loop(client, time="23:59", scheduleType="daily")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    await client.add_routine_trigger(
        "echo-loop", {"type": "file", "watchPath": str(inbox / "*.md"), "fireExisting": True}
    )
    _touch(inbox / "a.md", "a", 1_000_000_000_000_000_000)

    forced = await client.run_due_routines(name="echo-loop", force=True)
    assert [item.get("status") for item in forced["results"]] == [None]

    swept = await client.run_due_routines()
    assert [item["status"] for item in swept["results"]] == ["delivered"]
    assert json.loads(swept["results"][0]["run"]["stdout"].strip())["payload"]["name"] == "a.md"
