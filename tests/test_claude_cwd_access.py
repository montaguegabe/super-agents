from __future__ import annotations

from pathlib import Path

import pytest

from super_agents import claude_cwd_access
from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_cwd_access import (
    BLOCKED_PERMISSION_DIALOG,
    CwdAccessBlocked,
    await_cwd_access,
    blocked_message,
    protected_folder_name,
)
from super_agents.claude_sdk import ClaudeAgentSdkClient
from test_claude_sdk import fake_sdk_loader, reset_fake_claude_sdk, wait_for  # noqa: F401  (autouse fixture)


class FakeProbe:
    def __init__(self, *, done: bool = False) -> None:
        self._done = done
        self.error = None

    @property
    def done(self) -> bool:
        return self._done

    def release(self) -> None:
        self._done = True

    async def wait(self, seconds: float) -> bool:
        return self._done


def test_protected_folder_name_recognises_home_folders(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert protected_folder_name(str(tmp_path / "Desktop" / "proj")) == "Desktop"
    assert protected_folder_name(str(tmp_path / "Documents")) == "Documents"
    assert protected_folder_name(str(tmp_path / "Downloads" / "a" / "b")) == "Downloads"
    assert protected_folder_name(str(tmp_path / "code" / "proj")) is None
    assert protected_folder_name("/Volumes/USB/proj") == "an external volume"
    assert protected_folder_name(str(tmp_path / "Library" / "Mobile Documents" / "x")) == "iCloud Drive"
    assert "your Desktop folder" in blocked_message(str(tmp_path / "Desktop" / "proj"))
    assert "/elsewhere/proj" in blocked_message("/elsewhere/proj")


@pytest.mark.asyncio
async def test_accessible_folder_reports_nothing():
    calls: list[str] = []

    async def on_blocked(message: str) -> None:
        calls.append("blocked")

    async def on_unblocked() -> None:
        calls.append("unblocked")

    await await_cwd_access("/tmp", on_blocked=on_blocked, on_unblocked=on_unblocked, probe=FakeProbe(done=True))
    assert calls == []


@pytest.mark.asyncio
async def test_blocked_folder_reports_then_continues_once_released(monkeypatch):
    monkeypatch.setattr(claude_cwd_access, "POLL_SECONDS", 0.01)
    probe = FakeProbe()
    calls: list[str] = []

    async def on_blocked(message: str) -> None:
        calls.append(message)
        probe.release()

    async def on_unblocked() -> None:
        calls.append("unblocked")

    await await_cwd_access(
        "/tmp/x", on_blocked=on_blocked, on_unblocked=on_unblocked, probe=probe, grace_seconds=0, max_wait_seconds=1
    )
    assert calls == [blocked_message("/tmp/x"), "unblocked"]


@pytest.mark.asyncio
async def test_blocked_folder_fails_after_the_wait_limit(monkeypatch):
    monkeypatch.setattr(claude_cwd_access, "POLL_SECONDS", 0.01)

    async def on_blocked(message: str) -> None:
        pass

    async def on_unblocked() -> None:
        raise AssertionError("never released")

    with pytest.raises(CwdAccessBlocked, match="waiting for permission"):
        await await_cwd_access(
            "/tmp/x",
            on_blocked=on_blocked,
            on_unblocked=on_unblocked,
            probe=FakeProbe(),
            grace_seconds=0,
            max_wait_seconds=0.05,
        )


@pytest.mark.asyncio
async def test_turn_waits_out_the_consent_dialog_and_completes(monkeypatch, tmp_path):
    """VM2, 2026-10-11: a pending Desktop consent prompt hung every Claude
    Code start for a minute and failed the turn. The turn now reports the
    block within seconds and finishes by itself once the prompt is answered."""
    monkeypatch.setattr(claude_cwd_access, "POLL_SECONDS", 0.01)
    probe = FakeProbe()
    real_wait = claude_cwd_access.await_cwd_access

    async def wait_with_fake_probe(cwd, **kwargs):
        return await real_wait(cwd, probe=probe, **kwargs)

    monkeypatch.setattr("super_agents.claude_sdk.await_cwd_access", wait_with_fake_probe)
    events = []

    class EventClient(ClaudeAgentSdkClient):
        def handle_notification(self, method, params):
            events.append((method, params))

    store = Store(tmp_path / "consent.sqlite3")
    client = EventClient(store=store, sdk_loader=fake_sdk_loader)
    client._cwd_access_grace_seconds = 0
    client._cwd_access_max_wait_seconds = 2
    thread_id = (await client.start_thread({"name": "consent", "cwd": str(tmp_path)}))["threadId"]
    result = await client.start_turn_by_label(LabelQueryInput(thread_id=thread_id), {"prompt": "hello"})
    turn_id = result["turnId"]

    await wait_for(lambda: store.get_turn(turn_id).error_kind == BLOCKED_PERMISSION_DIALOG)
    assert store.get_turn(turn_id).status == "running"
    assert "Openbase Services needs access" in (store.get_session(thread_id).last_observed_state or "")
    blocked = [params for method, params in events if method == "turn/updated" and params.get("blocked")]
    assert blocked and blocked[0]["blocked"]["errorKind"] == BLOCKED_PERMISSION_DIALOG

    probe.release()
    await wait_for(lambda: store.get_turn(turn_id).status == "completed")
    assert store.get_turn(turn_id).error_kind is None
    await client.close()


@pytest.mark.asyncio
async def test_turn_fails_with_the_blocked_kind_when_the_dialog_is_never_answered(monkeypatch, tmp_path):
    monkeypatch.setattr(claude_cwd_access, "POLL_SECONDS", 0.01)
    real_wait = claude_cwd_access.await_cwd_access

    async def wait_with_stuck_probe(cwd, **kwargs):
        return await real_wait(cwd, probe=FakeProbe(), **kwargs)

    monkeypatch.setattr("super_agents.claude_sdk.await_cwd_access", wait_with_stuck_probe)
    store = Store(tmp_path / "stuck.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=fake_sdk_loader)
    client._cwd_access_grace_seconds = 0
    client._cwd_access_max_wait_seconds = 0.05
    thread_id = (await client.start_thread({"name": "stuck", "cwd": str(tmp_path)}))["threadId"]
    result = await client.start_turn_by_label(LabelQueryInput(thread_id=thread_id), {"prompt": "hello"})

    await wait_for(lambda: store.get_turn(result["turnId"]).status == "failed")
    turn = store.get_turn(result["turnId"])
    assert turn.error_kind == BLOCKED_PERMISSION_DIALOG
    assert "Click Allow" in (turn.last_error or "")
    await client.close()
