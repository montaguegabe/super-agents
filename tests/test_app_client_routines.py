from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from super_agents.app_client_routines import RoutineClientMixin
from super_agents.state import JsonObject, read_state_file


class RoutineClientStub(RoutineClientMixin):
    def __init__(self, state_file: Path, thread_results: dict[str, JsonObject]) -> None:
        self.state_file = state_file
        self._state_lock = asyncio.Lock()
        self.thread_results = thread_results
        self.read_thread_calls: list[str] = []

    async def read_state(self):
        return read_state_file(self.state_file)

    async def read_thread(
        self,
        thread_id: str,
        include_turns: bool = True,
    ) -> JsonObject:
        self.read_thread_calls.append(thread_id)
        return self.thread_results[thread_id]


@pytest.mark.asyncio
async def test_reconcile_reads_terminal_status_from_app_server(tmp_path: Path) -> None:
    client = RoutineClientStub(
        tmp_path / "state.json",
        {
            "thread-routine": {
                "thread": {
                    "id": "thread-routine",
                    "turns": [{"id": "turn-routine", "status": "completed"}],
                }
            }
        },
    )
    await client.save_routine(
        {
            "name": "daily-check",
            "prompt": "Inspect project health.",
            "lastStatus": "started",
            "lastThreadId": "thread-routine",
            "lastTurnId": "turn-routine",
        }
    )

    listed = await client.list_routines()

    assert listed["routines"][0]["lastStatus"] == "completed"
    assert listed["routines"][0].get("lastError") is None
    assert client.read_thread_calls == ["thread-routine"]


@pytest.mark.asyncio
async def test_reconcile_caches_thread_reads_for_shared_thread(tmp_path: Path) -> None:
    client = RoutineClientStub(
        tmp_path / "state.json",
        {
            "shared-thread": {
                "thread": {
                    "id": "shared-thread",
                    "turns": [
                        {"id": "turn-one", "status": "completed"},
                        {"id": "turn-two", "status": "failed"},
                    ],
                }
            }
        },
    )
    for name, turn_id in (("first", "turn-one"), ("second", "turn-two")):
        await client.save_routine(
            {
                "name": name,
                "prompt": "Inspect project health.",
                "lastStatus": "started",
                "lastThreadId": "shared-thread",
                "lastTurnId": turn_id,
            }
        )

    listed = await client.list_routines()

    statuses = {item["name"]: item["lastStatus"] for item in listed["routines"]}
    assert statuses == {"first": "completed", "second": "failed"}
    assert client.read_thread_calls == ["shared-thread"]
