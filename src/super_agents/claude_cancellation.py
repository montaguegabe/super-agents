"""Turn-scoped cancellation across clients sharing the Claude session store."""

from __future__ import annotations

import asyncio
import contextlib
from typing import Any

from super_agents.agent_store import iso_now
from super_agents.app_models import LabelQueryInput


class TurnCancellationMixin:
    async def cancel_by_label(self, input_data: LabelQueryInput) -> dict[str, Any]:
        session = self._resolve_session(input_data)
        turn_id = session.active_turn_id
        cancelled = False
        if turn_id and (not input_data.turn_id or input_data.turn_id == turn_id):
            # Persist first: the owner can be a different client/process and
            # must stop even if an SDK interrupt acknowledgement never arrives.
            turn = self.store.update_turn(
                turn_id, only_if_active=True, status="cancelled", finished_at=iso_now()
            )
            cancelled = turn.status == "cancelled"
            if cancelled:
                self._permission_gate.cancel_scope(thread_id=session.id, turn_id=turn_id)
                self._finish_cancelled_turn(session.id, turn_id)
        return {"backend": self.backend, "cancelled": cancelled, "threadId": session.id, "name": session.name}

    async def _watch_turn_cancellation(self, turn_id: str, owner: asyncio.Task) -> None:
        # A silent receive_response(), connect(), or query() cannot observe a
        # store-only interrupt itself. Wake the owning task so it can stop its
        # own SDK client; never signal another process by PID.
        while not owner.done():
            await asyncio.sleep(0.1)
            if await asyncio.to_thread(self._turn_was_cancelled, turn_id):
                owner.cancel()
                return

    async def _stop_cancelled_sdk_client(self, session_id: str) -> None:
        client = self._sdk_clients.get(session_id)
        if client is not None and hasattr(client, "interrupt"):
            # A hung acknowledgement must not prevent transport teardown.
            with contextlib.suppress(Exception):
                async with asyncio.timeout(0.5):
                    await client.interrupt()
        await self._disconnect_sdk_client(session_id)
