"""Active Claude SDK ownership and steering, including durable queue fallback."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any
from weakref import WeakValueDictionary

from super_agents.agent_store import Session
from super_agents.app_models import LabelQueryInput

JsonObject = dict[str, Any]
logger = logging.getLogger(__name__)
# A store row is shared across clients, but SDK transports are not. Retain the
# owner across initial responses AND background-task drain cycles, and never
# use another event loop's transport. Foreign processes use the durable queue.
_ACTIVE_OWNERS: WeakValueDictionary = WeakValueDictionary()
_TRANSPORT_OWNERS: WeakValueDictionary = WeakValueDictionary()


class ActiveSteeringMixin:
    def _owner_key(self, session_id: str) -> tuple:
        return (os.getpid(), str(self.store.path.resolve()), self.backend, session_id, asyncio.get_running_loop())

    def _register_active_owner(self, session_id: str) -> None:
        _ACTIVE_OWNERS[self._owner_key(session_id)] = self

    def _unregister_active_owner(self, session_id: str) -> None:
        key = self._owner_key(session_id)
        if _ACTIVE_OWNERS.get(key) is self:
            _ACTIVE_OWNERS.pop(key, None)

    def _active_owner(self, session_id: str) -> Any | None:
        return _ACTIVE_OWNERS.get(self._owner_key(session_id))

    def _transport_context_matches(self, session_id: str) -> bool:
        context = self._sdk_client_contexts.get(session_id)
        return context is None or context == (os.getpid(), asyncio.get_running_loop())

    def _register_transport_owner(self, session_id: str) -> None:
        # Transport lifetime extends beyond a turn, including idle continuations.
        _TRANSPORT_OWNERS[self._owner_key(session_id)] = self

    def _unregister_transport_owner(self, session_id: str) -> None:
        key = self._owner_key(session_id)
        if _TRANSPORT_OWNERS.get(key) is self:
            _TRANSPORT_OWNERS.pop(key, None)

    def _transport_owner(self, session: Session) -> Any | None:
        owner = _TRANSPORT_OWNERS.get(self._owner_key(session.id))
        if owner is not None and not owner._closed and owner._sdk_clients.get(session.id) is not None:
            if owner._owns_session_leaf(session):
                return owner
        return None

    def _has_other_loop_transport(self, session: Session) -> bool:
        key = self._owner_key(session.id)
        # This is only in-process evidence. Never use a foreign loop's client,
        # and exclude registries inherited across fork by comparing process IDs.
        return any(
            other_key[:-1] == key[:-1] and other_key[-1] is not key[-1]
            and owner._sdk_clients.get(session.id) is not None
            for other_key, owner in list(_TRANSPORT_OWNERS.items())
        )

    def _transport_outside_context(self, session: Session) -> bool:
        return not self._transport_context_matches(session.id) or self._has_other_loop_transport(session)

    def _unavailable_transport_result(self, session: Session) -> JsonObject:
        return {
            "backend": self.backend, "threadId": session.id, "name": session.name,
            "turnId": None, "queued": False, "steered": False,
            "startedImmediately": False, "confirmed": False,
            "delivery": "unavailable", "reason": "owner_outside_execution_context",
            "message": "Nothing was delivered or queued. The managed SDK transport belongs to another process or event loop.",
        }

    async def _steer_active_turn(
        self,
        session: Session,
        prompt: str,
        turn_input: JsonObject,
        *,
        requested_turn_id: str | None = None,
    ) -> JsonObject:
        owner = self._active_owner(session.id)
        if owner is not None and owner is not self:
            return await owner._steer_active_turn(
                session, prompt, turn_input, requested_turn_id=requested_turn_id
            )
        active_turn_id = session.active_turn_id
        if not active_turn_id:
            return await self.start_turn_by_label(
                LabelQueryInput(thread_id=session.id),
                {**turn_input, "prompt": prompt},
            )

        if requested_turn_id and requested_turn_id != active_turn_id:
            raise RuntimeError(
                f"Expected active turn id `{requested_turn_id}` but found `{active_turn_id}`. "
                "Nothing was delivered or queued."
            )

        sdk_client = await self._wait_for_active_sdk_client(session.id)
        owner = self._active_owner(session.id)
        if owner is not None and owner is not self:
            return await owner._steer_active_turn(
                self.store.get_session(session.id), prompt, turn_input, requested_turn_id=requested_turn_id
            )
        refreshed = self.store.get_session(session.id)
        if refreshed.active_turn_id != active_turn_id or not self._session_is_busy(refreshed):
            return await self.start_turn_by_label(
                LabelQueryInput(thread_id=session.id),
                {**turn_input, "prompt": prompt},
            )
        if sdk_client is None:
            # No client in this process: either the turn runs in another
            # process (its flock is held — queue a follow-up there)
            # or the owning process died and left a ghost row. Reclaim the
            # ghost so the user's message becomes a fresh turn instead of
            # black-holing against a dead turn.
            if self._reclaim_orphaned_turn(session.id, active_turn_id):
                logger.info(
                    "Steer reclaimed orphaned Claude Code turn session_id=%s turn_id=%s",
                    session.id,
                    active_turn_id,
                )
                return await self.start_turn_by_label(
                    LabelQueryInput(thread_id=session.id),
                    {**turn_input, "prompt": prompt},
                )
            return await self._queue_unavailable_steer(session, prompt, turn_input)

        # The steered query produces its own response on the shared stream;
        # register it so the active turn's reader consumes it instead of
        # leaving it to shift the next turn's answer (off-by-one).
        self._register_pending_result(session.id)
        try:
            if turn_input.get("interruptCurrentWork") is True:
                self._session_interrupted_steer_followups.add(session.id)
                await sdk_client.interrupt()
                logger.info(
                    "dispatch_timing stage=super_agent_steer_interrupt_ack thread_id=%s turn_id=%s",
                    session.id,
                    active_turn_id,
                )
            await sdk_client.query(self._prompt_for_session(refreshed, {**turn_input, "prompt": prompt}))
            logger.info(
                "dispatch_timing stage=super_agent_steer_correction_sent thread_id=%s turn_id=%s interrupted=%s",
                session.id,
                active_turn_id,
                turn_input.get("interruptCurrentWork") is True,
            )
        except BaseException:
            self._consume_pending_result(session.id)
            self._session_interrupted_steer_followups.discard(session.id)
            raise
        self._record_session_leaf_owner(session.id)
        # Persist the steering text on the turn row so thread reads (and other
        # processes' reads — the voice pipeline steers from a different process
        # than the one serving history) can render every user input in order.
        self.store.append_turn_steer(active_turn_id, prompt)
        current_turn = self.store.get_turn(active_turn_id)
        self.store.update_session(
            session.id,
            status="running",
            active_turn_id=active_turn_id,
            last_turn_id=active_turn_id,
            last_observed_state="steering active turn via Claude Code",
        )
        return {
            "backend": self.backend,
            "threadId": session.id,
            "name": session.name,
            "turnId": active_turn_id,
            "turn": current_turn.to_json(),
            "queued": False,
            "steered": True,
            "nativeSteer": True,
            "delivery": "sdk",
            "confirmed": True,
            "interruptedCurrentWork": turn_input.get("interruptCurrentWork") is True,
            "startedImmediately": False,
            "drain": "steered_active_turn",
        }

    async def _queue_unavailable_steer(
        self, session: Session, prompt: str, turn_input: JsonObject
    ) -> JsonObject:
        result = await self.queue_turn_by_label(
            LabelQueryInput(thread_id=session.id), {**turn_input, "prompt": prompt}
        )
        return {
            **result,
            "steered": False,
            "nativeSteer": False,
            "interruptedCurrentWork": False,
            "reason": "active_sdk_client_unavailable",
            "message": (
                "Steering was unavailable. The instruction was saved as a queued follow-up turn; "
                "the current work was not interrupted."
                if result.get("queued") else
                "The active turn finished. The instruction started as a new turn."
            ),
        }

    async def _wait_for_active_sdk_client(self, session_id: str) -> Any | None:
        for _ in range(100):
            client = self._sdk_clients.get(session_id)
            owner = self._active_owner(session_id)
            if owner is not None and owner is not self:
                return None
            if client is not None and self._owns_session_leaf(self.store.get_session(session_id)):
                return client
            await asyncio.sleep(0.01)
        return None

