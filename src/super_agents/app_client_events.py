from __future__ import annotations

import hmac
import logging
import secrets

from .app_events import (
    DEFAULT_HMAC_HEADER,
    MAX_EVENT_BODY_BYTES,
    RECENT_EVENT_IDS_LIMIT,
    event_id_from_headers,
    event_sender,
    new_webhook_trigger,
    parse_event_payload,
    sender_is_authorized,
    trigger_matches_event,
    validate_trigger_input,
    verify_hmac_signature,
)
from .app_time import iso_now
from .state import (
    JsonObject,
    RoutineRecord,
    StateFile,
    TriggerRecord,
    routine_record_from_json,
    update_state_file,
)

logger = logging.getLogger(__name__)


class EventClientMixin:
    """Trigger management and event delivery for routines (loops).

    Webhook triggers make a routine runnable by delivered events instead of
    (or in addition to) its schedule. Event runs deliberately leave
    lastRunDate/lastRunAt untouched so they never displace a scheduled run.
    """

    async def add_routine_trigger(self, name: str, trigger_input: JsonObject) -> JsonObject:
        async with self._state_lock:

            def update(state: StateFile) -> JsonObject:
                routine = state.routines.get(name)
                if routine is None:
                    raise ValueError(f"No Super Agents routine found for name {name}.")
                validate_trigger_input(routine, trigger_input)
                trigger = new_webhook_trigger(trigger_input)
                raw = routine.to_json()
                raw["triggers"] = [*(raw.get("triggers") or []), trigger]
                raw["updatedAt"] = iso_now()
                state.routines[name] = _routine_from_raw(raw)
                return {"routine": state.routines[name].to_json(), "trigger": trigger}

            result = update_state_file(self.state_file, update)
        return {**result, "nativeSupport": False, "scheduler": "super-agents-local-wrapper"}

    async def remove_routine_trigger(self, name: str, trigger_id: str) -> JsonObject:
        async with self._state_lock:

            def update(state: StateFile) -> JsonObject:
                routine = state.routines.get(name)
                if routine is None:
                    raise ValueError(f"No Super Agents routine found for name {name}.")
                triggers = [trigger.to_json() for trigger in routine.triggers or []]
                remaining = [trigger for trigger in triggers if trigger.get("id") != trigger_id]
                if len(remaining) == len(triggers):
                    raise ValueError(f"No trigger {trigger_id} found on routine {name}.")
                raw = routine.to_json()
                raw["triggers"] = remaining
                if not remaining:
                    raw.pop("triggers", None)
                raw["updatedAt"] = iso_now()
                state.routines[name] = _routine_from_raw(raw)
                return {"deleted": True, "routine": state.routines[name].to_json()}

            result = update_state_file(self.state_file, update)
        return result

    async def deliver_webhook_event(
        self,
        token: str,
        *,
        headers: JsonObject | None = None,
        body: bytes | str = b"",
        origin: str = "external",
    ) -> JsonObject:
        headers = headers or {}
        body_bytes = body.encode("utf-8") if isinstance(body, str) else body
        if len(body_bytes) > MAX_EVENT_BODY_BYTES:
            return {"status": "rejected", "reason": "payload_too_large"}
        state = await self.read_state()
        found = _find_webhook_trigger(state, token)
        if found is None:
            return {"status": "unknown_token"}
        routine, trigger = found
        if trigger.hmac_secret:
            header_name = (trigger.hmac_header or DEFAULT_HMAC_HEADER).lower()
            normalized_headers = {str(key).lower(): value for key, value in headers.items()}
            header_value = normalized_headers.get(header_name)
            if not verify_hmac_signature(
                trigger.hmac_secret,
                header_value if isinstance(header_value, str) else None,
                body_bytes,
            ):
                return {"status": "rejected", "reason": "invalid_signature"}
        event_id = event_id_from_headers(headers, body_bytes)
        payload = parse_event_payload(body_bytes)
        event: JsonObject = {
            "id": event_id,
            "origin": origin,
            "triggerId": trigger.id,
            "receivedAt": iso_now(),
            "payload": payload,
        }
        decision = await self._reserve_event_run(routine.name, trigger.id, event, origin)
        status = decision["status"]
        response: JsonObject = {
            "status": status,
            "routine": routine.name,
            "triggerId": trigger.id,
            "eventId": event_id,
        }
        if status != "accepted":
            return response
        event["sender"] = decision.get("sender")
        reserved = decision["reservedRoutine"]
        result = await self.run_routine(reserved, force=True, event=event)
        return {**response, "status": "delivered", "run": result}

    async def emit_routine_event(
        self,
        name: str,
        payload: JsonObject | None = None,
        event_id: str | None = None,
    ) -> JsonObject:
        event: JsonObject = {
            "id": event_id or f"local-{secrets.token_hex(8)}",
            "origin": "local",
            "receivedAt": iso_now(),
            "payload": payload,
        }
        reserved = await self._reserve_named_routine(name)
        result = await self.run_routine(reserved, force=True, event=event)
        return {"status": "delivered", "routine": name, "eventId": event["id"], "run": result}

    async def _reserve_event_run(
        self,
        name: str,
        trigger_id: str,
        event: JsonObject,
        origin: str,
    ) -> JsonObject:
        async with self._state_lock:

            def update(state: StateFile) -> JsonObject:
                routine = state.routines.get(name)
                trigger_raw = None
                if routine is not None:
                    for candidate in routine.triggers or []:
                        if candidate.id == trigger_id:
                            trigger_raw = candidate
                            break
                if routine is None or trigger_raw is None:
                    return {"status": "unknown_token"}
                if event["id"] in (trigger_raw.recent_event_ids or []):
                    return {"status": "duplicate"}
                now = iso_now()
                recent = [*(trigger_raw.recent_event_ids or []), event["id"]][-RECENT_EVENT_IDS_LIMIT:]
                trigger_patch = {
                    **trigger_raw.to_json(),
                    "lastEventAt": now,
                    "lastEventId": event["id"],
                    "eventCount": (trigger_raw.event_count or 0) + 1,
                    "recentEventIds": recent,
                }
                raw = routine.to_json()
                raw["triggers"] = [
                    trigger_patch if item.get("id") == trigger_id else item for item in raw.get("triggers") or []
                ]
                if not routine.enabled or not trigger_raw.enabled:
                    status: JsonObject = {"status": "disabled"}
                elif not trigger_matches_event(trigger_raw, event.get("payload")):
                    status = {"status": "filtered"}
                elif trigger_raw.sender_allowlist or (origin == "external" and routine.kind == "agent"):
                    # Allowlists are enforced whenever configured; external
                    # events may never start an agent turn without one.
                    if sender_is_authorized(trigger_raw, event.get("payload")):
                        status = {"status": "accepted", "sender": event_sender(trigger_raw, event.get("payload"))}
                    else:
                        status = {"status": "unauthorized_sender"}
                        logger.warning(
                            "event_delivery unauthorized_sender routine=%s trigger=%s event=%s",
                            name,
                            trigger_id,
                            event["id"],
                        )
                else:
                    status = {"status": "accepted"}
                if status["status"] == "accepted":
                    raw["lastStartedAt"] = now
                    raw["lastStatus"] = "starting"
                    raw.pop("lastError", None)
                raw["updatedAt"] = now
                state.routines[name] = _routine_from_raw(raw)
                if status["status"] == "accepted":
                    status["reservedRoutine"] = state.routines[name]
                return status

            return update_state_file(self.state_file, update)

    async def _reserve_named_routine(self, name: str) -> RoutineRecord:
        async with self._state_lock:

            def update(state: StateFile) -> RoutineRecord:
                routine = state.routines.get(name)
                if routine is None:
                    raise ValueError(f"No Super Agents routine found for name {name}.")
                now = iso_now()
                raw = {
                    **routine.to_json(),
                    "lastStartedAt": now,
                    "lastStatus": "starting",
                    "updatedAt": now,
                }
                raw.pop("lastError", None)
                state.routines[name] = _routine_from_raw(raw)
                return state.routines[name]

            return update_state_file(self.state_file, update)


def _find_webhook_trigger(state: StateFile, token: str) -> tuple[RoutineRecord, TriggerRecord] | None:
    for routine in state.routines.values():
        for trigger in routine.triggers or []:
            if trigger.token and hmac.compare_digest(trigger.token, token):
                return routine, trigger
    return None


def _routine_from_raw(raw: JsonObject) -> RoutineRecord:
    routine = routine_record_from_json(raw, default_updated_at=iso_now())
    if routine is None:
        raise ValueError("Invalid routine payload.")
    return routine
