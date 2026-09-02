from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from typing import Any

from .app_time import iso_now
from .state import JsonObject, RoutineRecord, TriggerRecord

MAX_EVENT_BODY_BYTES = 256 * 1024
RECENT_EVENT_IDS_LIMIT = 50
MAX_EVENT_PROMPT_CHARS = 8000
DEFAULT_HMAC_HEADER = "X-Hub-Signature-256"
EVENT_ID_HEADERS = ("x-github-delivery", "x-delivery-id", "x-request-id", "x-event-id")
FILTER_OPS = {"equals", "notEquals", "contains", "startsWith", "endsWith", "exists", "regex"}


def new_webhook_trigger(input_data: JsonObject) -> JsonObject:
    now = iso_now()
    return {
        "id": f"trg-{secrets.token_hex(4)}",
        "type": "webhook",
        "enabled": True,
        "token": secrets.token_hex(16),
        "description": input_data.get("description"),
        "hmacSecret": input_data.get("hmacSecret"),
        "hmacHeader": input_data.get("hmacHeader"),
        "relayEndpointId": input_data.get("relayEndpointId"),
        "relayUrl": input_data.get("relayUrl"),
        "senderPath": input_data.get("senderPath"),
        "senderAllowlist": input_data.get("senderAllowlist"),
        "filters": input_data.get("filters"),
        "createdAt": now,
    }


def validate_trigger_input(routine: RoutineRecord, input_data: JsonObject) -> None:
    filters = input_data.get("filters") or []
    for item in filters:
        if not isinstance(item, dict) or not item.get("path") or item.get("op") not in FILTER_OPS:
            raise ValueError(f"Trigger filters need a path and an op in {sorted(FILTER_OPS)}.")
        if item.get("op") == "regex":
            re.compile(str(item.get("value") or ""))
    if routine.kind == "agent":
        # An externally reachable trigger that can start an agent turn is a
        # prompt-injection port unless deliveries are pinned to known senders.
        if not input_data.get("senderPath") or not input_data.get("senderAllowlist"):
            raise ValueError(
                "Webhook triggers on agent loops require a senderPath and a non-empty "
                "senderAllowlist so only known senders can start agent runs."
            )


def verify_hmac_signature(secret: str, header_value: str | None, body: bytes) -> bool:
    if not header_value:
        return False
    provided = header_value.strip()
    if "=" in provided:
        provided = provided.split("=", 1)[1]
    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, provided.lower())


def json_path_value(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list) and part.isdigit() and int(part) < len(current):
            current = current[int(part)]
        else:
            return None
    return current


def event_filter_matches(filter_spec: JsonObject, payload: Any) -> bool:
    path = filter_spec.get("path")
    op = filter_spec.get("op")
    if not isinstance(path, str) or op not in FILTER_OPS:
        return False
    actual = json_path_value(payload, path)
    if op == "exists":
        return actual is not None
    expected = filter_spec.get("value")
    if op == "equals":
        return actual == expected or _as_text(actual) == _as_text(expected)
    if op == "notEquals":
        return not (actual == expected or _as_text(actual) == _as_text(expected))
    actual_text = _as_text(actual)
    expected_text = _as_text(expected)
    if actual_text is None or expected_text is None:
        return False
    if op == "contains":
        return expected_text in actual_text
    if op == "startsWith":
        return actual_text.startswith(expected_text)
    if op == "endsWith":
        return actual_text.endswith(expected_text)
    try:
        return re.search(expected_text, actual_text) is not None
    except re.error:
        return False


def trigger_matches_event(trigger: TriggerRecord, payload: Any) -> bool:
    return all(event_filter_matches(filter_spec, payload) for filter_spec in trigger.filters or [])


def event_sender(trigger: TriggerRecord, payload: Any) -> str | None:
    if not trigger.sender_path:
        return None
    return _as_text(json_path_value(payload, trigger.sender_path))


def sender_is_authorized(trigger: TriggerRecord, payload: Any) -> bool:
    if not trigger.sender_allowlist:
        return False
    sender = event_sender(trigger, payload)
    return sender is not None and sender in trigger.sender_allowlist


def event_id_from_headers(headers: JsonObject, body: bytes) -> str:
    normalized = {str(key).lower(): value for key, value in headers.items()}
    for header in EVENT_ID_HEADERS:
        value = normalized.get(header)
        if isinstance(value, str) and value:
            return value
    return f"sha256-{hashlib.sha256(body).hexdigest()}"


def parse_event_payload(body: bytes) -> Any:
    try:
        return json.loads(body.decode("utf-8"))
    except Exception:
        return None


def render_event_prompt_context(event: JsonObject) -> str:
    payload = event.get("payload")
    payload_text = json.dumps(payload, indent=2) if payload is not None else str(event.get("rawBody") or "")
    if len(payload_text) > MAX_EVENT_PROMPT_CHARS:
        payload_text = payload_text[:MAX_EVENT_PROMPT_CHARS] + "\n… (payload truncated)"
    lines = [
        "",
        "",
        "## Triggering event",
        f"- event id: {event.get('id')}",
        f"- trigger: {event.get('triggerId') or 'manual'}",
        f"- received at: {event.get('receivedAt')}",
    ]
    sender = event.get("sender")
    if sender:
        lines.append(f"- verified sender: {sender}")
    lines.extend(["", "```json", payload_text, "```"])
    return "\n".join(lines)


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return str(value)
    return None
