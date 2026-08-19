from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from .app_formatting import preview_text, without_none
from .state import JsonObject, SessionRecord, StoredStatus, TrackedStatus

LabelResolutionPrefer = Literal["latest_active", "latest_any"]
Mode = Literal["default", "plan"]

# Default result-list caps shared by every SuperAgentsClient backend.
DEFAULT_ACTIVE_AGENTS_LIMIT = 50
DEFAULT_RECENT_AGENTS_LIMIT = 20
DEFAULT_COMPACT_STATUS_LIMIT = 50


@dataclass(slots=True)
class PendingServerRequest:
    id: str | int
    method: str
    params: JsonObject
    received_at: str

    def to_json(self) -> JsonObject:
        return {
            "id": self.id,
            "method": self.method,
            "params": self.params,
            "receivedAt": self.received_at,
        }


PermissionRequestCallback = Callable[[PendingServerRequest], JsonObject | Awaitable[JsonObject | None] | None]


@dataclass(slots=True)
class TurnState:
    thread_id: str
    turn_id: str
    status: TrackedStatus
    started_at: str
    reasoning_effort: str | None = None
    events: list[JsonObject] = field(default_factory=list)
    pending_requests: list[PendingServerRequest] = field(default_factory=list)
    finished_at: str | None = None
    # Set whenever a notification advances this turn toward a state a waiter
    # cares about (terminal, or an approval request arriving/clearing). Lets
    # callers await turn progress instead of polling. Not serialized.
    update_event: asyncio.Event = field(default_factory=asyncio.Event, compare=False, repr=False)

    def to_json(self) -> JsonObject:
        return without_none(
            {
                "threadId": self.thread_id,
                "turnId": self.turn_id,
                "status": self.status,
                "startedAt": self.started_at,
                "reasoningEffort": self.reasoning_effort,
                "finishedAt": self.finished_at,
                "events": self.events,
                "pendingRequests": [request.to_json() for request in self.pending_requests],
            }
        )


@dataclass(slots=True)
class QueuedTurn:
    id: str
    thread_id: str
    label: str | None
    agent_name: str | None
    input_data: JsonObject
    queued_at: str
    status: str = "queued"
    started_at: str | None = None
    attempts: int = 0
    last_error: str | None = None

    def to_json(self) -> JsonObject:
        return without_none(
            {
                "id": self.id,
                "threadId": self.thread_id,
                "label": self.label,
                "agentName": self.agent_name,
                "inputData": self.input_data,
                "queuedAt": self.queued_at,
                "status": self.status,
                "startedAt": self.started_at,
                "attempts": self.attempts,
                "lastError": self.last_error,
                "promptPreview": preview_text(str(self.input_data.get("prompt") or "")),
            }
        )


@dataclass(slots=True)
class LabelQueryInput:
    label: str | None = None
    backend: str | None = None
    cwd: str | None = None
    group: str | None = None
    status: str | None = None
    favorite: bool | None = None
    limit: int | None = None
    include_inactive: bool | None = None
    prefer: LabelResolutionPrefer | None = None
    thread_id: str | None = None
    turn_id: str | None = None
    include_turn: bool | None = None
    include_items: bool | None = None
    full: bool | None = None
    final_only: bool | None = None
    max_items: int | None = None
    max_output_chars: int | None = None
    include_preview: bool | None = None
    preview_length: int | None = None
    fields: list[str] | None = None


@dataclass(slots=True)
class QueueCancelInput:
    queue_item_id: str | None = None
    backend: str | None = None
    label: str | None = None
    thread_id: str | None = None
    cwd: str | None = None
    position: int | None = None


@dataclass(slots=True)
class ResolvedSession:
    session: SessionRecord
    status: StoredStatus
    turn_id: str | None = None
