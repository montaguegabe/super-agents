"""JSON view formatting for Claude Code sessions and turns."""

from __future__ import annotations

from typing import Any

from super_agents.agent_store import Session
from super_agents.app_formatting import apply_field_selection, without_none
from super_agents.app_models import LabelQueryInput

JsonObject = dict[str, Any]


class SessionViewMixin:
    def _session_view(self, session: Session) -> JsonObject:
        return {"backend": self.backend, **session.to_json()}

    def _agent_item(self, session: Session, query: LabelQueryInput) -> JsonObject:
        turns = self.store.list_turns(session.id, limit=1)
        return apply_field_selection(
            without_none(
                {
                    "backend": self.backend,
                    "name": session.name,
                    "agentName": session.agent_name,
                    "threadId": session.id,
                    "turnId": session.active_turn_id or session.last_turn_id,
                    "cwd": session.cwd,
                    "status": session.status,
                    "model": session.model,
                    "updatedAt": session.updated_at,
                    "lastObservedState": session.last_observed_state,
                    "queueDepth": len(self.store.queued_turns(session.id)),
                    "preview": turns[0].to_json().get("promptPreview")
                    if turns and query.include_preview is not False
                    else None,
                }
            ),
            query.fields,
        )

    def _status_item(self, session: Session) -> JsonObject:
        queued = self.store.queued_turns(session.id)
        return without_none(
            {
                "backend": self.backend,
                "name": session.name,
                "agentName": session.agent_name,
                "threadId": session.id,
                "turnId": session.active_turn_id or session.last_turn_id,
                "cwd": session.cwd,
                "status": session.status,
                "model": session.model,
                "activeTurnId": session.active_turn_id,
                "lastTurnId": session.last_turn_id,
                "lastObservedState": session.last_observed_state,
                "lastUsefulMessage": session.last_useful_message,
                "queueDepth": len(queued),
                "updatedAt": session.updated_at,
            }
        )

    def _turn_view(self, session: Session, turn: Any) -> JsonObject:
        data = turn.to_json()
        if "lastUsefulMessage" not in data and turn.id == session.last_turn_id and session.last_useful_message:
            data["lastUsefulMessage"] = session.last_useful_message
        return data
