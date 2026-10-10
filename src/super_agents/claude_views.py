"""JSON view formatting for Claude Code sessions and turns."""

from __future__ import annotations

from typing import Any

from super_agents.agent_store import Session
from super_agents.app_formatting import apply_field_selection, without_none
from super_agents.app_models import LabelQueryInput
from super_agents.claude_turn_output import ordered_turn_items

JsonObject = dict[str, Any]


class SessionViewMixin:
    def _session_view(self, session: Session, latest: Any | None = None) -> JsonObject:
        view = {**session.to_json(), "backend": self._session_backend(session)}
        # Session rows do not record reasoning effort (or, for imported
        # sessions, a model); surface the latest turn's values so list
        # consumers can show them without fetching turns. Listings pass the
        # prefetched latest turn so a 500-session list is not 500 queries.
        if latest is None:
            turns = self.store.list_turns(session.id, limit=1)
            latest = turns[0] if turns else None
        if latest is not None:
            if latest.reasoning_effort:
                view.setdefault("reasoningEffort", latest.reasoning_effort)
            if latest.model:
                view.setdefault("model", latest.model)
        return view

    def _agent_item(self, session: Session, query: LabelQueryInput) -> JsonObject:
        turns = self.store.list_turns(session.id, limit=1)
        return apply_field_selection(
            without_none(
                {
                    "backend": self._session_backend(session),
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
                "backend": self._session_backend(session),
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

    def _turn_view(self, session: Session, turn: Any, *, include_prompt: bool = False) -> JsonObject:
        data = turn.to_json()
        if include_prompt:
            data["items"] = ordered_turn_items(self.store, turn)
        if include_prompt and turn.prompt:
            # Thread reads feed history UIs that render what the user actually
            # said; the 180-char promptPreview alone cuts transport envelopes
            # like <voice>...</voice> in half and loses the rest of the prompt.
            # Progress/status payloads keep the compact preview.
            data["prompt"] = turn.prompt
        return data

    def _session_backend(self, session: Session) -> str:
        return session.backend or self.backend
