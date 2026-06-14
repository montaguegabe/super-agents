from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class Session:
    id: str
    name: str
    cwd: str
    command: list[str]
    status: str
    created_at: str
    updated_at: str
    agent_name: str | None = None
    model: str | None = None
    pid: int | None = None
    active_turn_id: str | None = None
    last_turn_id: str | None = None
    last_observed_state: str | None = None
    last_useful_message: str | None = None
    last_exit_code: int | None = None
    log_path: str | None = None
    raw_log_path: str | None = None

    def to_json(self, include_paths: bool = True) -> JsonObject:
        data: JsonObject = {
            "id": self.id,
            "name": self.name,
            "agentName": self.agent_name,
            "cwd": self.cwd,
            "command": self.command,
            "model": self.model,
            "status": self.status,
            "pid": self.pid,
            "activeTurnId": self.active_turn_id,
            "lastTurnId": self.last_turn_id,
            "lastObservedState": self.last_observed_state,
            "lastUsefulMessage": self.last_useful_message,
            "lastExitCode": self.last_exit_code,
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
        }
        if include_paths:
            data["logPath"] = self.log_path
            data["rawLogPath"] = self.raw_log_path
        return {key: value for key, value in data.items() if value is not None}


@dataclass(frozen=True)
class Turn:
    id: str
    session_id: str
    prompt: str
    status: str
    created_at: str
    updated_at: str
    mode: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    finished_at: str | None = None
    attempts: int = 0
    last_error: str | None = None

    def to_json(self) -> JsonObject:
        return {
            key: value
            for key, value in {
                "turnId": self.id,
                "sessionId": self.session_id,
                "promptPreview": preview(self.prompt),
                "status": self.status,
                "mode": self.mode,
                "model": self.model,
                "reasoningEffort": self.reasoning_effort,
                "createdAt": self.created_at,
                "updatedAt": self.updated_at,
                "finishedAt": self.finished_at,
                "attempts": self.attempts,
                "lastError": self.last_error,
            }.items()
            if value is not None
        }


def command_to_json(command: list[str]) -> str:
    return json.dumps(command)


def command_from_json(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        raw = json.loads(value)
    except json.JSONDecodeError:
        return []
    return [str(part) for part in raw] if isinstance(raw, list) else []


def preview(text: str | None, limit: int = 180) -> str | None:
    if text is None:
        return None
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "..."
