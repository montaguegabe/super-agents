"""Persistent configured-backend ownership for Super Agents identifiers."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backend_config import normalize_backend
from .state import state_file_lock

BACKEND_PROVENANCE_FILE_ENV = "SUPER_AGENTS_BACKEND_PROVENANCE_FILE"
DEFAULT_BACKEND_PROVENANCE_FILE = Path.home() / ".super-agents" / "backend-provenance.json"


@dataclass(slots=True)
class BackendProvenance:
    threads: dict[str, str] = field(default_factory=dict)
    turns: dict[str, str] = field(default_factory=dict)
    requests: dict[str, set[str]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 2,
            "threads": self.threads,
            "turns": self.turns,
            "requests": {request_id: sorted(backends) for request_id, backends in self.requests.items()},
        }


class BackendProvenanceStore:
    """Atomic local ownership index used across MCP server restarts."""

    def __init__(self, path: str | Path | None = None) -> None:
        configured = path or os.environ.get(BACKEND_PROVENANCE_FILE_ENV)
        self.path = Path(configured).expanduser() if configured else DEFAULT_BACKEND_PROVENANCE_FILE

    def read(self) -> BackendProvenance:
        with state_file_lock(self.path):
            return self._read_unlocked()

    def backend_for_thread(self, thread_id: str) -> str | None:
        return self.read().threads.get(thread_id)

    def backend_for_turn(self, turn_id: str) -> str | None:
        return self.read().turns.get(turn_id)

    def backend_for_request(self, request_id: str | int) -> str | None:
        backends = self.backends_for_request(request_id)
        return next(iter(backends)) if len(backends) == 1 else None

    def backends_for_request(self, request_id: str | int) -> set[str]:
        return self.read().requests.get(str(request_id), set())

    def engaged_backends(self) -> set[str]:
        state = self.read()
        request_backends = {backend for backends in state.requests.values() for backend in backends}
        return set(state.threads.values()) | set(state.turns.values()) | request_backends

    def remember(
        self,
        backend: str,
        *,
        thread_ids: set[str] | None = None,
        turn_ids: set[str] | None = None,
        request_ids: set[str] | None = None,
    ) -> None:
        identity = normalize_backend(backend)
        with state_file_lock(self.path):
            state = self._read_unlocked()
            for thread_id in thread_ids or ():
                _remember_owner(state.threads, thread_id, identity)
            for turn_id in turn_ids or ():
                _remember_owner(state.turns, turn_id, identity)
            for request_id in request_ids or ():
                state.requests.setdefault(str(request_id), set()).add(identity)
            self._write_unlocked(state)

    def _read_unlocked(self) -> BackendProvenance:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return BackendProvenance()
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Backend provenance file is unreadable: {self.path}") from exc
        if not isinstance(raw, dict):
            raise RuntimeError(f"Backend provenance file must contain an object: {self.path}")
        return BackendProvenance(
            threads=_identity_map(raw.get("threads")),
            turns=_identity_map(raw.get("turns")),
            requests=_request_identity_map(raw.get("requests")),
        )

    def _write_unlocked(self, state: BackendProvenance) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.path.parent, 0o700)
        tmp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w",
                encoding="utf-8",
                dir=self.path.parent,
                delete=False,
            ) as handle:
                os.chmod(handle.name, 0o600)
                json.dump(state.to_json(), handle, indent=2, sort_keys=True)
                handle.write("\n")
                tmp_name = handle.name
            os.replace(tmp_name, self.path)
            os.chmod(self.path, 0o600)
        except BaseException:
            if tmp_name:
                try:
                    os.unlink(tmp_name)
                except FileNotFoundError:
                    pass
            raise


def _identity_map(value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, str] = {}
    for identifier, raw_backend in value.items():
        if not isinstance(identifier, str) or not isinstance(raw_backend, str):
            continue
        try:
            result[identifier] = normalize_backend(raw_backend)
        except ValueError:
            continue
    return result


def _request_identity_map(value: Any) -> dict[str, set[str]]:
    if not isinstance(value, dict):
        return {}
    result: dict[str, set[str]] = {}
    for identifier, raw_backends in value.items():
        if not isinstance(identifier, str):
            continue
        candidates = [raw_backends] if isinstance(raw_backends, str) else raw_backends
        if not isinstance(candidates, list):
            continue
        identities: set[str] = set()
        for raw_backend in candidates:
            if not isinstance(raw_backend, str):
                continue
            try:
                identities.add(normalize_backend(raw_backend))
            except ValueError:
                continue
        if identities:
            result[identifier] = identities
    return result


def _remember_owner(owners: dict[str, str], identifier: str, backend: str) -> None:
    existing = owners.get(identifier)
    if existing and existing != backend:
        raise ValueError(
            f"Identifier {identifier} is already owned by backend {existing}; refusing to reassign it to {backend}."
        )
    owners[identifier] = backend
