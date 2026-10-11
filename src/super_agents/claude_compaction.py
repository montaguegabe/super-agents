"""Compact a long-lived Claude Code session between requests.

Claude Code compacts a conversation by itself only when a request arrives
with the context nearly full, and that compaction runs on the request's
critical path: a voice Dispatcher on Haiku (200k window) auto-compacted about
once an hour and each time stalled the caller's request for 30-40 s (VM2,
2026-10-11 08:01Z: a steer relayed 40 s after it was spoken, 34 s of it
compacting). Compacting while the session is idle moves that cost off every
request.

Compaction is a maintenance operation, not a turn: Claude Code answers a raw
``/compact`` with a ``num_turns == 0`` result, which the turn machinery
treats as a resume handshake. It holds the same in-process and cross-process
session locks as a turn, so a request that arrives meanwhile waits behind it
and is never dropped.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

from super_agents.agent_store import Session
from super_agents.app_models import LabelQueryInput
from super_agents.claude_transcript import transcript_path

logger = logging.getLogger(__name__)

COMPACT_COMMAND = "/compact"
DEFAULT_COMPACT_TIMEOUT_SECONDS = 300.0
#: Bytes read from the end of a transcript to find the latest usage.
_TAIL_BYTES = 2_000_000


def context_tokens_from_transcript(path: Path | None) -> int | None:
    """Tokens the conversation's latest model call carried, or None if unknown.

    Uses the last main-thread assistant message's ``input``, ``cache_read``
    and ``cache_creation`` tokens. A ``compact_boundary`` after the last such
    message means the context was just compacted: returns 0.
    """
    if path is None:
        return None
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            size = handle.tell()
            handle.seek(max(0, size - _TAIL_BYTES))
            tail = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    for line in reversed(tail.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("subtype") == "compact_boundary":
            return 0
        if entry.get("type") != "assistant" or entry.get("isSidechain"):
            continue
        message = entry.get("message")
        usage = message.get("usage") if isinstance(message, dict) else None
        if not isinstance(usage, dict):
            continue
        return sum(
            int(usage.get(key) or 0)
            for key in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
        )
    return None


class SessionCompactionMixin:
    """``compact_session_by_label`` for :class:`ClaudeAgentSdkClient`."""

    def session_context_tokens(self, session: Session) -> int | None:
        return context_tokens_from_transcript(transcript_path(session))

    async def compact_session_by_label(
        self,
        input_data: LabelQueryInput,
        turn_input: dict[str, Any] | None = None,
        *,
        min_context_tokens: int = 0,
        timeout_seconds: float = DEFAULT_COMPACT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        """Compact the session now if it is idle and its context is large enough.

        ``turn_input`` carries the model, effort and service tier the next turn
        will use, so the connected client is the one that turn reuses (see
        ``warm_session_by_label``).
        """
        session = self._resolve_session(input_data)
        if self._transport_outside_context(session):
            return self._compact_result(session, compacted=False, reason="owner_outside_execution_context")
        if self._session_is_busy(session):
            return self._compact_result(session, compacted=False, reason="busy")
        tokens = await asyncio.to_thread(self.session_context_tokens, session)
        if tokens is None or tokens < min_context_tokens:
            return self._compact_result(session, compacted=False, reason="below_threshold", contextTokens=tokens)
        turn_input = turn_input or {}
        sdk = self._require_sdk()
        model = turn_input.get("model") or session.model or self._default_model()
        reasoning_effort = turn_input.get("reasoningEffort") or self._default_reasoning_effort()
        service_tier = turn_input.get("serviceTier")
        lock = self._session_locks.setdefault(session.id, asyncio.Lock())
        started = time.monotonic()
        async with lock, self._cross_process_session_lock(session.id):
            session = self.store.get_session(session.id)
            if self._session_is_busy(session):
                return self._compact_result(session, compacted=False, reason="busy")
            previous_state = session.last_observed_state
            await asyncio.to_thread(
                self.store.update_session, session.id, last_observed_state="Compacting conversation between requests"
            )
            client = await self._sdk_client_for(session, model, reasoning_effort, service_tier, sdk)
            outcome = "success"
            try:
                await client.query(COMPACT_COMMAND)
                await asyncio.wait_for(_read_until_result(client), timeout=timeout_seconds)
            except TimeoutError:
                outcome = "timeout"
                # The stream may still owe the compaction's result; never hand
                # it to the next turn.
                await self._disconnect_sdk_client(session.id)
            except Exception:
                outcome = "error"
                await self._disconnect_sdk_client(session.id)
                logger.warning("Claude session compaction failed thread_id=%s", session.id, exc_info=True)
            finally:
                with contextlib.suppress(Exception):
                    await asyncio.to_thread(self.store.update_session, session.id, last_observed_state=previous_state)
        elapsed_ms = int((time.monotonic() - started) * 1000)
        logger.info(
            "dispatch_timing stage=super_agent_session_compacted thread_id=%s outcome=%s "
            "context_tokens=%s elapsed_ms=%d",
            session.id,
            outcome,
            tokens,
            elapsed_ms,
        )
        return self._compact_result(
            session,
            compacted=outcome == "success",
            reason=None if outcome == "success" else outcome,
            contextTokens=tokens,
            elapsedMs=elapsed_ms,
        )

    def _compact_result(self, session: Session, *, compacted: bool, reason: str | None = None, **fields: Any):
        result = {"backend": self.backend, "threadId": session.id, "name": session.name, "compacted": compacted}
        if reason:
            result["reason"] = reason
        result.update({key: value for key, value in fields.items() if value is not None})
        return result


async def _read_until_result(client: Any) -> None:
    async for message in client.receive_response():
        if getattr(message, "num_turns", None) is not None:
            return
