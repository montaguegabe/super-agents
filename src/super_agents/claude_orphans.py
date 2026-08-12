"""Orphaned-turn reconciliation and cross-process session locking.

A service restart kills in-flight turn tasks without touching the store,
leaving sessions active/running forever; those ghost rows then black-hole
steers and mislead status consumers. This mixin sweeps and reclaims them, and
provides the flock that serializes a session's turns across processes.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from datetime import datetime, timezone

try:  # POSIX advisory locking; unavailable on Windows.
    import fcntl
except ImportError:  # pragma: no cover - platform dependent
    fcntl = None  # type: ignore[assignment]

from super_agents.agent_store import Turn, iso_now

logger = logging.getLogger(__name__)

# Minimum age before a running turn may be treated as orphaned. Rows are
# written before the turn task acquires the session flock, so very fresh
# turns can look unowned for a moment.
_ORPHAN_SWEEP_MIN_AGE_SECONDS = 60.0
_ORPHAN_RECLAIM_MIN_AGE_SECONDS = 5.0


class OrphanReconciliationMixin:
    def _session_turn_lock_is_free(self, session_id: str) -> bool:
        """True when no live process holds the session's turn lock.

        A running turn holds the session flock for its whole duration and
        flock releases on process death, so an acquirable lock while the
        store claims a running turn proves the owning process is gone.
        Without flock support, orphanhood cannot be proven; stay conservative.
        """
        if fcntl is None:
            return False
        lock_dir = self.store.path.parent / "session-locks"
        try:
            lock_dir.mkdir(parents=True, exist_ok=True)
            fd = os.open(lock_dir / f"{session_id}.lock", os.O_CREAT | os.O_RDWR, 0o644)
        except OSError:
            return False
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return False
            fcntl.flock(fd, fcntl.LOCK_UN)
            return True
        finally:
            os.close(fd)

    def _turn_age_seconds(self, turn: Turn | None) -> float | None:
        if turn is None or not turn.updated_at:
            return None
        try:
            updated = datetime.fromisoformat(turn.updated_at.replace("Z", "+00:00"))
        except ValueError:
            return None
        return (datetime.now(timezone.utc) - updated).total_seconds()

    def _fail_orphaned_turn(self, session_id: str, turn_id: str | None) -> None:
        if turn_id:
            with contextlib.suppress(KeyError):
                turn = self.store.get_turn(turn_id)
                if turn.status not in {"completed", "failed", "cancelled"}:
                    self.store.update_turn(
                        turn_id,
                        status="failed",
                        finished_at=iso_now(),
                        last_error="interrupted: owning process exited mid-turn",
                    )
        # Reconciliation is bookkeeping, not activity: keep the session's real
        # last-activity time so UIs don't surface long-dead threads as recent.
        previous_updated_at = None
        with contextlib.suppress(KeyError):
            previous_updated_at = self.store.get_session(session_id).updated_at
        self.store.update_session(
            session_id,
            status="failed",
            active_turn_id=None,
            last_observed_state="turn orphaned by process exit",
            **({"updated_at": previous_updated_at} if previous_updated_at else {}),
        )

    def _reclaim_orphaned_turn(self, session_id: str, turn_id: str) -> bool:
        """Terminalize a turn whose owning process died; True if reclaimed."""
        turn = None
        with contextlib.suppress(KeyError):
            turn = self.store.get_turn(turn_id)
        age = self._turn_age_seconds(turn)
        if age is not None and age < _ORPHAN_RECLAIM_MIN_AGE_SECONDS:
            return False
        if not self._session_turn_lock_is_free(session_id):
            return False
        self._fail_orphaned_turn(session_id, turn_id)
        return True

    def reconcile_orphaned_turns(self) -> int:
        """Startup sweep: fail running turns whose owning process is gone.

        A service restart kills in-flight turn tasks without touching the
        store, leaving sessions active/running forever; those ghost rows
        then black-hole steers and mislead status consumers.
        """
        reconciled = 0
        for session in self.store.list_sessions(include_inactive=True):
            if not session.active_turn_id and session.status != "running":
                continue
            turn = None
            if session.active_turn_id:
                with contextlib.suppress(KeyError):
                    turn = self.store.get_turn(session.active_turn_id)
            age = self._turn_age_seconds(turn)
            if age is not None and age < _ORPHAN_SWEEP_MIN_AGE_SECONDS:
                continue
            if not self._session_turn_lock_is_free(session.id):
                continue
            self._fail_orphaned_turn(session.id, session.active_turn_id)
            reconciled += 1
            logger.info(
                "Reconciled orphaned Claude Code turn session_id=%s turn_id=%s",
                session.id,
                session.active_turn_id or "",
            )
        return reconciled

    def _reconcile_orphaned_turns_once(self) -> None:
        if self._orphan_sweep_done:
            return
        self._orphan_sweep_done = True
        try:
            self.reconcile_orphaned_turns()
        except Exception:
            logger.warning("Orphaned-turn reconciliation failed", exc_info=True)

    @contextlib.asynccontextmanager
    async def _cross_process_session_lock(self, session_id: str):
        """Serialize a session's turns across processes.

        Several processes can hold client instances for the same store (a
        pool of voice workers plus long-lived MCP servers). Each keeps its
        own Claude CLI subprocess, and every CLI appends conversation entries
        to the one shared session transcript; unserialized turns interleave
        parent chains, and the next resume then forks from a stale leaf and
        silently drops the other writer's turns from the visible
        conversation. flock releases on process death, so a crashed worker
        cannot wedge the session.
        """
        if fcntl is None:
            yield
            return
        lock_dir = self.store.path.parent / "session-locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(lock_dir / f"{session_id}.lock", os.O_CREAT | os.O_RDWR, 0o644)
        try:
            await asyncio.to_thread(fcntl.flock, fd, fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(OSError):
                fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
