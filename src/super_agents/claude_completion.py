"""Interruptible application validation at a provisional Claude drain boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Awaitable


@dataclass
class _Validation:
    task: asyncio.Future[None]
    invalidated: bool = False


class ResultValidation:
    """Cancel stale validation and await its cleanup before reading new work.

    Embedding callbacks must propagate cancellation and revoke any pending
    publication in their cleanup. Already committed external side effects
    cannot be rolled back by the SDK client.
    """

    def __init__(self) -> None:
        self._active: dict[str, _Validation] = {}

    def invalidate(self, session_id: str) -> None:
        validation = self._active.get(session_id)
        if validation is not None and not validation.invalidated:
            validation.invalidated = True
            # Only cancel once: another steer must not interrupt the callback's
            # cancellation cleanup (for example, revoking a speech receipt).
            validation.task.cancel()

    async def run(self, session_id: str, callback: Awaitable[None]) -> bool:
        validation = _Validation(asyncio.ensure_future(callback))
        self._active[session_id] = validation
        try:
            await validation.task
        except asyncio.CancelledError:
            owner = asyncio.current_task()
            if not validation.invalidated or (owner is not None and owner.cancelling()):
                raise
        except Exception:
            # Errors from an obsolete validation cannot fail the accepted new
            # work. The fresh terminal result will be validated independently.
            if not validation.invalidated:
                raise
        finally:
            self._active.pop(session_id, None)
        return not validation.invalidated
