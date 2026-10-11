"""Wait for a thread's working folder to be openable before Claude Code starts.

On macOS the first access to a privacy-protected folder (Desktop, Documents,
Downloads, removable volumes, iCloud Drive) by the Openbase service blocks
inside the kernel until the user answers the consent dialog. A Claude Code
process spawned with such a cwd hangs in ``getcwd()`` before it can answer the
SDK's initialize request, which only surfaces a minute later as "Control
request timeout: initialize" (VM2, 2026-10-11: six voice turns, the user heard
"didn't reach the coding agent" and nothing said why). Probing the folder from
the spawning process first means the very same prompt appears immediately,
the turn is marked blocked within seconds (``error_kind`` =
:data:`BLOCKED_PERMISSION_DIALOG` plus a spoken-ready state message), and the
turn resumes by itself once the dialog is answered, with nothing for the user
to repeat.

The probe is a plain ``os.listdir`` on a daemon thread. A thread stuck in the
kernel cannot be interrupted, so it is deliberately not run on the shared
executor; when the dialog is finally answered the thread exits on its own.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

#: ``Turn.error_kind`` while (or after) a turn is blocked on a consent dialog.
BLOCKED_PERMISSION_DIALOG = "blocked_permission_dialog"

#: How long an ordinary folder listing may take before the turn is reported as
#: blocked. A local directory answers in microseconds; the only thing that
#: makes it take seconds is a pending consent prompt (or a dead network mount).
PROBE_GRACE_SECONDS = 2.0
#: How long a blocked turn waits for the dialog to be answered before failing.
MAX_WAIT_SECONDS = 600.0
POLL_SECONDS = 0.5

_HOME_FOLDERS = ("Desktop", "Documents", "Downloads")


class CwdAccessBlocked(RuntimeError):
    """The working folder stayed inaccessible past :data:`MAX_WAIT_SECONDS`."""


def protected_folder_name(cwd: str) -> str | None:
    """Human name of the protected location ``cwd`` lives in, if any."""
    try:
        path = Path(cwd).expanduser()
        home = Path.home()
    except (OSError, RuntimeError):
        return None
    parts = path.parts
    if "Mobile Documents" in parts and "Library" in parts:
        return "iCloud Drive"
    if len(parts) > 1 and parts[0] == os.sep and parts[1] == "Volumes":
        return "an external volume"
    try:
        relative = path.relative_to(home)
    except ValueError:
        return None
    if relative.parts and relative.parts[0] in _HOME_FOLDERS:
        return relative.parts[0]
    return None


def blocked_message(cwd: str) -> str:
    """What to tell the user while the folder waits on a consent dialog."""
    folder = protected_folder_name(cwd)
    where = f"your {folder} folder" if folder else f"the folder {cwd}"
    return (
        f"macOS is waiting for permission: Openbase Services needs access to {where}. "
        "Click Allow in the dialog on your Mac and the agent will continue."
    )


class CwdAccessProbe:
    """List ``cwd`` on a daemon thread and expose whether it has finished."""

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd
        self.error: OSError | None = None
        self._finished = threading.Event()
        thread = threading.Thread(target=self._run, name=f"cwd-access-probe:{cwd}", daemon=True)
        thread.start()

    def _run(self) -> None:
        try:
            os.listdir(self.cwd)
        except OSError as exc:
            self.error = exc
        finally:
            self._finished.set()

    @property
    def done(self) -> bool:
        return self._finished.is_set()

    async def wait(self, seconds: float) -> bool:
        deadline = time.monotonic() + seconds
        while not self.done:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(POLL_SECONDS, remaining))
        return True


async def await_cwd_access(
    cwd: str,
    *,
    on_blocked: Callable[[str], Awaitable[None]],
    on_unblocked: Callable[[], Awaitable[None]],
    is_cancelled: Callable[[], bool] = lambda: False,
    grace_seconds: float = PROBE_GRACE_SECONDS,
    max_wait_seconds: float = MAX_WAIT_SECONDS,
    probe: CwdAccessProbe | None = None,
) -> None:
    """Return once ``cwd`` can be listed; report a stall through the callbacks.

    A listing that fails outright (missing folder, denied by a dismissed
    dialog) is not a stall: the caller's own start path reports that the way
    it always has. Raises :class:`CwdAccessBlocked` when the folder is still
    inaccessible after ``max_wait_seconds`` and ``asyncio.CancelledError`` when
    ``is_cancelled`` turns true while waiting.
    """
    probe = probe or CwdAccessProbe(cwd)
    if await probe.wait(grace_seconds):
        return
    message = blocked_message(cwd)
    await on_blocked(message)
    started = time.monotonic()
    while not probe.done:
        if is_cancelled():
            raise asyncio.CancelledError
        if time.monotonic() - started >= max_wait_seconds:
            raise CwdAccessBlocked(message)
        await asyncio.sleep(POLL_SECONDS)
    await on_unblocked()
