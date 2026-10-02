"""Runtime-driven thread introduction.

A named Super Agent thread announces itself on its first turn by running a
host-configured command, so the introduction never depends on the model
obeying an instruction. The hook is generic: Super Agents only knows a
command template; the product that embeds it decides what the command does.

Configure with ``SUPER_AGENTS_THREAD_INTRO_COMMAND``, a shell-style command
template. ``{agent_name}``, ``{thread_name}`` and ``{thread_id}`` are replaced
inside each argument after the template is split, so values are never
re-parsed by a shell. Example::

    SUPER_AGENTS_THREAD_INTRO_COMMAND='openbase-coder user say {agent_name} "Hey there, I'"'"'m {agent_name}."'

Unset or blank disables the hook. The command runs with the login-shell
environment, is bounded by a timeout, and never raises into the turn.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
from collections.abc import Mapping

from .app_environment import login_shell_environment

logger = logging.getLogger(__name__)

THREAD_INTRO_COMMAND_ENV = "SUPER_AGENTS_THREAD_INTRO_COMMAND"
THREAD_INTRO_TIMEOUT_ENV = "SUPER_AGENTS_THREAD_INTRO_TIMEOUT_SECONDS"
DEFAULT_THREAD_INTRO_TIMEOUT_SECONDS = 15.0
_PLACEHOLDERS = ("agent_name", "thread_name", "thread_id")


def thread_intro_command_template(env: Mapping[str, str] | None = None) -> str | None:
    """The configured template, or None when the hook is disabled."""
    source = os.environ if env is None else env
    raw = (source.get(THREAD_INTRO_COMMAND_ENV) or "").strip()
    return raw or None


def thread_intro_timeout_seconds(env: Mapping[str, str] | None = None) -> float:
    source = os.environ if env is None else env
    raw = (source.get(THREAD_INTRO_TIMEOUT_ENV) or "").strip()
    if not raw:
        return DEFAULT_THREAD_INTRO_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_THREAD_INTRO_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_THREAD_INTRO_TIMEOUT_SECONDS


def render_thread_intro_command(
    template: str,
    *,
    agent_name: str,
    thread_name: str | None = None,
    thread_id: str | None = None,
) -> list[str]:
    """Split the template shell-style, then substitute placeholders per argument.

    Substituting after the split keeps an agent name such as ``O'Brien`` from
    being re-parsed as shell syntax.
    """
    values = {"agent_name": agent_name, "thread_name": thread_name or "", "thread_id": thread_id or ""}
    argv = shlex.split(template)
    if not argv:
        raise ValueError(f"{THREAD_INTRO_COMMAND_ENV} is blank")
    rendered: list[str] = []
    for arg in argv:
        for key in _PLACEHOLDERS:
            arg = arg.replace("{" + key + "}", values[key])
        rendered.append(arg)
    return rendered


def is_first_turn(*, last_turn_id: str | None, active_turn_id: str | None) -> bool:
    """A session that has never recorded a turn is about to run its first one."""
    return not last_turn_id and not active_turn_id


async def announce_thread_intro(
    *,
    agent_name: str | None,
    thread_name: str | None,
    thread_id: str | None,
    env: Mapping[str, str] | None = None,
    timeout_seconds: float | None = None,
) -> bool:
    """Run the configured intro command for a named thread. Returns True when it ran successfully.

    Silent no-op when the hook is disabled or the thread has no agent name.
    Failures and timeouts are logged, never raised: a missing greeting must not
    cost the user their turn.
    """
    template = thread_intro_command_template(env)
    if not template or not agent_name:
        return False
    try:
        argv = render_thread_intro_command(
            template, agent_name=agent_name, thread_name=thread_name, thread_id=thread_id
        )
    except ValueError as exc:
        logger.warning("thread_intro skipped: %s", exc)
        return False
    timeout = timeout_seconds if timeout_seconds is not None else thread_intro_timeout_seconds(env)
    subprocess_env = dict(env) if env is not None else await login_shell_environment()
    logger.info(
        "dispatch_timing stage=thread_intro_start thread_id=%s agent_name=%s argv0=%s",
        thread_id or "",
        agent_name,
        argv[0],
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            env=subprocess_env,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except (OSError, ValueError) as exc:
        logger.warning("thread_intro failed to start for %s: %s", agent_name, exc)
        return False
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        logger.warning("thread_intro timed out after %.1fs for %s", timeout, agent_name)
        return False
    if proc.returncode != 0:
        detail = (stderr or stdout).decode("utf-8", errors="replace").strip()[-400:]
        logger.warning("thread_intro exited %s for %s: %s", proc.returncode, agent_name, detail)
        return False
    logger.info("dispatch_timing stage=thread_intro_done thread_id=%s agent_name=%s", thread_id or "", agent_name)
    return True
