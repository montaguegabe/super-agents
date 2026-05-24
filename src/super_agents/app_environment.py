from __future__ import annotations

import asyncio
import os
import subprocess
import urllib.request

from .state import JsonObject

LOGIN_ENV_TIMEOUT_SECONDS = 5


def _check_ready_sync(url: str) -> bool:
    with urllib.request.urlopen(url, timeout=1) as response:
        return 200 <= response.status < 300


_login_shell_environment_task: asyncio.Task[dict[str, str]] | None = None


async def login_shell_config_override(
    *,
    thread_id: str | None = None,
    label: str | None = None,
    agent_name: str | None = None,
    include_super_agent_identity: bool = True,
) -> JsonObject:
    env = await login_shell_environment()
    set_values = {key: value for key in ["PATH", "SHELL", "HOME", "USER", "LOGNAME"] if (value := env.get(key))}
    return {"shell_environment_policy": {"inherit": "all", "set": set_values}}


async def login_shell_environment() -> dict[str, str]:
    global _login_shell_environment_task
    if _login_shell_environment_task is None:
        _login_shell_environment_task = asyncio.create_task(read_login_shell_environment())
    try:
        return await _login_shell_environment_task
    except Exception as exc:
        print(f"[super-agents] Failed to read login shell environment: {exc}", file=os.sys.stderr)
        return dict(os.environ)


async def read_login_shell_environment() -> dict[str, str]:
    shell = os.environ.get("SHELL") if os.environ.get("SHELL", "").startswith("/") else "/bin/zsh"

    def run() -> dict[str, str]:
        proc = subprocess.run(
            [shell, "-lic", "/usr/bin/env -0"],
            env=os.environ,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=LOGIN_ENV_TIMEOUT_SECONDS,
            check=True,
        )
        return {**os.environ, **parse_null_separated_env(proc.stdout.decode("utf-8", errors="replace"))}

    return await asyncio.to_thread(run)


def parse_null_separated_env(output: str) -> dict[str, str]:
    env: dict[str, str] = {}
    for entry in output.split("\0"):
        if "=" not in entry:
            continue
        key, value = entry.split("=", 1)
        if key:
            env[key] = value
    return env


def websocket_is_open(ws: object | None) -> bool:
    if ws is None:
        return False
    closed = getattr(ws, "closed", None)
    if isinstance(closed, bool):
        return not closed
    state = getattr(ws, "state", None)
    return state == 1 or str(state).endswith(".OPEN")
