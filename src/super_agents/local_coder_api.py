"""Read-only helpers for the local Openbase Coder server's HTTP API.

The local server proxies team data from openbase-cloud with its own
credentials, so this module never handles cloud tokens.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_LOCAL_SERVER_URL = "http://127.0.0.1:7999"


def local_server_url() -> str:
    return os.environ.get(
        "OPENBASE_CODER_LOCAL_SERVER_URL", DEFAULT_LOCAL_SERVER_URL
    ).rstrip("/")


async def fetch_team_activity() -> dict[str, Any]:
    """Fetch the merged teammate activity feed via the local proxy.

    Never raises: unreachable or unsupported backends return
    {"supported": False, "error": ...} so agent turns degrade gracefully.
    """
    url = f"{local_server_url()}/api/team/activity/"
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.get(url)
    except httpx.HTTPError as exc:
        return {"supported": False, "error": f"local server unreachable: {exc}"}
    if response.status_code >= 400:
        return {"supported": False, "error": f"HTTP {response.status_code}"}
    try:
        data = response.json()
    except ValueError:
        return {"supported": False, "error": "invalid response from local server"}
    if not isinstance(data, dict):
        return {"supported": False, "error": "unexpected response shape"}
    return data


def filter_team_activity(
    data: dict[str, Any],
    repo: str | None = None,
    include_offline: bool = False,
) -> dict[str, Any]:
    """Client-side filtering for the team_activity tool."""
    if not data.get("supported", True):
        return data
    members = []
    for member in data.get("members", []):
        if not include_offline and not member.get("online"):
            continue
        if repo:
            devices = []
            for device in member.get("devices", []):
                threads = [
                    thread
                    for thread in device.get("threads", [])
                    if thread.get("repo") == repo
                ]
                repos = [
                    entry
                    for entry in device.get("repos", [])
                    if entry.get("name") == repo
                ]
                if threads or repos:
                    devices.append({**device, "threads": threads, "repos": repos})
            if not devices:
                continue
            member = {**member, "devices": devices}
        members.append(member)
    return {**data, "members": members}
