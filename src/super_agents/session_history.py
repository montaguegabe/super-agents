"""Conversation export through supported vendor interfaces only."""

from __future__ import annotations

import asyncio
import json
from typing import Any

MAX_HISTORY_BYTES = 8_000_000


async def read_session_messages(client: Any, thread_id: str) -> list[dict[str, str]]:
    """Return chronological visible messages; never parse vendor session files."""
    messages: list[dict[str, str]] = []
    size = 0

    def append(role: str, text: str) -> None:
        nonlocal size
        if not text:
            return
        size += len(text.encode("utf-8"))
        if size > MAX_HISTORY_BYTES:
            raise ValueError("Conversation is too large to transfer safely (8 MB limit).")
        messages.append({"role": role, "text": text})

    if callable(getattr(client, "read_thread_page", None)):
        cursor = None
        seen: set[str] = set()
        while True:
            page = await client.read_thread_page(
                thread_id, limit=50, cursor=cursor, items_view="full", sort_direction="asc"
            )
            thread = page.get("thread", {})
            for turn in thread.get("turns", []):
                for item in turn.get("items", []):
                    kind = item.get("type")
                    if kind == "userMessage":
                        append("user", _content_text(item.get("content", [])))
                    elif kind == "agentMessage":
                        append("assistant", str(item.get("text") or ""))
                    elif kind in {"commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall"}:
                        # Keep outcomes, without copying hidden reasoning or native tool handles.
                        append(
                            "tool",
                            json.dumps(
                                {
                                    k: item[k]
                                    for k in (
                                        "type",
                                        "command",
                                        "status",
                                        "exitCode",
                                        "aggregatedOutput",
                                        "changes",
                                        "result",
                                    )
                                    if k in item
                                },
                                ensure_ascii=False,
                            ),
                        )
            cursor = thread.get("historyNextCursor")
            if not cursor:
                break
            if cursor in seen:
                raise RuntimeError("Backend returned a repeated history cursor.")
            seen.add(cursor)
        return messages

    session = client.store.get_session(thread_id)
    if not session.backend_session_id:
        if client.store.list_turns(thread_id, limit=1):
            raise RuntimeError("Claude session history is not available yet. Wait for the turn to finish.")
        return []
    sdk = client._sdk_loader()
    reader = getattr(sdk, "get_session_messages", None)
    if not callable(reader):
        raise RuntimeError(
            "Update claude-agent-sdk to a version with get_session_messages before transferring history."
        )
    offset = 0
    while True:
        page = await asyncio.to_thread(
            reader, session.backend_session_id, directory=session.cwd, limit=50, offset=offset
        )
        for message in page:
            append(message.type, _content_text(message.message.get("content", [])))
        if len(page) < 50:
            break
        offset += len(page)
    if not messages and client.store.list_turns(thread_id, limit=1):
        raise RuntimeError("Claude has not exposed this conversation through its history API yet.")
    return messages


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for block in content if isinstance(content, list) else []:
        kind = block.get("type")
        if kind == "text":
            parts.append(str(block.get("text") or ""))
        elif kind == "tool_result":
            parts.append(_content_text(block.get("content", [])))
        elif kind in {"image", "image_url", "localImage", "document"}:
            parts.append("[Attachment omitted from conversation transfer]")
    return "\n".join(parts)
