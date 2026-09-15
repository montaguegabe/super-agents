from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Sequence
from pathlib import Path

from .app_formatting import turn_text_preview, without_none
from .app_protocol import (
    extract_model,
    extract_thread_cwd,
    extract_thread_id,
    super_agent_label,
    with_base_instructions,
    with_super_agent_identity_instructions,
)
from .app_time import iso_now, path_basename
from .rollout_history import needs_rollout_turn_fallback, rollout_fallback_turns
from .state import JsonObject, get_string

logger = logging.getLogger(__name__)


class ThreadLifecycleMixin:
    async def start_thread(self, input_data: JsonObject) -> JsonObject:
        dispatch_id = str(input_data.get("_mcpCallId") or "")
        started = time.monotonic()
        logger.info(
            "dispatch_timing stage=super_agents_thread_start_request dispatch_id=%s "
            "name=%s agent_name=%s cwd_basename=%s",
            dispatch_id,
            input_data.get("name") or input_data.get("label") or "",
            input_data.get("agentName") or "",
            path_basename(str(input_data.get("cwd") or Path.home())),
        )
        await self.ensure_connected()
        name = super_agent_label(input_data.get("name") or input_data.get("label"))
        agent_name = super_agent_label(input_data.get("agentName"))
        model = input_data.get("model") or self.default_model
        params: JsonObject = {
            "cwd": input_data.get("cwd") or str(Path.home()),
            "model": model,
            "config": await self._login_shell_config_override(None, model if isinstance(model, str) else None),
            **self.permission_overrides(input_data),
        }
        if developer_instructions := with_super_agent_identity_instructions(
            with_base_instructions(get_string(input_data, "developerInstructions")),
            name,
            agent_name=agent_name,
        ):
            params["developerInstructions"] = developer_instructions

        result = await self.request(
            "thread/start",
            params,
            context={
                "dispatchId": dispatch_id,
                "name": name,
            },
        )
        thread_id = extract_thread_id(result)
        logger.info(
            "dispatch_timing stage=super_agents_thread_start_response dispatch_id=%s "
            "thread_id=%s name=%s agent_name=%s elapsed_ms=%d",
            dispatch_id,
            thread_id or "",
            name or "",
            agent_name or "",
            int((time.monotonic() - started) * 1000),
        )
        if thread_id:
            if name:
                await self.set_thread_name(thread_id, name)
            now = iso_now()
            await self.remember_session(
                thread_id,
                {
                    "label": name,
                    "agentName": agent_name,
                    "threadId": thread_id,
                    "cwd": extract_thread_cwd(result) or str(params["cwd"]),
                    "group": input_data.get("group"),
                    "model": extract_model(result) or self.default_model,
                    "createdAt": now,
                    "lastStatus": "unknown",
                },
            )
        return result

    async def set_thread_name(self, thread_id: str, name: str) -> JsonObject:
        await self.ensure_connected()
        return await self.request("thread/name/set", {"threadId": thread_id, "name": name})

    async def resume_thread(
        self,
        thread_id: str,
        *,
        label: str | None = None,
        agent_name: str | None = None,
        developer_instructions: str | None = None,
    ) -> JsonObject:
        started = time.monotonic()
        logger.info(
            "dispatch_timing stage=super_agents_thread_resume_request thread_id=%s",
            thread_id,
        )
        await self.ensure_connected()
        params: JsonObject = {
            "threadId": thread_id,
            "config": await self._login_shell_config_override(thread_id),
        }
        if identity_instructions := with_super_agent_identity_instructions(
            with_base_instructions(developer_instructions),
            label,
            thread_id,
            agent_name,
        ):
            params["developerInstructions"] = identity_instructions
        result = await self.request(
            "thread/resume",
            params,
        )
        await self.merge_session(
            thread_id,
            {
                "threadId": thread_id,
                "agentName": agent_name,
                "model": extract_model(result) or self.default_model,
                "lastUsefulMessage": turn_text_preview(result),
            },
        )
        logger.info(
            "dispatch_timing stage=super_agents_thread_resume_response thread_id=%s elapsed_ms=%d",
            thread_id,
            int((time.monotonic() - started) * 1000),
        )
        return result

    async def list_threads(
        self,
        use_state_db_only: bool = True,
        search_term: str | None = None,
        cwd: str | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        model_providers: Sequence[str] | None = (),
    ) -> JsonObject:
        """List threads via thread/list.

        ``model_providers`` defaults to an empty list, which disables the
        app server's default filter to the active model provider — threads
        belong to the user regardless of which provider ran them. Pass a
        list of provider ids to scope, or ``None`` to keep the server's
        active-provider default. Older servers ignore the parameter.
        """
        await self.ensure_connected()
        providers = list(model_providers) if model_providers is not None else None
        return await self.request(
            "thread/list",
            without_none(
                {
                    "useStateDbOnly": use_state_db_only,
                    "searchTerm": search_term,
                    "cwd": cwd,
                    "limit": limit,
                    "cursor": cursor,
                    "modelProviders": providers,
                }
            ),
        )

    async def read_thread(self, thread_id: str, include_turns: bool = True) -> JsonObject:
        started = time.monotonic()
        await self.ensure_connected()
        try:
            result = await self.request("thread/read", {"threadId": thread_id, "includeTurns": include_turns})
        except RuntimeError as exc:
            if not include_turns or "list_turns is not supported yet" not in str(exc):
                raise
            # Some Codex builds expose metadata but cannot hydrate a newly
            # created thread's turns. Read its existing rollout without
            # resuming it or changing the user's session configuration.
            result = await self.request("thread/read", {"threadId": thread_id, "includeTurns": False})
            thread = result.get("thread")
            if isinstance(thread, dict):
                path = thread.get("path")
                thread["turns"] = await asyncio.to_thread(rollout_fallback_turns, path) if path else []
        thread = result.get("thread") if isinstance(result, dict) else None
        if needs_rollout_turn_fallback(thread, include_turns):
            turns = await asyncio.to_thread(rollout_fallback_turns, thread["path"])
            if turns:
                thread["turns"] = turns
        logger.info(
            "dispatch_timing stage=super_agents_thread_read_response thread_id=%s include_turns=%s elapsed_ms=%d",
            thread_id,
            include_turns,
            int((time.monotonic() - started) * 1000),
        )
        return result

    async def read_thread_page(
        self,
        thread_id: str,
        *,
        limit: int,
        cursor: str | None = None,
        items_view: str = "summary",
        sort_direction: str = "desc",
    ) -> JsonObject:
        """Read thread metadata plus one bounded page of retained turns.

        Paginated Codex threads can be hundreds of megabytes on disk. Asking
        ``thread/read`` to inline every turn can therefore exceed the app-server
        transport limit before a caller has a chance to trim the result.
        ``thread/turns/list`` performs that bound inside Codex and is the
        supported history path for paginated threads.
        """
        started = time.monotonic()
        await self.ensure_connected()
        result = await self.request(
            "thread/read",
            {"threadId": thread_id, "includeTurns": False},
        )
        thread = result.get("thread") if isinstance(result, dict) else None
        if not isinstance(thread, dict):
            return result

        page = await self.request(
            "thread/turns/list",
            without_none(
                {
                    "threadId": thread_id,
                    "limit": limit,
                    "cursor": cursor,
                    "itemsView": items_view,
                    "sortDirection": sort_direction,
                }
            ),
        )
        turns = page.get("data") if isinstance(page, dict) else None
        thread["turns"] = turns if isinstance(turns, list) else []
        thread["historyNextCursor"] = page.get("nextCursor")
        thread["historyBackwardsCursor"] = page.get("backwardsCursor")
        thread["historyItemsView"] = items_view
        logger.info(
            "dispatch_timing stage=super_agents_thread_page_response "
            "thread_id=%s turn_count=%d has_next=%s elapsed_ms=%d",
            thread_id,
            len(thread["turns"]),
            bool(thread["historyNextCursor"]),
            int((time.monotonic() - started) * 1000),
        )
        return result
