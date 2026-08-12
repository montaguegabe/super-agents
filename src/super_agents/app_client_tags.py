from __future__ import annotations

from typing import Any

from .item_tags import (
    report_tags,
    set_report_tags,
    set_thread_tags,
    tag_options,
    thread_tags,
)
from .state import JsonObject
from .thread_favorites import favorite_status


class TagsFavoritesMixin:
    async def thread_favorite(self, thread_id: str) -> JsonObject:
        return favorite_status(thread_id)

    async def tags(self) -> JsonObject:
        return tag_options()

    async def thread_tags(self, thread_id: str, tags: list[Any] | None = None) -> JsonObject:
        if tags is None:
            return thread_tags(thread_id)
        return set_thread_tags(thread_id, tags)

    async def report_tags(
        self,
        project_path: str,
        path: str,
        tags: list[Any] | None = None,
    ) -> JsonObject:
        if tags is None:
            return report_tags(project_path, path)
        return set_report_tags(project_path, path, tags)
