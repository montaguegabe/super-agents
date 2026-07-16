"""Read and write Openbase's local shared tag metadata."""

from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

TAGS_FILE = "item-tags.json"
ItemKind = Literal["thread", "report"]


def tag_options() -> dict[str, list[dict[str, Any]]]:
    payload = read_tags()
    usage = _tag_usage_counts(payload)
    return {
        "tags": [
            {**option, "usageCount": usage.get(str(option["slug"]), 0)}
            for option in sorted(
                payload["options"].values(),
                key=lambda item: str(item["label"]).lower(),
            )
        ]
    }


def thread_tags(thread_id: str | None) -> dict[str, Any]:
    return _item_tags("thread", _normalize_item_id(thread_id))


def set_thread_tags(thread_id: str, tags: list[Any]) -> dict[str, Any]:
    normalized = _normalize_item_id(thread_id)
    if not normalized:
        raise ValueError("threadId is required.")
    return _set_item_tags("thread", normalized, tags)


def report_tags(project_path: str | None, path: str | None) -> dict[str, Any]:
    project = _normalize_item_id(project_path)
    report_path = _normalize_report_path(path)
    payload = _item_tags("report", _report_item_id(project, report_path))
    payload.update({"projectPath": project, "path": report_path})
    return payload


def set_report_tags(project_path: str, path: str, tags: list[Any]) -> dict[str, Any]:
    project = _normalize_item_id(project_path)
    report_path = _normalize_report_path(path)
    if not project:
        raise ValueError("projectPath is required.")
    if not report_path:
        raise ValueError("path is required.")
    payload = _set_item_tags("report", _report_item_id(project, report_path), tags)
    payload.update({"projectPath": project, "path": report_path})
    return payload


def read_tags() -> dict[str, Any]:
    path = tags_path()
    try:
        raw_payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        raw_payload = {}
    if not isinstance(raw_payload, dict):
        raw_payload = {}
    return {
        "options": _read_options(raw_payload.get("options")),
        "threads": _read_assignments(raw_payload.get("threads"), "thread"),
        "reports": _read_assignments(raw_payload.get("reports"), "report"),
    }


def tags_path() -> Path:
    data_dir = os.environ.get("OPENBASE_CODER_CLI_DATA_DIR")
    return Path(data_dir).expanduser() / TAGS_FILE if data_dir else Path.home() / ".openbase" / TAGS_FILE


def _item_tags(kind: ItemKind, item_id: str) -> dict[str, Any]:
    payload = read_tags()
    entry = payload[f"{kind}s"].get(item_id) if item_id else None
    slugs = entry.get("tags") if isinstance(entry, dict) else []
    tags = [payload["options"][slug]["label"] for slug in slugs if isinstance(slug, str) and slug in payload["options"]]
    return {
        f"{kind}Id": item_id,
        "tags": tags,
        "updatedAt": entry.get("updated_at") if isinstance(entry, dict) else None,
        "tagOptions": tag_options()["tags"],
    }


def _set_item_tags(kind: ItemKind, item_id: str, raw_tags: list[Any]) -> dict[str, Any]:
    if not isinstance(raw_tags, list):
        raise ValueError("tags must be a list.")
    payload = read_tags()
    normalized_tags = _normalize_tag_inputs(raw_tags, payload["options"])
    now = _utc_now()
    for tag in normalized_tags:
        current = payload["options"].get(tag["slug"])
        payload["options"][tag["slug"]] = {
            "slug": tag["slug"],
            "label": current.get("label") if current else tag["label"],
            "created_at": current.get("created_at") if current else now,
            "updated_at": now,
        }
    assignments = payload[f"{kind}s"]
    if normalized_tags:
        assignments[item_id] = {
            f"{kind}_id": item_id,
            "tags": [tag["slug"] for tag in normalized_tags],
            "updated_at": now,
        }
    else:
        assignments.pop(item_id, None)
    _write_tags(payload)
    return _item_tags(kind, item_id)


def _write_tags(payload: dict[str, Any]) -> None:
    path = tags_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as tmp:
        json.dump(payload, tmp, indent=2, sort_keys=True)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def _normalize_tag_inputs(
    raw_tags: list[Any],
    options: dict[str, dict[str, Any]],
) -> list[dict[str, str]]:
    tags: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_tag in raw_tags:
        label = raw_tag.strip() if isinstance(raw_tag, str) else ""
        if not label:
            continue
        slug = _slugify(label)
        if not slug or slug in seen:
            continue
        seen.add(slug)
        existing = options.get(slug)
        tags.append(
            {
                "slug": slug,
                "label": str(existing.get("label")) if existing else label,
            }
        )
    return tags


def _read_options(raw_options: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_options, dict):
        return {}
    options: dict[str, dict[str, Any]] = {}
    for raw_slug, raw_option in raw_options.items():
        if not isinstance(raw_option, dict):
            continue
        slug = _slugify(raw_option.get("slug") or raw_slug)
        label = raw_option.get("label")
        if not slug or not isinstance(label, str) or not label.strip():
            continue
        options[slug] = {
            "slug": slug,
            "label": label.strip(),
            "created_at": _optional_string(raw_option.get("created_at")),
            "updated_at": _optional_string(raw_option.get("updated_at")),
        }
    return options


def _read_assignments(raw_assignments: Any, kind: ItemKind) -> dict[str, dict[str, Any]]:
    if not isinstance(raw_assignments, dict):
        return {}
    assignments: dict[str, dict[str, Any]] = {}
    for raw_item_id, raw_entry in raw_assignments.items():
        item_id = _normalize_item_id(raw_item_id)
        if not item_id or not isinstance(raw_entry, dict):
            continue
        tags = [slug for raw_slug in raw_entry.get("tags", []) if (slug := _slugify(raw_slug))]
        if not tags:
            continue
        assignments[item_id] = {
            f"{kind}_id": item_id,
            "tags": list(dict.fromkeys(tags)),
            "updated_at": _optional_string(raw_entry.get("updated_at")),
        }
    return assignments


def _tag_usage_counts(payload: dict[str, Any]) -> dict[str, int]:
    usage: dict[str, int] = {}
    for assignments in (payload["threads"], payload["reports"]):
        for entry in assignments.values():
            for slug in entry.get("tags", []):
                usage[slug] = usage.get(slug, 0) + 1
    return usage


def _report_item_id(project_path: str, relative_path: str) -> str:
    return f"{project_path}\n{relative_path}" if project_path and relative_path else ""


def _normalize_report_path(value: Any) -> str:
    return str(value).strip().replace("\\", "/") if value is not None else ""


def _normalize_item_id(value: Any) -> str:
    return str(value).strip() if value is not None and str(value).strip() else ""


def _slugify(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")
