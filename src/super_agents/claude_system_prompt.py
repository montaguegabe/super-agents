"""Compose session policy without replacing Claude's configured base prompt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def supports_refreshable_system_prompt(sdk: Any) -> bool:
    # Older SDKs silently omit this initialize-protocol field. Keep their
    # existing per-query instruction path instead of freezing new policy in
    # a session's first system-prompt snapshot. No dependency uplift required.
    types = getattr(sdk, "types", sdk)
    preset = getattr(types, "SystemPromptPreset", None)
    custom = getattr(types, "SystemPromptCustom", None)
    return all("snapshot" in getattr(kind, "__annotations__", {}) for kind in (preset, custom))


def compose_system_prompt(base: Any, policy: str | None, sdk: Any) -> Any:
    if not policy or not supports_refreshable_system_prompt(sdk):
        return base
    if isinstance(base, dict) and base.get("type") == "preset":
        return {
            **base,
            "append": _join(base.get("append"), policy),
            "snapshot": False,
        }
    if isinstance(base, dict) and base.get("type") == "file":
        base = Path(base["path"]).read_text(encoding="utf-8")
    elif isinstance(base, dict) and base.get("type") == "custom":
        base = base.get("prompt")
    return {"type": "custom", "prompt": _join(base, policy), "snapshot": False}


def system_prompt_fingerprint(prompt: Any) -> str:
    # Fingerprint effective bytes, including a replace-mode file's contents.
    # Do not log policy text or leak it through cache diagnostics.
    if isinstance(prompt, dict) and prompt.get("type") == "file":
        prompt = {**prompt, "content": Path(prompt["path"]).read_text(encoding="utf-8")}
    return hashlib.sha256(json.dumps(prompt, sort_keys=True).encode()).hexdigest()


def _join(base: str | None, policy: str) -> str:
    return "\n\n".join(part for part in (base, policy) if part)
