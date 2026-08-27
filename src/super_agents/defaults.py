from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from .backend_config import (  # noqa: F401  (re-exported for compatibility)
    CLAUDE_CODE_BACKEND,
    CODEX_BACKEND,
    CODEX_COMPATIBLE_BACKENDS,
    CODING_BACKEND_ENV_KEY,
    DEFAULT_ENV_FILE,
    OPENBASE_CLOUD_BACKEND,
    backend_from_environment,
    configured_backend_from_environment,
    execution_backend,
    normalize_backend,
    CLAUDE_MODEL_ALIASES,
)

JsonObject = dict[str, Any]

REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}
CODEX_SERVICE_TIERS = {"fast", "standard"}
DEFAULT_CODEX_SERVICE_TIER = "standard"
DEFAULT_OPENBASE_CLOUD_CLAUDE_MODEL = "claude-sonnet-5"

logger = logging.getLogger(__name__)


def default_super_agents_reasoning_effort() -> str:
    payload = default_dispatcher_config()
    value = payload.get("super_agents_reasoning_effort") or payload.get("superAgentsReasoningEffort")
    return value if isinstance(value, str) and value in REASONING_EFFORTS else "high"


def default_service_tier() -> str:
    payload = default_dispatcher_config()
    value = payload.get("codex_service_tier") or payload.get("codexServiceTier")
    if isinstance(value, str) and value in CODEX_SERVICE_TIERS:
        return value
    env_value = os.environ.get("CODEX_SERVICE_TIER", "").strip()
    if env_value in CODEX_SERVICE_TIERS:
        return env_value
    return DEFAULT_CODEX_SERVICE_TIER


def default_super_agents_model(*, backend: str | None = None) -> str | None:
    payload = default_dispatcher_config()
    configured_backend = normalize_backend(backend or configured_backend_from_environment())
    selected_backend = execution_backend(configured_backend)
    configured_model = _backend_model(
        payload,
        "super_agents",
        backend=configured_backend,
    ) or _backend_model(
        payload,
        "super_agents",
        backend=selected_backend,
    )
    if configured_backend == OPENBASE_CLOUD_BACKEND and not configured_model:
        configured_model = DEFAULT_OPENBASE_CLOUD_CLAUDE_MODEL
    return _model_for_backend(
        configured_model,
        backend=selected_backend,
    )


def default_dispatcher_config() -> JsonObject:
    path = default_dispatcher_config_path()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def default_dispatcher_config_path() -> Path:
    configured = os.environ.get("SUPER_AGENTS_DEFAULT_CONFIG_PATH") or os.environ.get("LIVEKIT_DISPATCHER_CONFIG_PATH")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".openbase" / "dispatcher-config.json"


def _backend_model(payload: JsonObject, role: str, *, backend: str) -> str | None:
    backend_models = payload.get("backend_models") or payload.get("backendModels")
    if not isinstance(backend_models, dict):
        return None
    model_config = backend_models.get(backend)
    if not isinstance(model_config, dict):
        return None
    value = model_config.get(role)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _model_for_backend(model: str | None, *, backend: str | None = None) -> str | None:
    if not model:
        return None
    selected_backend = backend or backend_from_environment()
    normalized_model = model.strip().lower()
    if selected_backend in CODEX_COMPATIBLE_BACKENDS and (
        normalized_model in CLAUDE_MODEL_ALIASES or normalized_model.startswith("claude-")
    ):
        logger.warning(
            "Ignoring Claude Super Agents model %s for Codex backend; using Codex default model",
            model,
        )
        return None
    return model
