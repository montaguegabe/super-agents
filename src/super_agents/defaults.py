from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

JsonObject = dict[str, Any]

CODEX_BACKEND = "codex"
OPENBASE_CLOUD_BACKEND = "openbase_cloud"
CLAUDE_CODE_BACKEND = "claude_code"
CODEX_COMPATIBLE_BACKENDS = {CODEX_BACKEND, OPENBASE_CLOUD_BACKEND}
CODING_BACKEND_ENV_KEY = "OPENBASE_CODING_BACKEND"
LEGACY_CODEX_BACKEND_ENV_KEY = "OPENBASE_CODEX_BACKEND"
DEFAULT_ENV_FILE = Path.home() / ".openbase" / ".env"
REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}
CODEX_SERVICE_TIERS = {"fast", "standard"}
DEFAULT_CODEX_SERVICE_TIER = "standard"
CLAUDE_MODEL_ALIASES = {"fable", "opus", "sonnet", "haiku"}

logger = logging.getLogger(__name__)


def default_super_agents_reasoning_effort() -> str:
    payload = default_dispatcher_config()
    value = payload.get("super_agents_reasoning_effort") or payload.get(
        "superAgentsReasoningEffort"
    )
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
    selected_backend = execution_backend(backend or backend_from_environment())
    return _model_for_backend(
        _backend_model(
            payload,
            "super_agents",
            backend=selected_backend,
        ),
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
    configured = os.environ.get("SUPER_AGENTS_DEFAULT_CONFIG_PATH") or os.environ.get(
        "LIVEKIT_DISPATCHER_CONFIG_PATH"
    )
    if configured:
        return Path(configured).expanduser()
    current = Path.home() / ".openbase" / "dispatcher-config.json"
    legacy = Path.home() / ".openbase" / "codex_home" / "dispatcher-config.json"
    return current if current.exists() else legacy


def backend_from_environment() -> str:
    env_values = _env_file_values(DEFAULT_ENV_FILE)
    return execution_backend(
        normalize_backend(
            os.environ.get(CODING_BACKEND_ENV_KEY)
            or os.environ.get(LEGACY_CODEX_BACKEND_ENV_KEY)
            or env_values.get(CODING_BACKEND_ENV_KEY)
            or env_values.get(LEGACY_CODEX_BACKEND_ENV_KEY)
        )
    )


def execution_backend(backend: str) -> str:
    return CODEX_BACKEND if backend in CODEX_COMPATIBLE_BACKENDS else backend


def normalize_backend(value: str | None) -> str:
    raw = (value or "").strip().lower()
    aliases = {
        "": CODEX_BACKEND,
        "openai": CODEX_BACKEND,
        "codex": CODEX_BACKEND,
        "openbase": OPENBASE_CLOUD_BACKEND,
        "openbase-cloud": OPENBASE_CLOUD_BACKEND,
        "openbase_cloud": OPENBASE_CLOUD_BACKEND,
        "cloud": OPENBASE_CLOUD_BACKEND,
        "claude": CLAUDE_CODE_BACKEND,
        "claude-code": CLAUDE_CODE_BACKEND,
        "claude_code": CLAUDE_CODE_BACKEND,
        "claude-agent": CLAUDE_CODE_BACKEND,
        "claude-agent-sdk": CLAUDE_CODE_BACKEND,
        "claude_agent_sdk": CLAUDE_CODE_BACKEND,
        "claude-sdk": CLAUDE_CODE_BACKEND,
        "claude-tui": CLAUDE_CODE_BACKEND,
        "claude-code-tui": CLAUDE_CODE_BACKEND,
    }
    try:
        return aliases[raw]
    except KeyError as exc:
        supported = ", ".join(
            sorted({CODEX_BACKEND, OPENBASE_CLOUD_BACKEND, CLAUDE_CODE_BACKEND})
        )
        raise ValueError(
            f"Unsupported {CODING_BACKEND_ENV_KEY}: {value}. Supported backends: {supported}."
        ) from exc


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
    if selected_backend in {CODEX_BACKEND, OPENBASE_CLOUD_BACKEND} and (
        normalized_model in CLAUDE_MODEL_ALIASES or normalized_model.startswith("claude-")
    ):
        logger.warning(
            "Ignoring Claude Super Agents model %s for Codex backend; using Codex default model",
            model,
        )
        return None
    return model


def _env_file_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values
