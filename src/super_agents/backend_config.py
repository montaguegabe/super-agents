"""Single source of truth for Super Agents coding-backend selection.

Backend constants, the alias map, ``.env`` parsing, and environment-based
backend resolution live here. ``defaults``, ``backend_clients``, and
``backend`` re-export these names for compatibility.
"""

from __future__ import annotations

import os
from pathlib import Path

CODEX_BACKEND = "codex"
OPENBASE_CLOUD_BACKEND = "openbase_cloud"
OPENBASE_CLOUD_CODEX_BACKEND = "openbase_cloud_codex"
CLAUDE_CODE_BACKEND = "claude_code"
BACKENDS = {
    CODEX_BACKEND,
    OPENBASE_CLOUD_BACKEND,
    CLAUDE_CODE_BACKEND,
    OPENBASE_CLOUD_CODEX_BACKEND,
}
CODEX_COMPATIBLE_BACKENDS = {CODEX_BACKEND, OPENBASE_CLOUD_CODEX_BACKEND}
CODING_BACKEND_ENV_KEY = "OPENBASE_CODING_BACKEND"
DEFAULT_ENV_FILE = Path.home() / ".openbase" / ".env"
BACKEND_ALIASES = {
    "": CODEX_BACKEND,
    "codex": CODEX_BACKEND,
    "codecs": CODEX_BACKEND,
    "openbase cloud": OPENBASE_CLOUD_BACKEND,
    "claude code": CLAUDE_CODE_BACKEND,
    "cloud code": CLAUDE_CODE_BACKEND,
    "openbase cloud codex": OPENBASE_CLOUD_CODEX_BACKEND,
    "openbase cloud codecs": OPENBASE_CLOUD_CODEX_BACKEND,
    "codex via openbase cloud": OPENBASE_CLOUD_CODEX_BACKEND,
}


def normalize_backend(value: str | None) -> str:
    raw = " ".join((value or "").strip().lower().replace("_", " ").replace("-", " ").split())
    try:
        return BACKEND_ALIASES[raw]
    except KeyError as exc:
        supported = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unsupported {CODING_BACKEND_ENV_KEY}: {value}. Supported backends: {supported}.") from exc


def execution_backend(backend: str) -> str:
    if backend in CODEX_COMPATIBLE_BACKENDS:
        return CODEX_BACKEND
    if backend == OPENBASE_CLOUD_BACKEND:
        return CLAUDE_CODE_BACKEND
    return backend


def configured_backend_from_environment() -> str:
    env_values = read_env_values(DEFAULT_ENV_FILE)
    return normalize_backend(os.environ.get(CODING_BACKEND_ENV_KEY) or env_values.get(CODING_BACKEND_ENV_KEY))


def backend_from_environment() -> str:
    return execution_backend(configured_backend_from_environment())


def read_env_values(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        values[key.strip()] = _unquote_env_value(value.strip())
    return values


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value
