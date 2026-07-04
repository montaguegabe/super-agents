from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from ._package_version import package_version
from .backend_config import (  # noqa: F401  (re-exported for compatibility)
    BACKEND_ALIASES,
    BACKENDS,
    CLAUDE_CODE_BACKEND,
    CODEX_BACKEND,
    CODING_BACKEND_ENV_KEY,
    DEFAULT_ENV_FILE,
    OPENBASE_CLOUD_BACKEND,
    normalize_backend,
    read_env_values,
)


@dataclass(frozen=True)
class BackendStatus:
    env_file: Path
    backend: str
    exists: bool


def current_backend(path: Path = DEFAULT_ENV_FILE) -> BackendStatus:
    values = read_env_values(path)
    backend = (
        values.get(CODING_BACKEND_ENV_KEY)
        or os.environ.get(CODING_BACKEND_ENV_KEY)
        or CODEX_BACKEND
    )
    try:
        backend = normalize_backend(backend)
    except ValueError:
        backend = f"unsupported:{backend}"
    if backend == OPENBASE_CLOUD_BACKEND:
        backend = CODEX_BACKEND
    return BackendStatus(env_file=path, backend=backend, exists=path.is_file())


def set_backend(backend: str, path: Path = DEFAULT_ENV_FILE) -> BackendStatus:
    normalized = normalize_backend(backend)

    update_env_file(path, {CODING_BACKEND_ENV_KEY: normalized})
    return current_backend(path)


def update_env_file(path: Path, values: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    remaining = dict(values)
    updated: list[str] = []
    for line in lines:
        key = _active_env_key(line)
        if key in remaining:
            updated.append(f"{key}={_format_env_value(remaining.pop(key))}")
        else:
            updated.append(line)
    if updated and updated[-1].strip():
        updated.append("")
    for key, value in remaining.items():
        updated.append(f"{key}={_format_env_value(value)}")
    path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")


def _active_env_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    key, _value = stripped.split("=", 1)
    key = key.strip()
    return key if key else None


def _format_env_value(value: str) -> str:
    if not value or any(char.isspace() for char in value) or any(char in value for char in ['"', "'", "#"]):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="super-agents-backend")
    parser.add_argument(
        "--version", action="version", version=f"super-agents {package_version()}"
    )
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    use_parser = subparsers.add_parser("use")
    use_parser.add_argument("backend")
    args = parser.parse_args(argv)

    if args.command == "status":
        status = current_backend(args.env_file)
        exists = "exists" if status.exists else "missing"
        print(f"Backend: {status.backend}")
        print(f"Env file: {status.env_file} ({exists})")
        return 0

    if args.command == "use":
        status = set_backend(args.backend, args.env_file)
        print(f"Backend set to {status.backend} in {status.env_file}.")
        if status.backend == CLAUDE_CODE_BACKEND:
            print(
                "Restart the Super Agents MCP server for Claude Code mode. codex-app-server is not used by this backend."
            )
        else:
            print("Restart codex-app-server for the change to apply.")
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
