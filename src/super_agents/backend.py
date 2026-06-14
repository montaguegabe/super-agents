from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_ENV_FILE = Path.home() / ".openbase" / ".env"
CODEX_BACKEND = "codex"
OPENBASE_CLOUD_BACKEND = "openbase_cloud"
CLAUDE_CODE_BACKEND = "claude_code"
CODING_BACKEND_ENV_KEY = "OPENBASE_CODING_BACKEND"
LEGACY_CODEX_BACKEND_ENV_KEY = "OPENBASE_CODEX_BACKEND"
BACKENDS = {CODEX_BACKEND, OPENBASE_CLOUD_BACKEND, CLAUDE_CODE_BACKEND}
BACKEND_ALIASES = {
    "codex": CODEX_BACKEND,
    "openai": CODEX_BACKEND,
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


@dataclass(frozen=True)
class BackendStatus:
    env_file: Path
    backend: str
    exists: bool


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


def current_backend(path: Path = DEFAULT_ENV_FILE) -> BackendStatus:
    values = read_env_values(path)
    backend = (
        values.get(CODING_BACKEND_ENV_KEY)
        or values.get(LEGACY_CODEX_BACKEND_ENV_KEY)
        or os.environ.get(CODING_BACKEND_ENV_KEY)
        or os.environ.get(LEGACY_CODEX_BACKEND_ENV_KEY)
        or CODEX_BACKEND
    )
    backend = BACKEND_ALIASES.get(backend, backend)
    if backend not in BACKENDS:
        backend = f"unsupported:{backend}"
    return BackendStatus(env_file=path, backend=backend, exists=path.is_file())


def set_backend(backend: str, path: Path = DEFAULT_ENV_FILE) -> BackendStatus:
    normalized = BACKEND_ALIASES.get(backend, backend)
    if normalized not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")

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


def _unquote_env_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _format_env_value(value: str) -> str:
    if not value or any(char.isspace() for char in value) or any(char in value for char in ['"', "'", "#"]):
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="super-agents-backend")
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
        elif status.backend == OPENBASE_CLOUD_BACKEND:
            print(
                "Restart codex-app-server for Openbase Cloud model proxy mode."
            )
        else:
            print("Restart codex-app-server for the change to apply.")
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
