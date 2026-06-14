from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from .config import model_catalog_path

DEFAULT_ENV_FILE = Path.home() / ".openbase" / ".env"
CODEX_BACKEND = "codex"
CLAUDE_CODE_PROXY_BACKEND = "claude-code-proxy"
CLAUDE_TUI_BACKEND = "claude-tui"
BACKENDS = {CODEX_BACKEND, CLAUDE_CODE_PROXY_BACKEND, CLAUDE_TUI_BACKEND}
BACKEND_ALIASES = {
    "codex": CODEX_BACKEND,
    "openai": CODEX_BACKEND,
    "claude": CLAUDE_CODE_PROXY_BACKEND,
    "claude-code": CLAUDE_CODE_PROXY_BACKEND,
    "claude-code-proxy": CLAUDE_CODE_PROXY_BACKEND,
    "claude-proxy": CLAUDE_CODE_PROXY_BACKEND,
    "claude-tui": CLAUDE_TUI_BACKEND,
    "claude-code-tui": CLAUDE_TUI_BACKEND,
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
    backend = values.get("OPENBASE_CODEX_BACKEND") or os.environ.get("OPENBASE_CODEX_BACKEND") or "codex"
    backend = BACKEND_ALIASES.get(backend, backend)
    if backend not in BACKENDS:
        backend = f"unsupported:{backend}"
    return BackendStatus(env_file=path, backend=backend, exists=path.is_file())


def set_backend(backend: str, path: Path = DEFAULT_ENV_FILE) -> BackendStatus:
    normalized = BACKEND_ALIASES.get(backend, backend)
    if normalized not in BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}")

    values = {"OPENBASE_CODEX_BACKEND": normalized}
    if normalized == CLAUDE_CODE_PROXY_BACKEND:
        values.update(
            {
                "CODEX_CLAUDE_PROXY_COMMAND": "super-agents-claude-proxy",
                "CODEX_CLAUDE_MODEL_CATALOG_JSON": str(model_catalog_path()),
            }
        )
    update_env_file(path, values)
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
    use_parser.add_argument("backend", choices=sorted(BACKENDS | {"claude-code"}))
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
        if status.backend == CLAUDE_CODE_PROXY_BACKEND:
            print("Restart codex-app-server; keep codex-claude-proxy running for proxy mode.")
        elif status.backend == CLAUDE_TUI_BACKEND:
            print("Restart the Super Agents MCP server for claude-tui mode. codex-app-server is not used by this backend.")
        else:
            print("Restart codex-app-server for the change to apply.")
        return 0

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
