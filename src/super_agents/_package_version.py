"""Package version lookup for CLI --version flags."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version


def package_version() -> str:
    try:
        return version("super-agents")
    except PackageNotFoundError:
        return "unknown"
