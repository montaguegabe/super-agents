from __future__ import annotations

from pathlib import Path

__version__ = "0.1.0"

_sidecar_package = Path(__file__).resolve().parents[3] / "claude-tui" / "src" / "super_agents" / "claude_tui"
__path__ = [str(_sidecar_package)]
