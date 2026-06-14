from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field


ANSI_RE = re.compile(r"\x1b(?:\][^\x07]*(?:\x07|\x1b\\)|\[[0-?]*[ -/]*[@-~]|[@-Z\\-_]|[78]|\([A-Za-z0-9])")


def strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", value).replace("\r", "\n")


@dataclass
class TerminalObservation:
    max_lines: int = 1000
    raw_tail: deque[str] = field(default_factory=lambda: deque(maxlen=400))
    text_lines: deque[str] = field(default_factory=lambda: deque(maxlen=1000))
    _partial: str = ""

    def feed(self, data: bytes) -> str:
        raw = data.decode("utf-8", errors="replace")
        self.raw_tail.append(raw)
        text = strip_ansi(raw)
        combined = self._partial + text
        parts = combined.split("\n")
        self._partial = parts.pop() if parts else ""
        for line in parts:
            clean = line.rstrip()
            if clean:
                self.text_lines.append(clean)
        if len(self.text_lines) > self.max_lines:
            while len(self.text_lines) > self.max_lines:
                self.text_lines.popleft()
        return text

    def text(self) -> str:
        lines = list(self.text_lines)
        if self._partial.strip():
            lines.append(self._partial.strip())
        return "\n".join(lines)

    def tail(self, line_count: int = 80) -> list[str]:
        lines = list(self.text_lines)
        if self._partial.strip():
            lines.append(self._partial.strip())
        return lines[-line_count:]
