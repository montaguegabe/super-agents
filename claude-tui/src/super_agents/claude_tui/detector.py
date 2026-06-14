from __future__ import annotations

import re
from dataclasses import dataclass

from .ansi import strip_ansi


@dataclass(frozen=True)
class ObservedState:
    status: str
    detail: str
    wants_input: bool = False
    wants_approval: bool = False


APPROVAL_PATTERNS = [
    re.compile(r"\b(do you want|would you like).{0,80}\b(proceed|continue|allow|approve|run)\b", re.I | re.S),
    re.compile(r"\b(yes/no|y/n|allow|deny|approve|reject)\b", re.I),
    re.compile(r"\bpermission\b.{0,80}\b(required|requested|needed)\b", re.I | re.S),
    re.compile(r"\b(yes,?\s*i\s*trust\s*this\s*folder|yes,?itrustthisfolder)\b", re.I),
    re.compile(r"\b(enter\s*to\s*confirm|entertoconfirm)\b", re.I),
]

BUSY_PATTERNS = [
    re.compile(r"\b(thinking|working|running|executing|calling tool|reading|editing)\b", re.I),
    re.compile(r"\besc\b.{0,40}\binterrupt\b", re.I),
]

IDLE_PATTERNS = [
    re.compile(r"(^|\n)\s*(>|❯|›)\s*$"),
    re.compile(r"\bready\s*(>|❯|›)\s*$", re.I),
    re.compile(r"❯\s*(try|$)", re.I),
    re.compile(r"\btry\s*\"?how\s*does\b", re.I),
    re.compile(r"\btry\"howdoes\b", re.I),
    re.compile(r"\b(how can i help|what would you like|ask me anything)\b", re.I),
]

ERROR_PATTERNS = [
    re.compile(r"command not found:?\s+claude", re.I),
    re.compile(r"no such file or directory:?\s+claude", re.I),
]


def infer_state(text: str, process_running: bool = True) -> ObservedState:
    tail_lines = strip_ansi(text).splitlines()[-30:]
    tail = "\n".join(tail_lines)
    non_empty_tail = [line.strip() for line in tail_lines if line.strip()]
    for pattern in ERROR_PATTERNS:
        if pattern.search(tail):
            return ObservedState("failed", "Claude CLI is not available")
    for pattern in APPROVAL_PATTERNS:
        if pattern.search(tail):
            return ObservedState("waiting", "approval prompt detected", wants_approval=True)
    if not process_running:
        return ObservedState("completed", "process exited")
    if non_empty_tail and re.fullmatch(r"(>|❯|›)\s*", non_empty_tail[-1]):
        return ObservedState("waiting", "Claude appears ready for input", wants_input=True)
    if "⏺" in tail and re.search(r"(←\s*for\s*agents|for\s+agents)", tail, re.I):
        return ObservedState("waiting", "Claude appears ready after an assistant response", wants_input=True)
    if re.search(r"\b(crunched|sautéed|sauteed|\w+ed)\s+for\s*\d+s\b", tail, re.I):
        return ObservedState("waiting", "Claude appears ready after completing a turn", wants_input=True)
    for pattern in BUSY_PATTERNS:
        if pattern.search(tail):
            return ObservedState("running", "Claude appears busy")
    for pattern in IDLE_PATTERNS:
        if pattern.search(tail):
            return ObservedState("waiting", "Claude appears ready for input", wants_input=True)
    if tail.strip():
        return ObservedState("running", "output observed")
    return ObservedState("running", "process running")
