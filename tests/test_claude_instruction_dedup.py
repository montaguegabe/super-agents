from __future__ import annotations

import json
from pathlib import Path

from super_agents.claude_compaction import transcript_recently_contains
from super_agents.claude_prompts import (
    INSTRUCTIONS_FINGERPRINT_PREFIX,
    instructions_fingerprint,
    with_claude_turn_context,
)

RULES = "You are the Dispatcher. " * 2000  # ~48k characters, like the real rules


def _user(text: str) -> str:
    return json.dumps({"type": "user", "message": {"content": text}})


def test_full_block_then_reference_then_full_again_after_compaction(tmp_path: Path) -> None:
    """VM2, 2026-10-11: ~63k characters of Dispatcher instructions were re-sent
    with every request, filling Haiku's 200k window in about ten requests."""
    marker = INSTRUCTIONS_FINGERPRINT_PREFIX + instructions_fingerprint(RULES)
    full = with_claude_turn_context("first", cwd="/w", developer_instructions=RULES)
    assert marker in full and RULES.strip() in full

    transcript = tmp_path / "t.jsonl"
    transcript.write_text(_user(full) + "\n")
    assert transcript_recently_contains(transcript, marker)

    reference = with_claude_turn_context(
        "second", cwd="/w", developer_instructions=RULES, instructions_in_recent_context=True
    )
    assert RULES.strip() not in reference and len(reference) < 600
    assert instructions_fingerprint(RULES) in reference
    # A reference must never satisfy the check on its own.
    assert marker not in reference

    with transcript.open("a") as handle:
        handle.write(json.dumps({"type": "system", "subtype": "compact_boundary"}) + "\n")
        handle.write(_user(reference) + "\n")
    assert not transcript_recently_contains(transcript, marker)


def test_block_far_back_in_the_transcript_is_sent_again(tmp_path: Path) -> None:
    marker = INSTRUCTIONS_FINGERPRINT_PREFIX + instructions_fingerprint(RULES)
    transcript = tmp_path / "t.jsonl"
    full = with_claude_turn_context("first", cwd="/w", developer_instructions=RULES)
    filler = _user("x" * 500_000)
    transcript.write_text(_user(full) + "\n" + filler + "\n")
    assert not transcript_recently_contains(transcript, marker)
    assert not transcript_recently_contains(None, marker)
