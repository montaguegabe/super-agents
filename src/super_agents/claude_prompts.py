"""Prompt templating for the Claude Code backend."""

from __future__ import annotations


CLAUDE_CONTEXT_OPEN = "<openbase-claude-code-context>"
CLAUDE_CONTEXT_CLOSE = "</openbase-claude-code-context>"


def user_prompt_for_title(prompt: str) -> str:
    """Remove leading transport framing before shortening a user request."""
    return user_prompt_for_display(prompt.strip()).strip()


def user_prompt_for_display(prompt: str) -> str:
    """Present imported user text without optional integration transport framing.

    Strip only known boundary envelopes, never arbitrary XML or inline examples.
    Work on a read-only view: the transcript used to resume Claude stays intact.
    """
    text = _strip_internal_envelopes(prompt)
    if text.startswith("<voice>") and text.endswith("</voice>"):
        spoken = text[len("<voice>") : -len("</voice>")]
        if "<voice>" not in spoken and "</voice>" not in spoken:
            return spoken.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return text


def _strip_internal_envelopes(value: str) -> str:
    envelopes = (
        (CLAUDE_CONTEXT_OPEN, CLAUDE_CONTEXT_CLOSE),
        ("[Openbase system note:", "]"),
        ("<system-reminder>", "</system-reminder>"),
    )
    text = value
    while True:
        trimmed = text.strip()
        for opening, closing in envelopes:
            if trimmed.startswith(opening):
                end = _envelope_end(trimmed, opening, closing)
                if end is None:
                    return ""
                text = trimmed[end:].lstrip()
                break
            # Claude can append system reminders as separate content blocks.
            start = trimmed.rfind("\n" + opening)
            if start >= 0:
                block = trimmed[start + 1 :]
                end = _envelope_end(block, opening, closing)
                if end is None or end == len(block):
                    text = trimmed[:start].rstrip()
                    break
        else:
            return text


def _envelope_end(text: str, opening: str, closing: str) -> int | None:
    if closing == "]":
        depth = 0
        for index, char in enumerate(text):
            if char == "[":
                depth += 1
            elif char == "]":
                depth -= 1
                if depth == 0:
                    return index + 1
        return None
    end = text.find(closing, len(opening))
    return end + len(closing) if end >= 0 else None


def with_claude_turn_context(
    prompt: str,
    *,
    cwd: str,
    developer_instructions: str | None,
) -> str:
    context_parts = [
        CLAUDE_CONTEXT_OPEN,
        f"Current working directory: {cwd}",
        (
            "When the user asks you to create or edit files in the current working directory, "
            "interpret that as this directory and prefer relative paths or paths under it."
        ),
    ]
    if developer_instructions:
        context_parts.extend(
            [
                "",
                "Developer instructions for this Openbase thread:",
                developer_instructions.strip(),
            ]
        )
    context_parts.append(CLAUDE_CONTEXT_CLOSE)
    return "\n".join(context_parts) + "\n\n" + prompt


def combine_developer_instructions(base: str | None, overlay: str | None) -> str | None:
    parts = [part.strip() for part in (base, overlay) if part and part.strip()]
    if not parts:
        return None
    if len(parts) == 2 and parts[1] in parts[0]:
        return parts[0]
    return "\n\n".join(parts)
