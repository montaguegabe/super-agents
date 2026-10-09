"""Prompt templating for the Claude Code backend."""

from __future__ import annotations


CLAUDE_CONTEXT_OPEN = "<openbase-claude-code-context>"
CLAUDE_CONTEXT_CLOSE = "</openbase-claude-code-context>"


def user_prompt_for_title(prompt: str) -> str:
    """Remove leading transport framing before shortening a user request."""
    text = prompt.strip()
    while text.startswith(CLAUDE_CONTEXT_OPEN):
        _, closing_tag, text = text.partition(CLAUDE_CONTEXT_CLOSE)
        if not closing_tag:
            # A truncated context block contains no safe user title text.
            return ""
        text = text.strip()
    if text.startswith("<voice>") and text.endswith("</voice>"):
        text = text[len("<voice>") : -len("</voice>")]
        text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    return text.strip()


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
