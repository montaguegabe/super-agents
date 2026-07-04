"""Prompt templating for the Claude Code backend."""

from __future__ import annotations


def with_claude_turn_context(
    prompt: str,
    *,
    cwd: str,
    developer_instructions: str | None,
) -> str:
    context_parts = [
        "<openbase-claude-code-context>",
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
    context_parts.append("</openbase-claude-code-context>")
    return "\n".join(context_parts) + "\n\n" + prompt


def combine_developer_instructions(base: str | None, overlay: str | None) -> str | None:
    parts = [part.strip() for part in (base, overlay) if part and part.strip()]
    if not parts:
        return None
    if len(parts) == 2 and parts[1] in parts[0]:
        return parts[0]
    return "\n\n".join(parts)
