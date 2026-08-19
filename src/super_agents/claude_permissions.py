"""Claude Agent SDK adapter for the backend-neutral approval gate."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .approval_gate import ToolApprovalGate
from .state import JsonObject

CLAUDE_APPROVAL_METHOD = "claudeCode/requestApproval"

TurnIdGetter = Callable[[], str | None]


def can_use_tool_handler(
    sdk: Any,
    gate: ToolApprovalGate,
    thread_id: str,
    turn_id_getter: TurnIdGetter,
) -> Any:
    """Build a Claude SDK callback that allows tools only after approval."""

    async def can_use_tool(tool_name: str, tool_input: JsonObject, context: Any) -> Any:
        outcome = await gate.request_permission(
            tool_name=tool_name,
            tool_input=tool_input,
            thread_id=thread_id,
            turn_id=turn_id_getter(),
            tool_call_id=_context_string(context, "tool_use_id"),
            description=_context_description(context),
        )
        if outcome.decision == "accept":
            # Explicit pass-through avoids older SDK/CLI combinations treating
            # an omitted updated_input as an empty tool argument object.
            return sdk.PermissionResultAllow(updated_input=tool_input)
        if outcome.reason == "timeout":
            message = "Tool use denied because approval timed out."
        elif outcome.decision == "cancel":
            message = "Tool use cancelled by the approver."
        else:
            message = "Tool use denied by the approver."
        return sdk.PermissionResultDeny(
            message=message,
            interrupt=outcome.decision == "cancel",
        )

    return can_use_tool


def _context_description(context: Any) -> str | None:
    for attribute in ("title", "description", "display_name"):
        if value := _context_string(context, attribute):
            return value
    return None


def _context_string(context: Any, attribute: str) -> str | None:
    value = getattr(context, attribute, None)
    return value.strip() if isinstance(value, str) and value.strip() else None
