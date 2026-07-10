from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from .app_server_client import LabelQueryInput
from .backend_clients import SuperAgentsClient
from .backend_config import normalize_backend
from .defaults import (
    default_service_tier,
    default_super_agents_model,
    default_super_agents_reasoning_effort,
)
from .multi_backend import MultiBackendClient
from .permission_guard import apply_permission_guard
from .state import without_none

JsonObject = dict[str, Any]
Handler = Callable[[JsonObject], Awaitable[Any]]
logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "Control Openbase Super Agents threads asynchronously. Each thread launch may pick its coding backend — "
    "Codex-compatible app-server sessions or Claude Code Agent SDK sessions — via the optional backend "
    "parameter on super_agents_start; without it the configured default backend is used, and threads on "
    "different backends can run side by side. Tools start, inspect, steer, cancel, and answer callbacks; they "
    "do not wait for turns to finish. Do not silently approve app-server callbacks; use codex_answer_request "
    "when a callback is pending."
)

SUPER_AGENT_INSTRUCTIONS_FILENAME = "SUPER_AGENT_INSTRUCTIONS.md"


@dataclass(slots=True)
class ToolDefinition:
    name: str
    title: str
    description: str
    input_schema: JsonObject
    handler: Handler
    annotations: JsonObject | None = None

    def to_mcp_tool(self) -> types.Tool:
        kwargs: JsonObject = {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "inputSchema": self.input_schema,
        }
        if self.annotations:
            kwargs["annotations"] = self.annotations
        return types.Tool(**kwargs)


def create_server(client: SuperAgentsClient | None = None) -> Server:
    app_client = client or MultiBackendClient()
    server = Server("super-agents", instructions=INSTRUCTIONS)
    tool_by_name = {tool.name: tool for tool in build_tools(app_client)}

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [tool.to_mcp_tool() for tool in tool_by_name.values()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        started = time.monotonic()
        mcp_call_id = f"mcp-{uuid.uuid4().hex[:12]}"
        safe_arguments = dict(arguments or {})
        safe_arguments["_mcpCallId"] = mcp_call_id
        logger.info(
            "dispatch_timing stage=mcp_tool_request mcp_call_id=%s tool=%s name=%s cwd_basename=%s",
            mcp_call_id,
            name,
            safe_arguments.get("name") or safe_arguments.get("label") or "",
            _cwd_basename(safe_arguments.get("cwd")),
        )
        try:
            tool = tool_by_name.get(name)
            if not tool:
                raise ValueError(f"Unknown tool: {name}")
            output = await tool.handler(safe_arguments)
            logger.info(
                "dispatch_timing stage=mcp_tool_response mcp_call_id=%s tool=%s elapsed_ms=%d status=ok",
                mcp_call_id,
                name,
                int((time.monotonic() - started) * 1000),
            )
            return text_tool_result(output)
        except Exception as exc:
            logger.info(
                "dispatch_timing stage=mcp_tool_response mcp_call_id=%s tool=%s "
                "elapsed_ms=%d status=error error_type=%s",
                mcp_call_id,
                name,
                int((time.monotonic() - started) * 1000),
                type(exc).__name__,
            )
            return text_tool_result({"error": str(exc)}, is_error=True)

    return server


def build_tools(client: SuperAgentsClient) -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="codex_app_server_status",
            title="Super Agents Backend Status",
            description=(
                "Check Super Agents backend readiness, pending requests, queued turns, and active turns. "
                "The backend may be Codex-compatible or Claude Code."
            ),
            input_schema=object_schema({}),
            annotations={"readOnlyHint": True, "idempotentHint": True},
            handler=lambda _input: client.status(),
        ),
        ToolDefinition(
            name="super_agents_start",
            title="Start Super Agents Thread",
            description=(
                "Create a named Super Agents thread. Each launch may pick its own coding backend via the "
                "backend parameter; the configured default backend is used when omitted."
            ),
            input_schema=object_schema(
                {
                    "name": {"type": "string", "description": "Human-friendly thread name for future operations."},
                    "cwd": {
                        "type": "string",
                        "description": "Project working directory. Defaults to the user's home directory.",
                    },
                    "developerInstructions": {"type": "string"},
                    "agentName": {
                        "type": "string",
                        "description": 'Optional agent name/persona, e.g. "Carl" or "Dottie".',
                    },
                    "backend": {
                        "type": "string",
                        "description": (
                            'Coding backend for this launch: "codex", "openbase_cloud", or "claude_code". '
                            "Defaults to the configured backend; launches may mix backends freely."
                        ),
                    },
                    **permission_option_properties(include_sandbox=True),
                },
                ["name"],
            ),
            handler=lambda input_data: client.start_thread(clean_thread_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_resume",
            title="Resume Super Agents Thread",
            description="Resume a named Super Agents thread.",
            input_schema=name_query_schema(["name"]),
            annotations={"readOnlyHint": True},
            handler=lambda input_data: client.resume_by_label(clean_name_query_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_read",
            title="Read Super Agents Thread",
            description=(
                "Read a named or id-addressed Super Agents thread. Compact by default; "
                "pass includeTurns=true to include full turns."
            ),
            input_schema=object_schema(
                {**name_query_properties(), "includeTurns": {"type": "boolean", "default": False}},
            ),
            annotations={"readOnlyHint": True},
            handler=lambda input_data: client.read_by_label(
                clean_name_query_input(input_data),
                optional_boolean(input_data, "includeTurns", False),
            ),
        ),
        ToolDefinition(
            name="super_agents_rename",
            title="Rename Super Agents Thread",
            description="Rename a Super Agents thread using its current name.",
            input_schema=object_schema(
                {
                    **name_query_properties(include_ids=False, include_output_options=False),
                    "newName": {"type": "string"},
                },
                ["name", "newName"],
            ),
            handler=lambda input_data: client.rename_by_label(
                clean_name_query_input(input_data),
                required_string(input_data, "newName"),
            ),
        ),
        ToolDefinition(
            name="codex_answer_request",
            title="Answer Super Agents Callback",
            description=(
                "Answer a pending Super Agents backend callback. For plan questions, pass result "
                "{ answers: { question_id: { answers: [...] } } }. For approvals, pass result "
                "{ decision: 'accept' | 'decline' | 'cancel' }. Some backends may not support callbacks."
            ),
            input_schema=object_schema(
                {
                    "requestId": {"anyOf": [{"type": "string"}, {"type": "number"}]},
                    "result": {"type": "object", "additionalProperties": True},
                },
                ["requestId", "result"],
            ),
            handler=lambda input_data: client.answer_request(
                required_request_id(input_data), required_object(input_data, "result")
            ),
        ),
        ToolDefinition(
            name="super_agents_sessions",
            title="Super Agents Sessions",
            description="List named Super Agents threads from the configured backend.",
            input_schema=object_schema({}),
            annotations={"readOnlyHint": True, "idempotentHint": True},
            handler=lambda _input: client.sessions(),
        ),
        ToolDefinition(
            name="super_agents_thread_favorite",
            title="Super Agents Thread Favorite",
            description="Query whether one local Openbase Coder thread is favorited.",
            input_schema=object_schema(
                {
                    "threadId": {
                        "type": "string",
                        "description": "Super Agents thread id to inspect.",
                    },
                },
                ["threadId"],
            ),
            annotations={"readOnlyHint": True, "idempotentHint": True},
            handler=lambda input_data: client.thread_favorite(required_string(input_data, "threadId")),
        ),
        ToolDefinition(
            name="super_agents_tags",
            title="Super Agents Tags",
            description="List local Openbase Coder tag options shared by threads and reports.",
            input_schema=object_schema({}),
            annotations={"readOnlyHint": True, "idempotentHint": True},
            handler=lambda _input: client.tags(),
        ),
        ToolDefinition(
            name="super_agents_thread_tags",
            title="Super Agents Thread Tags",
            description=(
                "Read or replace local Openbase Coder tags for one thread. Omit tags to read; "
                "pass tags to apply shared tag options."
            ),
            input_schema=object_schema(
                {
                    "threadId": {
                        "type": "string",
                        "description": "Super Agents thread id to inspect or tag.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Complete tag label list to apply. Omit to read current tags.",
                    },
                },
                ["threadId"],
            ),
            handler=lambda input_data: client.thread_tags(
                required_string(input_data, "threadId"),
                optional_string_array(input_data, "tags"),
            ),
        ),
        ToolDefinition(
            name="super_agents_report_tags",
            title="Super Agents Report Tags",
            description=(
                "Read or replace local Openbase Coder tags for one report file. Omit tags to read; "
                "pass tags to apply shared tag options."
            ),
            input_schema=object_schema(
                {
                    "projectPath": {
                        "type": "string",
                        "description": "Project directory containing the .reports folder.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Report path relative to the project's .reports folder.",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Complete tag label list to apply. Omit to read current tags.",
                    },
                },
                ["projectPath", "path"],
            ),
            handler=lambda input_data: client.report_tags(
                required_string(input_data, "projectPath"),
                required_string(input_data, "path"),
                optional_string_array(input_data, "tags"),
            ),
        ),
        ToolDefinition(
            name="super_agents_active",
            title="Active Super Agents",
            description=(
                "List active tracked Super Agents with names, cwd, status, age, and short previews. "
                "Previews default to 160 chars; pass previewLength or includePreview=false to control them."
            ),
            input_schema=name_query_schema(),
            annotations={"readOnlyHint": True, "idempotentHint": True},
            handler=lambda input_data: client.active(clean_name_query_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_status",
            title="Super Agents Compact Status",
            description=(
                "Compact status list for voice/status checks. Returns active threads by default with name, thread id, "
                "turn id, status, update times, pending request count, cwd, and stale indicators. No transcripts, diffs, or previews."
            ),
            input_schema=name_query_schema(),
            annotations={"readOnlyHint": True, "idempotentHint": True},
            handler=lambda input_data: client.compact_status(clean_name_query_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_resolve",
            title="Resolve Super Agents Name",
            description="Resolve a thread name to the latest active matching Super Agents session by default.",
            input_schema=name_query_schema(["name"]),
            annotations={"readOnlyHint": True},
            handler=lambda input_data: client.resolve_label(clean_name_query_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_progress",
            title="Super Agents Progress By Name",
            description=(
                "Check progress for a Super Agents turn by name or threadId/turnId. Compact by default: status, summary, "
                "pending requests, and stale indicators only. Pass full=true for raw turn/tracked-turn output; includeTurn=true "
                "or includeItems=true for bounded extra details."
            ),
            input_schema=name_query_schema(),
            annotations={"readOnlyHint": True},
            handler=lambda input_data: client.progress_by_label(clean_name_query_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_steer",
            title="Steer Super Agents By Name",
            description=(
                "Send steering input to the latest active Super Agents turn matching a thread name. "
                "If no active turn exists, starts a new turn on the same thread."
            ),
            input_schema=object_schema(
                {**name_query_properties(include_output_options=False), "prompt": {"type": "string"}},
                ["name", "prompt"],
            ),
            handler=lambda input_data: client.steer_by_label(
                clean_name_query_input(input_data),
                required_string(input_data, "prompt"),
            ),
        ),
        ToolDefinition(
            name="super_agents_cancel",
            title="Cancel Super Agents By Name",
            description="Cancel the latest active Super Agents turn matching a thread name.",
            input_schema=name_query_schema(["name"]),
            handler=lambda input_data: client.cancel_by_label(clean_name_query_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_start_turn",
            title="Start Super Agents Turn By Name",
            description=(
                "Submit follow-up input to the latest matching named thread. If a turn is active, this steers "
                "the active turn; otherwise it starts a new turn. Use super_agents_queue_turn for an explicit "
                "separate follow-up after the active turn finishes."
            ),
            input_schema=object_schema(
                {
                    **name_query_properties(include_ids=False, include_output_options=False),
                    "prompt": {"type": "string"},
                    "cwd": {"type": "string"},
                    "mode": {"type": "string", "enum": ["default", "plan"], "default": "default"},
                    "model": {
                        "type": "string",
                        "description": "Defaults to the model for the selected backend's Super Agents role.",
                    },
                    "developerInstructions": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    **permission_option_properties(include_sandbox_type=True),
                },
                ["name", "prompt"],
            ),
            handler=lambda input_data: client.start_turn_by_label(
                clean_name_query_input(input_data),
                clean_start_turn_by_name_input(
                    input_data, backend=optional_backend(input_data) or client_backend(client)
                ),
            ),
        ),
        ToolDefinition(
            name="super_agents_queue_turn",
            title="Queue Super Agents Turn",
            description=(
                "Queue a follow-up prompt in Super Agents' per-thread filesystem queue so it starts as a separate "
                "turn after the target thread's active turn finishes. If no active turn exists, starts immediately. "
                "Some backends do not expose native queued-next-turn semantics for normal user prompts."
            ),
            input_schema=object_schema(
                {
                    **name_query_properties(include_ids=True, include_output_options=False),
                    "prompt": {"type": "string"},
                    "mode": {"type": "string", "enum": ["default", "plan"], "default": "default"},
                    "model": {
                        "type": "string",
                        "description": "Defaults to the model for the selected backend's Super Agents role.",
                    },
                    "developerInstructions": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "agentName": {
                        "type": "string",
                        "description": "Optional agent name/persona to store for this thread.",
                    },
                    **permission_option_properties(include_sandbox_type=True),
                },
                ["prompt"],
            ),
            handler=lambda input_data: client.queue_turn_by_label(
                clean_name_query_input(input_data),
                clean_queue_turn_input(input_data, backend=optional_backend(input_data) or client_backend(client)),
            ),
        ),
        ToolDefinition(
            name="super_agents_recent",
            title="Recent Super Agents",
            description="List recent named Super Agents threads from the configured backend.",
            input_schema=name_query_schema(),
            annotations={"readOnlyHint": True, "idempotentHint": True},
            handler=lambda input_data: client.recent(clean_name_query_input(input_data)),
        ),
    ]


def object_schema(properties: JsonObject, required: list[str] | None = None) -> JsonObject:
    schema: JsonObject = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": properties,
    }
    if required:
        schema["required"] = required
    return schema


def name_query_properties(include_ids: bool = True, include_output_options: bool = True) -> JsonObject:
    properties: JsonObject = {
        "name": {"type": "string"},
        "backend": {
            "type": "string",
            "description": (
                'Pin the operation to one coding backend ("codex", "openbase_cloud", or "claude_code"). '
                "When omitted, the owning backend is found automatically."
            ),
        },
        "cwd": {"type": "string"},
        "status": {"type": "string", "enum": ["running", "waiting", "completed", "failed", "cancelled", "unknown"]},
        "favorite": {
            "type": "boolean",
            "description": "Filter listed threads by local Openbase Coder favorite state.",
        },
        "limit": {"type": "number"},
        "includeInactive": {"type": "boolean", "default": False},
        "prefer": {"type": "string", "enum": ["latest_active", "latest_any"], "default": "latest_active"},
    }
    if include_ids:
        properties.update(
            {
                "threadId": {
                    "type": "string",
                    "description": "Inspect a specific Super Agents thread without resolving a name.",
                },
                "turnId": {
                    "type": "string",
                    "description": "Inspect a specific Super Agents turn when used with name or threadId.",
                },
            }
        )
    if include_output_options:
        properties.update(
            {
                "includePreview": {"type": "boolean", "default": True},
                "previewLength": {"type": "number", "default": 160},
                "includeTurn": {"type": "boolean", "default": False},
                "includeItems": {"type": "boolean", "default": False},
                "full": {"type": "boolean", "default": False},
                "finalOnly": {"type": "boolean", "default": False},
                "maxItems": {"type": "number", "default": 10},
                "maxOutputChars": {"type": "number", "default": 1200},
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional top-level field allowlist for concise output.",
                },
            }
        )
    return properties


def permission_option_properties(*, include_sandbox: bool = False, include_sandbox_type: bool = False) -> JsonObject:
    properties: JsonObject = {
        "approvalPolicy": {
            "type": "string",
            "description": 'Codex approval policy override, e.g. "never".',
        },
        "sandboxPolicy": {
            "type": "string",
            "description": 'Codex sandbox policy override, e.g. "danger-full-access".',
        },
        "permissionMode": {
            "type": "string",
            "description": (
                'Claude Code permission mode override, e.g. "bypassPermissions" (full access) or '
                '"default" (tool permissions gated through the shared approval queue).'
            ),
        },
    }
    if include_sandbox:
        properties["sandbox"] = {
            "type": "string",
            "description": 'Alias for sandboxPolicy, e.g. "danger-full-access".',
        }
    if include_sandbox_type:
        properties["sandboxType"] = {
            "type": "string",
            "description": 'Openbase sandbox type alias, e.g. "dangerFullAccess".',
        }
    return properties


def name_query_schema(required: list[str] | None = None) -> JsonObject:
    return object_schema(name_query_properties(), required)


def text_tool_result(value: Any, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(value, indent=2))],
        isError=is_error,
    )


def client_backend(client: SuperAgentsClient) -> str | None:
    backend = getattr(client, "backend", None)
    return backend if isinstance(backend, str) else None


def clean_thread_input(input_data: JsonObject) -> JsonObject:
    cwd = optional_string(input_data, "cwd")
    validate_thread_cwd(cwd)
    return apply_permission_guard(
        without_none(
            {
                "cwd": cwd,
                "developerInstructions": developer_instructions_or_default(input_data),
                "name": optional_string(input_data, "name"),
                "agentName": optional_string(input_data, "agentName"),
                "backend": optional_backend(input_data),
                "approvalPolicy": optional_string(input_data, "approvalPolicy"),
                "sandbox": optional_string(input_data, "sandbox"),
                "sandboxPolicy": optional_string(input_data, "sandboxPolicy"),
                "permissionMode": optional_string(input_data, "permissionMode"),
                "_mcpCallId": optional_string(input_data, "_mcpCallId"),
            }
        )
    )


def validate_thread_cwd(cwd: str | None) -> None:
    if cwd is None:
        return
    path = Path(cwd).expanduser()
    if not path.is_dir():
        raise ValueError(f"cwd must be an existing directory: {cwd}")


def clean_turn_input(input_data: JsonObject, *, backend: str | None = None) -> JsonObject:
    prompt = required_string(input_data, "prompt")
    return apply_permission_guard(
        without_none(
            {
                "threadId": required_string(input_data, "threadId"),
                "prompt": prompt,
                "cwd": optional_string(input_data, "cwd"),
                "mode": optional_mode(input_data, "mode") or "default",
                "model": optional_string(input_data, "model") or default_super_agents_model(backend=backend),
                "reasoningEffort": optional_string(input_data, "reasoningEffort")
                or default_super_agents_reasoning_effort(),
                "serviceTier": optional_string(input_data, "serviceTier") or default_service_tier(),
                "developerInstructions": developer_instructions_or_default(input_data, allow_explicit_null=True),
                "name": optional_string(input_data, "name"),
                "label": optional_string(input_data, "name") or optional_string(input_data, "label"),
                "agentName": optional_string(input_data, "agentName"),
                "approvalPolicy": optional_string(input_data, "approvalPolicy"),
                "sandboxType": optional_string(input_data, "sandboxType"),
                "sandboxPolicy": optional_string(input_data, "sandboxPolicy"),
                "permissionMode": optional_string(input_data, "permissionMode"),
                "_mcpCallId": optional_string(input_data, "_mcpCallId"),
            }
        )
    )


def developer_instructions_or_default(input_data: JsonObject, allow_explicit_null: bool = False) -> str | None:
    default_instructions = default_super_agent_instructions()
    if "developerInstructions" in input_data:
        if input_data["developerInstructions"] is None and allow_explicit_null:
            return None
        return combine_developer_instructions(
            default_instructions, optional_string(input_data, "developerInstructions")
        )
    return default_instructions


def combine_developer_instructions(default_instructions: str | None, explicit_instructions: str | None) -> str | None:
    if default_instructions and explicit_instructions:
        return f"{default_instructions.rstrip()}\n\n{explicit_instructions}"
    return explicit_instructions or default_instructions


def default_super_agent_instructions() -> str | None:
    path = default_super_agent_instructions_path()
    try:
        text = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
    return text if text.strip() else None


def default_super_agent_instructions_path() -> Path:
    if configured := os.environ.get("CODEX_SUPER_AGENT_INSTRUCTIONS_PATH"):
        return Path(configured).expanduser()
    if codex_home := os.environ.get("CODEX_HOME"):
        return Path(codex_home).expanduser() / SUPER_AGENT_INSTRUCTIONS_FILENAME
    return Path.home() / ".openbase" / "instructions" / SUPER_AGENT_INSTRUCTIONS_FILENAME


def clean_name_query_input(input_data: JsonObject) -> LabelQueryInput:
    return LabelQueryInput(
        label=optional_string(input_data, "name") or optional_string(input_data, "label"),
        backend=optional_backend(input_data),
        cwd=optional_string(input_data, "cwd"),
        status=optional_string(input_data, "status"),
        favorite=optional_boolean_or_none(input_data, "favorite"),
        limit=optional_number(input_data, "limit"),
        include_inactive=optional_boolean_or_none(input_data, "includeInactive"),
        prefer=optional_prefer(input_data, "prefer"),
        thread_id=optional_string(input_data, "threadId"),
        turn_id=optional_string(input_data, "turnId"),
        include_turn=optional_boolean_or_none(input_data, "includeTurn"),
        include_items=optional_boolean_or_none(input_data, "includeItems"),
        full=optional_boolean_or_none(input_data, "full"),
        final_only=optional_boolean_or_none(input_data, "finalOnly"),
        max_items=optional_number(input_data, "maxItems"),
        max_output_chars=optional_number(input_data, "maxOutputChars"),
        include_preview=optional_boolean_or_none(input_data, "includePreview"),
        preview_length=optional_number(input_data, "previewLength"),
        fields=optional_string_array(input_data, "fields"),
    )


def clean_start_turn_by_name_input(input_data: JsonObject, *, backend: str | None = None) -> JsonObject:
    cleaned = clean_turn_input({"threadId": "__placeholder__", **input_data}, backend=backend)
    cleaned.pop("threadId", None)
    cleaned.pop("agentName", None)
    cleaned["name"] = required_string(input_data, "name")
    cleaned["label"] = required_string(input_data, "name")
    return cleaned


def clean_queue_turn_input(input_data: JsonObject, *, backend: str | None = None) -> JsonObject:
    if not optional_string(input_data, "name") and not optional_string(input_data, "threadId"):
        raise ValueError("name or threadId must be provided.")
    cleaned = clean_turn_input(
        {"threadId": optional_string(input_data, "threadId") or "__placeholder__", **input_data},
        backend=backend,
    )
    cleaned.pop("threadId", None)
    if name := optional_string(input_data, "name"):
        cleaned["name"] = name
        cleaned["label"] = name
    return cleaned


def required_string(value: JsonObject, key: str) -> str:
    result = value.get(key)
    if not isinstance(result, str) or not result:
        raise ValueError(f"{key} must be a non-empty string.")
    return result


def required_object(value: JsonObject, key: str) -> JsonObject:
    result = value.get(key)
    if not isinstance(result, dict):
        raise ValueError(f"{key} must be an object.")
    return result


def required_request_id(value: JsonObject) -> str | int:
    result = value.get("requestId")
    if isinstance(result, str | int) and not isinstance(result, bool):
        return result
    raise ValueError("requestId must be a string or number.")


def optional_string(value: JsonObject, key: str) -> str | None:
    result = value.get(key)
    return result if isinstance(result, str) and result else None


def optional_backend(value: JsonObject, key: str = "backend") -> str | None:
    result = optional_string(value, key)
    return normalize_backend(result) if result else None


def optional_boolean_or_none(value: JsonObject, key: str) -> bool | None:
    result = value.get(key)
    return result if isinstance(result, bool) else None


def optional_boolean(value: JsonObject, key: str, default: bool) -> bool:
    result = value.get(key)
    return result if isinstance(result, bool) else default


def optional_number(value: JsonObject, key: str) -> int | None:
    result = value.get(key)
    if isinstance(result, int | float) and not isinstance(result, bool) and result > 0:
        return int(result)
    return None


def optional_mode(value: JsonObject, key: str) -> str | None:
    result = optional_string(value, key)
    return result if result in {"default", "plan"} else None


def optional_prefer(value: JsonObject, key: str) -> str | None:
    result = optional_string(value, key)
    return result if result in {"latest_active", "latest_any"} else None


def optional_string_array(value: JsonObject, key: str) -> list[str] | None:
    result = value.get(key)
    if not isinstance(result, list):
        return None
    return [item for item in result if isinstance(item, str) and item]


def _cwd_basename(value: Any) -> str:
    if not isinstance(value, str) or not value:
        return ""
    return value.rstrip("/").rsplit("/", 1)[-1]


async def run_stdio() -> None:
    server = create_server()
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="super-agents",
                server_version="0.1.0",
                instructions=INSTRUCTIONS,
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(tools_changed=True),
                    experimental_capabilities={},
                ),
            ),
        )


def main() -> None:
    if "--version" in sys.argv[1:]:
        from ._package_version import package_version

        print(f"super-agents {package_version()}")
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print("Super Agents MCP running on stdio.", file=sys.stderr)
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
