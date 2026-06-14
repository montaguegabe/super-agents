from __future__ import annotations

import asyncio
import json
import logging
import os
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
from .backend_clients import SuperAgentsClient, backend_from_environment, client_from_environment

JsonObject = dict[str, Any]
Handler = Callable[[JsonObject], Awaitable[Any]]
logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "Control local Codex app-server threads asynchronously. Tools start, inspect, steer, cancel, and answer "
    "callbacks; they do not wait for turns to finish. Do not silently approve app-server callbacks; use "
    "codex_answer_request when a callback is pending."
)

OPENBASE_DISPATCHER_CONFIG_PATH = Path.home() / ".openbase" / "dispatcher-config.json"
LEGACY_OPENBASE_DISPATCHER_CONFIG_PATH = Path.home() / ".openbase" / "codex_home" / "dispatcher-config.json"
SUPER_AGENT_INSTRUCTIONS_FILENAME = "SUPER_AGENT_INSTRUCTIONS.md"
REASONING_EFFORTS = {"low", "medium", "high", "xhigh"}
CLAUDE_MODEL_ALIASES = {"opus", "sonnet", "haiku"}


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
    app_client = client or client_from_environment()
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
            title="Codex App Server Status",
            description="Check local Codex app-server readiness, websocket connection, pending requests, and active turns.",
            input_schema=object_schema({}),
            annotations={"readOnlyHint": True, "idempotentHint": True},
            handler=lambda _input: client.status(),
        ),
        ToolDefinition(
            name="super_agents_start",
            title="Start Super Agents Thread",
            description="Create a named Codex app-server thread using Codex's native thread-name store.",
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
                },
                ["name"],
            ),
            handler=lambda input_data: client.start_thread(clean_thread_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_resume",
            title="Resume Super Agents Thread",
            description="Resume a named Codex app-server thread.",
            input_schema=name_query_schema(["name"]),
            annotations={"readOnlyHint": True},
            handler=lambda input_data: client.resume_by_label(clean_name_query_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_read",
            title="Read Super Agents Thread",
            description="Read a named or id-addressed Codex app-server thread. Compact by default; pass includeTurns=true to include full turns.",
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
            description="Rename a Codex app-server thread using its current name.",
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
            title="Answer Codex Request",
            description=(
                "Answer a pending app-server callback. For plan questions, pass result { answers: { question_id: "
                "{ answers: [...] } } }. For approvals, pass result { decision: 'accept' | 'decline' | 'cancel' }."
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
            description="List named Codex app-server threads.",
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
                        "description": "App-server thread id to inspect.",
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
                        "description": "App-server thread id to inspect or tag.",
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
                },
                ["name", "prompt"],
            ),
            handler=lambda input_data: client.start_turn_by_label(
                clean_name_query_input(input_data),
                clean_start_turn_by_name_input(input_data, backend=client_backend(client)),
            ),
        ),
        ToolDefinition(
            name="super_agents_queue_turn",
            title="Queue Super Agents Turn",
            description=(
                "Queue a follow-up prompt in Super Agents' per-thread filesystem queue so it starts as a separate "
                "turn after the target thread's active turn finishes. If no active turn exists, starts immediately. "
                "Codex app-server does not expose native queued-next-turn semantics for normal user prompts."
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
                },
                ["prompt"],
            ),
            handler=lambda input_data: client.queue_turn_by_label(
                clean_name_query_input(input_data),
                clean_queue_turn_input(input_data, backend=client_backend(client)),
            ),
        ),
        ToolDefinition(
            name="super_agents_recent",
            title="Recent Super Agents",
            description="List recent named Codex app-server threads.",
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


def turn_start_properties() -> JsonObject:
    return {
        "threadId": {"type": "string"},
        "prompt": {"type": "string"},
        "cwd": {"type": "string"},
        "mode": {"type": "string", "enum": ["default", "plan"], "default": "default"},
        "model": {
            "type": "string",
            "description": "Defaults to the model for the selected backend's Super Agents role.",
        },
        "developerInstructions": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "name": {"type": "string"},
        "agentName": {
            "type": "string",
            "description": "Optional agent name/persona to store for this thread.",
        },
    }


def name_query_properties(include_ids: bool = True, include_output_options: bool = True) -> JsonObject:
    properties: JsonObject = {
        "name": {"type": "string"},
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
                    "description": "Inspect a specific app-server thread without resolving a name.",
                },
                "turnId": {
                    "type": "string",
                    "description": "Inspect a specific app-server turn when used with name or threadId.",
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
    return without_none(
        {
            "cwd": cwd,
            "developerInstructions": developer_instructions_or_default(input_data),
            "name": optional_string(input_data, "name"),
            "agentName": optional_string(input_data, "agentName"),
            "_mcpCallId": optional_string(input_data, "_mcpCallId"),
        }
    )


def validate_thread_cwd(cwd: str | None) -> None:
    if cwd is None:
        return
    path = Path(cwd).expanduser()
    if not path.is_dir():
        raise ValueError(f"cwd must be an existing directory: {cwd}")


def clean_turn_input(input_data: JsonObject, *, backend: str | None = None) -> JsonObject:
    prompt = required_string(input_data, "prompt")
    return without_none(
        {
            "threadId": required_string(input_data, "threadId"),
            "prompt": prompt,
            "cwd": optional_string(input_data, "cwd"),
            "mode": optional_mode(input_data, "mode") or "default",
            "model": optional_string(input_data, "model")
            or default_super_agents_model(backend=backend),
            "reasoningEffort": optional_string(input_data, "reasoningEffort") or default_reasoning_effort(),
            "serviceTier": optional_string(input_data, "serviceTier") or "fast",
            "developerInstructions": developer_instructions_or_default(input_data, allow_explicit_null=True),
            "name": optional_string(input_data, "name"),
            "label": optional_string(input_data, "name") or optional_string(input_data, "label"),
            "agentName": optional_string(input_data, "agentName"),
            "_mcpCallId": optional_string(input_data, "_mcpCallId"),
        }
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


def default_reasoning_effort() -> str:
    payload = default_dispatcher_config()
    value = payload.get("super_agents_reasoning_effort") or payload.get("superAgentsReasoningEffort")
    return value if isinstance(value, str) and value in REASONING_EFFORTS else "high"


def default_super_agents_model(*, backend: str | None = None) -> str | None:
    payload = default_dispatcher_config()
    return _model_for_backend(
        _backend_model(
            payload,
            "super_agents",
            backend=backend or backend_from_environment(),
        ),
        backend=backend,
    )


def _backend_model(payload: JsonObject, role: str, *, backend: str) -> str | None:
    backend_models = payload.get("backend_models") or payload.get("backendModels")
    if not isinstance(backend_models, dict):
        return None
    model_config = backend_models.get(backend)
    if not isinstance(model_config, dict):
        return None
    value = model_config.get(role)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _model_for_backend(model: str | None, *, backend: str | None = None) -> str | None:
    if not model:
        return None
    selected_backend = backend or backend_from_environment()
    normalized_model = model.strip().lower()
    if selected_backend in {"codex", "openbase_cloud"} and (
        normalized_model in CLAUDE_MODEL_ALIASES or normalized_model.startswith("claude-")
    ):
        logger.warning(
            "Ignoring Claude Super Agents model %s for Codex backend; using Codex default model",
            model,
        )
        return None
    return model


def default_dispatcher_config() -> JsonObject:
    configured = os.environ.get("SUPER_AGENTS_DEFAULT_CONFIG_PATH") or os.environ.get("LIVEKIT_DISPATCHER_CONFIG_PATH")
    paths = [Path(configured).expanduser()] if configured else []
    paths.extend([OPENBASE_DISPATCHER_CONFIG_PATH, LEGACY_OPENBASE_DISPATCHER_CONFIG_PATH])
    for config_path in paths:
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            return payload
    return {}


def clean_name_query_input(input_data: JsonObject) -> LabelQueryInput:
    return LabelQueryInput(
        label=optional_string(input_data, "name") or optional_string(input_data, "label"),
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


def without_none(value: JsonObject) -> JsonObject:
    return {key: item for key, item in value.items() if item is not None}


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
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    print("Super Agents MCP running on stdio.", file=__import__("sys").stderr)
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
