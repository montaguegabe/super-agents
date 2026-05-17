from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

import mcp.server.stdio
import mcp.types as types
from mcp.server.lowlevel import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from .app_server_client import CodexAppServerClient, LabelQueryInput

JsonObject = dict[str, Any]
Handler = Callable[[JsonObject], Awaitable[Any]]

INSTRUCTIONS = (
    "Control local Codex app-server threads asynchronously. Tools start, inspect, steer, cancel, and answer "
    "callbacks; they do not wait for turns to finish. Do not silently approve app-server callbacks; use "
    "codex_answer_request when a callback is pending."
)


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


def create_server(client: CodexAppServerClient | None = None) -> Server:
    app_client = client or CodexAppServerClient()
    server = Server("super-agents", instructions=INSTRUCTIONS)
    tool_by_name = {tool.name: tool for tool in build_tools(app_client)}

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [tool.to_mcp_tool() for tool in tool_by_name.values()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any] | None) -> types.CallToolResult:
        try:
            tool = tool_by_name.get(name)
            if not tool:
                raise ValueError(f"Unknown tool: {name}")
            output = await tool.handler(arguments or {})
            return text_tool_result(output)
        except Exception as exc:
            return text_tool_result({"error": str(exc)}, is_error=True)

    return server


def build_tools(client: CodexAppServerClient) -> list[ToolDefinition]:
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
            name="codex_thread_start",
            title="Start Codex Thread",
            description="Create a Codex app-server thread and optionally remember a human label for it.",
            input_schema=object_schema(
                {
                    "cwd": {"type": "string", "description": "Project working directory. Defaults to the user's home directory."},
                    "approvalPolicy": {"type": "string", "default": "never"},
                    "sandbox": {
                        "type": "string",
                        "enum": ["read-only", "workspace-write", "danger-full-access"],
                        "default": "danger-full-access",
                    },
                    "developerInstructions": {"type": "string"},
                    "label": {"type": "string", "description": "Friendly label stored by Super Agents for later lookup."},
                    "group": {"type": "string", "description": "Optional group name for related Super Agents sessions."},
                }
            ),
            handler=lambda input_data: client.start_thread(clean_thread_input(input_data)),
        ),
        ToolDefinition(
            name="codex_thread_resume",
            title="Resume Codex Thread",
            description="Resume an existing Codex thread and refresh the local session record.",
            input_schema=object_schema({"threadId": {"type": "string"}}, ["threadId"]),
            annotations={"readOnlyHint": True},
            handler=lambda input_data: client.resume_thread(required_string(input_data, "threadId")),
        ),
        ToolDefinition(
            name="codex_thread_list",
            title="List Codex Threads",
            description="List known Codex threads from the app server.",
            input_schema=object_schema({"useStateDbOnly": {"type": "boolean", "default": True}}),
            annotations={"readOnlyHint": True},
            handler=lambda input_data: client.list_threads(optional_boolean(input_data, "useStateDbOnly", True)),
        ),
        ToolDefinition(
            name="codex_thread_read",
            title="Read Codex Thread",
            description="Read a Codex thread, optionally including turns.",
            input_schema=object_schema(
                {"threadId": {"type": "string"}, "includeTurns": {"type": "boolean", "default": True}},
                ["threadId"],
            ),
            annotations={"readOnlyHint": True},
            handler=lambda input_data: client.read_thread(
                required_string(input_data, "threadId"),
                optional_boolean(input_data, "includeTurns", True),
            ),
        ),
        ToolDefinition(
            name="codex_turn_start",
            title="Start Codex Turn",
            description="Start a normal or plan-mode turn on an existing Codex thread and return immediately with the turn id.",
            input_schema=object_schema(turn_start_properties(), ["threadId", "prompt"]),
            handler=lambda input_data: client.start_turn(clean_turn_input(input_data)),
        ),
        ToolDefinition(
            name="codex_turn_progress",
            title="Check Codex Turn Progress",
            description="Check the current state of a turn without waiting for it to finish.",
            input_schema=object_schema({"threadId": {"type": "string"}, "turnId": {"type": "string"}}, ["threadId", "turnId"]),
            annotations={"readOnlyHint": True},
            handler=lambda input_data: client.turn_progress(
                required_string(input_data, "threadId"),
                required_string(input_data, "turnId"),
            ),
        ),
        ToolDefinition(
            name="codex_turn_steer",
            title="Steer Codex Turn",
            description="Send steering input to an active running Codex turn.",
            input_schema=object_schema(
                {"threadId": {"type": "string"}, "turnId": {"type": "string"}, "prompt": {"type": "string"}},
                ["threadId", "turnId", "prompt"],
            ),
            handler=lambda input_data: client.steer_turn(
                required_string(input_data, "threadId"),
                required_string(input_data, "turnId"),
                required_string(input_data, "prompt"),
            ),
        ),
        ToolDefinition(
            name="codex_turn_cancel",
            title="Cancel Codex Turn",
            description="Interrupt a running Codex turn.",
            input_schema=object_schema({"threadId": {"type": "string"}, "turnId": {"type": "string"}}, ["threadId", "turnId"]),
            handler=lambda input_data: client.cancel_turn(
                required_string(input_data, "threadId"),
                required_string(input_data, "turnId"),
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
            handler=lambda input_data: client.answer_request(required_request_id(input_data), required_object(input_data, "result")),
        ),
        ToolDefinition(
            name="super_agents_sessions",
            title="Super Agents Sessions",
            description="List thread labels remembered by this MCP wrapper.",
            input_schema=object_schema({}),
            annotations={"readOnlyHint": True, "idempotentHint": True},
            handler=lambda _input: client.sessions(),
        ),
        ToolDefinition(
            name="super_agents_active",
            title="Active Super Agents",
            description="List active tracked Super Agents with labels, cwd, thread ids, running turn ids, status, age, and previews.",
            input_schema=label_query_schema(),
            annotations={"readOnlyHint": True, "idempotentHint": True},
            handler=lambda input_data: client.active(clean_label_query_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_resolve",
            title="Resolve Super Agents Label",
            description="Resolve a label to the latest active matching Super Agents thread and turn by default.",
            input_schema=label_query_schema(["label"]),
            annotations={"readOnlyHint": True},
            handler=lambda input_data: client.resolve_label(clean_label_query_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_progress",
            title="Super Agents Progress By Label",
            description="Check progress for the latest active Super Agents turn matching a label.",
            input_schema=label_query_schema(["label"]),
            annotations={"readOnlyHint": True},
            handler=lambda input_data: client.progress_by_label(clean_label_query_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_steer",
            title="Steer Super Agents By Label",
            description="Send steering input to the latest active Super Agents turn matching a label.",
            input_schema=object_schema({**label_query_properties(), "prompt": {"type": "string"}}, ["label", "prompt"]),
            handler=lambda input_data: client.steer_by_label(
                clean_label_query_input(input_data),
                required_string(input_data, "prompt"),
            ),
        ),
        ToolDefinition(
            name="super_agents_cancel",
            title="Cancel Super Agents By Label",
            description="Cancel the latest active Super Agents turn matching a label.",
            input_schema=label_query_schema(["label"]),
            handler=lambda input_data: client.cancel_by_label(clean_label_query_input(input_data)),
        ),
        ToolDefinition(
            name="super_agents_start_turn",
            title="Start Super Agents Turn By Label",
            description="Start a follow-up turn on the latest matching Super Agents thread for a label.",
            input_schema=object_schema(
                {
                    **label_query_properties(),
                    "prompt": {"type": "string"},
                    "cwd": {"type": "string"},
                    "approvalPolicy": {"type": "string", "default": "never"},
                    "sandboxType": {
                        "type": "string",
                        "enum": ["readOnly", "workspaceWrite", "dangerFullAccess"],
                        "default": "dangerFullAccess",
                    },
                    "mode": {"type": "string", "enum": ["default", "plan"], "default": "default"},
                    "model": {"type": "string", "description": "Defaults to thread model or SUPER_AGENTS_MODEL."},
                    "reasoningEffort": {"type": "string", "default": "medium"},
                    "developerInstructions": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                ["label", "prompt"],
            ),
            handler=lambda input_data: client.start_turn_by_label(
                clean_label_query_input({**input_data, "label": required_string(input_data, "label")}),
                clean_start_turn_by_label_input(input_data),
            ),
        ),
        ToolDefinition(
            name="super_agents_recent",
            title="Recent Super Agents",
            description="List recent tracked Super Agents by label, cwd, group, and status.",
            input_schema=label_query_schema(),
            annotations={"readOnlyHint": True, "idempotentHint": True},
            handler=lambda input_data: client.recent(clean_label_query_input(input_data)),
        ),
    ]


def object_schema(properties: JsonObject, required: list[str] | None = None) -> JsonObject:
    schema: JsonObject = {"$schema": "http://json-schema.org/draft-07/schema#", "type": "object", "properties": properties}
    if required:
        schema["required"] = required
    return schema


def turn_start_properties() -> JsonObject:
    return {
        "threadId": {"type": "string"},
        "prompt": {"type": "string"},
        "cwd": {"type": "string"},
        "approvalPolicy": {"type": "string", "default": "never"},
        "sandboxType": {
            "type": "string",
            "enum": ["readOnly", "workspaceWrite", "dangerFullAccess"],
            "default": "dangerFullAccess",
        },
        "mode": {"type": "string", "enum": ["default", "plan"], "default": "default"},
        "model": {"type": "string", "description": "Defaults to thread model or SUPER_AGENTS_MODEL."},
        "reasoningEffort": {"type": "string", "default": "medium"},
        "developerInstructions": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "label": {"type": "string"},
        "group": {"type": "string"},
    }


def label_query_properties() -> JsonObject:
    return {
        "label": {"type": "string"},
        "cwd": {"type": "string"},
        "group": {"type": "string"},
        "status": {"type": "string", "enum": ["running", "waiting", "completed", "failed", "cancelled", "unknown"]},
        "limit": {"type": "number"},
        "includeInactive": {"type": "boolean", "default": False},
        "prefer": {"type": "string", "enum": ["latest_active", "latest_any"], "default": "latest_active"},
        "turnId": {"type": "string"},
    }


def label_query_schema(required: list[str] | None = None) -> JsonObject:
    return object_schema(label_query_properties(), required)


def text_tool_result(value: Any, is_error: bool = False) -> types.CallToolResult:
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(value, indent=2))],
        isError=is_error,
    )


def clean_thread_input(input_data: JsonObject) -> JsonObject:
    return without_none(
        {
            "cwd": optional_string(input_data, "cwd"),
            "approvalPolicy": optional_string(input_data, "approvalPolicy") or "never",
            "sandbox": optional_string(input_data, "sandbox") or "danger-full-access",
            "developerInstructions": optional_string(input_data, "developerInstructions"),
            "label": optional_string(input_data, "label"),
            "group": optional_string(input_data, "group"),
        }
    )


def clean_turn_input(input_data: JsonObject) -> JsonObject:
    return without_none(
        {
            "threadId": required_string(input_data, "threadId"),
            "prompt": required_string(input_data, "prompt"),
            "cwd": optional_string(input_data, "cwd"),
            "approvalPolicy": optional_string(input_data, "approvalPolicy") or "never",
            "sandboxType": optional_string(input_data, "sandboxType") or "dangerFullAccess",
            "mode": optional_mode(input_data, "mode") or "default",
            "model": optional_string(input_data, "model"),
            "reasoningEffort": optional_string(input_data, "reasoningEffort") or "medium",
            "developerInstructions": optional_nullable_string(input_data, "developerInstructions"),
            "label": optional_string(input_data, "label"),
            "group": optional_string(input_data, "group"),
        }
    )


def clean_label_query_input(input_data: JsonObject) -> LabelQueryInput:
    return LabelQueryInput(
        label=optional_string(input_data, "label"),
        cwd=optional_string(input_data, "cwd"),
        group=optional_string(input_data, "group"),
        status=optional_string(input_data, "status"),
        limit=optional_number(input_data, "limit"),
        include_inactive=optional_boolean_or_none(input_data, "includeInactive"),
        prefer=optional_prefer(input_data, "prefer"),
        turn_id=optional_string(input_data, "turnId"),
    )


def clean_start_turn_by_label_input(input_data: JsonObject) -> JsonObject:
    cleaned = clean_turn_input({"threadId": "__placeholder__", **input_data})
    cleaned.pop("threadId", None)
    cleaned["label"] = required_string(input_data, "label")
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


def optional_nullable_string(value: JsonObject, key: str) -> str | None:
    if key in value and value[key] is None:
        return None
    return optional_string(value, key)


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
    print("Super Agents MCP running on stdio.", file=__import__("sys").stderr)
    asyncio.run(run_stdio())


if __name__ == "__main__":
    main()
