#!/usr/bin/env node
import { createInterface } from "node:readline";

import { CodexAppServerClient, type LabelQueryInput, type StartThreadInput, type StartTurnInput } from "./appServerClient.js";

type JsonObject = Record<string, unknown>;

type ToolDefinition = {
  name: string;
  title: string;
  description: string;
  inputSchema: JsonObject;
  annotations?: JsonObject;
  handler: (input: JsonObject) => Promise<unknown>;
};

const client = new CodexAppServerClient();

const instructions =
  "Control local Codex app-server threads asynchronously. Tools start, inspect, steer, cancel, and answer callbacks; they do not wait for turns to finish. Do not silently approve app-server callbacks; use codex_answer_request when a callback is pending.";

const tools: ToolDefinition[] = [
  {
    name: "codex_app_server_status",
    title: "Codex App Server Status",
    description: "Check local Codex app-server readiness, websocket connection, pending requests, and active turns.",
    inputSchema: objectSchema({}),
    annotations: { readOnlyHint: true, idempotentHint: true },
    handler: async () => client.status(),
  },
  {
    name: "codex_thread_start",
    title: "Start Codex Thread",
    description: "Create a Codex app-server thread and optionally remember a human label for it.",
    inputSchema: objectSchema({
      cwd: { type: "string", description: "Project working directory. Defaults to the user's home directory." },
      approvalPolicy: { type: "string", default: "never" },
      sandbox: { type: "string", enum: ["read-only", "workspace-write", "danger-full-access"], default: "danger-full-access" },
      developerInstructions: { type: "string" },
      label: { type: "string", description: "Friendly label stored by Super Agents for later lookup." },
      group: { type: "string", description: "Optional group name for related Super Agents sessions." },
    }),
    handler: async (input) => client.startThread(cleanThreadInput(input)),
  },
  {
    name: "codex_thread_resume",
    title: "Resume Codex Thread",
    description: "Resume an existing Codex thread and refresh the local session record.",
    inputSchema: objectSchema({ threadId: { type: "string" } }, ["threadId"]),
    annotations: { readOnlyHint: true },
    handler: async (input) => client.resumeThread(requiredString(input, "threadId")),
  },
  {
    name: "codex_thread_list",
    title: "List Codex Threads",
    description: "List known Codex threads from the app server.",
    inputSchema: objectSchema({ useStateDbOnly: { type: "boolean", default: true } }),
    annotations: { readOnlyHint: true },
    handler: async (input) => client.listThreads(optionalBoolean(input, "useStateDbOnly") ?? true),
  },
  {
    name: "codex_thread_read",
    title: "Read Codex Thread",
    description: "Read a Codex thread, optionally including turns.",
    inputSchema: objectSchema(
      {
        threadId: { type: "string" },
        includeTurns: { type: "boolean", default: true },
      },
      ["threadId"],
    ),
    annotations: { readOnlyHint: true },
    handler: async (input) => client.readThread(requiredString(input, "threadId"), optionalBoolean(input, "includeTurns") ?? true),
  },
  {
    name: "codex_turn_start",
    title: "Start Codex Turn",
    description: "Start a normal or plan-mode turn on an existing Codex thread and return immediately with the turn id.",
    inputSchema: objectSchema(
      {
        threadId: { type: "string" },
        prompt: { type: "string" },
        cwd: { type: "string" },
        approvalPolicy: { type: "string", default: "never" },
        sandboxType: { type: "string", enum: ["readOnly", "workspaceWrite", "dangerFullAccess"], default: "dangerFullAccess" },
        mode: { type: "string", enum: ["default", "plan"], default: "default" },
        model: { type: "string", description: "Defaults to thread model or SUPER_AGENTS_MODEL." },
        reasoningEffort: { type: "string", default: "medium" },
        developerInstructions: { anyOf: [{ type: "string" }, { type: "null" }] },
        label: { type: "string" },
        group: { type: "string" },
      },
      ["threadId", "prompt"],
    ),
    handler: async (input) => client.startTurn(cleanTurnInput(input)),
  },
  {
    name: "codex_turn_progress",
    title: "Check Codex Turn Progress",
    description: "Check the current state of a turn without waiting for it to finish.",
    inputSchema: objectSchema(
      {
        threadId: { type: "string" },
        turnId: { type: "string" },
      },
      ["threadId", "turnId"],
    ),
    annotations: { readOnlyHint: true },
    handler: async (input) =>
      client.turnProgress(requiredString(input, "threadId"), requiredString(input, "turnId")),
  },
  {
    name: "codex_turn_steer",
    title: "Steer Codex Turn",
    description: "Send steering input to an active running Codex turn.",
    inputSchema: objectSchema(
      {
        threadId: { type: "string" },
        turnId: { type: "string" },
        prompt: { type: "string" },
      },
      ["threadId", "turnId", "prompt"],
    ),
    handler: async (input) =>
      client.steerTurn(requiredString(input, "threadId"), requiredString(input, "turnId"), requiredString(input, "prompt")),
  },
  {
    name: "codex_turn_cancel",
    title: "Cancel Codex Turn",
    description: "Interrupt a running Codex turn.",
    inputSchema: objectSchema(
      {
        threadId: { type: "string" },
        turnId: { type: "string" },
      },
      ["threadId", "turnId"],
    ),
    handler: async (input) =>
      client.cancelTurn(requiredString(input, "threadId"), requiredString(input, "turnId")),
  },
  {
    name: "codex_answer_request",
    title: "Answer Codex Request",
    description:
      "Answer a pending app-server callback. For plan questions, pass result { answers: { question_id: { answers: [...] } } }. For approvals, pass result { decision: 'accept' | 'decline' | 'cancel' }.",
    inputSchema: objectSchema(
      {
        requestId: { anyOf: [{ type: "string" }, { type: "number" }] },
        result: { type: "object", additionalProperties: true },
      },
      ["requestId", "result"],
    ),
    handler: async (input) => {
      const requestId = input.requestId;
      if (typeof requestId !== "string" && typeof requestId !== "number") {
        throw new Error("requestId must be a string or number.");
      }
      return client.answerRequest(requestId, requiredObject(input, "result"));
    },
  },
  {
    name: "super_agents_sessions",
    title: "Super Agents Sessions",
    description: "List thread labels remembered by this MCP wrapper.",
    inputSchema: objectSchema({}),
    annotations: { readOnlyHint: true, idempotentHint: true },
    handler: async () => client.sessions(),
  },
  {
    name: "super_agents_active",
    title: "Active Super Agents",
    description: "List active tracked Super Agents with labels, cwd, thread ids, running turn ids, status, age, and previews.",
    inputSchema: labelQuerySchema(),
    annotations: { readOnlyHint: true, idempotentHint: true },
    handler: async (input) => client.active(cleanLabelQueryInput(input)),
  },
  {
    name: "super_agents_resolve",
    title: "Resolve Super Agents Label",
    description: "Resolve a label to the latest active matching Super Agents thread and turn by default.",
    inputSchema: labelQuerySchema(["label"]),
    annotations: { readOnlyHint: true },
    handler: async (input) => client.resolveLabel(cleanLabelQueryInput(input)),
  },
  {
    name: "super_agents_progress",
    title: "Super Agents Progress By Label",
    description: "Check progress for the latest active Super Agents turn matching a label.",
    inputSchema: labelQuerySchema(["label"]),
    annotations: { readOnlyHint: true },
    handler: async (input) => client.progressByLabel(cleanLabelQueryInput(input)),
  },
  {
    name: "super_agents_steer",
    title: "Steer Super Agents By Label",
    description: "Send steering input to the latest active Super Agents turn matching a label.",
    inputSchema: objectSchema({ ...labelQueryProperties(), prompt: { type: "string" } }, ["label", "prompt"]),
    handler: async (input) => client.steerByLabel({ ...cleanLabelQueryInput(input), prompt: requiredString(input, "prompt") }),
  },
  {
    name: "super_agents_cancel",
    title: "Cancel Super Agents By Label",
    description: "Cancel the latest active Super Agents turn matching a label.",
    inputSchema: labelQuerySchema(["label"]),
    handler: async (input) => client.cancelByLabel(cleanLabelQueryInput(input)),
  },
  {
    name: "super_agents_start_turn",
    title: "Start Super Agents Turn By Label",
    description: "Start a follow-up turn on the latest matching Super Agents thread for a label.",
    inputSchema: objectSchema(
      {
        ...labelQueryProperties(),
        prompt: { type: "string" },
        cwd: { type: "string" },
        approvalPolicy: { type: "string", default: "never" },
        sandboxType: { type: "string", enum: ["readOnly", "workspaceWrite", "dangerFullAccess"], default: "dangerFullAccess" },
        mode: { type: "string", enum: ["default", "plan"], default: "default" },
        model: { type: "string", description: "Defaults to thread model or SUPER_AGENTS_MODEL." },
        reasoningEffort: { type: "string", default: "medium" },
        developerInstructions: { anyOf: [{ type: "string" }, { type: "null" }] },
      },
      ["label", "prompt"],
    ),
    handler: async (input) => client.startTurnByLabel(cleanStartTurnByLabelInput(input)),
  },
  {
    name: "super_agents_recent",
    title: "Recent Super Agents",
    description: "List recent tracked Super Agents by label, cwd, group, and status.",
    inputSchema: labelQuerySchema(),
    annotations: { readOnlyHint: true, idempotentHint: true },
    handler: async (input) => client.recent(cleanLabelQueryInput(input)),
  },
];

const toolByName = new Map(tools.map((tool) => [tool.name, tool]));

const lineReader = createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

console.error("Super Agents MCP running on stdio.");

lineReader.on("line", (line) => {
  if (!line.trim()) {
    return;
  }
  void handleMessage(line);
});

async function handleMessage(line: string): Promise<void> {
  let message: JsonObject;
  try {
    message = JSON.parse(line) as JsonObject;
  } catch (error) {
    sendError(null, -32700, errorMessage(error));
    return;
  }

  const id = typeof message.id === "string" || typeof message.id === "number" || message.id === null ? message.id : null;
  const method = typeof message.method === "string" ? message.method : undefined;
  if (!method) {
    if (id !== null) {
      sendError(id, -32600, "Missing method.");
    }
    return;
  }

  try {
    if (method === "initialize") {
      sendResult(id, {
        protocolVersion: protocolVersion(message.params),
        capabilities: { tools: { listChanged: true } },
        serverInfo: { name: "super-agents", version: "0.1.0" },
        instructions,
      });
      return;
    }
    if (method === "notifications/initialized") {
      return;
    }
    if (method === "ping") {
      sendResult(id, {});
      return;
    }
    if (method === "tools/list") {
      sendResult(id, {
        tools: tools.map(({ handler: _handler, ...tool }) => tool),
      });
      return;
    }
    if (method === "tools/call") {
      const params = asObject(message.params);
      const name = requiredString(params, "name");
      const tool = toolByName.get(name);
      if (!tool) {
        throw new Error(`Unknown tool: ${name}`);
      }
      const output = await tool.handler(asObject(params.arguments));
      sendResult(id, textToolResult(output));
      return;
    }
    sendError(id, -32601, `Method not found: ${method}`);
  } catch (error) {
    sendResult(id, textToolResult({ error: errorMessage(error) }, true));
  }
}

function objectSchema(properties: JsonObject, required: string[] = []): JsonObject {
  return {
    $schema: "http://json-schema.org/draft-07/schema#",
    type: "object",
    properties,
    ...(required.length ? { required } : {}),
  };
}

function labelQueryProperties(): JsonObject {
  return {
    label: { type: "string" },
    cwd: { type: "string" },
    group: { type: "string" },
    status: { type: "string", enum: ["running", "waiting", "completed", "failed", "cancelled", "unknown"] },
    limit: { type: "number" },
    includeInactive: { type: "boolean", default: false },
    prefer: { type: "string", enum: ["latest_active", "latest_any"], default: "latest_active" },
    turnId: { type: "string" },
  };
}

function labelQuerySchema(required: string[] = []): JsonObject {
  return objectSchema(labelQueryProperties(), required);
}

function textToolResult(value: unknown, isError = false): JsonObject {
  return {
    content: [
      {
        type: "text",
        text: JSON.stringify(value, null, 2),
      },
    ],
    isError,
  };
}

function cleanThreadInput(input: JsonObject): StartThreadInput {
  return withoutUndefined({
    cwd: optionalString(input, "cwd"),
    approvalPolicy: optionalString(input, "approvalPolicy") ?? "never",
    sandbox: optionalString(input, "sandbox") ?? "danger-full-access",
    developerInstructions: optionalString(input, "developerInstructions"),
    label: optionalString(input, "label"),
    group: optionalString(input, "group"),
  });
}

function cleanTurnInput(input: JsonObject): StartTurnInput {
  return withoutUndefined({
    threadId: requiredString(input, "threadId"),
    prompt: requiredString(input, "prompt"),
    cwd: optionalString(input, "cwd"),
    approvalPolicy: optionalString(input, "approvalPolicy") ?? "never",
    sandboxType: optionalString(input, "sandboxType") ?? "dangerFullAccess",
    mode: optionalMode(input, "mode") ?? "default",
    model: optionalString(input, "model"),
    reasoningEffort: optionalString(input, "reasoningEffort") ?? "medium",
    developerInstructions: optionalNullableString(input, "developerInstructions"),
    label: optionalString(input, "label"),
    group: optionalString(input, "group"),
  });
}

function cleanLabelQueryInput(input: JsonObject): LabelQueryInput {
  return withoutUndefined({
    label: optionalString(input, "label"),
    cwd: optionalString(input, "cwd"),
    group: optionalString(input, "group"),
    status: optionalString(input, "status"),
    limit: optionalNumber(input, "limit"),
    includeInactive: optionalBoolean(input, "includeInactive"),
    prefer: optionalPrefer(input, "prefer"),
    turnId: optionalString(input, "turnId"),
  });
}

function cleanStartTurnByLabelInput(input: JsonObject): Omit<StartTurnInput, "threadId"> & LabelQueryInput {
  return withoutUndefined({
    ...cleanLabelQueryInput(input),
    label: requiredString(input, "label"),
    prompt: requiredString(input, "prompt"),
    cwd: optionalString(input, "cwd"),
    approvalPolicy: optionalString(input, "approvalPolicy") ?? "never",
    sandboxType: optionalString(input, "sandboxType") ?? "dangerFullAccess",
    mode: optionalMode(input, "mode") ?? "default",
    model: optionalString(input, "model"),
    reasoningEffort: optionalString(input, "reasoningEffort") ?? "medium",
    developerInstructions: optionalNullableString(input, "developerInstructions"),
    group: optionalString(input, "group"),
  });
}

function sendResult(id: string | number | null, result: unknown): void {
  if (id === null) {
    return;
  }
  process.stdout.write(JSON.stringify({ result, jsonrpc: "2.0", id }) + "\n");
}

function sendError(id: string | number | null, code: number, message: string): void {
  if (id === null) {
    return;
  }
  process.stdout.write(JSON.stringify({ error: { code, message }, jsonrpc: "2.0", id }) + "\n");
}

function protocolVersion(params: unknown): string {
  const version = asObject(params).protocolVersion;
  return typeof version === "string" ? version : "2025-06-18";
}

function requiredString(value: JsonObject, key: string): string {
  const result = value[key];
  if (typeof result !== "string" || !result) {
    throw new Error(`${key} must be a non-empty string.`);
  }
  return result;
}

function requiredObject(value: JsonObject, key: string): JsonObject {
  return asObject(value[key]);
}

function optionalString(value: JsonObject, key: string): string | undefined {
  const result = value[key];
  return typeof result === "string" && result ? result : undefined;
}

function optionalNullableString(value: JsonObject, key: string): string | null | undefined {
  if (value[key] === null) {
    return null;
  }
  return optionalString(value, key);
}

function optionalBoolean(value: JsonObject, key: string): boolean | undefined {
  const result = value[key];
  return typeof result === "boolean" ? result : undefined;
}

function optionalNumber(value: JsonObject, key: string): number | undefined {
  const result = value[key];
  return typeof result === "number" && Number.isFinite(result) && result > 0 ? Math.floor(result) : undefined;
}

function optionalMode(value: JsonObject, key: string): "default" | "plan" | undefined {
  const result = optionalString(value, key);
  if (result === "default" || result === "plan") {
    return result;
  }
  return undefined;
}

function optionalPrefer(value: JsonObject, key: string): "latest_active" | "latest_any" | undefined {
  const result = optionalString(value, key);
  if (result === "latest_active" || result === "latest_any") {
    return result;
  }
  return undefined;
}

function asObject(value: unknown): JsonObject {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonObject) : {};
}

function withoutUndefined<T extends JsonObject>(value: T): T {
  return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== undefined)) as T;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
