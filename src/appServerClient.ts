import { spawn, type ChildProcess } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { homedir } from "node:os";
import WebSocket from "ws";

type JsonObject = Record<string, unknown>;

type PendingRpc = {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
};

export type PendingServerRequest = {
  id: string | number;
  method: string;
  params: JsonObject;
  receivedAt: string;
};

export type TurnState = {
  threadId: string;
  turnId: string;
  status: "running" | "completed" | "failed" | "waiting" | "cancelled";
  startedAt: string;
  finishedAt?: string;
  events: JsonObject[];
  pendingRequests: PendingServerRequest[];
};

type TrackedStatus = TurnState["status"];

type TurnSummary = {
  turnId: string;
  status: TrackedStatus;
  mode?: "default" | "plan" | undefined;
  startedAt: string;
  updatedAt: string;
  finishedAt?: string | undefined;
  promptPreview?: string | undefined;
  lastUsefulMessage?: string | undefined;
  pendingRequestIds?: Array<string | number> | undefined;
  eventCount?: number | undefined;
};

type SessionRecord = {
  label?: string | undefined;
  threadId: string;
  cwd?: string | undefined;
  group?: string | undefined;
  model?: string | undefined;
  lastTurnId?: string | undefined;
  activeTurnId?: string | undefined;
  createdAt?: string | undefined;
  lastStartedAt?: string | undefined;
  lastFinishedAt?: string | undefined;
  lastStatus?: TrackedStatus | "unknown" | undefined;
  lastUsefulMessage?: string | undefined;
  lastEventAt?: string | undefined;
  turns?: Record<string, TurnSummary> | undefined;
  updatedAt: string;
};

type StateFile = {
  sessions: Record<string, SessionRecord>;
};

export type StartThreadInput = {
  cwd?: string | undefined;
  approvalPolicy?: string | undefined;
  sandbox?: string | undefined;
  developerInstructions?: string | undefined;
  label?: string | undefined;
  group?: string | undefined;
};

export type StartTurnInput = {
  threadId: string;
  prompt: string;
  cwd?: string | undefined;
  approvalPolicy?: string | undefined;
  sandboxType?: string | undefined;
  mode?: "default" | "plan" | undefined;
  model?: string | undefined;
  reasoningEffort?: string | undefined;
  developerInstructions?: string | null | undefined;
  label?: string | undefined;
  group?: string | undefined;
};

type LabelResolutionPrefer = "latest_active" | "latest_any";

export type LabelQueryInput = {
  label?: string | undefined;
  cwd?: string | undefined;
  group?: string | undefined;
  status?: string | undefined;
  limit?: number | undefined;
  includeInactive?: boolean | undefined;
  prefer?: LabelResolutionPrefer | undefined;
  turnId?: string | undefined;
};

type ResolvedSession = {
  session: SessionRecord;
  turnId?: string | undefined;
  status: TrackedStatus | "unknown";
};

const DEFAULT_WS_URL = "ws://127.0.0.1:4500";
const DEFAULT_MODEL = "gpt-5.4";
const DEFAULT_STATE_FILE = join(homedir(), ".super-agents", "state.json");
const LOGIN_ENV_TIMEOUT_MS = 5_000;

let loginShellEnvironmentPromise: Promise<NodeJS.ProcessEnv> | undefined;

export class CodexAppServerClient {
  private ws: WebSocket | undefined;
  private nextId = 1;
  private pending = new Map<string | number, PendingRpc>();
  private pendingServerRequests = new Map<string | number, PendingServerRequest>();
  private turns = new Map<string, TurnState>();
  private child: ChildProcess | undefined;
  private initializePromise: Promise<void> | undefined;

  constructor(
    private readonly wsUrl = process.env.SUPER_AGENTS_WS_URL ?? DEFAULT_WS_URL,
    private readonly stateFile = process.env.SUPER_AGENTS_STATE_FILE ?? DEFAULT_STATE_FILE,
    private readonly defaultModel = process.env.SUPER_AGENTS_MODEL ?? DEFAULT_MODEL,
  ) {}

  async status(): Promise<JsonObject> {
    const ready = await this.checkReady();
    return {
      ready,
      websocketUrl: this.wsUrl,
      websocketConnected: this.ws?.readyState === WebSocket.OPEN,
      managedProcess: Boolean(this.child && !this.child.killed),
      pendingRequests: [...this.pendingServerRequests.values()],
      activeTurns: [...this.turns.values()].filter((turn) => turn.status === "running" || turn.status === "waiting"),
    };
  }

  async ensureConnected(): Promise<void> {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }
    if (this.initializePromise) {
      return this.initializePromise;
    }

    this.initializePromise = this.connect();
    try {
      await this.initializePromise;
    } finally {
      this.initializePromise = undefined;
    }
  }

  async startThread(input: StartThreadInput): Promise<JsonObject> {
    await this.ensureConnected();
    const params: JsonObject = {
      cwd: input.cwd ?? homedir(),
      approvalPolicy: input.approvalPolicy ?? "never",
      sandbox: input.sandbox ?? "danger-full-access",
      config: await loginShellConfigOverride(),
    };
    if (input.developerInstructions) {
      params.developerInstructions = input.developerInstructions;
    }

    const result = await this.request<JsonObject>("thread/start", params);
    const threadId = extractThreadId(result);
    if (threadId) {
      const now = new Date().toISOString();
      await this.rememberSession(threadId, withoutUndefined({
        label: input.label,
        threadId,
        cwd: extractThreadCwd(result) ?? String(params.cwd),
        group: input.group,
        model: extractModel(result) ?? this.defaultModel,
        createdAt: now,
        lastStatus: "unknown",
      }));
    }
    return result;
  }

  async resumeThread(threadId: string): Promise<JsonObject> {
    await this.ensureConnected();
    const result = await this.request<JsonObject>("thread/resume", {
      threadId,
      approvalPolicy: "never",
      sandbox: "danger-full-access",
      config: await loginShellConfigOverride(),
    });
    await this.mergeSession(threadId, {
      threadId,
      model: extractModel(result) ?? this.defaultModel,
      lastUsefulMessage: textPreview(result),
    });
    return result;
  }

  async listThreads(useStateDbOnly = true): Promise<JsonObject> {
    await this.ensureConnected();
    return this.request<JsonObject>("thread/list", { useStateDbOnly });
  }

  async readThread(threadId: string, includeTurns = true): Promise<JsonObject> {
    await this.ensureConnected();
    return this.request<JsonObject>("thread/read", { threadId, includeTurns });
  }

  async startTurn(input: StartTurnInput): Promise<JsonObject> {
    await this.ensureConnected();
    const session = await this.getSession(input.threadId);
    const mode = input.mode ?? "default";
    const model = input.model ?? session?.model ?? this.defaultModel;
    const params: JsonObject = {
      threadId: input.threadId,
      cwd: input.cwd ?? session?.cwd ?? homedir(),
      approvalPolicy: input.approvalPolicy ?? "never",
      sandboxPolicy: { type: input.sandboxType ?? "dangerFullAccess" },
      collaborationMode: collaborationMode(mode, model, input.reasoningEffort, input.developerInstructions),
      input: [{ type: "text", text: input.prompt }],
    };

    const result = await this.request<JsonObject>("turn/start", params);
    const turnId = extractTurnId(result) ?? `${input.threadId}:unknown:${Date.now()}`;
    const now = new Date().toISOString();
    this.turns.set(turnKey(input.threadId, turnId), {
      threadId: input.threadId,
      turnId,
      status: "running",
      startedAt: now,
      events: [],
      pendingRequests: [],
    });
    await this.mergeSession(input.threadId, {
      label: input.label,
      threadId: input.threadId,
      cwd: String(params.cwd),
      group: input.group,
      model,
      lastTurnId: turnId,
      activeTurnId: turnId,
      lastStartedAt: now,
      lastStatus: "running",
      lastUsefulMessage: textPreview(result),
      turns: {
        [turnId]: withoutUndefined({
          turnId,
          status: "running",
          mode,
          startedAt: now,
          updatedAt: now,
          promptPreview: previewText(input.prompt),
          lastUsefulMessage: textPreview(result),
          pendingRequestIds: [],
          eventCount: 0,
        }),
      },
    });
    return { ...result, threadId: input.threadId, turnId, mode };
  }

  async steerTurn(threadId: string, turnId: string, prompt: string): Promise<JsonObject> {
    await this.ensureConnected();
    return this.request<JsonObject>("turn/steer", {
      threadId,
      expectedTurnId: turnId,
      input: [{ type: "text", text: prompt }],
    });
  }

  async turnProgress(threadId: string, turnId: string): Promise<JsonObject> {
    await this.ensureConnected();
    const key = turnKey(threadId, turnId);
    const trackedTurn = this.turns.get(key);
    const thread = await this.readThread(threadId, true);
    const persistedTurn = findTurn(thread, turnId);
    const pendingRequests = [...this.pendingServerRequests.values()].filter(
      (request) => request.params.threadId === threadId && request.params.turnId === turnId,
    );
    const status = pendingRequests.length
      ? "waiting"
      : normalizeTurnStatus(persistedTurn) ?? trackedTurn?.status ?? "unknown";
    if (trackedTurn && status !== "unknown") {
      trackedTurn.status = toTrackedTurnStatus(status);
      if ((trackedTurn.status === "completed" || trackedTurn.status === "failed") && !trackedTurn.finishedAt) {
        trackedTurn.finishedAt = new Date().toISOString();
      }
    }
    await this.recordTurnProgress(threadId, turnId, status, persistedTurn, pendingRequests);

    return {
      status,
      threadId,
      turnId,
      turn: persistedTurn,
      trackedTurn,
      pendingRequests,
    };
  }

  async cancelTurn(threadId: string, turnId: string): Promise<JsonObject> {
    await this.ensureConnected();
    const result = await this.request<JsonObject>("turn/interrupt", { threadId, turnId });
    const turn = this.ensureTurn(threadId, turnId);
    turn.status = "cancelled";
    turn.finishedAt = new Date().toISOString();
    await this.mergeSession(
      threadId,
      {
        threadId,
        lastTurnId: turnId,
        lastStatus: "cancelled",
        lastFinishedAt: turn.finishedAt,
        turns: {
          [turnId]: withoutUndefined({
            turnId,
            status: "cancelled",
            startedAt: turn.startedAt,
            updatedAt: turn.finishedAt,
            finishedAt: turn.finishedAt,
            eventCount: turn.events.length,
            pendingRequestIds: [],
          }),
        },
      },
      ["activeTurnId"],
    );
    return { cancelled: true, threadId, turnId, result };
  }

  async answerRequest(id: string | number, result: JsonObject): Promise<JsonObject> {
    await this.ensureConnected();
    if (!this.pendingServerRequests.has(id)) {
      throw new Error(`No pending app-server request found for id ${String(id)}.`);
    }
    this.send({ id, result });
    const request = this.pendingServerRequests.get(id);
    this.pendingServerRequests.delete(id);
    this.removePendingRequestFromTurn(id);
    return { answered: true, request };
  }

  async sessions(): Promise<SessionRecord[]> {
    const state = await this.readState();
    return Object.values(state.sessions).sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
  }

  async active(input: LabelQueryInput = {}): Promise<JsonObject> {
    const sessions = await this.filteredSessions(input);
    const items = sessions
      .map((session) => this.sessionView(session))
      .filter((item) => isActiveStatus(typeof item.status === "string" ? item.status : undefined))
      .slice(0, input.limit ?? 50);
    return { count: items.length, agents: items };
  }

  async recent(input: LabelQueryInput = {}): Promise<JsonObject> {
    const sessions = await this.filteredSessions(input);
    const items = sessions
      .map((session) => this.sessionView(session))
      .filter((item) => input.includeInactive || isActiveStatus(typeof item.status === "string" ? item.status : undefined))
      .slice(0, input.limit ?? 20);
    return { count: items.length, agents: items };
  }

  async resolveLabel(input: LabelQueryInput): Promise<JsonObject> {
    const resolved = await this.resolveSession(requiredLabel(input), input);
    return {
      label: resolved.session.label,
      group: resolved.session.group,
      cwd: resolved.session.cwd,
      threadId: resolved.session.threadId,
      turnId: resolved.turnId,
      status: resolved.status,
      updatedAt: resolved.session.updatedAt,
      lastUsefulMessage: resolved.session.lastUsefulMessage,
    };
  }

  async progressByLabel(input: LabelQueryInput): Promise<JsonObject> {
    const resolved = await this.resolveSession(requiredLabel(input), input);
    const turnId = input.turnId ?? resolved.turnId;
    if (!turnId) {
      throw new Error(`No turn is known for label ${requiredLabel(input)}.`);
    }
    return this.turnProgress(resolved.session.threadId, turnId);
  }

  async steerByLabel(input: LabelQueryInput & { prompt?: string | undefined }): Promise<JsonObject> {
    const resolved = await this.resolveSession(requiredLabel(input), input);
    const turnId = input.turnId ?? resolved.turnId;
    if (!turnId || !isActiveStatus(resolved.status)) {
      throw new Error(`No active turn is known for label ${requiredLabel(input)}.`);
    }
    return this.steerTurn(resolved.session.threadId, turnId, requiredPrompt(input));
  }

  async cancelByLabel(input: LabelQueryInput): Promise<JsonObject> {
    const resolved = await this.resolveSession(requiredLabel(input), input);
    const turnId = input.turnId ?? resolved.turnId;
    if (!turnId || !isActiveStatus(resolved.status)) {
      throw new Error(`No active turn is known for label ${requiredLabel(input)}.`);
    }
    return this.cancelTurn(resolved.session.threadId, turnId);
  }

  async startTurnByLabel(input: Omit<StartTurnInput, "threadId"> & LabelQueryInput): Promise<JsonObject> {
    const resolved = await this.resolveSession(requiredLabel(input), { ...input, prefer: input.prefer ?? "latest_any" });
    return this.startTurn({
      threadId: resolved.session.threadId,
      prompt: requiredPrompt(input),
      cwd: input.cwd,
      approvalPolicy: input.approvalPolicy,
      sandboxType: input.sandboxType,
      mode: input.mode,
      model: input.model,
      reasoningEffort: input.reasoningEffort,
      developerInstructions: input.developerInstructions,
      label: input.label,
      group: input.group,
    });
  }

  defaultModelName(): string {
    return this.defaultModel;
  }

  private async connect(): Promise<void> {
    if (!(await this.checkReady())) {
      await this.startManagedServer();
    }

    this.ws = await this.openWebsocket();
    this.ws.on("message", (data) => this.handleMessage(data.toString()));
    this.ws.on("close", () => {
      this.rejectPending(new Error("Codex app-server websocket closed."));
      this.ws = undefined;
    });
    this.ws.on("error", (error) => {
      this.rejectPending(error instanceof Error ? error : new Error(String(error)));
    });

    await this.request("initialize", {
      clientInfo: {
        name: "super-agents-mcp",
        title: "Super Agents MCP",
        version: "0.1.0",
      },
      capabilities: {
        experimentalApi: true,
      },
    });
    this.send({ method: "initialized", params: {} });
  }

  private async openWebsocket(): Promise<WebSocket> {
    return new Promise((resolve, reject) => {
      const ws = new WebSocket(this.wsUrl);
      const timer = setTimeout(() => {
        ws.close();
        reject(new Error("Timed out connecting to Codex app-server websocket."));
      }, 5_000);
      ws.once("open", () => {
        clearTimeout(timer);
        resolve(ws);
      });
      ws.once("error", (error) => {
        clearTimeout(timer);
        reject(error);
      });
    });
  }

  private async startManagedServer(): Promise<void> {
    const env = await loginShellEnvironment();
    this.child = spawn("codex", ["app-server", "--listen", this.wsUrl], {
      stdio: ["ignore", "ignore", "pipe"],
      env,
    });
    this.child.stderr?.on("data", (data) => {
      console.error(`[codex app-server] ${data.toString().trim()}`);
    });

    const started = Date.now();
    while (Date.now() - started < 10_000) {
      if (await this.checkReady()) {
        return;
      }
      await sleep(250);
    }
    throw new Error("Codex app-server did not become ready.");
  }

  private async checkReady(): Promise<boolean> {
    try {
      const readyUrl = this.wsUrl.replace(/^ws:/, "http:").replace(/^wss:/, "https:").replace(/\/$/, "") + "/readyz";
      const response = await fetch(readyUrl, { signal: AbortSignal.timeout(1_000) });
      return response.ok;
    } catch {
      return false;
    }
  }

  private request<T>(method: string, params: JsonObject = {}, timeoutMs = 30_000): Promise<T> {
    const id = this.nextId++;
    this.send({ id, method, params });
    return new Promise<T>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error(`Timed out waiting for app-server response to ${method}.`));
      }, timeoutMs);
      this.pending.set(id, {
        resolve: (value) => resolve(value as T),
        reject,
        timer,
      });
    });
  }

  private send(message: JsonObject): void {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) {
      throw new Error("Codex app-server websocket is not connected.");
    }
    this.ws.send(JSON.stringify(message));
  }

  private handleMessage(raw: string): void {
    let message: JsonObject;
    try {
      message = JSON.parse(raw) as JsonObject;
    } catch {
      return;
    }

    const id = message.id as string | number | undefined;
    const method = typeof message.method === "string" ? message.method : undefined;

    if (id !== undefined && method) {
      this.handleServerRequest(id, method, asObject(message.params));
      return;
    }

    if (id !== undefined) {
      this.handleRpcResponse(id, message);
      return;
    }

    if (method) {
      this.handleNotification(method, asObject(message.params));
    }
  }

  private handleRpcResponse(id: string | number, message: JsonObject): void {
    const pending = this.pending.get(id);
    if (!pending) {
      return;
    }
    clearTimeout(pending.timer);
    this.pending.delete(id);
    if (message.error) {
      pending.reject(new Error(JSON.stringify(message.error)));
      return;
    }
    pending.resolve(message.result);
  }

  private handleServerRequest(id: string | number, method: string, params: JsonObject): void {
    const pendingRequest: PendingServerRequest = {
      id,
      method,
      params,
      receivedAt: new Date().toISOString(),
    };
    this.pendingServerRequests.set(id, pendingRequest);

    const threadId = extractNotificationThreadId(params);
    const turnId = extractNotificationTurnId(params);
    if (threadId && turnId) {
      const turn = this.ensureTurn(threadId, turnId);
      turn.status = "waiting";
      turn.pendingRequests.push(pendingRequest);
      void this.mergeSession(threadId, {
        threadId,
        activeTurnId: turnId,
        lastTurnId: turnId,
        lastStatus: "waiting",
        lastEventAt: pendingRequest.receivedAt,
        lastUsefulMessage: textPreview(params) ?? method,
        turns: {
          [turnId]: withoutUndefined({
            turnId,
            status: "waiting",
            startedAt: turn.startedAt,
            updatedAt: pendingRequest.receivedAt,
            lastUsefulMessage: textPreview(params) ?? method,
            pendingRequestIds: turn.pendingRequests.map((request) => request.id),
            eventCount: turn.events.length,
          }),
        },
      });
    }
  }

  private handleNotification(method: string, params: JsonObject): void {
    const threadId = extractNotificationThreadId(params);
    const turnId = extractNotificationTurnId(params);
    if (threadId && turnId) {
      const turn = this.ensureTurn(threadId, turnId);
      turn.events.push({ method, params, receivedAt: new Date().toISOString() });
      if (turn.events.length > 200) {
        turn.events.shift();
      }
      if (method === "turn/completed") {
        turn.status = "completed";
        turn.finishedAt = new Date().toISOString();
      } else if (method === "turn/failed") {
        turn.status = "failed";
        turn.finishedAt = new Date().toISOString();
      } else if (turn.status !== "waiting" && turn.status !== "completed" && turn.status !== "failed") {
        turn.status = "running";
      }
      const receivedAt = new Date().toISOString();
      const lastUsefulMessage = textPreview(params) ?? method;
      const clearFields: Array<keyof SessionRecord> =
        turn.status === "completed" || turn.status === "failed" ? ["activeTurnId"] : [];
      void this.mergeSession(
        threadId,
        {
          threadId,
          activeTurnId: turn.status === "completed" || turn.status === "failed" ? undefined : turnId,
          lastTurnId: turnId,
          lastStatus: turn.status,
          lastEventAt: receivedAt,
          lastFinishedAt: turn.finishedAt,
          lastUsefulMessage,
          turns: {
            [turnId]: withoutUndefined({
              turnId,
              status: turn.status,
              startedAt: turn.startedAt,
              updatedAt: receivedAt,
              finishedAt: turn.finishedAt,
              lastUsefulMessage,
              pendingRequestIds: turn.pendingRequests.map((request) => request.id),
              eventCount: turn.events.length,
            }),
          },
        },
        clearFields,
      );
    }
  }

  private ensureTurn(threadId: string, turnId: string): TurnState {
    const key = turnKey(threadId, turnId);
    let turn = this.turns.get(key);
    if (!turn) {
      turn = {
        threadId,
        turnId,
        status: "running",
        startedAt: new Date().toISOString(),
        events: [],
        pendingRequests: [],
      };
      this.turns.set(key, turn);
    }
    return turn;
  }

  private removePendingRequestFromTurn(id: string | number): void {
    for (const turn of this.turns.values()) {
      turn.pendingRequests = turn.pendingRequests.filter((request) => request.id !== id);
      if (!turn.pendingRequests.length && turn.status === "waiting") {
        turn.status = "running";
      }
    }
  }

  private rejectPending(error: Error): void {
    for (const [id, pending] of this.pending.entries()) {
      clearTimeout(pending.timer);
      pending.reject(error);
      this.pending.delete(id);
    }
  }

  private async filteredSessions(input: LabelQueryInput): Promise<SessionRecord[]> {
    const state = await this.readState();
    return Object.values(state.sessions)
      .filter((session) => !input.label || session.label === input.label)
      .filter((session) => !input.cwd || session.cwd === input.cwd)
      .filter((session) => !input.group || session.group === input.group)
      .filter((session) => !input.status || this.sessionStatus(session) === input.status)
      .sort(compareSessionsByRecency);
  }

  private async resolveSession(label: string, input: LabelQueryInput): Promise<ResolvedSession> {
    const prefer = input.prefer ?? "latest_active";
    const candidates = await this.filteredSessions({ ...input, label });
    if (!candidates.length) {
      throw new Error(`No Super Agents session found for label ${label}.`);
    }

    const activeCandidates = candidates.filter((session) => isActiveStatus(this.sessionStatus(session)));
    const scopedCandidates = prefer === "latest_any" ? candidates : activeCandidates;
    if (!scopedCandidates.length) {
      throw new Error(
        `No active Super Agents session found for label ${label}. Recent inactive candidates: ${JSON.stringify(
          candidates.slice(0, 5).map((session) => this.sessionView(session)),
        )}`,
      );
    }

    const [first, second] = scopedCandidates;
    if (!first) {
      throw new Error(`No Super Agents session found for label ${label}.`);
    }
    if (second && sessionRecency(first) === sessionRecency(second)) {
      throw new Error(
        `Ambiguous Super Agents label ${label}. Candidates: ${JSON.stringify(
          scopedCandidates.slice(0, 5).map((session) => this.sessionView(session)),
        )}`,
      );
    }
    return {
      session: first,
      turnId: input.turnId ?? first.activeTurnId ?? first.lastTurnId,
      status: this.sessionStatus(first),
    };
  }

  private sessionView(session: SessionRecord): JsonObject {
    const status = this.sessionStatus(session);
    const runningTurnId = isActiveStatus(status) ? session.activeTurnId ?? session.lastTurnId : undefined;
    return withoutUndefined({
      label: session.label,
      group: session.group,
      cwd: session.cwd,
      threadId: session.threadId,
      runningTurnId,
      lastTurnId: session.lastTurnId,
      status,
      ageMs: Date.now() - Date.parse(session.lastStartedAt ?? session.updatedAt),
      updatedAt: session.updatedAt,
      lastUsefulMessage: session.lastUsefulMessage,
      pendingRequestCount: this.pendingRequestCount(session.threadId, runningTurnId),
    });
  }

  private sessionStatus(session: SessionRecord): TrackedStatus | "unknown" {
    const turnId = session.activeTurnId ?? session.lastTurnId;
    const runtimeTurn = turnId ? this.turns.get(turnKey(session.threadId, turnId)) : undefined;
    return runtimeTurn?.status ?? session.lastStatus ?? "unknown";
  }

  private pendingRequestCount(threadId: string, turnId?: string): number {
    return [...this.pendingServerRequests.values()].filter(
      (request) => request.params.threadId === threadId && (!turnId || request.params.turnId === turnId),
    ).length;
  }

  private async recordTurnProgress(
    threadId: string,
    turnId: string,
    status: string,
    persistedTurn: JsonObject | undefined,
    pendingRequests: PendingServerRequest[],
  ): Promise<void> {
    const trackedStatus = status === "unknown" ? "unknown" : toTrackedTurnStatus(status);
    const trackedTurn = this.turns.get(turnKey(threadId, turnId));
    const finishedAt =
      trackedStatus === "completed" || trackedStatus === "failed" || trackedStatus === "cancelled"
        ? trackedTurn?.finishedAt ?? new Date().toISOString()
        : undefined;
    await this.mergeSession(
      threadId,
      {
        threadId,
        activeTurnId: isActiveStatus(trackedStatus) ? turnId : undefined,
        lastTurnId: turnId,
        lastStatus: trackedStatus,
        lastFinishedAt: finishedAt,
        lastUsefulMessage: textPreview(persistedTurn),
        turns: {
          [turnId]: withoutUndefined({
            turnId,
            status: trackedStatus === "unknown" ? "running" : trackedStatus,
            startedAt: trackedTurn?.startedAt ?? new Date().toISOString(),
            updatedAt: new Date().toISOString(),
            finishedAt,
            lastUsefulMessage: textPreview(persistedTurn),
            pendingRequestIds: pendingRequests.map((request) => request.id),
            eventCount: trackedTurn?.events.length ?? 0,
          }),
        },
      },
      isActiveStatus(trackedStatus) ? [] : ["activeTurnId"],
    );
  }

  private async rememberSession(threadId: string, patch: Omit<SessionRecord, "updatedAt">): Promise<void> {
    const state = await this.readState();
    const now = new Date().toISOString();
    state.sessions[threadId] = {
      ...patch,
      threadId,
      createdAt: patch.createdAt ?? now,
      updatedAt: now,
    };
    await this.writeState(state);
  }

  private async mergeSession(
    threadId: string,
    patch: Partial<Omit<SessionRecord, "updatedAt">>,
    clearFields: Array<keyof SessionRecord> = [],
  ): Promise<void> {
    const state = await this.readState();
    const now = new Date().toISOString();
    const current = state.sessions[threadId] ?? { threadId, createdAt: now, updatedAt: now };
    const next = {
      ...current,
      ...withoutUndefined(patch),
      turns: mergeTurns(current.turns, patch.turns),
      threadId,
      createdAt: current.createdAt ?? now,
      updatedAt: now,
    };
    for (const field of clearFields) {
      delete next[field];
    }
    state.sessions[threadId] = {
      ...next,
    };
    await this.writeState(state);
  }

  private async getSession(threadId: string): Promise<SessionRecord | undefined> {
    const state = await this.readState();
    return state.sessions[threadId];
  }

  private async readState(): Promise<StateFile> {
    try {
      const parsed = JSON.parse(await readFile(this.stateFile, "utf8")) as Partial<StateFile>;
      return {
        sessions: asSessionRecordMap(parsed.sessions),
      };
    } catch {
      return { sessions: {} };
    }
  }

  private async writeState(state: StateFile): Promise<void> {
    await mkdir(dirname(this.stateFile), { recursive: true });
    await writeFile(this.stateFile, JSON.stringify(state, null, 2) + "\n");
  }
}

async function loginShellConfigOverride(): Promise<JsonObject> {
  const env = await loginShellEnvironment();
  const set: Record<string, string> = {};
  for (const key of ["PATH", "SHELL", "HOME", "USER", "LOGNAME"]) {
    const value = env[key];
    if (typeof value === "string" && value) {
      set[key] = value;
    }
  }
  return {
    shell_environment_policy: {
      inherit: "all",
      set,
    },
  };
}

async function loginShellEnvironment(): Promise<NodeJS.ProcessEnv> {
  loginShellEnvironmentPromise ??= readLoginShellEnvironment().catch((error) => {
    console.error(`[super-agents] Failed to read login shell environment: ${errorMessage(error)}`);
    return process.env;
  });
  return loginShellEnvironmentPromise;
}

function readLoginShellEnvironment(): Promise<NodeJS.ProcessEnv> {
  const shell = process.env.SHELL && process.env.SHELL.startsWith("/") ? process.env.SHELL : "/bin/zsh";
  return new Promise((resolve, reject) => {
    const child = spawn(shell, ["-lic", "/usr/bin/env -0"], {
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });
    const chunks: Buffer[] = [];
    const stderrChunks: Buffer[] = [];
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      reject(new Error(`Timed out reading login shell environment from ${shell}.`));
    }, LOGIN_ENV_TIMEOUT_MS);

    child.stdout.on("data", (chunk: Buffer) => chunks.push(chunk));
    child.stderr.on("data", (chunk: Buffer) => stderrChunks.push(chunk));
    child.on("error", (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      if (code !== 0) {
        const stderr = Buffer.concat(stderrChunks).toString("utf8").trim();
        reject(new Error(`Login shell exited with code ${code}${stderr ? `: ${stderr}` : ""}`));
        return;
      }
      resolve({ ...process.env, ...parseNullSeparatedEnv(Buffer.concat(chunks).toString("utf8")) });
    });
  });
}

function parseNullSeparatedEnv(output: string): NodeJS.ProcessEnv {
  const env: NodeJS.ProcessEnv = {};
  for (const entry of output.split("\0")) {
    const separator = entry.indexOf("=");
    if (separator <= 0) {
      continue;
    }
    env[entry.slice(0, separator)] = entry.slice(separator + 1);
  }
  return env;
}

function collaborationMode(
  mode: "default" | "plan",
  model: string,
  reasoningEffort?: string,
  developerInstructions?: string | null,
): JsonObject {
  const settings: JsonObject = {
    model,
    developer_instructions: developerInstructions ?? null,
  };
  if (mode === "plan") {
    settings.reasoning_effort = reasoningEffort ?? "medium";
  }
  return { mode, settings };
}

function getString(value: JsonObject, key: string): string | undefined {
  const result = value[key];
  return typeof result === "string" ? result : undefined;
}

function extractModel(value: JsonObject): string | undefined {
  return getString(value, "model") ?? getString(asObject(value.thread), "model");
}

function extractThreadId(value: JsonObject): string | undefined {
  return getString(value, "threadId") ?? getString(value, "id") ?? getString(asObject(value.thread), "id");
}

function extractThreadCwd(value: JsonObject): string | undefined {
  return getString(value, "cwd") ?? getString(asObject(value.thread), "cwd");
}

function extractTurnId(value: JsonObject): string | undefined {
  return getString(value, "turnId") ?? getString(value, "id") ?? getString(asObject(value.turn), "id");
}

function extractNotificationThreadId(value: JsonObject): string | undefined {
  return (
    getString(value, "threadId") ??
    getString(asObject(value.thread), "id") ??
    getString(asObject(value.turn), "threadId") ??
    getString(asObject(value.item), "threadId")
  );
}

function extractNotificationTurnId(value: JsonObject): string | undefined {
  return (
    getString(value, "turnId") ??
    getString(asObject(value.turn), "id") ??
    getString(asObject(value.item), "turnId")
  );
}

function findTurn(value: unknown, turnId: string): JsonObject | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      const result = findTurn(item, turnId);
      if (result) {
        return result;
      }
    }
    return undefined;
  }

  const object = value as JsonObject;
  if (getString(object, "id") === turnId) {
    return object;
  }
  for (const item of Object.values(object)) {
    const result = findTurn(item, turnId);
    if (result) {
      return result;
    }
  }
  return undefined;
}

function normalizeTurnStatus(turn: JsonObject | undefined): string | undefined {
  const status = turn ? getString(turn, "status") : undefined;
  if (!status) {
    return undefined;
  }
  if (status === "inProgress" || status === "active") {
    return "running";
  }
  return status;
}

function toTrackedTurnStatus(status: string): TurnState["status"] {
  if (status === "completed" || status === "failed" || status === "waiting" || status === "cancelled") {
    return status;
  }
  return "running";
}

function isActiveStatus(status: string | undefined): boolean {
  return status === "running" || status === "waiting";
}

function requiredLabel(input: LabelQueryInput): string {
  if (!input.label) {
    throw new Error("label must be a non-empty string.");
  }
  return input.label;
}

function requiredPrompt(input: { prompt?: string | undefined }): string {
  if (!input.prompt) {
    throw new Error("prompt must be a non-empty string.");
  }
  return input.prompt;
}

function compareSessionsByRecency(a: SessionRecord, b: SessionRecord): number {
  return sessionRecency(b) - sessionRecency(a);
}

function sessionRecency(session: SessionRecord): number {
  return Date.parse(session.lastEventAt ?? session.updatedAt) || 0;
}

function mergeTurns(
  current: Record<string, TurnSummary> | undefined,
  patch: Record<string, TurnSummary> | undefined,
): Record<string, TurnSummary> | undefined {
  if (!current && !patch) {
    return undefined;
  }
  const result: Record<string, TurnSummary> = { ...(current ?? {}) };
  for (const [turnId, summary] of Object.entries(patch ?? {})) {
    result[turnId] = {
      ...result[turnId],
      ...summary,
      turnId,
    };
  }
  return result;
}

function asSessionRecordMap(value: unknown): Record<string, SessionRecord> {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {};
  }
  const sessions: Record<string, SessionRecord> = {};
  for (const [threadId, rawSession] of Object.entries(value as Record<string, unknown>)) {
    if (!rawSession || typeof rawSession !== "object" || Array.isArray(rawSession)) {
      continue;
    }
    const session = rawSession as JsonObject;
    const normalizedThreadId = getString(session, "threadId") ?? threadId;
    const updatedAt = getString(session, "updatedAt") ?? new Date(0).toISOString();
    sessions[normalizedThreadId] = withoutUndefined({
      label: getString(session, "label"),
      threadId: normalizedThreadId,
      cwd: getString(session, "cwd"),
      group: getString(session, "group"),
      model: getString(session, "model"),
      lastTurnId: getString(session, "lastTurnId"),
      activeTurnId: getString(session, "activeTurnId"),
      createdAt: getString(session, "createdAt"),
      lastStartedAt: getString(session, "lastStartedAt"),
      lastFinishedAt: getString(session, "lastFinishedAt"),
      lastStatus: asStoredStatus(getString(session, "lastStatus")),
      lastUsefulMessage: getString(session, "lastUsefulMessage"),
      lastEventAt: getString(session, "lastEventAt"),
      turns: asTurnSummaryMap(session.turns),
      updatedAt,
    });
  }
  return sessions;
}

function asTurnSummaryMap(value: unknown): Record<string, TurnSummary> | undefined {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return undefined;
  }
  const turns: Record<string, TurnSummary> = {};
  for (const [turnId, rawTurn] of Object.entries(value as Record<string, unknown>)) {
    if (!rawTurn || typeof rawTurn !== "object" || Array.isArray(rawTurn)) {
      continue;
    }
    const turn = rawTurn as JsonObject;
    const normalizedTurnId = getString(turn, "turnId") ?? turnId;
    const status = asStoredStatus(getString(turn, "status"));
    if (!status || status === "unknown") {
      continue;
    }
    turns[normalizedTurnId] = withoutUndefined({
      turnId: normalizedTurnId,
      status,
      mode: asMode(getString(turn, "mode")),
      startedAt: getString(turn, "startedAt") ?? getString(turn, "updatedAt") ?? new Date(0).toISOString(),
      updatedAt: getString(turn, "updatedAt") ?? new Date(0).toISOString(),
      finishedAt: getString(turn, "finishedAt"),
      promptPreview: getString(turn, "promptPreview"),
      lastUsefulMessage: getString(turn, "lastUsefulMessage"),
      pendingRequestIds: asStringOrNumberArray(turn.pendingRequestIds),
      eventCount: typeof turn.eventCount === "number" ? turn.eventCount : undefined,
    });
  }
  return Object.keys(turns).length ? turns : undefined;
}

function asStoredStatus(status: string | undefined): TrackedStatus | "unknown" | undefined {
  if (status === "running" || status === "waiting" || status === "completed" || status === "failed" || status === "cancelled") {
    return status;
  }
  if (status === "unknown") {
    return "unknown";
  }
  return undefined;
}

function asMode(mode: string | undefined): "default" | "plan" | undefined {
  if (mode === "default" || mode === "plan") {
    return mode;
  }
  return undefined;
}

function asStringOrNumberArray(value: unknown): Array<string | number> | undefined {
  if (!Array.isArray(value)) {
    return undefined;
  }
  return value.filter((item): item is string | number => typeof item === "string" || typeof item === "number");
}

function previewText(value: string, maxLength = 240): string {
  const normalized = value.replace(/\s+/g, " ").trim();
  return normalized.length > maxLength ? `${normalized.slice(0, maxLength - 3)}...` : normalized;
}

function textPreview(value: unknown): string | undefined {
  const text = findUsefulText(value);
  return text ? previewText(text) : undefined;
}

function findUsefulText(value: unknown, depth = 0): string | undefined {
  if (depth > 6 || value === null || value === undefined) {
    return undefined;
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed.length >= 8 ? trimmed : undefined;
  }
  if (Array.isArray(value)) {
    for (const item of value) {
      const result = findUsefulText(item, depth + 1);
      if (result) {
        return result;
      }
    }
    return undefined;
  }
  if (typeof value !== "object") {
    return undefined;
  }
  const object = value as JsonObject;
  for (const key of ["text", "message", "content", "summary", "output", "preview"]) {
    const result = findUsefulText(object[key], depth + 1);
    if (result) {
      return result;
    }
  }
  for (const item of Object.values(object)) {
    const result = findUsefulText(item, depth + 1);
    if (result) {
      return result;
    }
  }
  return undefined;
}

function asObject(value: unknown): JsonObject {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonObject) : {};
}

function withoutUndefined<T extends JsonObject>(value: T): T {
  return Object.fromEntries(Object.entries(value).filter(([, item]) => item !== undefined)) as T;
}

function turnKey(threadId: string, turnId: string): string {
  return `${threadId}:${turnId}`;
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
