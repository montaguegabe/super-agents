import { createServer } from "node:http";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import assert from "node:assert/strict";
import { WebSocketServer } from "ws";

import { CodexAppServerClient } from "../dist/appServerClient.js";

test("steerTurn sends expectedTurnId to the app server", async () => {
  const captured = [];
  const { close, wsUrl } = await startFakeAppServer(captured);
  const stateDir = await mkdtemp(join(tmpdir(), "super-agents-test-"));

  try {
    const client = new CodexAppServerClient(wsUrl, join(stateDir, "state.json"), "gpt-test");

    await client.steerTurn("thread-1", "turn-1", "narrow the scope");

    const steerRequest = captured.find((message) => message.method === "turn/steer");
    assert.ok(steerRequest);
    assert.deepEqual(steerRequest.params, {
      threadId: "thread-1",
      expectedTurnId: "turn-1",
      input: [{ type: "text", text: "narrow the scope" }],
    });
    assert.equal(Object.hasOwn(steerRequest.params, "turnId"), false);
  } finally {
    await close();
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("label tools resolve latest active session and list active agents", async () => {
  const captured = [];
  let threadIndex = 0;
  let turnIndex = 0;
  const { close, wsUrl } = await startFakeAppServer(captured, (message) => {
    if (message.method === "thread/start") {
      threadIndex += 1;
      return { threadId: `thread-${threadIndex}`, cwd: message.params.cwd, model: "gpt-test" };
    }
    if (message.method === "turn/start") {
      turnIndex += 1;
      return { turnId: `turn-${turnIndex}`, text: `started ${turnIndex}` };
    }
    return { ok: true };
  });
  const stateDir = await mkdtemp(join(tmpdir(), "super-agents-test-"));

  try {
    const client = new CodexAppServerClient(wsUrl, join(stateDir, "state.json"), "gpt-test");

    await client.startThread({ label: "build", cwd: "/tmp/one" });
    await client.startTurn({ threadId: "thread-1", label: "build", prompt: "first" });
    await sleep(5);
    await client.startThread({ label: "build", cwd: "/tmp/two" });
    await client.startTurn({ threadId: "thread-2", label: "build", prompt: "second" });

    const resolved = await client.resolveLabel({ label: "build" });
    assert.equal(resolved.threadId, "thread-2");
    assert.equal(resolved.turnId, "turn-2");
    assert.equal(resolved.status, "running");

    const active = await client.active({ label: "build" });
    assert.equal(active.count, 2);
    assert.equal(active.agents[0].threadId, "thread-2");
    assert.equal(active.agents[0].runningTurnId, "turn-2");
  } finally {
    await close();
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("progress by label reuses resolved thread and turn ids", async () => {
  const captured = [];
  const { close, wsUrl } = await startFakeAppServer(captured, (message) => {
    if (message.method === "thread/start") {
      return { threadId: "thread-progress", cwd: message.params.cwd, model: "gpt-test" };
    }
    if (message.method === "turn/start") {
      return { turnId: "turn-progress" };
    }
    if (message.method === "thread/read") {
      return {
        threadId: message.params.threadId,
        turns: [{ id: "turn-progress", status: "inProgress", message: "still working" }],
      };
    }
    return { ok: true };
  });
  const stateDir = await mkdtemp(join(tmpdir(), "super-agents-test-"));

  try {
    const client = new CodexAppServerClient(wsUrl, join(stateDir, "state.json"), "gpt-test");

    await client.startThread({ label: "progress" });
    await client.startTurn({ threadId: "thread-progress", label: "progress", prompt: "work" });
    const progress = await client.progressByLabel({ label: "progress" });

    assert.equal(progress.threadId, "thread-progress");
    assert.equal(progress.turnId, "turn-progress");
    assert.equal(progress.status, "running");
    assert.ok(captured.find((message) => message.method === "thread/read" && message.params.threadId === "thread-progress"));
  } finally {
    await close();
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("steer and cancel by label call existing id-based app-server methods", async () => {
  const captured = [];
  const { close, wsUrl } = await startFakeAppServer(captured, (message) => {
    if (message.method === "thread/start") {
      return { threadId: "thread-control", cwd: message.params.cwd, model: "gpt-test" };
    }
    if (message.method === "turn/start") {
      return { turnId: "turn-control" };
    }
    return { ok: true };
  });
  const stateDir = await mkdtemp(join(tmpdir(), "super-agents-test-"));

  try {
    const client = new CodexAppServerClient(wsUrl, join(stateDir, "state.json"), "gpt-test");

    await client.startThread({ label: "control" });
    await client.startTurn({ threadId: "thread-control", label: "control", prompt: "work" });
    await client.steerByLabel({ label: "control", prompt: "adjust" });
    await client.cancelByLabel({ label: "control" });

    const steerRequest = captured.find((message) => message.method === "turn/steer");
    assert.deepEqual(steerRequest.params, {
      threadId: "thread-control",
      expectedTurnId: "turn-control",
      input: [{ type: "text", text: "adjust" }],
    });

    const cancelRequest = captured.find((message) => message.method === "turn/interrupt");
    assert.deepEqual(cancelRequest.params, {
      threadId: "thread-control",
      turnId: "turn-control",
    });

    const active = await client.active({ label: "control" });
    assert.equal(active.count, 0);
  } finally {
    await close();
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("startTurnByLabel can start a follow-up on latest inactive matching label", async () => {
  const captured = [];
  const { close, wsUrl } = await startFakeAppServer(captured, (message) => {
    if (message.method === "thread/start") {
      return { threadId: "thread-follow-up", cwd: message.params.cwd, model: "gpt-test" };
    }
    if (message.method === "turn/start") {
      return { turnId: "turn-follow-up" };
    }
    return { ok: true };
  });
  const stateDir = await mkdtemp(join(tmpdir(), "super-agents-test-"));

  try {
    const client = new CodexAppServerClient(wsUrl, join(stateDir, "state.json"), "gpt-test");

    await client.startThread({ label: "follow-up", cwd: "/tmp/project" });
    const result = await client.startTurnByLabel({ label: "follow-up", prompt: "continue" });

    assert.equal(result.threadId, "thread-follow-up");
    assert.equal(result.turnId, "turn-follow-up");
    const startRequest = captured.find((message) => message.method === "turn/start");
    assert.equal(startRequest.params.threadId, "thread-follow-up");
    assert.equal(startRequest.params.cwd, "/tmp/project");
  } finally {
    await close();
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("old state files are tolerated by recent and label resolution", async () => {
  const stateDir = await mkdtemp(join(tmpdir(), "super-agents-test-"));
  const stateFile = join(stateDir, "state.json");

  try {
    await writeFile(
      stateFile,
      JSON.stringify({
        sessions: {
          "thread-old": {
            label: "old",
            threadId: "thread-old",
            cwd: "/tmp/old",
            model: "gpt-test",
            lastTurnId: "turn-old",
            updatedAt: "2026-01-01T00:00:00.000Z",
          },
        },
      }),
    );
    const client = new CodexAppServerClient("ws://127.0.0.1:1", stateFile, "gpt-test");

    const recent = await client.recent({ label: "old", includeInactive: true });
    assert.equal(recent.count, 1);
    assert.equal(recent.agents[0].threadId, "thread-old");

    const resolved = await client.resolveLabel({ label: "old", prefer: "latest_any" });
    assert.equal(resolved.threadId, "thread-old");
    assert.equal(resolved.turnId, "turn-old");
    assert.equal(resolved.status, "unknown");
  } finally {
    await rm(stateDir, { recursive: true, force: true });
  }
});

test("session state is enriched with turn metadata compatibly", async () => {
  const captured = [];
  const { close, wsUrl } = await startFakeAppServer(captured, (message) => {
    if (message.method === "thread/start") {
      return { threadId: "thread-state", cwd: message.params.cwd, model: "gpt-test" };
    }
    if (message.method === "turn/start") {
      return { turnId: "turn-state", text: "started state work" };
    }
    return { ok: true };
  });
  const stateDir = await mkdtemp(join(tmpdir(), "super-agents-test-"));
  const stateFile = join(stateDir, "state.json");

  try {
    const client = new CodexAppServerClient(wsUrl, stateFile, "gpt-test");

    await client.startThread({ label: "state", group: "batch", cwd: "/tmp/state" });
    await client.startTurn({ threadId: "thread-state", label: "state", group: "batch", prompt: "state prompt" });

    const state = JSON.parse(await readFile(stateFile, "utf8"));
    assert.equal(state.sessions["thread-state"].label, "state");
    assert.equal(state.sessions["thread-state"].group, "batch");
    assert.equal(state.sessions["thread-state"].activeTurnId, "turn-state");
    assert.equal(state.sessions["thread-state"].turns["turn-state"].promptPreview, "state prompt");
  } finally {
    await close();
    await rm(stateDir, { recursive: true, force: true });
  }
});

async function startFakeAppServer(captured, handler = () => ({ ok: true })) {
  const httpServer = createServer((request, response) => {
    if (request.url === "/readyz") {
      response.writeHead(200);
      response.end("ok");
      return;
    }
    response.writeHead(404);
    response.end();
  });
  const wsServer = new WebSocketServer({ server: httpServer });
  const sockets = new Set();

  wsServer.on("connection", (socket) => {
    sockets.add(socket);
    socket.on("close", () => sockets.delete(socket));
    socket.on("message", (data) => {
      const message = JSON.parse(data.toString());
      captured.push(message);
      socket.send(JSON.stringify({ id: message.id, result: handler(message, socket) }));
    });
  });

  await new Promise((resolve) => httpServer.listen(0, "127.0.0.1", resolve));
  const address = httpServer.address();
  assert.ok(address && typeof address === "object");

  return {
    wsUrl: `ws://127.0.0.1:${address.port}`,
    close: () =>
      new Promise((resolve, reject) => {
        for (const socket of sockets) {
          socket.terminate();
        }
        wsServer.close((wsError) => {
          httpServer.close((httpError) => {
            const error = wsError ?? httpError;
            if (error) {
              reject(error);
            } else {
              resolve();
            }
          });
        });
      }),
  };
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
