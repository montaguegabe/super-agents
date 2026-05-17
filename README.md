# Super Agents MCP

Super Agents is a local MCP server that wraps the Codex app server. It gives agents a smaller asynchronous tool surface for creating Codex threads, spawning turns, checking progress, steering active turns, cancelling turns, and answering plan-mode questions.

## Install

Python/uv entrypoint:

```bash
uv sync --extra dev
uv run super-agents-mcp
```

Add an MCP server entry that runs:

```bash
uv --directory /path/to/super-agents run super-agents-mcp
```

Legacy TypeScript entrypoint, retained during the parallel transition:

```bash
npm install
npm run build
```

Then add an MCP server entry that runs:

```bash
node /path/to/super-agents/dist/index.js
```

The server starts `codex app-server` automatically when it cannot reach the configured websocket endpoint.

## Configuration

- `SUPER_AGENTS_WS_URL`: Codex app-server websocket URL. Defaults to `ws://127.0.0.1:4500`.
- `SUPER_AGENTS_MODEL`: fallback model for plan/default collaboration settings. Defaults to `gpt-5.4`.
- `SUPER_AGENTS_STATE_FILE`: optional JSON state file for remembered thread labels.

## Tools

- `codex_app_server_status`: check local app-server readiness and websocket connection.
- `codex_thread_start`: create a Codex thread and optionally label or group it.
- `codex_thread_resume`: resume a persisted Codex thread.
- `codex_thread_list`: list Codex threads from the app server.
- `codex_thread_read`: read a thread, optionally including turns.
- `codex_turn_start`: start a normal or plan-mode turn and return immediately.
- `codex_turn_progress`: check a turn's current state without waiting.
- `codex_turn_steer`: steer an active running turn.
- `codex_turn_cancel`: interrupt a running turn.
- `codex_answer_request`: answer a pending app-server callback such as plan-mode user input or command approval.
- `super_agents_sessions`: list the wrapper's remembered thread labels.
- `super_agents_active`: list active tracked agents only, including label, cwd, thread id, running turn id, status, age, and preview.
- `super_agents_resolve`: resolve a label to the latest active matching thread and turn by default.
- `super_agents_progress`: check progress for the latest active turn matching a label.
- `super_agents_steer`: send steering input to the latest active turn matching a label.
- `super_agents_cancel`: cancel the latest active turn matching a label.
- `super_agents_start_turn`: start a follow-up turn on the latest matching thread for a label.
- `super_agents_recent`: list recent tracked agents by label, cwd, group, status, and activity.

## Label Tracking

Labels are stable human handles, not unique aliases. If a label is reused, label-first operations prefer the most recently active matching session. Use optional `cwd` or `group` filters when a label intentionally has several active agents.

The state file remains keyed by thread id and is backward compatible with older files. New runs add metadata such as group, active turn id, last status, timestamps, turn summaries, and a short useful preview.

## Notes

This wrapper does not silently approve app-server callbacks. If plan mode asks a question or a sandboxed turn asks for approval, use `codex_turn_progress` to see the pending request and then call `codex_answer_request` explicitly.

## Python Port Transition

The Python implementation is now available in parallel with the existing TypeScript implementation. It uses the official low-level Python MCP SDK for stdio MCP handling, while preserving the existing websocket transport to `codex app-server`, tool names, tool schemas, callback handling, label/session behavior, and the `~/.super-agents/state.json` format.

Transition phases:

1. Keep both implementations available and run both test suites.
2. Point MCP clients at the uv command after parity is verified in local use.
3. Remove the TypeScript package flow only after the Python entrypoint has replaced it in all active MCP configs.

Python modules are split so Openbase Coder code can later import the app-server websocket client and session metadata/state helpers directly, without depending on MCP stdio startup.
