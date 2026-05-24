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
cd legacy
npm install
npm run build
```

Then add an MCP server entry that runs:

```bash
node /path/to/super-agents/legacy/dist/index.js
```

The server starts `codex app-server` automatically when it cannot reach the configured websocket endpoint.

## Configuration

- `SUPER_AGENTS_WS_URL`: Codex app-server websocket URL. Defaults to `ws://127.0.0.1:4500`.
- `SUPER_AGENTS_MODEL`: fallback model for plan/default collaboration settings. Defaults to `gpt-5.4`.
- `SUPER_AGENTS_STATE_FILE`: optional JSON state file for local turn metadata.

Super Agents MCP tools do not accept or set approval or sandbox options on app-server thread/turn requests. Codex uses the defaults from the configured Codex home.

## Tools

- `codex_app_server_status`: check local app-server readiness and websocket connection with compact active-turn summaries.
- `super_agents_start`: create a named Codex app-server thread.
- `super_agents_resume`: resume a named Codex app-server thread.
- `super_agents_read`: read a named or `threadId` thread. `includeTurns` defaults to false; pass `includeTurns=true` for full turns.
- `super_agents_rename`: rename a Codex app-server thread using its current name.
- `codex_answer_request`: answer a pending app-server callback such as plan-mode user input or command approval.
- `super_agents_sessions`: list named Codex app-server threads.
- `super_agents_active`: list active tracked agents by name, cwd, status, age, and a short preview. `previewLength` defaults to 160; pass `includePreview=false` to omit previews.
- `super_agents_status`: compact voice-friendly status list with name, thread id, turn id, status, update times, pending request count, cwd, and stale indicators. It never includes transcripts, diffs, or previews.
- `super_agents_resolve`: resolve a name to the latest active matching thread and turn by default.
- `super_agents_progress`: check progress by `name` or `threadId`/`turnId`. Defaults to status and summary only; pass `full=true` for raw turn/tracked-turn output, or `includeTurn=true` / `includeItems=true` for bounded detail.
- `super_agents_steer`: send steering input to the latest active turn matching a name.
- `super_agents_cancel`: cancel the latest active turn matching a name.
- `super_agents_start_turn`: submit follow-up input to the latest matching named thread with app-server `turn/start`.
- `super_agents_queue_turn`: queue a follow-up prompt in Super Agents memory so it starts as a separate turn after the target thread's active turn finishes.
- `super_agents_recent`: list recent named Codex app-server threads.

Codex app-server 0.133.0 does not expose a separate queued-next-turn method. Its protocol includes `turn/start`, `turn/steer`, and `turn/interrupt`; `turn/start` is still the native submission path and the app-server decides whether it starts a new turn or is accepted as pending input for the current active turn. `super_agents_queue_turn` implements CLI-style client-side queueing in memory and drains queued prompts from app-server completion notifications.

Codex app-server also does not expose native routine storage, cron-like scheduled prompts, or persisted scheduled task execution. Super Agents routines are stored in the local Super Agents state file and managed through the Openbase Coder CLI and console, not the MCP tool surface. Routine execution ultimately calls the same app-server `turn/start` path as other Super Agents turns.

Routine fields include `name`, `prompt`, daily `time` in `HH:MM` format, `timezone` (default `America/New_York`), `enabled`, `targetName` or `threadId`, `cwd`, `mode`, `model`, `reasoningEffort`, `serviceTier`, and optional `developerInstructions`. If a routine has no existing target thread, Super Agents starts a thread named after the routine and then starts the turn.

Openbase Coder exposes routines through commands such as:

```bash
openbase-coder routines list
openbase-coder routines create daily --prompt "Inspect project health." --time 09:00
openbase-coder routines update daily --disable
openbase-coder routines run-due
openbase-coder routines run-due --name daily --force
openbase-coder routines delete daily
```

## Name Tracking

Super Agents uses Codex app-server's native thread names. The public MCP tools accept names and selected inspection tools also accept raw `threadId` / `turnId` so unnamed threads can be inspected directly.

Names are human handles, not guaranteed unique aliases. If a name is reused, name-first operations prefer the most recent matching thread. Use optional `cwd` filters when a name intentionally has several matching sessions.

The local state file remains keyed by thread id and is backward compatible with older files. New runs add local turn metadata such as active turn id, last status, timestamps, turn summaries, and a short useful preview.

## Notes

This wrapper does not silently approve app-server callbacks. If plan mode asks a question or a sandboxed turn asks for approval, use `codex_app_server_status` or `super_agents_progress` to see the pending request and then call `codex_answer_request` explicitly.

Verbose data is opt-in. Default tool responses avoid full turns, tracked event transcripts, diffs, and large previews. Use `fields`, `maxItems`, and `maxOutputChars` on compact tools when a status check needs an even smaller response.

## Python Port Transition

The Python implementation is now available in parallel with the legacy TypeScript implementation. It uses the official low-level Python MCP SDK for stdio MCP handling, while preserving the existing websocket transport to `codex app-server`, callback handling, name/session behavior, and the `~/.super-agents/state.json` format.

Transition phases:

1. Keep both implementations available and run both test suites.
2. Point MCP clients at the uv command after parity is verified in local use.
3. Remove the TypeScript package flow only after the Python entrypoint has replaced it in all active MCP configs.

Python modules are split so Openbase Coder code can later import the app-server websocket client and session metadata/state helpers directly, without depending on MCP stdio startup.
