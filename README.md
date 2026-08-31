# Super Agents

Super Agents is a Python MCP server and library for controlling local
[Codex app-server](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)
sessions asynchronously.

It gives AI agents a compact tool surface for creating named Codex threads,
starting turns, checking progress, steering active work, cancelling turns,
answering app-server callbacks, and tracking lightweight local session state.

Super Agents is used by Openbase Coder, but it can also be run directly by any
MCP client that needs to coordinate Codex app-server threads without blocking on
long-running turns.

It also supports Claude Code for Super Agents UI-driver sessions. A small
non-MCP command switches Openbase's default backend.

## Install

The recommended install path is `uv tool install`, which installs the
`super-agents-mcp` command in an isolated tool environment:

```bash
uv tool install super-agents
```

Then register `super-agents-mcp` with your MCP client.

`pipx` also works:

```bash
pipx install super-agents
```

For library-only use inside an existing Python environment:

```bash
python -m pip install super-agents
```

## Requirements

- Python 3.11+
- A running local
  [`codex app-server`](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)
- An MCP-compatible client such as Codex, Claude Desktop, or Openbase Coder

On macOS and Linux, Super Agents defaults to Codex's standard Unix control
socket at `$CODEX_HOME/app-server-control/app-server-control.sock`. Windows
retains the loopback WebSocket default until Codex provides an equivalent
local daemon transport there.

## MCP Server

Run the MCP server with:

```bash
super-agents-mcp
```

For local development from a checkout:

```bash
uv sync --extra dev
uv run super-agents-mcp
```

Example MCP server command:

```bash
uvx --from super-agents super-agents-mcp
```

If you are running from a source checkout instead of an installed package:

```bash
uv --directory /path/to/super-agents run super-agents-mcp
```

## Start Codex App Server

Super Agents talks to the
[Codex app-server](https://github.com/openai/codex/blob/main/codex-rs/app-server/README.md)
over WebSocket frames. On Unix, start Codex on its standard Unix control socket
before using the MCP tools:

```bash
codex app-server --listen unix://
```

Openbase Coder users usually do not need to run this by hand; the
`codex-app-server` background service owns that process.

## Backends

Super Agents supports four configured backend identities:

- `codex`: native Codex app-server over a Unix socket or TCP WebSocket.
- `openbase_cloud`: Claude Code sessions through the Openbase Cloud Anthropic proxy.
- `claude_code`: Claude Code sessions using local Claude auth/billing.
- `openbase_cloud_codex`: Codex-compatible sessions routed through an
  Openbase Cloud integration.

The MCP server can mix these identities per thread. Pass `backend` to
`super_agents_start` to override the launch default. If it is omitted,
`SUPER_AGENTS_DEFAULT_BACKEND` wins when set, then `OPENBASE_CODING_BACKEND`.
Explicit and default identities are persisted against returned thread, turn,
and approval-request ids, so follow-up operations keep using the owning
backend after a server restart. If the same name exists on multiple backends,
name-only operations fail with an ambiguity error; pass `backend` or the
returned `threadId`.

`codex` and `openbase_cloud_codex` can coexist only when their identity-specific
websocket variables point to distinct app-server processes. The server rejects
a second Codex identity on the same endpoint instead of pretending it is an
independent backend.

Switch modes without MCP:

```bash
super-agents-backend use codex
super-agents-backend use openbase-cloud
super-agents-backend use claude-code
super-agents-backend status
```

Restart the process that owns Super Agents after switching. For `codex`, restart
`codex-app-server`; for Openbase Cloud and Claude Code, restart the MCP host
running `super-agents-mcp`.

## Claude Code Backend

The Claude Code backend uses the `claude-agent-sdk` package directly. It does
not run a local Anthropic Messages API adapter, does not expose `/v1/responses`,
and does not require Codex app-server. Direct `claude_code` billing/auth comes
from the local Claude setup on the computer. The `openbase_cloud` backend uses
the same Claude Code execution path, but points Claude Code at Openbase's
Anthropic proxy with `ANTHROPIC_BASE_URL` and an Openbase machine token in
`ANTHROPIC_AUTH_TOKEN`.

Set `SUPER_AGENTS_CLAUDE_EXTRA_ARGS` to a JSON object to pass extra Claude
Code CLI flags to every session, e.g. `{"chrome": null}` to enable the Claude
in Chrome browser tools (`null` means a bare flag; string values are passed as
the flag's argument).

```bash
uv tool install 'super-agents[claude]'
super-agents-backend use claude-code
```

or, from a source checkout:

```bash
uv sync --extra dev --extra claude
uv run super-agents-backend use claude-code
```

Set `OPENBASE_CODING_BACKEND=claude_code` to make `super-agents-mcp` use
this backend. `OPENBASE_CODEX_BACKEND` is still read as a legacy fallback.
Follow-up turns preserve live SDK conversation context while the MCP process
remains running; persisted metadata and logs survive restarts.

On this backend, `start_thread` reuses an existing session with the same
`name` (refreshing its cwd, instructions, and model). Library callers that
need a brand-new session and conversation under an existing name can pass
`"fresh": true`, which retires the old session by renaming it aside before
creating the new one.

If the SDK package is not installed, the backend reports `ready=false` with an
install hint.

## Add To Codex

Install the package, then register the MCP server:

```bash
uv tool install super-agents
codex mcp add super-agents -- super-agents-mcp
```

If your Codex app-server is listening somewhere other than the standard Unix
socket, pass the neutral endpoint setting when registering the server:

```bash
codex mcp add \
  --env SUPER_AGENTS_APP_SERVER_ENDPOINT=unix:///var/run/codex.sock \
  super-agents -- super-agents-mcp
```

`SUPER_AGENTS_WS_URL` and `CODEX_APP_SERVER_URL` remain compatibility aliases
for existing TCP WebSocket deployments. Conflicting values are rejected so two
variables cannot silently select different app-server owners.

Check that Codex can see the server:

```bash
codex mcp list
codex mcp get super-agents
```

## Add To Claude Code

Install the package, then register the MCP server:

```bash
uv tool install super-agents
claude mcp add --scope user super-agents -- super-agents-mcp
```

For a non-default Codex app-server endpoint:

```bash
claude mcp add \
  --scope user \
  -e SUPER_AGENTS_APP_SERVER_ENDPOINT=ws://127.0.0.1:4500 \
  super-agents -- super-agents-mcp
```

Check that Claude Code can see the server:

```bash
claude mcp list
claude mcp get super-agents
```

For project-local installs, use `--scope project` instead of `--scope user`.

## Configuration

Super Agents is configured with environment variables and, when running under
Openbase, `~/.openbase/dispatcher-config.json`.

| Variable | Default | Description |
| --- | --- | --- |
| `SUPER_AGENTS_APP_SERVER_ENDPOINT` | `unix://` on Unix; `ws://127.0.0.1:4500` on Windows | Codex app-server endpoint (`unix://`, `unix://PATH`, `ws://`, or `wss://`) |
| `SUPER_AGENTS_WS_URL` | unset | Deprecated compatibility alias for TCP WebSocket deployments |
| `CODEX_APP_SERVER_URL` | unset | Openbase compatibility alias; must not conflict with another endpoint selector |
| `SUPER_AGENTS_CODEX_APP_SERVER_ENDPOINT` | unset | Identity-specific Codex endpoint; required to differ when local and Cloud Codex identities coexist |
| `SUPER_AGENTS_OPENBASE_CLOUD_CODEX_APP_SERVER_ENDPOINT` | unset | Identity-specific Cloud Codex endpoint; required to differ when local and Cloud Codex identities coexist |
| `OPENBASE_CODING_BACKEND` | unset | Machine backend identity: `codex`, `openbase_cloud`, `claude_code`, or `openbase_cloud_codex` |
| `SUPER_AGENTS_DEFAULT_BACKEND` | unset | Per-process default for newly launched threads; falls back to `OPENBASE_CODING_BACKEND` |
| `OPENBASE_CODEX_BACKEND` | unset | Legacy fallback for `OPENBASE_CODING_BACKEND` |
| `SUPER_AGENTS_STATE_FILE` | `~/.super-agents/state.json` | Local session metadata file |
| `SUPER_AGENTS_BACKEND_PROVENANCE_FILE` | `~/.super-agents/backend-provenance.json` | Configured-backend ownership for thread, turn, and approval ids |
| `SUPER_AGENTS_CLAUDE_PERMISSION_MODE` | `bypassPermissions` | Claude SDK permission mode. Gated modes route tool requests through the native approval store. |

Openbase-specific defaults:

| Config key | Description |
| --- | --- |
| `super_agents_reasoning_effort` | Default reasoning effort for Super Agents turns |
| `backend_models` | Model defaults keyed by backend and role |
| `SUPER_AGENTS_QUEUE_DIR` | next to the state file | Directory for queued turn files |

Super Agents does not silently approve app-server callbacks. If plan mode asks a
question or a sandboxed turn asks for approval, inspect the pending request and
answer it explicitly with `codex_answer_request`.

The same flow applies to Claude Code when `SUPER_AGENTS_CLAUDE_PERMISSION_MODE`
is a gated SDK mode such as `default` or `acceptEdits`. Pending tool requests
appear in status and can be answered with `codex_answer_request`. Approval waits
are bounded and fail closed; persisted details are redacted, and stale requests
from exited processes are discarded rather than reused after restart.

## Tools

The MCP server exposes these tools:

- `codex_app_server_status`: check app-server readiness, websocket connection,
  pending callbacks, and active turns.
- `super_agents_start`: create a named thread, optionally on an explicit backend.
- `super_agents_resume`: resume a named thread.
- `super_agents_read`: read a named or id-addressed thread.
- `super_agents_rename`: rename a Super Agents thread.
- `codex_answer_request`: answer a pending callback on its owning backend.
- `super_agents_sessions`: list named threads across engaged backends.
- `super_agents_thread_favorite`: check whether one local Openbase Coder thread
  is favorited.
- `super_agents_tags`: list local Openbase Coder tag options shared by threads
  and reports.
- `super_agents_thread_tags`: read or replace local tags for one thread.
- `super_agents_report_tags`: read or replace local tags for one report file.
- `super_agents_active`: list active tracked agents with compact previews.
- `super_agents_status`: return a compact status list for voice/status checks.
- `super_agents_resolve`: resolve a name to the latest matching active thread.
- `super_agents_progress`: inspect progress by name, thread id, or turn id.
- `super_agents_steer`: send steering input to an active turn.
- `super_agents_cancel`: cancel an active turn.
- `super_agents_start_turn`: submit follow-up input through app-server
  `turn/start`.
- `super_agents_queue_turn`: queue a follow-up prompt to run after the active
  turn completes.
- `super_agents_cancel_queued_turn`: remove a queued follow-up prompt before
  it starts.
- `super_agents_recent`: list recent named Codex app-server threads.

List-style tools accept `favorite=true` or `favorite=false` to filter by
Openbase Coder's local per-machine favorite metadata.

Default responses are intentionally compact. Full turns, tracked event
transcripts, diffs, and large previews are opt-in through each tool's detail
flags.

### Cancelling Queued Turns

Queued follow-up prompts are visible in `codex_app_server_status` under
`queuedTurns`, and `super_agents_queue_turn` returns the queued item under
`item`. To remove a queued item before it starts, call
`super_agents_cancel_queued_turn` with the queued item id:

```json
{
  "queueItemId": "q_abc123"
}
```

The Claude Code backend exposes queued turns as turn ids, so `turnId` is also
accepted as an alias:

```json
{
  "turnId": "t_abc123"
}
```

If the id is not available, cancel by target thread and 1-based queue position:

```json
{
  "name": "agent-name",
  "position": 1
}
```

Use `threadId` instead of `name` when resolving by id is clearer. This tool only
removes queued items whose status is still `queued`; active or already-started
turns are rejected. Use `super_agents_cancel` for active turns.

## Openbase Coder Report Tags

Openbase Coder stores thread and report tags in the same local tag registry.
When an agent writes a report under a project's `.reports` directory, it can
tag that report by calling `super_agents_report_tags` with:

- `projectPath`: the project directory that contains `.reports`.
- `path`: the report path relative to `.reports`, such as `summary.md` or
  `audits/security.md`.
- `tags`: the complete list of tag labels that should be assigned.

For example, after creating `/workspace/app/.reports/audit.md`, call
`super_agents_report_tags` with `projectPath=/workspace/app`,
`path=audit.md`, and `tags=["Needs Review", "Security"]`. Omitting `tags`
reads the current assignment. New tag labels are added to the shared options
and can then be reused by thread and report tag pickers in Openbase Coder.

## Python API

Openbase Coder and other Python applications can use the app-server client
directly:

```python
from super_agents.app_server_client import CodexAppServerClient

client = CodexAppServerClient()
try:
    status = await client.status()
finally:
    await client.close()
```

The public modules are organized so applications can reuse the websocket client,
session metadata helpers, routine state, and queue handling without starting the
stdio MCP server.

## Development

From this repository:

```bash
uv sync --extra dev
uv run pytest
uv run ruff check .
uv build
```

## License

Super Agents is licensed under the [MIT License](LICENSE).
