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

By default, Super Agents connects to Codex at `ws://127.0.0.1:4500`.

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
over a websocket. Start Codex with a local websocket listener before using the
MCP tools:

```bash
codex app-server --listen ws://127.0.0.1:4500
```

Openbase Coder users usually do not need to run this by hand; the
`codex-app-server` background service owns that process.

## Backends

Super Agents supports three backend modes:

- `codex`: native Codex app-server over websocket.
- `openbase_cloud`: Codex-compatible sessions through Openbase Cloud.
- `claude_code`: Claude Code sessions using local Claude auth/billing.

The configured mode is only the default. Each `super_agents_start` call may
pass a `backend` parameter to launch that thread on a different backend, and
threads on different backends run side by side; name- and id-addressed tools
find the owning backend automatically (pass `backend` to pin a lookup when
the same name exists on more than one).

A host can also pin the launch default for one `super-agents-mcp` process
without changing the machine-wide configuration: set
`SUPER_AGENTS_DEFAULT_BACKEND` in that process's environment (for example in
the MCP server registration's `env`). Explicit per-launch `backend`
parameters still win. This lets each MCP host default new threads to its own
backend type while other hosts on the same machine default differently.

Switch the default mode without MCP:

```bash
super-agents-backend use codex
super-agents-backend use openbase-cloud
super-agents-backend use claude-code
super-agents-backend status
```

The aliases `claude` and `claude-sdk` also select `claude_code`. Restart the
process that owns Super Agents after switching. For `codex` and
`openbase_cloud`, restart `codex-app-server`; for Claude Code, restart the MCP
host running `super-agents-mcp`.

## Claude Code Backend

The Claude Code backend uses the `claude-agent-sdk` package directly. It
does not run a local Anthropic Messages API adapter, does not expose
`/v1/responses`, does not require Codex app-server, and does not support
`ANTHROPIC_API_KEY`. Billing/auth comes from the local Claude setup on the
computer.

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

If the SDK package is not installed, the backend reports `ready=false` with an
install hint.

## Add To Codex

Install the package, then register the MCP server:

```bash
uv tool install super-agents
codex mcp add super-agents -- super-agents-mcp
```

If your Codex app-server is listening somewhere other than the default
`ws://127.0.0.1:4500`, pass `SUPER_AGENTS_WS_URL` when registering the server:

```bash
codex mcp add \
  --env SUPER_AGENTS_WS_URL=ws://127.0.0.1:4500 \
  super-agents -- super-agents-mcp
```

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

For a non-default Codex app-server websocket URL:

```bash
claude mcp add \
  --scope user \
  -e SUPER_AGENTS_WS_URL=ws://127.0.0.1:4500 \
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
| `SUPER_AGENTS_WS_URL` | `ws://127.0.0.1:4500` | Codex app-server websocket URL |
| `OPENBASE_CODING_BACKEND` | unset | Backend mode: `codex`, `openbase_cloud`, or `claude_code` |
| `OPENBASE_CODEX_BACKEND` | unset | Legacy fallback for `OPENBASE_CODING_BACKEND` |
| `SUPER_AGENTS_DEFAULT_BACKEND` | unset | Per-process launch-default override; per-launch `backend` parameters still win |
| `SUPER_AGENTS_STATE_FILE` | `~/.super-agents/state.json` | Local session metadata file |

Openbase-specific defaults:

| Config key | Description |
| --- | --- |
| `super_agents_reasoning_effort` | Default reasoning effort for Super Agents turns |
| `backend_models` | Model defaults keyed by backend and role |
| `SUPER_AGENTS_QUEUE_DIR` | next to the state file | Directory for queued turn files |

Super Agents does not silently approve app-server callbacks. If plan mode asks a
question or a sandboxed turn asks for approval, inspect the pending request and
answer it explicitly with `codex_answer_request`.

The Claude Code backend supports the same approval flow. By default Claude
sessions run with the `bypassPermissions` permission mode; set
`SUPER_AGENTS_CLAUDE_PERMISSION_MODE` to a gated mode (for example
`acceptEdits` or `default`) and the SDK's permission prompts are recorded as
`claudeCode/requestApproval` requests in the shared open-approvals queue,
reported by `codex_app_server_status`, and answerable with
`codex_answer_request` or any open-approvals approver surface.

Permission posture is also a per-launch decision: `super_agents_start` and
the turn tools accept `approvalPolicy`/`sandboxPolicy` (Codex) and
`permissionMode` (Claude Code) overrides, so one thread can run fully
sandboxed and gated while another runs with full access. A Claude thread
started with a gated `permissionMode` keeps that mode for later turns unless
a turn passes an explicit override.

A supervising process can also impose a machine-wide **permission guard**: a
JSON file (default `~/.super-agents/permission-guard.json`, overridable via
`SUPER_AGENTS_PERMISSION_GUARD_FILE`) with `{"restricted": true}`. While
restricted, launches through the MCP server cannot use permission bypasses —
never-ask approval policies, full-access sandboxes, and `bypassPermissions`
are downgraded to gated equivalents (explicitly stricter values are kept).
The supervisor controls the file, so a prompt-injected agent cannot talk its
way into a bypass. See `permission_guard.py` for the file shape and
per-backend override fields.

## Tools

The MCP server exposes these tools:

- `codex_app_server_status`: check app-server readiness, websocket connection,
  pending callbacks, and active turns.
- `super_agents_start`: create a named thread on any backend (optional
  `backend` parameter; defaults to the process's default backend).
- `super_agents_resume`: resume a named thread.
- `super_agents_read`: read a named or id-addressed thread.
- `super_agents_rename`: rename a Codex app-server thread.
- `codex_answer_request`: answer a pending app-server callback.
- `super_agents_sessions`: list named Codex app-server threads.
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
- `super_agents_recent`: list recent named Codex app-server threads.

List-style tools accept `favorite=true` or `favorite=false` to filter by
Openbase Coder's local per-machine favorite metadata.

Default responses are intentionally compact. Full turns, tracked event
transcripts, diffs, and large previews are opt-in through each tool's detail
flags.

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
session metadata helpers, and queue handling without starting the stdio MCP
server. Product schedulers, recurring jobs, and command automation belong in
the integrating application, not in Super Agents.

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
