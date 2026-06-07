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

Super Agents is configured with environment variables:

| Variable | Default | Description |
| --- | --- | --- |
| `SUPER_AGENTS_WS_URL` | `ws://127.0.0.1:4500` | Codex app-server websocket URL |
| `SUPER_AGENTS_MODEL` | `gpt-5.4` | Default model for new plan/default turns |
| `SUPER_AGENTS_STATE_FILE` | `~/.super-agents/state.json` | Local session metadata file |
| `SUPER_AGENTS_QUEUE_DIR` | next to the state file | Directory for queued turn files |

Super Agents does not silently approve app-server callbacks. If plan mode asks a
question or a sandboxed turn asks for approval, inspect the pending request and
answer it explicitly with `codex_answer_request`.

## Tools

The MCP server exposes these tools:

- `codex_app_server_status`: check app-server readiness, websocket connection,
  pending callbacks, and active turns.
- `super_agents_start`: create a named Codex app-server thread.
- `super_agents_resume`: resume a named thread.
- `super_agents_read`: read a named or id-addressed thread.
- `super_agents_rename`: rename a Codex app-server thread.
- `codex_answer_request`: answer a pending app-server callback.
- `super_agents_sessions`: list named Codex app-server threads.
- `super_agents_thread_favorite`: check whether one local Openbase Coder thread
  is favorited.
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
