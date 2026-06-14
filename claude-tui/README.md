# Super Agents Claude

Super Agents Claude is a local, self-contained alternative to the Codex app-server based Super Agents workflow. It creates named agent sessions, launches Claude CLI/TUI processes under PTYs, observes their terminal output, infers coarse state, sends prompts or keystrokes, and persists metadata in a local SQLite database.

This project is intentionally local. It does not publish, deploy, push, or call a remote management server. Claude usage happens only through the Claude CLI you configure.

## Status

This is an MVP. The core runtime works as a terminal owner around Claude:

1. Named sessions are stored locally.
2. Each live session can launch a Claude CLI process in its own working directory.
3. Output is captured as raw PTY logs and cleaned text logs.
4. The runtime infers states such as running, waiting for input, waiting for approval, completed, failed, and missing CLI.
5. The TUI can send prompts, raw keys, approval yes/no answers, and interrupts.
6. Queued turns are persisted and the TUI drains them when a session appears idle.

The live PTY controller only exists while the TUI or foreground command is running. Persisted sessions and logs survive restarts, but live Claude processes are not reattached after the manager exits.

## Install

Use Python 3.9 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Install the Claude CLI separately, then confirm it is on `PATH`:

```bash
claude --help
```

If Claude is installed somewhere else, configure the command:

```bash
export SUPER_AGENTS_CLAUDE_TUI_CMD="claude"
```

Optional CLI args and model selection can be configured with:

```bash
export SUPER_AGENTS_CLAUDE_TUI_ARGS="--some-claude-flag"
export SUPER_AGENTS_CLAUDE_TUI_MODEL="sonnet"
```

Use the cheapest suitable Claude Code model available to your account for smoke tests. The default recommendation in this project is `sonnet`; you can override that guidance with `SUPER_AGENTS_CLAUDE_TUI_SMOKE_MODEL`. Do not use `fable` for smoke testing because it is too expensive for this purpose. The manager simply appends `--model <value>` when a model is set.

For low-cost smoke testing, also prefer low-effort and safe-mode startup flags:

```bash
export SUPER_AGENTS_CLAUDE_TUI_ARGS="--effort low --safe-mode --permission-mode dontAsk"
export SUPER_AGENTS_CLAUDE_TUI_MODEL="sonnet"
```

## Run

Open the TUI:

```bash
super-agents-claude-tui
```

or:

```bash
super-agents-claude-tui tui
```

Useful keys:

```text
n  create a named session
l  launch Claude for the selected session
p  send a prompt
x  send a named key such as enter, escape, ctrl-c, up, down
a  answer an approval prompt with yes
d  answer an approval prompt with no
c  send ctrl-c
r  refresh
q  quit
```

## CLI

The CLI mirrors the useful parts of the current Super Agents surface for local use:

```bash
super-agents-claude-tui doctor
super-agents-claude-tui start "frontend-fix" --cwd "$PWD" --model sonnet
super-agents-claude-tui sessions
super-agents-claude-tui active
super-agents-claude-tui status
super-agents-claude-tui read "frontend-fix" --lines 120
super-agents-claude-tui queue-turn "frontend-fix" "Continue with the next failing test."
super-agents-claude-tui send "frontend-fix" "Start by inspecting the repo." --wait 3
super-agents-claude-tui rename "frontend-fix" "frontend-agent"
super-agents-claude-tui cancel "frontend-agent"
```

Claude Code does not provide a separate queue API. The reliable follow-up flow is steering: keep the TUI running, wait until the selected Claude session is ready for input, press `p`, type the follow-up prompt, and press Enter. The runner sends that text into the same live Claude TUI/PTTY.

`queue-turn` is only a local convenience for delayed steering while the TUI is open. The TUI owns the PTY and drains queued turns by typing into Claude when the session appears ready; Claude itself does not know about this queue.

## How It Differs From Codex App-Server Super Agents

Codex app-server Super Agents talks to the Codex app server through structured APIs. It can inspect native app-server thread and turn state, answer callbacks, steer active turns, cancel turns, and queue follow-ups against server-owned threads.

Super Agents Claude cannot rely on those structured internals. It treats Claude as an interactive terminal program:

1. It starts Claude under a pseudo-terminal.
2. It records the raw terminal byte stream.
3. It strips ANSI sequences into readable logs.
4. It scans recent output for patterns that imply waiting, running, approval, or failure.
5. It sends text, enter, arrow keys, escape, ctrl-c, or yes/no approval keystrokes back into the PTY.

That means it is more general but less certain. If Claude changes its TUI wording, state inference may need new patterns in `src/super_agents/claude_tui/detector.py`.

## Local Data

By default, data is stored under:

```text
~/.local/share/super-agents-claude-tui/
```

Set `SUPER_AGENTS_CLAUDE_TUI_HOME` to use a different local directory.

The directory contains:

```text
state.sqlite3       session and turn metadata
logs/*.log          cleaned terminal logs
logs/*.raw.log      raw PTY output
```

## Smoke Testing

Run paid Claude smoke tests sparingly. For the cheapest practical smoke path, use `sonnet` or the lowest-cost model your Claude Code account supports, not `fable`.

```bash
export SUPER_AGENTS_CLAUDE_TUI_ARGS="--effort low --safe-mode --permission-mode dontAsk"
super-agents-claude-tui start smoke --cwd "$PWD" --model sonnet
super-agents-claude-tui send smoke "Reply with exactly SAC_SMOKE_OK and no other text." --wait 30
```

The expected evidence is a cleaned log containing `SAC_SMOKE_OK` and a post-run detector state of `waiting` with `wants_input=true`.

Follow-up steering can usually be verified without another paid Claude call by running the test suite. `tests/test_controller.py` uses a fake Claude process and confirms that two prompts are sent through the same live PTY session.

## Testing Without Claude

The test suite uses a fake Python process to exercise PTY send/read behavior, so it does not spend Claude credits. From an editable install, run:

```bash
python3 -m unittest
```

From a fresh checkout without installing the package first, run:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```
