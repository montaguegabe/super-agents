from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from super_agents.thread_intro import (
    DEFAULT_THREAD_INTRO_TIMEOUT_SECONDS,
    THREAD_INTRO_COMMAND_ENV,
    THREAD_INTRO_TIMEOUT_ENV,
    announce_thread_intro,
    is_first_turn,
    render_thread_intro_command,
    thread_intro_command_template,
    thread_intro_timeout_seconds,
)

# A tiny "announcer" that records its argv so tests can assert what ran.
RECORDER = "import json, sys; open(sys.argv[1], 'w').write(json.dumps(sys.argv[2:]))"


def recorder_template(out: Path) -> str:
    return f'{sys.executable} -c "{RECORDER}" {out} say {{agent_name}} "Hey there, I\'m {{agent_name}}." {{thread_name}} {{thread_id}}'


def test_template_is_disabled_when_unset_or_blank() -> None:
    assert thread_intro_command_template({}) is None
    assert thread_intro_command_template({THREAD_INTRO_COMMAND_ENV: "   "}) is None
    assert thread_intro_command_template({THREAD_INTRO_COMMAND_ENV: "say {agent_name}"}) == "say {agent_name}"


def test_timeout_falls_back_to_default_for_bad_values() -> None:
    assert thread_intro_timeout_seconds({}) == DEFAULT_THREAD_INTRO_TIMEOUT_SECONDS
    assert thread_intro_timeout_seconds({THREAD_INTRO_TIMEOUT_ENV: "nope"}) == DEFAULT_THREAD_INTRO_TIMEOUT_SECONDS
    assert thread_intro_timeout_seconds({THREAD_INTRO_TIMEOUT_ENV: "-3"}) == DEFAULT_THREAD_INTRO_TIMEOUT_SECONDS
    assert thread_intro_timeout_seconds({THREAD_INTRO_TIMEOUT_ENV: "2.5"}) == 2.5


def test_render_substitutes_inside_arguments_without_reparsing() -> None:
    argv = render_thread_intro_command(
        'openbase-coder user say {agent_name} "Hey there, I\'m {agent_name}." {thread_id}',
        agent_name="O'Brien",
        thread_name="react chess",
        thread_id="s_1",
    )
    assert argv == ["openbase-coder", "user", "say", "O'Brien", "Hey there, I'm O'Brien.", "s_1"]


def test_render_rejects_blank_template() -> None:
    with pytest.raises(ValueError):
        render_thread_intro_command("   ", agent_name="Connie")


def test_is_first_turn_only_before_any_turn_is_recorded() -> None:
    assert is_first_turn(last_turn_id=None, active_turn_id=None)
    assert is_first_turn(last_turn_id="", active_turn_id=None)
    assert not is_first_turn(last_turn_id="t_1", active_turn_id=None)
    assert not is_first_turn(last_turn_id=None, active_turn_id="t_1")


@pytest.mark.asyncio
async def test_announce_runs_configured_command_with_placeholders(tmp_path: Path) -> None:
    out = tmp_path / "argv.json"
    env = {THREAD_INTRO_COMMAND_ENV: recorder_template(out), "PATH": "/usr/bin:/bin"}
    ran = await announce_thread_intro(agent_name="Connie", thread_name="react-chess-game", thread_id="s_42", env=env)
    assert ran is True
    assert json.loads(out.read_text()) == ["say", "Connie", "Hey there, I'm Connie.", "react-chess-game", "s_42"]


@pytest.mark.asyncio
async def test_announce_is_a_no_op_without_agent_name_or_template(tmp_path: Path) -> None:
    out = tmp_path / "argv.json"
    env = {THREAD_INTRO_COMMAND_ENV: recorder_template(out), "PATH": "/usr/bin:/bin"}
    assert await announce_thread_intro(agent_name=None, thread_name="t", thread_id="s", env=env) is False
    assert await announce_thread_intro(agent_name="Connie", thread_name="t", thread_id="s", env={}) is False
    assert not out.exists()


@pytest.mark.asyncio
async def test_announce_failures_are_logged_not_raised(tmp_path: Path) -> None:
    failing = {THREAD_INTRO_COMMAND_ENV: f'{sys.executable} -c "import sys; sys.exit(3)"', "PATH": "/usr/bin:/bin"}
    assert await announce_thread_intro(agent_name="Connie", thread_name="t", thread_id="s", env=failing) is False
    missing = {THREAD_INTRO_COMMAND_ENV: str(tmp_path / "no-such-binary") + " {agent_name}", "PATH": "/usr/bin:/bin"}
    assert await announce_thread_intro(agent_name="Connie", thread_name="t", thread_id="s", env=missing) is False


@pytest.mark.asyncio
async def test_announce_times_out_and_kills_the_command() -> None:
    env = {
        THREAD_INTRO_COMMAND_ENV: f'{sys.executable} -c "import time; time.sleep(30)"',
        THREAD_INTRO_TIMEOUT_ENV: "0.2",
        "PATH": "/usr/bin:/bin",
    }
    assert await announce_thread_intro(agent_name="Connie", thread_name="t", thread_id="s", env=env) is False
