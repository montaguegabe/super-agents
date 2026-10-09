"""Optional display titles do not participate in session label uniqueness."""

import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.claude_sdk import ClaudeAgentSdkClient


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPER_AGENTS_CLAUDE_CODE_HOME", str(tmp_path))
    return Store(tmp_path / "sessions.sqlite3")


def test_title_survives_reopen_and_later_turns(store, tmp_path):
    session = store.create_session("internal-id", cwd=str(tmp_path / "project"), auto_title=True)
    assert session.title == f"project ({session.id[-8:]})"
    store.create_turn(session.id, "  what is 17\n times 23  ")
    reopened = Store(store.path)
    title = reopened.get_session(session.id).title
    assert title == f"what is 17 times 23 ({session.id[-8:]})"
    reopened.create_turn(session.id, "second prompt")
    assert reopened.get_session(session.id).title == title
    assert reopened.get_session(session.id).name == "internal-id"


def test_concurrent_first_turns_preserve_one_title(store):
    session = store.create_session("internal", auto_title=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: store.create_turn(session.id, f"prompt {n}"), range(4)))
    with store.connect() as conn:
        first = conn.execute("select prompt from turns order by rowid limit 1").fetchone()[0]
    assert store.get_session(session.id).title == f"{first} ({session.id[-8:]})"


@pytest.mark.parametrize("prompt", ["", " \n ", "long title " * 100])
def test_empty_or_long_first_prompt_has_bounded_title(store, prompt):
    session = store.create_session("internal", cwd="/workspace/project", auto_title=True)
    store.create_turn(session.id, prompt)
    title = store.get_session(session.id).title
    assert len(title) <= 91
    assert title.endswith(f"({session.id[-8:]})")
    if not prompt.strip():
        assert title == session.title


@pytest.mark.asyncio
async def test_manual_rename_before_first_turn_wins_and_allows_duplicate_titles(store):
    client = ClaudeAgentSdkClient(store=store)
    for name in ("one", "two"):
        session = store.create_session(name, auto_title=True)
        await client.rename_by_label(LabelQueryInput(thread_id=session.id), "My title")
        store.create_turn(session.id, "a different first message")
        saved = store.get_session(session.id)
        assert saved.title == "My title"
        assert saved.name == name


def test_named_sessions_keep_their_existing_semantics(store):
    session = store.create_session("dispatcher")
    store.create_turn(session.id, "do a task")
    assert store.get_session(session.id).title is None
    assert "title" not in store.get_session(session.id).to_json()
    store.rename_session(session.id, "custom")
    assert store.get_session(session.id).name == "custom"


def test_existing_database_gets_optional_title_columns(store):
    session = store.create_session("existing")
    with sqlite3.connect(store.path) as conn:
        conn.execute("alter table sessions drop column title")
        conn.execute("alter table sessions drop column auto_title")
    migrated = Store(store.path).get_session(session.id)
    assert migrated.name == "existing"
    assert migrated.title is None
    assert migrated.auto_title is False
