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
    assert session.title == "project"
    store.create_turn(session.id, "  what is 17\n times 23  ")
    reopened = Store(store.path)
    title = reopened.get_session(session.id).title
    assert title == "what is 17 times 23"
    reopened.create_turn(session.id, "second prompt")
    assert reopened.get_session(session.id).title == title
    assert reopened.get_session(session.id).name == "internal-id"


def test_concurrent_first_turns_preserve_one_title(store):
    session = store.create_session("internal", auto_title=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda n: store.create_turn(session.id, f"prompt {n}"), range(4)))
    with store.connect() as conn:
        first = conn.execute("select prompt from turns order by rowid limit 1").fetchone()[0]
    assert store.get_session(session.id).title == first


@pytest.mark.parametrize("prompt", ["", " \n ", "long title " * 100])
def test_empty_or_long_first_prompt_has_bounded_title(store, prompt):
    session = store.create_session("internal", cwd="/workspace/project", auto_title=True)
    store.create_turn(session.id, prompt)
    title = store.get_session(session.id).title
    assert len(title) <= 80
    assert session.id[-8:] not in title
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


@pytest.mark.parametrize("user_request", ["Are you there?", "<voice>Are you &lt;there&gt;?</voice>", ""])
def test_manual_title_strips_context_before_shortening(store, user_request):
    from super_agents.claude_prompts import with_claude_turn_context

    session = store.create_session("internal", cwd="/workspace/project", auto_title=True)
    prompt = with_claude_turn_context(user_request, cwd=session.cwd, developer_instructions="private context " * 50)
    store.create_turn(session.id, prompt)
    title = store.get_session(session.id).title
    expected = "Are you <there>?" if user_request.startswith("<voice>") else user_request or "project"
    assert title == expected
    assert "context" not in title


def test_titles_never_carry_the_session_id_and_may_repeat(store):
    sessions = [store.create_session(f"thread-{n}", auto_title=True) for n in range(2)]
    for session in sessions:
        store.create_turn(session.id, "Hi are you there?")
    assert [store.get_session(s.id).title for s in sessions] == ["Hi are you there?"] * 2


def test_reopening_strips_legacy_session_id_suffixes_once(store):
    titled = store.create_session("thread-a", auto_title=True)
    imported = store.create_session("placeholder")
    retired = store.create_session("other")
    store.create_session("project")  # holds the stripped name already
    with sqlite3.connect(store.path) as conn:
        conn.execute("update sessions set title = ? where id = ?", (f"Hi there ({titled.id[-8:]})", titled.id))
        conn.execute("update sessions set name = ? where id = ?", (f"project ({imported.id[-8:]})", imported.id))
        conn.execute(
            "update sessions set name = ? where id = ?", (f"dispatcher (retired {retired.id[-8:]})", retired.id)
        )
        conn.execute("pragma user_version = 0")

    reopened = Store(store.path)

    assert reopened.get_session(titled.id).title == "Hi there"
    assert reopened.get_session(imported.id).name == "project (2)"
    assert reopened.get_session(retired.id).name == "dispatcher (retired)"
    with reopened.connect() as conn:
        assert conn.execute("pragma user_version").fetchone()[0] == 1
    deliberate_title = f"Fix bug ({titled.id[-8:]})"
    deliberate_name = f"project ({imported.id[-8:]})"
    reopened.update_session(titled.id, title=deliberate_title)
    reopened.rename_session(imported.id, deliberate_name)
    reopened_again = Store(store.path)
    assert reopened_again.get_session(titled.id).title == deliberate_title
    assert reopened_again.get_session(imported.id).name == deliberate_name


def test_migration_preserves_other_ids_and_parenthesized_user_text(store):
    session = store.create_session("placeholder")
    other = store.create_session("other")
    for text in (f"Fix bug ({other.id[-8:]})", "Fix bug (2)", "Fix bug (retired)"):
        with store.connect() as conn:
            conn.execute("update sessions set name = ?, title = ? where id = ?", (text, text, session.id))
            conn.execute("pragma user_version = 0")
        migrated = Store(store.path).get_session(session.id)
        assert migrated.name == text
        assert migrated.title == text


def test_migration_numbers_colliding_imported_and_retired_names(store):
    for base in ("project", "dispatcher (retired)"):
        store.create_session(base)
        store.create_session(f"{base} (2)")
    imported = [store.create_session(f"import-{index}") for index in range(2)]
    retired = [store.create_session(f"retired-{index}") for index in range(2)]
    with store.connect() as conn:
        for session in imported:
            conn.execute("update sessions set name = ? where id = ?", (f"project ({session.id[-8:]})", session.id))
        for session in retired:
            conn.execute(
                "update sessions set name = ? where id = ?", (f"dispatcher (retired {session.id[-8:]})", session.id)
            )
        conn.execute("pragma user_version = 0")
    reopened = Store(store.path)
    assert {reopened.get_session(session.id).name for session in imported} == {"project (3)", "project (4)"}
    assert {reopened.get_session(session.id).name for session in retired} == {
        "dispatcher (retired) (3)",
        "dispatcher (retired) (4)",
    }
