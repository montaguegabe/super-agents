from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from super_agents import initial_context as context
from super_agents.session_history import read_session_messages
from super_agents.app_client_threads import ThreadLifecycleMixin


@pytest.fixture(autouse=True)
def context_db(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPER_AGENTS_INITIAL_CONTEXT_DB", str(tmp_path / "context.sqlite3"))


@pytest.mark.asyncio
async def test_context_is_ordinary_input_once_and_survives_new_client():
    context.register_initial_context("thread", "user: Remember the blue lighthouse.")
    async with context.initial_context_input(object(), "thread", "What color?") as prompt:
        assert "blue lighthouse" in prompt
        assert prompt.endswith("[Current user message]\nWhat color?")
        assert context.strip_initial_context(prompt) == "What color?"
    async with context.initial_context_input(object(), "thread", "Next") as prompt:
        assert prompt == "Next"


@pytest.mark.asyncio
async def test_ambiguous_submission_reconciles_without_repeating_user_request(monkeypatch):
    context.register_initial_context("thread", "old history")
    with pytest.raises(TimeoutError):
        async with context.initial_context_input(object(), "thread", "Do work"):
            raise TimeoutError("transport closed after submission")
    reader = AsyncMock(return_value=[{"text": context.context_marker("thread")}])
    monkeypatch.setattr(context, "read_session_messages", reader)
    with pytest.raises(RuntimeError, match="Refresh"):
        async with context.initial_context_input(object(), "thread", "Do work"):
            pytest.fail("Must not resubmit an uncertain prompt")
    async with context.initial_context_input(object(), "thread", "Follow-up") as prompt:
        assert prompt == "Follow-up"


@pytest.mark.asyncio
async def test_uncertain_missing_history_stays_blocked(monkeypatch):
    context.register_initial_context("thread", "history")
    with pytest.raises(TimeoutError):
        async with context.initial_context_input(object(), "thread", "work"):
            raise TimeoutError()
    monkeypatch.setattr(context, "read_session_messages", AsyncMock(return_value=[]))
    for _ in range(2):
        with pytest.raises(RuntimeError):
            async with context.initial_context_input(object(), "thread", "work"):
                pytest.fail("Cannot prove whether the previous submission was accepted")


@pytest.mark.asyncio
async def test_codex_export_reads_all_api_pages_and_excludes_reasoning():
    client = SimpleNamespace(
        read_thread_page=AsyncMock(
            side_effect=[
                {
                    "thread": {
                        "turns": [
                            {
                                "items": [
                                    {"type": "userMessage", "content": [{"type": "text", "text": "constraint"}]},
                                    {"type": "reasoning", "text": "hidden"},
                                ]
                            }
                        ],
                        "historyNextCursor": "next",
                    }
                },
                {"thread": {"turns": [{"items": [{"type": "agentMessage", "text": "answer"}]}]}},
            ]
        )
    )
    assert await read_session_messages(client, "thread") == [
        {"role": "user", "text": "constraint"},
        {"role": "assistant", "text": "answer"},
    ]
    assert client.read_thread_page.call_args_list[1].kwargs["cursor"] == "next"
    assert client.read_thread_page.call_args_list[0].kwargs["items_view"] == "full"


@pytest.mark.asyncio
async def test_claude_export_uses_official_sdk():
    calls = []

    def reader(session_id, **kwargs):
        calls.append((session_id, kwargs))
        return [SimpleNamespace(type="user", message={"content": [{"type": "text", "text": "hello"}]})]

    client = SimpleNamespace(
        store=SimpleNamespace(get_session=lambda _: SimpleNamespace(backend_session_id="native", cwd="project")),
        _sdk_loader=lambda: SimpleNamespace(get_session_messages=reader),
    )
    assert await read_session_messages(client, "thread") == [{"role": "user", "text": "hello"}]
    assert calls == [("native", {"directory": "project", "limit": 50, "offset": 0})]


@pytest.mark.asyncio
async def test_claude_export_missing_sdk_api_does_not_fallback():
    client = SimpleNamespace(
        store=SimpleNamespace(get_session=lambda _: SimpleNamespace(backend_session_id="native", cwd="project")),
        _sdk_loader=lambda: object(),
    )
    with pytest.raises(RuntimeError, match="Update claude-agent-sdk"):
        await read_session_messages(client, "thread")


@pytest.mark.asyncio
async def test_fresh_codex_thread_has_empty_history_through_api():
    client = ThreadLifecycleMixin()
    client.ensure_connected = AsyncMock()
    client.request = AsyncMock(
        side_effect=[
            {"thread": {"id": "new"}},
            RuntimeError(
                "thread new is not materialized yet; thread/turns/list is unavailable before first user message"
            ),
        ]
    )
    result = await client.read_thread_page("new", limit=50)
    assert result["thread"]["turns"] == []
    assert client.request.call_count == 2


@pytest.mark.asyncio
async def test_other_codex_history_errors_are_not_hidden():
    client = ThreadLifecycleMixin()
    client.ensure_connected = AsyncMock()
    client.request = AsyncMock(side_effect=[{"thread": {"id": "old"}}, RuntimeError("storage unavailable")])
    with pytest.raises(RuntimeError, match="storage unavailable"):
        await client.read_thread_page("old", limit=50)


@pytest.mark.asyncio
async def test_codex_context_is_injected_without_running_a_turn_and_not_duplicated():
    client = SimpleNamespace(backend="codex", ensure_connected=AsyncMock(), request=AsyncMock(return_value={}))
    await context.initialize_session_context(client, "native", "earlier facts")
    await context.initialize_session_context(client, "native", "earlier facts")
    client.request.assert_awaited_once()
    method, params = client.request.call_args.args
    assert method == "thread/inject_items"
    assert params["threadId"] == "native"
    assert params["items"][0]["role"] == "user"
    assert "earlier facts" in params["items"][0]["content"][0]["text"]
    async with context.initial_context_input(client, "native", "next") as prompt:
        assert prompt == "next"


@pytest.mark.asyncio
async def test_claude_context_is_deferred_until_normal_sdk_query():
    client = SimpleNamespace(backend="claude_code", request=AsyncMock())
    await context.initialize_session_context(client, "native", "earlier facts")
    client.request.assert_not_called()
    assert context.has_pending_initial_context("native")


@pytest.mark.asyncio
async def test_missing_rollout_is_empty_only_for_a_pending_context():
    context.register_initial_context("new", "history")
    client = ThreadLifecycleMixin()
    client.ensure_connected = AsyncMock()
    client.request = AsyncMock(
        side_effect=[
            {"thread": {"id": "new"}},
            RuntimeError("invalid paginated history lineage: missing source rollout"),
        ]
    )
    assert (await client.read_thread_page("new", limit=50))["thread"]["turns"] == []
    client.request.side_effect = [
        {"thread": {"id": "old"}},
        RuntimeError("invalid paginated history lineage: missing source rollout"),
    ]
    with pytest.raises(RuntimeError, match="missing source rollout"):
        await client.read_thread_page("old", limit=50)
