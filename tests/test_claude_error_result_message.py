import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from super_agents.agent_store import Store
from super_agents.app_models import LabelQueryInput
from super_agents.app_protocol import AUTHENTICATION_FAILED, turn_error_kind
from super_agents.claude_sdk import (
    _ERROR_RESULT_MESSAGE,
    ClaudeAgentSdkClient,
    _error_result_message,
    _turn_failure_text,
)


def test_error_result_keeps_claude_login_failure_text() -> None:
    message = SimpleNamespace(result="Not logged in · Please run /login", is_error=True)

    assert _error_result_message(message) == (f"Not logged in · Please run /login ({_ERROR_RESULT_MESSAGE})")


CLOUD_TIMEOUT_ENVELOPE = (
    "API Error: 524 "
    '{"error": {"type": "api_error", "code": "origin_response_timeout", '
    '"message": "The model provider did not respond in time. Retry the request.", '
    '"request_id": "req_123", "operator": "check the Cloud proxy logs for req_123"}}'
)


def test_error_result_reduces_provider_envelope_to_its_message() -> None:
    message = SimpleNamespace(result=CLOUD_TIMEOUT_ENVELOPE, is_error=True)

    assert _error_result_message(message) == (
        f"API Error: 524 The model provider did not respond in time. Retry the request. ({_ERROR_RESULT_MESSAGE})"
    )


def test_turn_failure_text_reduces_envelope_and_keeps_plain_errors() -> None:
    assert _turn_failure_text(RuntimeError(CLOUD_TIMEOUT_ENVELOPE)) == (
        "API Error: 524 The model provider did not respond in time. Retry the request."
    )
    assert _turn_failure_text(RuntimeError("boom")) == "boom"
    assert _turn_failure_text(RuntimeError("")) == "RuntimeError"


def test_error_result_without_text_falls_back_to_unverified() -> None:
    assert _error_result_message(SimpleNamespace(result="", is_error=True)) == _ERROR_RESULT_MESSAGE
    assert _error_result_message(SimpleNamespace(is_error=True)) == _ERROR_RESULT_MESSAGE


class _Options:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _SignedOutClient:
    """What the Claude Agent SDK yields when Claude Code is signed out
    (claude-agent-sdk 0.2.93, Claude Code 2.1.296)."""

    def __init__(self, options):
        self.prompts: asyncio.Queue[str] = asyncio.Queue()

    async def connect(self):
        pass

    async def query(self, prompt):
        await self.prompts.put(prompt)

    async def interrupt(self):
        pass

    async def disconnect(self):
        pass

    async def receive_response(self):
        await self.prompts.get()
        yield SimpleNamespace(
            content=[SimpleNamespace(text="Not logged in · Please run /login")],
            error="authentication_failed",
        )
        yield SimpleNamespace(result="Not logged in · Please run /login", num_turns=1, is_error=True)


@pytest.mark.asyncio
async def test_signed_out_claude_turn_records_the_sdk_error_kind(tmp_path: Path) -> None:
    sdk = SimpleNamespace(ClaudeSDKClient=_SignedOutClient, ClaudeAgentOptions=_Options)
    store = Store(tmp_path / "state.sqlite3")
    client = ClaudeAgentSdkClient(store=store, sdk_loader=lambda: sdk)
    notifications: list[tuple[str, dict]] = []
    original_notify = client._notify_turn

    def capture(method, session_id, turn_id, **params):
        notifications.append((method, params))
        return original_notify(method, session_id, turn_id, **params)

    client._notify_turn = capture
    try:
        await client.start_thread({"name": "auth", "cwd": str(tmp_path)})
        started = await client.start_turn_by_label(LabelQueryInput(label="auth"), {"prompt": "hi"})
        async with asyncio.timeout(2):
            while store.get_turn(started["turnId"]).status == "running":
                await asyncio.sleep(0.005)
        turn = store.get_turn(started["turnId"])
        assert turn.status == "failed"
        assert turn.error_kind == AUTHENTICATION_FAILED
        assert turn.to_json()["errorKind"] == AUTHENTICATION_FAILED
        failed = [params for method, params in notifications if method == "turn/failed"]
        assert failed and failed[-1]["error"]["errorKind"] == AUTHENTICATION_FAILED
    finally:
        await client.close()


def test_turn_error_kind_reads_only_structured_fields() -> None:
    codex_turn = {
        "status": "failed",
        "error": {
            "message": "unexpected status 401 Unauthorized: Missing bearer or basic authentication in header",
            "codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 401}},
        },
    }
    assert turn_error_kind(codex_turn) == AUTHENTICATION_FAILED
    assert turn_error_kind({"turn": codex_turn}) == AUTHENTICATION_FAILED
    assert turn_error_kind({"errorKind": "billing_error"}) == "billing_error"
    assert turn_error_kind({"error": {"message": "x", "errorKind": "rate_limit"}}) == "rate_limit"
    # Text alone never classifies a failure.
    assert turn_error_kind({"error": {"message": "Not logged in · Please run /login"}}) is None
    assert turn_error_kind({"error": {"codexErrorInfo": {"httpConnectionFailed": {"httpStatusCode": 500}}}}) is None
    assert turn_error_kind(None) is None
