import json
from types import SimpleNamespace

import pytest

from super_agents.app_protocol import turn_error_message, unwrap_error_message
from super_agents.claude_sdk import _ERROR_RESULT_MESSAGE, _error_result_message, _turn_failure_text

TIMEOUT_MESSAGE = "The model provider did not respond in time. Retry the request."
# Same shape as the field-test failure, with private diagnostics replaced.
PROBLEM = {
    "type": "https://example.com/problems/origin-response-timeout",
    "title": "Error 524: A timeout occurred",
    "status": 524,
    "detail": (
        "The origin web server did not return a complete response within the "
        "120-second Proxy Read Timeout window. The connection was established, "
        "but the origin took too long to respond."
    ),
    "instance": "redacted-request",
    "error_name": "origin_response_timeout",
    "retryable": True,
    "what_you_should_do": "Private operator instructions must not reach the turn UI.",
}


def test_recorded_timeout_shape_reaches_both_backend_error_paths():
    raw = "API Error: 524 " + json.dumps(PROBLEM)
    expected = "API Error: 524 " + TIMEOUT_MESSAGE
    assert unwrap_error_message(raw) == expected
    assert turn_error_message({"error": {"message": raw}}) == expected
    assert _turn_failure_text(RuntimeError(raw)) == expected
    assert _error_result_message(SimpleNamespace(result=raw, is_error=True)) == (
        f"{expected} ({_ERROR_RESULT_MESSAGE})"
    )


def test_nested_problem_preserves_prefix_suffix_and_hides_diagnostics():
    raw = "API Error: 524 " + json.dumps({"error": PROBLEM}) + " (retry)"
    assert unwrap_error_message(raw) == "API Error: 524 " + TIMEOUT_MESSAGE + " (retry)"


@pytest.mark.parametrize(
    ("problem", "expected"),
    [
        ({"status": 429, "title": "Too many requests", "detail": "Try again later."}, "Try again later."),
        ({"status": 503, "title": "Service unavailable"}, "Service unavailable"),
        ({"status": 503, "title": "Service unavailable", "detail": "  "}, "Service unavailable"),
        ({"status": 524, "title": "Timeout", "message": "A specific backend message."}, "A specific backend message."),
    ],
)
def test_other_problem_messages_and_existing_message_precedence(problem, expected):
    assert unwrap_error_message(json.dumps(problem)) == expected


@pytest.mark.parametrize(
    "problem",
    [
        {"status": "524", "title": "Timeout"},
        {"status": True, "title": "Timeout"},
        {"status": 200, "title": "Successful result"},
        {"status": 524, "title": [], "detail": {"message": "Unexpected shape"}},
        {"status": 524, "title": " ", "detail": " "},
        {"title": "A JSON result, not a structured HTTP error"},
    ],
)
def test_non_problem_json_retains_existing_fallback(problem):
    raw = json.dumps(problem)
    assert unwrap_error_message(raw) == raw
