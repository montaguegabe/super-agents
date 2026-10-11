from types import SimpleNamespace

from super_agents.claude_sdk import _ERROR_RESULT_MESSAGE, _error_result_message


def test_error_result_keeps_claude_login_failure_text() -> None:
    message = SimpleNamespace(result="Not logged in · Please run /login", is_error=True)

    assert _error_result_message(message) == (f"Not logged in · Please run /login ({_ERROR_RESULT_MESSAGE})")


def test_error_result_without_text_falls_back_to_unverified() -> None:
    assert _error_result_message(SimpleNamespace(result="", is_error=True)) == _ERROR_RESULT_MESSAGE
    assert _error_result_message(SimpleNamespace(is_error=True)) == _ERROR_RESULT_MESSAGE
