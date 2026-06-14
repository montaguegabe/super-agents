from __future__ import annotations

import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from super_agents.claude_proxy.backend import current_backend, main as backend_main, set_backend
from super_agents.claude_proxy.config import ProxyOptions, model_catalog_path
from super_agents.claude_proxy.server import make_handler
from super_agents.claude_proxy.translator import (
    AnthropicResponse,
    build_anthropic_request,
    failed_event,
    response_stream_events,
    run_anthropic_with_internal_tools,
)


def test_backend_switch_updates_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("KEEP_ME=1\nOPENBASE_CODEX_BACKEND=codex\n", encoding="utf-8")

    status = set_backend("claude-code", env_file)

    content = env_file.read_text(encoding="utf-8")
    assert status.backend == "claude-code-proxy"
    assert "KEEP_ME=1" in content
    assert "OPENBASE_CODEX_BACKEND=claude-code-proxy" in content
    assert "CODEX_CLAUDE_PROXY_COMMAND=super-agents-claude-proxy" in content
    assert f"CODEX_CLAUDE_MODEL_CATALOG_JSON={model_catalog_path()}" in content

    status = set_backend("codex", env_file)

    assert status.backend == "codex"
    assert "OPENBASE_CODEX_BACKEND=codex" in env_file.read_text(encoding="utf-8")


def test_backend_cli_status_and_use(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    env_file = tmp_path / ".env"

    assert backend_main(["--env-file", str(env_file), "use", "claude-code"]) == 0
    assert current_backend(env_file).backend == "claude-code-proxy"

    assert backend_main(["--env-file", str(env_file), "status"]) == 0
    output = capsys.readouterr().out
    assert "Backend set to claude-code-proxy" in output
    assert "Backend: claude-code-proxy" in output


def test_backend_switch_supports_claude_tui(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"

    status = set_backend("claude-tui", env_file)

    assert status.backend == "claude-tui"
    assert "OPENBASE_CODEX_BACKEND=claude-tui" in env_file.read_text(encoding="utf-8")


def test_model_catalog_resource_exists() -> None:
    catalog = json.loads(model_catalog_path().read_text(encoding="utf-8"))

    assert catalog["models"][0]["slug"] == "claude-code"


def test_build_anthropic_request_converts_messages_tools_and_thinking() -> None:
    body = {
        "model": "claude-code",
        "instructions": "Be brief.",
        "reasoning": {"effort": "medium"},
        "input": [{"type": "message", "role": "user", "content": "Say hi"}],
        "tools": [
            {
                "type": "function",
                "name": "exec_command",
                "description": "Run a command",
                "input_schema": {
                    "type": "object",
                    "properties": {"cmd": {"type": "string"}},
                    "required": ["cmd"],
                    "additionalProperties": False,
                },
            },
            {"type": "web_search"},
        ],
    }

    request, tool_index = build_anthropic_request(body, ProxyOptions(api_key="test"))

    assert request["model"] == "claude-sonnet-4-20250514"
    assert request["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Say hi"}]}]
    assert "Be brief." in request["system"]
    assert request["thinking"]["type"] == "enabled"
    assert request["thinking"]["budget_tokens"] == 2048
    assert tool_index["exec_command"]["kind"] == "function"
    assert [tool["name"] for tool in request["tools"]] == ["exec_command", "web_search"]


def test_build_anthropic_request_uses_camel_case_reasoning_effort() -> None:
    request, _tool_index = build_anthropic_request(
        {
            "model": "claude-code",
            "reasoningEffort": "low",
            "input": [{"type": "message", "role": "user", "content": "Say hi"}],
        },
        ProxyOptions(api_key="test"),
    )

    assert request["thinking"]["budget_tokens"] == 1024


def test_build_anthropic_request_clamps_large_reasoning_budget() -> None:
    request, _tool_index = build_anthropic_request(
        {
            "model": "claude-code",
            "reasoningEffort": "xhigh",
            "input": [{"type": "message", "role": "user", "content": "Say hi"}],
        },
        ProxyOptions(api_key="test", max_tokens=4096),
    )

    assert request["thinking"]["budget_tokens"] == 4095


def test_response_stream_events_include_reasoning_text_and_usage() -> None:
    events = response_stream_events(
        {"model": "claude-code"},
        AnthropicResponse(
            answer="Done",
            reasoning="Thought",
            tool_calls=[],
            usage={"input_tokens": 2, "cache_creation_input_tokens": 3, "cache_read_input_tokens": 1, "output_tokens": 5},
            raw=None,
        ),
    )

    assert events[0]["type"] == "response.created"
    assert any(event["type"] == "response.reasoning_summary_text.delta" for event in events)
    assert any(event.get("delta") == "Done" for event in events)
    completed = events[-1]
    assert completed["type"] == "response.completed"
    assert completed["response"]["end_turn"] is True
    assert completed["response"]["usage"]["total_tokens"] == 10


def test_response_stream_events_include_function_tool_call() -> None:
    events = response_stream_events(
        {
            "model": "claude-code",
            "tools": [
                {
                    "name": "exec_command",
                    "input_schema": {
                        "type": "object",
                        "properties": {"cmd": {"type": "string"}},
                        "required": ["cmd"],
                        "additionalProperties": False,
                    },
                }
            ],
        },
        AnthropicResponse(
            answer="",
            reasoning="",
            tool_calls=[{"type": "tool_use", "id": "toolu_1", "name": "exec_command", "input": {"command": "pwd"}}],
            usage=None,
            raw=None,
        ),
    )

    item = next(event["item"] for event in events if event["type"] == "response.output_item.done")
    assert item["type"] == "function_call"
    assert item["call_id"] == "toolu_1"
    assert json.loads(item["arguments"]) == {"cmd": "pwd"}
    assert events[-1]["response"]["end_turn"] is False


def test_missing_anthropic_api_key_can_be_returned_as_failed_event() -> None:
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        run_anthropic_with_internal_tools({"messages": []}, ProxyOptions(api_key=""))

    event = failed_event("ANTHROPIC_API_KEY is not set")
    assert event["type"] == "response.failed"
    assert event["response"]["error"]["code"] == "codex_claude_proxy_error"


def test_health_endpoint_returns_proxy_metadata() -> None:
    options = ProxyOptions(api_key="test-key")
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(options))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert payload["ok"] is True
    assert payload["endpoint"] == "/v1/responses"
    assert payload["hasAnthropicApiKey"] is True
