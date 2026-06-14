from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse
from uuid import uuid4

from .config import ProxyOptions, model_catalog_path, parse_options
from .translator import (
    build_anthropic_request,
    failed_event,
    response_stream_events,
    run_anthropic_with_internal_tools,
)

MAX_REQUEST_BYTES = 25 * 1024 * 1024


class ClaudeProxyHandler(BaseHTTPRequestHandler):
    options: ProxyOptions

    def log_message(self, format: str, *args: Any) -> None:
        if self.options.debug:
            super().log_message(format, *args)

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path in {"/", "/health"}:
            self._write_json(
                200,
                {
                    "ok": True,
                    "provider": "codex-claude-proxy",
                    "endpoint": "/v1/responses",
                    "anthropicApiUrl": self.options.api_url,
                    "anthropicModel": self.options.model,
                    "anthropicVersion": self.options.anthropic_version,
                    "maxTokens": self.options.max_tokens,
                    "thinking": self.options.thinking,
                    "thinkingBudgetTokens": self.options.thinking_budget_tokens,
                    "hasAnthropicApiKey": bool(self.options.api_key),
                    "modelCatalogJson": str(model_catalog_path()),
                },
            )
            return
        self._write_json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        if path != "/v1/responses":
            self._write_json(404, {"error": "not found"})
            return
        try:
            body = self._read_json_body()
        except ValueError as exc:
            self._write_json(400, {"error": str(exc)})
            return

        try:
            request, _tool_index = build_anthropic_request(body, self.options)
        except Exception as exc:
            self._write_json(400, {"error": str(exc)})
            return

        if self.options.debug:
            print(
                json.dumps(
                    {
                        "codexModel": body.get("model"),
                        "anthropicModel": request.get("model"),
                        "inputItems": len(body.get("input", [])) if isinstance(body.get("input"), list) else 0,
                        "anthropicMessages": len(request.get("messages", [])),
                        "anthropicTools": len(request.get("tools", [])),
                        "thinking": bool(request.get("thinking")),
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("x-request-id", f"req_{uuid4().hex}")
        self.send_header("openai-model", str(body.get("model") or "claude-code"))
        self.end_headers()

        try:
            anthropic = run_anthropic_with_internal_tools(request, self.options)
            events = response_stream_events(body, anthropic)
        except Exception as exc:
            events = [failed_event(str(exc))]
        for event in events:
            self._write_sse(event)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("content-length") or "0")
        if length > MAX_REQUEST_BYTES:
            raise ValueError("Request body is too large")
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw.decode("utf-8")) if raw else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON body: {exc}") from None
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _write_sse(self, payload: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(payload, separators=(',', ':'))}\n\n".encode("utf-8"))
        self.wfile.flush()

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def make_handler(options: ProxyOptions) -> type[ClaudeProxyHandler]:
    class BoundClaudeProxyHandler(ClaudeProxyHandler):
        pass

    BoundClaudeProxyHandler.options = options
    return BoundClaudeProxyHandler


def run_server(options: ProxyOptions) -> None:
    server = ThreadingHTTPServer((options.host, options.port), make_handler(options))
    print(f"codex-claude-proxy listening on http://{options.host}:{options.port}/v1/responses", file=sys.stderr)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    options, print_catalog = parse_options(argv)
    if print_catalog:
        print(model_catalog_path())
        return 0
    assert options is not None
    run_server(options)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
