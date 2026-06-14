from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

DEFAULT_PORT = 6066
DEFAULT_HOST = "127.0.0.1"
DEFAULT_ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_ANTHROPIC_MODEL = "claude-sonnet-4-20250514"
DEFAULT_MAX_TOKENS = 4096
DEFAULT_THINKING_BUDGET_TOKENS = 1024


@dataclass(frozen=True)
class ProxyOptions:
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    api_key: str = ""
    api_url: str = DEFAULT_ANTHROPIC_API_URL
    anthropic_version: str = DEFAULT_ANTHROPIC_VERSION
    model: str = DEFAULT_ANTHROPIC_MODEL
    max_tokens: int = DEFAULT_MAX_TOKENS
    thinking_budget_tokens: int = DEFAULT_THINKING_BUDGET_TOKENS
    thinking: bool = True
    debug: bool = False
    debug_last_anthropic_request_path: str = ""


def model_catalog_path() -> Path:
    return Path(str(resources.files("super_agents.claude_proxy").joinpath("model-catalog.json")))


def options_from_environment() -> ProxyOptions:
    return ProxyOptions(
        host=os.environ.get("CODEX_CLAUDE_PROXY_HOST", DEFAULT_HOST),
        port=int(os.environ.get("CODEX_CLAUDE_PROXY_PORT", str(DEFAULT_PORT))),
        api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        api_url=os.environ.get("ANTHROPIC_API_URL", DEFAULT_ANTHROPIC_API_URL),
        anthropic_version=os.environ.get("ANTHROPIC_VERSION", DEFAULT_ANTHROPIC_VERSION),
        model=os.environ.get("ANTHROPIC_MODEL", DEFAULT_ANTHROPIC_MODEL),
        max_tokens=int(os.environ.get("ANTHROPIC_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))),
        thinking_budget_tokens=int(
            os.environ.get("ANTHROPIC_THINKING_BUDGET_TOKENS", str(DEFAULT_THINKING_BUDGET_TOKENS))
        ),
        thinking=os.environ.get("ANTHROPIC_THINKING", "1") != "0",
        debug=os.environ.get("CODEX_CLAUDE_PROXY_DEBUG") == "1",
        debug_last_anthropic_request_path=os.environ.get("CODEX_CLAUDE_PROXY_DEBUG_REQUEST_PATH", ""),
    )


def parse_options(argv: list[str] | None = None) -> tuple[ProxyOptions | None, bool]:
    parser = argparse.ArgumentParser(prog="super-agents-claude-proxy")
    parser.add_argument("--host", default=None, help=f"Bind host. Default: {DEFAULT_HOST}")
    parser.add_argument("--port", type=int, default=None, help=f"Bind port. Default: {DEFAULT_PORT}")
    parser.add_argument("--model", default=None, help=f"Anthropic API model. Default: {DEFAULT_ANTHROPIC_MODEL}")
    parser.add_argument("--api-url", default=None, help=f"Anthropic Messages URL. Default: {DEFAULT_ANTHROPIC_API_URL}")
    parser.add_argument("--max-tokens", type=int, default=None, help=f"Max response tokens. Default: {DEFAULT_MAX_TOKENS}")
    parser.add_argument(
        "--thinking-budget-tokens",
        type=int,
        default=None,
        help=f"Extended thinking budget. Default: {DEFAULT_THINKING_BUDGET_TOKENS}",
    )
    parser.add_argument("--no-thinking", action="store_true", help="Disable Anthropic extended thinking")
    parser.add_argument("--debug", action="store_true", help="Log request summaries to stderr")
    parser.add_argument(
        "--print-model-catalog-path",
        action="store_true",
        help="Print the packaged Codex model catalog path and exit",
    )
    args = parser.parse_args(argv)

    if args.print_model_catalog_path:
        return None, True

    env_options = options_from_environment()
    options = ProxyOptions(
        host=args.host or env_options.host,
        port=args.port if args.port is not None else env_options.port,
        api_key=env_options.api_key,
        api_url=args.api_url or env_options.api_url,
        anthropic_version=env_options.anthropic_version,
        model=args.model or env_options.model,
        max_tokens=args.max_tokens if args.max_tokens is not None else env_options.max_tokens,
        thinking_budget_tokens=(
            args.thinking_budget_tokens
            if args.thinking_budget_tokens is not None
            else env_options.thinking_budget_tokens
        ),
        thinking=False if args.no_thinking else env_options.thinking,
        debug=args.debug or env_options.debug,
        debug_last_anthropic_request_path=env_options.debug_last_anthropic_request_path,
    )
    validate_options(options)
    return options, False


def validate_options(options: ProxyOptions) -> None:
    if options.port <= 0 or options.port > 65535:
        raise ValueError(f"Invalid port: {options.port}")
    if options.max_tokens < 1:
        raise ValueError(f"Invalid max tokens: {options.max_tokens}")
    if options.thinking_budget_tokens < 1024:
        raise ValueError(f"Invalid thinking budget tokens: {options.thinking_budget_tokens}")
