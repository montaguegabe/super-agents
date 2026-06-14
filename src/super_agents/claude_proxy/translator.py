from __future__ import annotations

import copy
import html
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ProxyOptions

JsonObject = dict[str, Any]
MAX_TOOL_CALL_REGISTRY_ENTRIES = 1000
MAX_INTERNAL_TOOL_ROUNDS = 3
REASONING_BUDGET_TOKENS = {
    "low": 1024,
    "medium": 2048,
    "high": 4096,
    "xhigh": 8192,
}

_assistant_turn_by_tool_call_id: dict[str, JsonObject] = {}
_tool_use_block_by_call_id: dict[str, JsonObject] = {}


@dataclass(frozen=True)
class AnthropicResponse:
    answer: str
    reasoning: str
    tool_calls: list[JsonObject]
    usage: JsonObject | None
    raw: JsonObject | None


def build_anthropic_request(body: JsonObject, options: ProxyOptions) -> tuple[JsonObject, dict[str, JsonObject]]:
    tool_index = tool_index_from_request(body.get("tools"))
    messages = codex_input_to_anthropic_messages(body.get("input"), tool_index)
    tools = codex_tools_to_anthropic_tools(body.get("tools"), tool_index)
    request: JsonObject = {
        "model": options.model,
        "max_tokens": options.max_tokens,
        "messages": messages,
    }

    system = build_system_prompt(body)
    if system:
        request["system"] = system
    if tools:
        request["tools"] = tools
        request["tool_choice"] = {"type": "auto"}
    thinking_budget = thinking_budget_tokens(body, request, options)
    if options.thinking and thinking_budget is not None:
        request["thinking"] = {
            "type": "enabled",
            "budget_tokens": thinking_budget,
        }
    return request, tool_index


def build_system_prompt(body: JsonObject) -> str:
    parts = [
        "You are replying to Codex through a local Anthropic Messages adapter.",
        "Use the provided tools when they are useful. After tool results are returned, finish with a concise answer.",
        "If web_search is available and current information is needed, use it with a concise query.",
    ]
    instructions = body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        parts.append(instructions.strip())
    return "\n\n".join(part for part in parts if part)


def should_enable_thinking(body: JsonObject, request: JsonObject) -> bool:
    return request.get("max_tokens", 0) >= 1025 and bool(
        body.get("reasoning")
        or body.get("reasoning_effort")
        or body.get("reasoningEffort")
        or body.get("reasoning_summary")
    )


def thinking_budget_tokens(body: JsonObject, request: JsonObject, options: ProxyOptions) -> int | None:
    if not should_enable_thinking(body, request):
        return None
    effort = reasoning_effort(body)
    configured = REASONING_BUDGET_TOKENS.get(effort, options.thinking_budget_tokens)
    return min(configured, options.max_tokens - 1)


def reasoning_effort(body: JsonObject) -> str:
    for key in ("reasoningEffort", "reasoning_effort"):
        value = body.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict):
        value = reasoning.get("effort")
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return ""


def codex_input_to_anthropic_messages(input_value: Any, tool_index: dict[str, JsonObject]) -> list[JsonObject]:
    messages: list[JsonObject] = []
    emitted_assistant_turn_keys: set[str] = set()

    def push(role: str, content: Any) -> None:
        blocks = normalize_anthropic_content(content)
        if not blocks:
            return
        if messages and messages[-1].get("role") == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    if isinstance(input_value, list):
        for item in input_value:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "message":
                role = item.get("role")
                if role in {"system", "developer"}:
                    push("user", [{"type": "text", "text": content_text(item.get("content"))}])
                else:
                    push("assistant" if role == "assistant" else "user", codex_content_to_anthropic(item.get("content")))
            elif item_type in {"function_call", "custom_tool_call"}:
                push("assistant", assistant_tool_call_to_anthropic(item, tool_index, emitted_assistant_turn_keys))
            elif item_type in {"function_call_output", "custom_tool_call_output"}:
                result = tool_result_from_output(item.get("output"))
                push(
                    "user",
                    [
                        {
                            "type": "tool_result",
                            "tool_use_id": item.get("call_id") or f"toolu_{_uuid()}",
                            **result,
                        }
                    ],
                )

    if not messages:
        messages.append({"role": "user", "content": [{"type": "text", "text": "Respond to an empty Codex request."}]})
    if messages[0].get("role") != "user":
        messages.insert(0, {"role": "user", "content": [{"type": "text", "text": "Continue."}]})
    return messages


def content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            parts.append(part["text"])
        elif isinstance(part.get("output_text"), str):
            parts.append(part["output_text"])
        elif part.get("type") == "input_image":
            parts.append(f"[image: {part.get('image_url') or 'inline'}]")
    return "\n".join(parts)


def codex_content_to_anthropic(content: Any) -> list[JsonObject]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []
    blocks: list[JsonObject] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if isinstance(part.get("text"), str):
            blocks.append({"type": "text", "text": part["text"]})
        elif isinstance(part.get("output_text"), str):
            blocks.append({"type": "text", "text": part["output_text"]})
        elif part.get("type") == "input_image":
            blocks.append(anthropic_image_block(part))
    return blocks


def anthropic_image_block(part: JsonObject) -> JsonObject:
    image_url = part.get("image_url") if isinstance(part.get("image_url"), str) else ""
    match = re.match(r"^data:([^;]+);base64,(.+)$", image_url)
    if match:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": match.group(1),
                "data": match.group(2),
            },
        }
    return {"type": "text", "text": f"[image omitted: {image_url or 'inline image'}]"}


def normalize_anthropic_content(content: Any) -> list[JsonObject]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list):
        return []
    return [block for block in content if isinstance(block, dict)]


def assistant_tool_call_to_anthropic(
    item: JsonObject,
    tool_index: dict[str, JsonObject],
    emitted_assistant_turn_keys: set[str],
) -> list[JsonObject]:
    call_id = item.get("call_id")
    if isinstance(call_id, str) and call_id in _assistant_turn_by_tool_call_id:
        stored_turn = _assistant_turn_by_tool_call_id[call_id]
        key = str(stored_turn.get("key") or "")
        if key in emitted_assistant_turn_keys:
            return []
        emitted_assistant_turn_keys.add(key)
        return copy.deepcopy(stored_turn.get("content") or [])
    if isinstance(call_id, str) and call_id in _tool_use_block_by_call_id:
        return [copy.deepcopy(_tool_use_block_by_call_id[call_id])]

    name = item.get("name") if isinstance(item.get("name"), str) else ""
    indexed = tool_index.get(name) if name else None
    if item.get("type") == "custom_tool_call" or indexed and indexed.get("kind") == "custom":
        raw_input = item.get("input")
        tool_input = {"input": raw_input if isinstance(raw_input, str) else str(raw_input or "")}
    else:
        tool_input = parse_json_object(item.get("arguments"))
    return [{"type": "tool_use", "id": call_id or f"toolu_{_uuid()}", "name": name, "input": tool_input}]


def parse_json_object(value: Any) -> JsonObject:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def output_payload_text(output: Any) -> str:
    if isinstance(output, str):
        return output
    if isinstance(output, list):
        return content_text(output)
    if isinstance(output, dict):
        if isinstance(output.get("content"), str):
            return output["content"]
        if isinstance(output.get("content"), list):
            return content_text(output["content"])
        return json.dumps(output)
    return ""


def tool_result_from_output(output: Any) -> JsonObject:
    normalized = normalize_tool_output(output)
    result: JsonObject = {"content": normalized["content"]}
    if normalized["is_error"]:
        result["is_error"] = True
    return result


def normalize_tool_output(output: Any) -> JsonObject:
    if isinstance(output, str):
        return {"content": clean_codex_tool_output_text(output), "is_error": False}
    if isinstance(output, list):
        return {"content": function_output_content_items_to_anthropic(output), "is_error": False}
    if isinstance(output, dict):
        if "body" in output:
            nested = normalize_tool_output(output["body"])
            return {**nested, "is_error": output.get("success") is False}
        if "output" in output:
            nested = normalize_tool_output(output["output"])
            return {**nested, "is_error": output.get("success") is False}
        if isinstance(output.get("content"), str):
            return {"content": clean_codex_tool_output_text(output["content"]), "is_error": output.get("success") is False}
        if isinstance(output.get("content"), list):
            return {
                "content": function_output_content_items_to_anthropic(output["content"]),
                "is_error": output.get("success") is False,
            }
        return {"content": json.dumps(output), "is_error": output.get("success") is False}
    return {"content": "", "is_error": False}


def clean_codex_tool_output_text(text: str) -> str:
    marker = "\nOutput:\n"
    if (text.startswith("Chunk ID:") or text.startswith("Wall time:")) and marker in text:
        return text[text.rfind(marker) + len(marker) :]
    return text


def function_output_content_items_to_anthropic(items: list[Any]) -> list[JsonObject] | str:
    content: list[JsonObject] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "input_text" and isinstance(item.get("text"), str):
            content.append({"type": "text", "text": clean_codex_tool_output_text(item["text"])})
        elif item.get("type") == "input_image" and isinstance(item.get("image_url"), str):
            image = anthropic_image_block(item)
            content.append(image if image.get("type") == "image" else {"type": "text", "text": image.get("text", "")})
    return content or ""


def codex_tools_to_anthropic_tools(tools: Any, tool_index: dict[str, JsonObject]) -> list[JsonObject]:
    if not isinstance(tools, list):
        return []
    converted: list[JsonObject] = []
    seen: set[str] = set()
    for tool in tools:
        collect_anthropic_tool(tool, tool_index, converted, seen)
    return converted


def collect_anthropic_tool(
    tool: Any,
    tool_index: dict[str, JsonObject],
    converted: list[JsonObject],
    seen: set[str],
) -> None:
    if not isinstance(tool, dict):
        return
    nested_tools = tool.get("tools")
    if isinstance(nested_tools, list):
        for nested in nested_tools:
            collect_anthropic_tool(nested, tool_index, converted, seen)

    name = tool_name(tool)
    if not name or name in seen:
        return
    seen.add(name)
    if name == "web_search":
        converted.append(
            {
                "name": name,
                "description": "Search the web for current information. Use concise natural-language queries.",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "The web search query."}},
                    "required": ["query"],
                    "additionalProperties": False,
                },
            }
        )
        return

    indexed = tool_index.get(name)
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    description = tool.get("description") or function.get("description") or f"Codex tool {name}"
    converted.append(
        {
            "name": name,
            "description": description,
            "input_schema": (
                {"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"], "additionalProperties": False}
                if indexed and indexed.get("kind") == "custom"
                else tool_schema(tool) or {"type": "object", "properties": {}}
            ),
        }
    )


def tool_index_from_request(tools: Any) -> dict[str, JsonObject]:
    index: dict[str, JsonObject] = {}
    if not isinstance(tools, list):
        return index

    def visit(tool: Any) -> None:
        if not isinstance(tool, dict):
            return
        nested_tools = tool.get("tools")
        if isinstance(nested_tools, list):
            for nested in nested_tools:
                visit(nested)
        name = tool_name(tool)
        if not name:
            return
        tool_type = str(tool.get("type") or "").lower()
        kind = (
            "custom"
            if "custom" in tool_type
            or "freeform" in tool_type
            or name == "apply_patch"
            or bool(tool.get("format") or tool.get("input_format"))
            else "function"
        )
        index[name] = {"kind": kind, "spec": tool}

    for tool in tools:
        visit(tool)
    return index


def tool_name(tool: JsonObject) -> str:
    if tool.get("type") == "web_search":
        return "web_search"
    if isinstance(tool.get("name"), str):
        return tool["name"]
    function = tool.get("function") if isinstance(tool.get("function"), dict) else {}
    if isinstance(function.get("name"), str):
        return function["name"]
    return ""


def tool_schema(spec: JsonObject | None) -> JsonObject | None:
    if not isinstance(spec, dict):
        return None
    function = spec.get("function") if isinstance(spec.get("function"), dict) else {}
    for key in ("input_schema", "parameters"):
        if isinstance(spec.get(key), dict):
            return spec[key]
    return function.get("parameters") if isinstance(function.get("parameters"), dict) else None


def normalize_anthropic_response(message: JsonObject) -> AnthropicResponse:
    content = message.get("content") if isinstance(message.get("content"), list) else []
    text_parts: list[str] = []
    reasoning_parts: list[str] = []
    tool_calls: list[JsonObject] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text_parts.append(block["text"])
        elif block.get("type") == "thinking" and isinstance(block.get("thinking"), str):
            reasoning_parts.append(block["thinking"])
        elif block.get("type") == "redacted_thinking":
            reasoning_parts.append("[redacted thinking]")
        elif block.get("type") == "tool_use":
            tool_calls.append(block)

    for tool_call in tool_calls:
        remember_tool_use_block(tool_call.get("id"), tool_call)
    remember_assistant_tool_turn(content, tool_calls)

    usage = message.get("usage") if isinstance(message.get("usage"), dict) else None
    return AnthropicResponse(
        answer="\n".join(text_parts),
        reasoning="\n".join(reasoning_parts),
        tool_calls=tool_calls,
        usage=usage,
        raw=message,
    )


def remember_tool_use_block(call_id: Any, block: JsonObject) -> None:
    if not isinstance(call_id, str) or not call_id:
        return
    _tool_use_block_by_call_id[call_id] = {
        "type": "tool_use",
        "id": call_id,
        "name": block.get("name"),
        "input": block.get("input") if isinstance(block.get("input"), dict) else {},
    }
    trim_registry(_tool_use_block_by_call_id)


def remember_assistant_tool_turn(content: list[Any], tool_calls: list[JsonObject]) -> None:
    call_ids = [call.get("id") for call in tool_calls if isinstance(call.get("id"), str)]
    if not call_ids:
        return
    stored_turn = {"key": ",".join(call_ids), "content": copy.deepcopy(content)}
    for call_id in call_ids:
        _assistant_turn_by_tool_call_id[call_id] = stored_turn
    trim_registry(_assistant_turn_by_tool_call_id)


def trim_registry(registry: dict[str, JsonObject]) -> None:
    while len(registry) > MAX_TOOL_CALL_REGISTRY_ENTRIES:
        oldest = next(iter(registry))
        del registry[oldest]


def run_anthropic_once(request: JsonObject, options: ProxyOptions) -> AnthropicResponse:
    if not options.api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")
    if options.debug_last_anthropic_request_path:
        Path(options.debug_last_anthropic_request_path).write_text(json.dumps(request, indent=2), encoding="utf-8")

    payload = json.dumps(request).encode("utf-8")
    http_request = urllib.request.Request(
        options.api_url,
        data=payload,
        method="POST",
        headers={
            "content-type": "application/json",
            "x-api-key": options.api_key,
            "anthropic-version": options.anthropic_version,
        },
    )
    try:
        with urllib.request.urlopen(http_request, timeout=120) as response:
            raw_text = response.read().decode("utf-8")
            status = response.status
    except urllib.error.HTTPError as exc:
        raw_text = exc.read().decode("utf-8", errors="replace")
        try:
            error_json = json.loads(raw_text) if raw_text else {}
        except json.JSONDecodeError:
            raise RuntimeError(f"Anthropic returned non-JSON HTTP {exc.code}: {raw_text[:500]}") from None
        message = error_json.get("error", {}).get("message") if isinstance(error_json.get("error"), dict) else None
        raise RuntimeError(f"Anthropic HTTP {exc.code}: {message or json.dumps(error_json)}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Anthropic request failed: {exc.reason}") from None

    try:
        json_response = json.loads(raw_text) if raw_text else {}
    except json.JSONDecodeError:
        raise RuntimeError(f"Anthropic returned non-JSON HTTP {status}: {raw_text[:500]}") from None
    return normalize_anthropic_response(json_response)


def run_anthropic_with_internal_tools(request: JsonObject, options: ProxyOptions) -> AnthropicResponse:
    current_request = request
    last_response: AnthropicResponse | None = None
    for _round in range(MAX_INTERNAL_TOOL_ROUNDS + 1):
        response = run_anthropic_once(current_request, options)
        last_response = response
        internal_tool_calls = [call for call in response.tool_calls if is_internal_tool_name(call.get("name"))]
        if not internal_tool_calls:
            return response

        non_internal_calls = [call for call in response.tool_calls if not is_internal_tool_name(call.get("name"))]
        if non_internal_calls:
            return AnthropicResponse(
                answer=response.answer,
                reasoning=response.reasoning,
                tool_calls=non_internal_calls,
                usage=response.usage,
                raw=response.raw,
            )

        tool_results = [run_internal_tool(tool_call, options) for tool_call in internal_tool_calls]
        current_request = {
            **current_request,
            "messages": [
                *current_request.get("messages", []),
                {"role": "assistant", "content": response.raw.get("content", []) if response.raw else []},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": result["tool_use_id"],
                            "content": result["content"],
                            **({"is_error": True} if result.get("is_error") else {}),
                        }
                        for result in tool_results
                    ],
                },
            ],
        }
        current_request.pop("thinking", None)

    return AnthropicResponse(
        answer="Web search did not complete within the proxy's internal tool round limit.",
        reasoning="",
        tool_calls=[],
        usage=last_response.usage if last_response else None,
        raw=last_response.raw if last_response else None,
    )


def is_internal_tool_name(name: Any) -> bool:
    return name == "web_search"


def run_internal_tool(tool_call: JsonObject, options: ProxyOptions) -> JsonObject:
    if tool_call.get("name") == "web_search":
        try:
            query = web_search_query_from_input(tool_call.get("input"))
            return {"tool_use_id": tool_call.get("id"), "content": run_web_search(query)}
        except Exception as exc:
            return {"tool_use_id": tool_call.get("id"), "content": f"web_search failed: {exc}", "is_error": True}
    return {"tool_use_id": tool_call.get("id"), "content": f"unsupported internal tool: {tool_call.get('name')}", "is_error": True}


def web_search_query_from_input(input_value: Any) -> str:
    if isinstance(input_value, dict):
        for key in ("query", "q", "search_query", "term"):
            value = input_value.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    if isinstance(input_value, str) and input_value.strip():
        return input_value.strip()
    raise ValueError("missing query")


def run_web_search(query: str) -> str:
    encoded = urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(
        f"https://html.duckduckgo.com/html/?{encoded}",
        headers={"user-agent": "codex-claude-proxy/0.1 (+https://duckduckgo.com)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            document = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from None
    results = parse_duckduckgo_results(document)[:5]
    if not results:
        return f"No web results found for: {query}"
    entries = [f"Web search results for: {query}"]
    for index, result in enumerate(results, start=1):
        entries.append("\n".join(part for part in [f"{index}. {result['title']}", result["url"], result.get("snippet", "")] if part))
    return "\n\n".join(entries)


def parse_duckduckgo_results(document: str) -> list[JsonObject]:
    results: list[JsonObject] = []
    pattern = re.compile(
        r'<div class="result[\s\S]*?<a rel="nofollow" class="result__a" href="([^"]+)"[\s\S]*?>([\s\S]*?)</a>[\s\S]*?(?:<a class="result__snippet"[\s\S]*?>([\s\S]*?)</a>|<div class="result__snippet"[\s\S]*?>([\s\S]*?)</div>)'
    )
    for match in pattern.finditer(document):
        url = decode_duckduckgo_url(html.unescape(match.group(1)))
        title = strip_html(match.group(2))
        snippet = strip_html(match.group(3) or match.group(4) or "")
        if title and url:
            results.append({"title": title, "url": url, "snippet": snippet})
    return results


def decode_duckduckgo_url(url: str) -> str:
    try:
        parsed = urllib.parse.urlparse(urllib.parse.urljoin("https://duckduckgo.com", url))
        query = urllib.parse.parse_qs(parsed.query)
        uddg = query.get("uddg", [""])[0]
        return uddg or urllib.parse.urlunparse(parsed)
    except Exception:
        return url


def strip_html(value: str) -> str:
    return html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]*>", " ", value)).strip())


def response_stream_events(request_body: JsonObject, anthropic: AnthropicResponse) -> list[JsonObject]:
    response_id = f"resp_{_uuid()}"
    created = int(time.time())
    model = request_body.get("model") or "claude-code"
    tool_index = tool_index_from_request(request_body.get("tools"))
    tool_items = [
        item
        for call in anthropic.tool_calls
        if (item := normalize_tool_call(call, tool_index)) is not None
    ]
    events: list[JsonObject] = [
        {
            "type": "response.created",
            "response": {
                "id": response_id,
                "type": "response",
                "created_at": created,
                "model": model,
                "headers": {"openai-model": model},
            },
        }
    ]
    if anthropic.reasoning:
        events.extend(reasoning_events(anthropic.reasoning))
    if tool_items:
        events.extend({"type": "response.output_item.done", "item": item} for item in tool_items)
        events.append(completed_event(response_id, model, anthropic.usage, False))
        return events

    events.extend(message_events(anthropic.answer or ""))
    events.append(completed_event(response_id, model, anthropic.usage, True))
    return events


def reasoning_events(text: str) -> list[JsonObject]:
    reasoning_id = f"rs_{_uuid()}"
    events: list[JsonObject] = [
        {
            "type": "response.output_item.added",
            "item": {"id": reasoning_id, "type": "reasoning", "summary": [], "content": [], "encrypted_content": None},
        },
        {
            "type": "response.reasoning_summary_part.added",
            "item_id": reasoning_id,
            "summary_index": 0,
            "part": {"type": "summary_text", "text": ""},
        },
    ]
    events.extend(
        {
            "type": "response.reasoning_summary_text.delta",
            "item_id": reasoning_id,
            "summary_index": 0,
            "delta": chunk,
        }
        for chunk in chunk_text(text, 1200)
    )
    events.append(
        {
            "type": "response.output_item.done",
            "item": {
                "id": reasoning_id,
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": text}],
                "content": [{"type": "reasoning_text", "text": text}],
                "encrypted_content": None,
            },
        }
    )
    return events


def message_events(text: str) -> list[JsonObject]:
    message_id = f"msg_{_uuid()}"
    events: list[JsonObject] = [
        {
            "type": "response.output_item.added",
            "item": {"id": message_id, "type": "message", "role": "assistant", "content": []},
        }
    ]
    events.extend({"type": "response.output_text.delta", "delta": chunk} for chunk in chunk_text(text, 1200))
    events.append(
        {
            "type": "response.output_item.done",
            "item": {
                "id": message_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        }
    )
    return events


def completed_event(response_id: str, model: str, usage: JsonObject | None, end_turn: bool) -> JsonObject:
    return {
        "type": "response.completed",
        "response": {
            "id": response_id,
            "status": "completed",
            "model": model,
            "end_turn": end_turn,
            "usage": codex_usage(usage),
        },
    }


def failed_event(message: str) -> JsonObject:
    return {
        "type": "response.failed",
        "response": {
            "id": f"resp_{_uuid()}",
            "error": {"type": "server_error", "code": "codex_claude_proxy_error", "message": message},
        },
    }


def codex_usage(usage: JsonObject | None) -> JsonObject:
    input_tokens = numeric_usage(usage, "input_tokens")
    output_tokens = numeric_usage(usage, "output_tokens")
    cache_read = numeric_usage(usage, "cache_read_input_tokens")
    cache_creation = numeric_usage(usage, "cache_creation_input_tokens")
    return {
        "input_tokens": input_tokens + cache_creation,
        "input_tokens_details": {"cached_tokens": cache_read},
        "output_tokens": output_tokens,
        "output_tokens_details": {"reasoning_tokens": 0},
        "total_tokens": input_tokens + cache_creation + output_tokens,
    }


def numeric_usage(usage: JsonObject | None, key: str) -> int:
    value = usage.get(key) if isinstance(usage, dict) else 0
    return value if isinstance(value, int | float) else 0


def normalize_tool_call(call: JsonObject, tool_index: dict[str, JsonObject]) -> JsonObject | None:
    name = call.get("name") if isinstance(call.get("name"), str) else ""
    if not name:
        return None
    indexed = tool_index.get(name)
    kind = indexed.get("kind", "function") if indexed else "function"
    call_id = call.get("id") if isinstance(call.get("id"), str) else f"toolu_{_uuid()}"
    if kind == "custom":
        input_value = call.get("input")
        if isinstance(input_value, dict) and isinstance(input_value.get("input"), str):
            custom_input = input_value["input"]
        elif isinstance(input_value, str):
            custom_input = input_value
        else:
            custom_input = stringify_arguments(input_value)
        return {"type": "custom_tool_call", "call_id": call_id, "name": name, "input": custom_input}
    return {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": stringify_arguments(repair_tool_input_for_codex(name, call.get("input"), indexed.get("spec") if indexed else None)),
    }


def repair_tool_input_for_codex(name: str, input_value: Any, spec: JsonObject | None) -> Any:
    if not isinstance(input_value, dict):
        return input_value
    repaired = dict(input_value)
    schema = tool_schema(spec) or {}
    properties = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    required = schema.get("required") if isinstance(schema.get("required"), list) else []
    if name == "exec_command" and isinstance(repaired.get("command"), str) and repaired.get("cmd") is None:
        repaired["cmd"] = repaired.pop("command")
    elif name == "shell_command" and isinstance(repaired.get("cmd"), str) and repaired.get("command") is None:
        repaired["command"] = repaired.pop("cmd")
    if len(required) == 1 and isinstance(repaired.get("command"), str) and repaired.get(required[0]) is None:
        repaired[required[0]] = repaired["command"]
        if required[0] != "command":
            repaired.pop("command", None)
    if schema.get("additionalProperties") is False and properties:
        repaired = {key: value for key, value in repaired.items() if key in properties}
    return repaired


def stringify_arguments(value: Any) -> str:
    if isinstance(value, str):
        try:
            json.loads(value)
            return value
        except json.JSONDecodeError:
            return json.dumps({"input": value})
    if isinstance(value, dict):
        return json.dumps(value)
    return "{}"


def chunk_text(text: str, size: int) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def _uuid() -> str:
    return uuid.uuid4().hex
