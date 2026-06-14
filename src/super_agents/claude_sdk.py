from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterator

from super_agents.app_formatting import apply_field_selection, without_none
from super_agents.app_models import LabelQueryInput
from super_agents.app_protocol import is_active_status, with_super_agent_identity_instructions
from super_agents.app_sessions import required_label
from super_agents.agent_store import Session, Store, iso_now

JsonObject = dict[str, Any]
SdkLoader = Callable[[], Any]
CLAUDE_PERMISSION_MODE = "bypassPermissions"
CLAUDE_CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"
CLAUDE_CONFIG_FILENAME = ".claude.json"
CLAUDE_SETTINGS_FILENAME = "settings.json"
CLAUDE_INSTRUCTIONS_FILENAME = "CLAUDE.md"

SDK_IMPORT_ERROR = (
    "Claude Code backend requires the claude-agent-sdk package. "
    "Install super-agents with the claude extra, or install claude-agent-sdk in this environment."
)


class ClaudeAgentSdkClient:
    """Super Agents backend using Claude Code without Anthropic API keys."""

    backend = "claude_code"

    def __init__(self, store: Store | None = None, sdk_loader: SdkLoader | None = None) -> None:
        self.store = store or Store()
        self._sdk_loader = sdk_loader or _load_sdk
        self._sdk_clients: dict[str, Any] = {}
        self._sdk_client_efforts: dict[str, str | None] = {}
        self._session_locks: dict[str, asyncio.Lock] = {}
        self._queue_tasks: dict[str, asyncio.Task[None]] = {}

    async def status(self) -> JsonObject:
        sessions = self.store.list_sessions(include_inactive=True)
        ready, error = self._sdk_ready()
        return without_none(
            {
                "ready": ready,
                "backend": self.backend,
                "managedProcess": False,
                "sdkPackage": "claude-agent-sdk",
                "sdkError": error,
                "dataStore": str(self.store.path),
                "pendingRequests": [],
                "pendingPermissionRequests": [],
                "queuedTurns": [turn.to_json() for session in sessions for turn in self.store.queued_turns(session.id)],
                "activeTurns": [self._status_item(session) for session in sessions if is_active_status(session.status)],
            }
        )

    async def start_thread(self, input_data: JsonObject) -> JsonObject:
        name = str(input_data.get("name") or input_data.get("label") or "").strip()
        if not name:
            raise ValueError("name is required.")
        agent_name = _optional_str(input_data.get("agentName"))
        developer_instructions = with_super_agent_identity_instructions(
            _optional_str(input_data.get("developerInstructions")),
            name,
            agent_name=agent_name,
        )
        existing = self.store.get_by_name(name)
        if existing is not None:
            effective_agent_name = agent_name or existing.agent_name
            effective_developer_instructions = with_super_agent_identity_instructions(
                _optional_str(input_data.get("developerInstructions")) or existing.developer_instructions,
                name,
                agent_name=effective_agent_name,
            )
            session = self.store.update_session(
                existing.id,
                cwd=str(Path(_optional_str(input_data.get("cwd")) or existing.cwd).expanduser()),
                agent_name=effective_agent_name,
                developer_instructions=effective_developer_instructions,
                model=_optional_str(input_data.get("model")) or existing.model,
                status="waiting",
                active_turn_id=None,
                last_observed_state="Claude Code thread refreshed",
            )
            return {"backend": self.backend, "threadId": session.id, "session": session.to_json()}
        session = self.store.create_session(
            name,
            cwd=_optional_str(input_data.get("cwd")),
            agent_name=agent_name,
            developer_instructions=developer_instructions,
            model=_optional_str(input_data.get("model")),
            command=["claude-agent-sdk"],
        )
        return {"backend": self.backend, "threadId": session.id, "session": session.to_json()}

    async def resume_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        session = self._resolve_session(input_data)
        return {
            "backend": self.backend,
            "name": session.name,
            "threadId": session.id,
            "session": session.to_json(),
        }

    async def read_by_label(self, input_data: LabelQueryInput, include_turns: bool = False) -> JsonObject:
        session = self._resolve_session(input_data)
        turns = self.store.list_turns(session.id, limit=input_data.max_items or 20)
        payload: JsonObject = {
            "backend": self.backend,
            "threadId": session.id,
            "name": session.name,
            "session": session.to_json(),
            "logTail": self.store.tail_log(session, lines=80),
        }
        if include_turns:
            payload["turns"] = [self._turn_view(session, turn) for turn in turns]
        else:
            payload["recentTurns"] = [self._turn_view(session, turn) for turn in turns[:5]]
        return payload

    async def rename_by_label(self, input_data: LabelQueryInput, new_name: str) -> JsonObject:
        session = self._resolve_session(input_data)
        renamed = self.store.rename_session(session.id, new_name)
        return {
            "backend": self.backend,
            "renamed": True,
            "name": renamed.name,
            "previousName": session.name,
            "threadId": renamed.id,
        }

    async def answer_request(self, request_id: str | int, result: JsonObject) -> JsonObject:
        return self._unsupported("codex_answer_request", requestId=request_id, result=result)

    async def sessions(self) -> list[JsonObject]:
        return [self._session_view(session) for session in self.store.list_sessions(include_inactive=True)]

    async def active(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        items = [self._agent_item(session, query) for session in self._query_sessions(query, include_inactive=False)]
        return {"backend": self.backend, "count": len(items), "agents": items[: query.limit or 50]}

    async def recent(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        items = [
            self._agent_item(session, query)
            for session in self._query_sessions(query, include_inactive=bool(query.include_inactive))
        ]
        return {"backend": self.backend, "count": len(items), "agents": items[: query.limit or 20]}

    async def compact_status(self, input_data: LabelQueryInput | None = None) -> JsonObject:
        query = input_data or LabelQueryInput()
        items = [
            self._status_item(session)
            for session in self._query_sessions(query, include_inactive=bool(query.include_inactive))
        ][: query.limit or 50]
        return {
            "backend": self.backend,
            "count": len(items),
            "agents": [apply_field_selection(item, query.fields) for item in items],
        }

    async def resolve_label(self, input_data: LabelQueryInput) -> JsonObject:
        return self._status_item(self._resolve_session(input_data))

    async def progress_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        session = self._resolve_session(input_data)
        turns = self.store.list_turns(session.id, limit=input_data.max_items or 10)
        payload = self._status_item(session)
        payload["turns"] = [self._turn_view(session, turn) for turn in turns]
        payload["logTail"] = self.store.tail_log(session, lines=40)
        return payload

    async def steer_by_label(
        self,
        input_data: LabelQueryInput,
        prompt: str,
        turn_input: JsonObject | None = None,
    ) -> JsonObject:
        return await self.start_turn_by_label(
            input_data,
            {**(turn_input or {}), "prompt": prompt},
        )

    async def cancel_by_label(self, input_data: LabelQueryInput) -> JsonObject:
        session = self._resolve_session(input_data)
        sdk_client = self._sdk_clients.get(session.id)
        if sdk_client and hasattr(sdk_client, "interrupt"):
            await sdk_client.interrupt()
        if session.active_turn_id:
            self.store.update_turn(session.active_turn_id, status="cancelled", finished_at=iso_now())
        refreshed = self.store.update_session(
            session.id,
            status="cancelled",
            active_turn_id=None,
            last_observed_state="interrupt sent",
        )
        return {"backend": self.backend, "cancelled": True, "threadId": refreshed.id, "name": refreshed.name}

    async def start_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject:
        session = self._resolve_session(input_data)
        if self._session_is_busy(session):
            interrupted_turn_id = await self._interrupt_active_turn(
                session,
                reason="steered by follow-up",
            )
            sdk = self._require_sdk()
            prompt = str(turn_input["prompt"])
            sdk_prompt = self._prompt_for_session(session, turn_input)
            model = _optional_str(turn_input.get("model")) or session.model
            reasoning_effort = _optional_str(turn_input.get("reasoningEffort"))
            turn = self.store.create_turn(
                session.id,
                prompt,
                status="running",
                mode=_optional_str(turn_input.get("mode")),
                model=model,
                reasoning_effort=reasoning_effort,
            )
            self.store.update_session(
                session.id,
                status="running",
                active_turn_id=turn.id,
                last_turn_id=turn.id,
                last_observed_state="steering active turn via Claude Code",
            )
            asyncio.create_task(
                self._run_turn(
                    session.id,
                    turn.id,
                    sdk_prompt,
                    model,
                    reasoning_effort,
                    sdk,
                )
            )
            return {
                "backend": self.backend,
                "threadId": session.id,
                "name": session.name,
                "turnId": turn.id,
                "turn": turn.to_json(),
                "queued": False,
                "steered": True,
                "interruptedTurnId": interrupted_turn_id,
                "startedImmediately": False,
                "drain": "steered_active_turn",
            }
        sdk = self._require_sdk()
        prompt = str(turn_input["prompt"])
        sdk_prompt = self._prompt_for_session(session, turn_input)
        model = _optional_str(turn_input.get("model")) or session.model
        reasoning_effort = _optional_str(turn_input.get("reasoningEffort"))
        turn = self.store.create_turn(
            session.id,
            prompt,
            status="running",
            mode=_optional_str(turn_input.get("mode")),
            model=model,
            reasoning_effort=reasoning_effort,
        )
        self.store.update_session(
            session.id,
            status="running",
            active_turn_id=turn.id,
            last_turn_id=turn.id,
            last_observed_state="running via Claude Code",
        )
        asyncio.create_task(
            self._run_turn(
                session.id,
                turn.id,
                sdk_prompt,
                model,
                reasoning_effort,
                sdk,
            )
        )
        return {
            "backend": self.backend,
            "threadId": session.id,
            "name": session.name,
            "turnId": turn.id,
            "turn": turn.to_json(),
            "queued": False,
            "startedImmediately": True,
            "drain": "started_immediately",
        }

    async def queue_turn_by_label(self, input_data: LabelQueryInput, turn_input: JsonObject) -> JsonObject:
        session = self._resolve_session(input_data)
        if not self._session_is_busy(session):
            return await self.start_turn_by_label(input_data, turn_input)
        turn = self.store.create_turn(
            session.id,
            str(turn_input["prompt"]),
            status="queued",
            mode=_optional_str(turn_input.get("mode")),
            model=_optional_str(turn_input.get("model")) or session.model,
            reasoning_effort=_optional_str(turn_input.get("reasoningEffort")),
        )
        position = len(self.store.queued_turns(session.id))
        self._schedule_queue_drain(session.id)
        return {
            "backend": self.backend,
            "queued": True,
            "threadId": session.id,
            "name": session.name,
            "turnId": turn.id,
            "position": position,
            "queueDepth": position,
            "item": turn.to_json(),
            "drain": "scheduled",
        }

    async def thread_favorite(self, thread_id: str) -> JsonObject:
        return self._unsupported("super_agents_thread_favorite", threadId=thread_id)

    async def tags(self) -> JsonObject:
        return self._unsupported("super_agents_tags")

    async def thread_tags(self, thread_id: str, tags: list[Any] | None = None) -> JsonObject:
        return self._unsupported("super_agents_thread_tags", threadId=thread_id, tags=tags)

    async def report_tags(self, project_path: str, path: str, tags: list[Any] | None = None) -> JsonObject:
        return self._unsupported("super_agents_report_tags", projectPath=project_path, path=path, tags=tags)

    async def _run_turn(
        self,
        session_id: str,
        turn_id: str,
        prompt: str,
        model: str | None,
        reasoning_effort: str | None,
        sdk: Any,
    ) -> None:
        lock = self._session_locks.setdefault(session_id, asyncio.Lock())
        async with lock:
            session = self.store.get_session(session_id)
            try:
                if self._turn_was_cancelled(turn_id):
                    self._finish_cancelled_turn(session_id, turn_id)
                    return
                sdk_client = await self._sdk_client_for(
                    session,
                    model,
                    reasoning_effort,
                    sdk,
                )
                last_useful_message = ""
                with _without_unsupported_anthropic_api_keys():
                    await sdk_client.query(prompt)
                    async for message in sdk_client.receive_response():
                        append_log(session.log_path, _message_to_log(message))
                        if claude_session_id := _message_session_id(message):
                            self.store.update_session(session_id, backend_session_id=claude_session_id)
                        if useful := _message_preview(message):
                            last_useful_message = useful
                            self.store.update_turn(turn_id, last_useful_message=last_useful_message)
                if self._turn_was_cancelled(turn_id):
                    self._finish_cancelled_turn(session_id, turn_id)
                else:
                    self.store.update_turn(
                        turn_id,
                        status="completed",
                        finished_at=iso_now(),
                        last_useful_message=last_useful_message or None,
                    )
                    self._finish_session_turn(
                        session_id,
                        turn_id,
                        status="waiting",
                        active_turn_id=None,
                        last_observed_state="Claude Code response completed",
                        last_useful_message=last_useful_message or None,
                    )
            except Exception as exc:
                append_log(session.log_path, f"[{iso_now()}] ERROR {type(exc).__name__}: {exc}\n")
                if self._turn_was_cancelled(turn_id):
                    self._finish_cancelled_turn(session_id, turn_id)
                else:
                    self.store.update_turn(turn_id, status="failed", finished_at=iso_now(), last_error=str(exc))
                    self._finish_session_turn(
                        session_id,
                        turn_id,
                        status="failed",
                        active_turn_id=None,
                        last_observed_state=str(exc),
                    )
            finally:
                self._schedule_queue_drain(session_id)

    async def _interrupt_active_turn(self, session: Session, *, reason: str) -> str | None:
        turn_id = session.active_turn_id
        if not turn_id:
            return None

        with contextlib.suppress(KeyError):
            self.store.update_turn(turn_id, status="cancelled", finished_at=iso_now())

        sdk_client = self._sdk_clients.get(session.id)
        if sdk_client is not None and hasattr(sdk_client, "interrupt"):
            await sdk_client.interrupt()

        self.store.update_session(
            session.id,
            status="running",
            active_turn_id=turn_id,
            last_observed_state=f"Claude Code turn interrupted: {reason}",
        )
        return turn_id

    def _finish_session_turn(self, session_id: str, turn_id: str, **fields: object) -> None:
        try:
            current = self.store.get_session(session_id)
        except KeyError:
            return
        if current.active_turn_id == turn_id:
            self.store.update_session(session_id, **fields)

    def _finish_cancelled_turn(self, session_id: str, turn_id: str) -> None:
        self._finish_session_turn(
            session_id,
            turn_id,
            status="cancelled",
            active_turn_id=None,
            last_observed_state="Claude Code response interrupted",
        )

    def _turn_was_cancelled(self, turn_id: str) -> bool:
        try:
            return self.store.get_turn(turn_id).status == "cancelled"
        except KeyError:
            return False

    async def _sdk_client_for(
        self,
        session: Session,
        model: str | None,
        reasoning_effort: str | None,
        sdk: Any,
    ) -> Any:
        existing = self._sdk_clients.get(session.id)
        if existing is not None and self._sdk_client_efforts.get(session.id) == reasoning_effort:
            if model and hasattr(existing, "set_model"):
                await existing.set_model(model)
            return existing
        options = _agent_options(
            sdk,
            session.cwd,
            model,
            reasoning_effort,
            resume=session.backend_session_id,
        )
        with _without_unsupported_anthropic_api_keys():
            client = sdk.ClaudeSDKClient(options=options)
            await client.connect()
        self._sdk_clients[session.id] = client
        self._sdk_client_efforts[session.id] = reasoning_effort
        return client

    def _schedule_queue_drain(self, session_id: str) -> None:
        task = self._queue_tasks.get(session_id)
        if task and not task.done():
            return
        self._queue_tasks[session_id] = asyncio.create_task(self._queue_drain_loop(session_id))

    async def _queue_drain_loop(self, session_id: str) -> None:
        while True:
            await asyncio.sleep(0.5)
            try:
                session = self.store.get_session(session_id)
            except KeyError:
                return
            if self._session_is_busy(session):
                continue
            queued = self.store.queued_turns(session_id)
            if not queued:
                return
            turn = queued[0]
            sdk = self._require_sdk()
            self.store.update_turn(turn.id, status="running", attempts=turn.attempts + 1)
            self.store.update_session(
                session_id,
                status="running",
                active_turn_id=turn.id,
                last_turn_id=turn.id,
                last_observed_state="running queued turn via Claude Code",
            )
            asyncio.create_task(
                self._run_turn(
                    session_id,
                    turn.id,
                    self._prompt_for_session(session, {"prompt": turn.prompt}),
                    turn.model,
                    turn.reasoning_effort,
                    sdk,
                )
            )
            return

    def _resolve_session(self, input_data: LabelQueryInput) -> Session:
        if input_data.thread_id:
            try:
                return self.store.get_session(input_data.thread_id)
            except KeyError:
                pass
        return self.store.require_by_name(required_label(input_data))

    def _query_sessions(self, query: LabelQueryInput, *, include_inactive: bool) -> list[Session]:
        if query.thread_id or query.label:
            sessions = [self._resolve_session(query)]
        else:
            sessions = self.store.list_sessions(include_inactive=True, status=query.status)
        if query.cwd:
            cwd = str(Path(query.cwd).expanduser())
            sessions = [session for session in sessions if session.cwd == cwd]
        if query.status:
            sessions = [session for session in sessions if session.status == query.status]
        if not include_inactive:
            sessions = [session for session in sessions if is_active_status(session.status)]
        return sessions

    def _session_view(self, session: Session) -> JsonObject:
        return {"backend": self.backend, **session.to_json()}

    def _agent_item(self, session: Session, query: LabelQueryInput) -> JsonObject:
        turns = self.store.list_turns(session.id, limit=1)
        return apply_field_selection(
            without_none(
                {
                    "backend": self.backend,
                    "name": session.name,
                    "agentName": session.agent_name,
                    "threadId": session.id,
                    "turnId": session.active_turn_id or session.last_turn_id,
                    "cwd": session.cwd,
                    "status": session.status,
                    "model": session.model,
                    "updatedAt": session.updated_at,
                    "lastObservedState": session.last_observed_state,
                    "queueDepth": len(self.store.queued_turns(session.id)),
                    "preview": turns[0].to_json().get("promptPreview")
                    if turns and query.include_preview is not False
                    else None,
                }
            ),
            query.fields,
        )

    def _status_item(self, session: Session) -> JsonObject:
        queued = self.store.queued_turns(session.id)
        return without_none(
            {
                "backend": self.backend,
                "name": session.name,
                "agentName": session.agent_name,
                "threadId": session.id,
                "turnId": session.active_turn_id or session.last_turn_id,
                "cwd": session.cwd,
                "status": session.status,
                "model": session.model,
                "activeTurnId": session.active_turn_id,
                "lastTurnId": session.last_turn_id,
                "lastObservedState": session.last_observed_state,
                "lastUsefulMessage": session.last_useful_message,
                "queueDepth": len(queued),
                "updatedAt": session.updated_at,
            }
        )

    def _turn_view(self, session: Session, turn: Any) -> JsonObject:
        data = turn.to_json()
        if (
            "lastUsefulMessage" not in data
            and turn.id == session.last_turn_id
            and session.last_useful_message
        ):
            data["lastUsefulMessage"] = session.last_useful_message
        return data

    def _sdk_ready(self) -> tuple[bool, str | None]:
        try:
            with _without_unsupported_anthropic_api_keys():
                self._sdk_loader()
        except Exception as exc:
            return False, f"{SDK_IMPORT_ERROR} ({type(exc).__name__}: {exc})"
        return True, None

    def _session_is_busy(self, session: Session) -> bool:
        return bool(session.active_turn_id or session.status == "running")

    def _prompt_for_session(self, session: Session, turn_input: JsonObject) -> str:
        prompt = str(turn_input["prompt"])
        developer_instructions = _combine_developer_instructions(
            session.developer_instructions,
            _optional_str(turn_input.get("developerInstructions")),
        )
        developer_instructions = with_super_agent_identity_instructions(
            developer_instructions,
            session.name,
            session.id,
            session.agent_name,
        )
        return _with_claude_turn_context(
            prompt,
            cwd=session.cwd,
            developer_instructions=developer_instructions,
        )

    def _require_sdk(self) -> Any:
        try:
            with _without_unsupported_anthropic_api_keys():
                return self._sdk_loader()
        except Exception as exc:
            raise RuntimeError(SDK_IMPORT_ERROR) from exc

    def _unsupported(self, tool: str, **extra: Any) -> JsonObject:
        return {
            "backend": self.backend,
            "supported": False,
            "tool": tool,
            "error": f"{tool} is only available through the Codex app-server backend.",
            **{key: value for key, value in extra.items() if value is not None},
        }


def _load_sdk() -> Any:
    return importlib.import_module("claude_agent_sdk")


@contextlib.contextmanager
def _without_unsupported_anthropic_api_keys() -> Iterator[None]:
    previous = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ["ANTHROPIC_API_KEY"] = previous


def _agent_options(
    sdk: Any,
    cwd: str,
    model: str | None,
    reasoning_effort: str | None,
    *,
    resume: str | None,
) -> Any:
    kwargs: JsonObject = {
        "cwd": cwd,
        "permission_mode": CLAUDE_PERMISSION_MODE,
        **_managed_claude_config_options(),
    }
    if model:
        kwargs["model"] = model
    if reasoning_effort:
        kwargs["effort"] = reasoning_effort
    if resume:
        kwargs["resume"] = resume
    return sdk.ClaudeAgentOptions(**kwargs)


def _with_claude_turn_context(
    prompt: str,
    *,
    cwd: str,
    developer_instructions: str | None,
) -> str:
    context_parts = [
        "<openbase-claude-code-context>",
        f"Current working directory: {cwd}",
        (
            "When the user asks you to create or edit files in the current working directory, "
            "interpret that as this directory and prefer relative paths or paths under it."
        ),
    ]
    if developer_instructions:
        context_parts.extend(
            [
                "",
                "Developer instructions for this Openbase thread:",
                developer_instructions.strip(),
            ]
        )
    context_parts.append("</openbase-claude-code-context>")
    return "\n".join(context_parts) + "\n\n" + prompt


def _managed_claude_config_options() -> JsonObject:
    config_dir_value = os.environ.get(CLAUDE_CONFIG_DIR_ENV)
    if not config_dir_value:
        return {}

    config_dir = Path(config_dir_value).expanduser()
    options: JsonObject = {
        "env": {CLAUDE_CONFIG_DIR_ENV: str(config_dir)},
        "setting_sources": ["project"],
    }

    settings_path = config_dir / CLAUDE_SETTINGS_FILENAME
    if settings_path.exists():
        options["settings"] = str(settings_path)

    instructions_path = config_dir / CLAUDE_INSTRUCTIONS_FILENAME
    if instructions_path.exists():
        options["system_prompt"] = {
            "type": "file",
            "path": str(instructions_path),
        }

    mcp_servers = _managed_claude_mcp_servers(config_dir)
    if mcp_servers:
        options["mcp_servers"] = mcp_servers

    return options


def _managed_claude_mcp_servers(config_dir: Path) -> JsonObject | None:
    config_path = config_dir / CLAUDE_CONFIG_FILENAME
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    mcp_servers = payload.get("mcpServers")
    return mcp_servers if isinstance(mcp_servers, dict) and mcp_servers else None


def _optional_str(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _combine_developer_instructions(base: str | None, overlay: str | None) -> str | None:
    parts = [part.strip() for part in (base, overlay) if part and part.strip()]
    if not parts:
        return None
    return "\n\n".join(parts)


def append_log(path_value: str | None, text: str) -> None:
    if not path_value:
        return
    path = Path(path_value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _message_to_log(message: Any) -> str:
    payload = _jsonable_message(message)
    return f"[{iso_now()}] {json.dumps(payload, ensure_ascii=True, sort_keys=True, default=str)}\n"


def _jsonable_message(message: Any) -> Any:
    if dataclasses.is_dataclass(message):
        return dataclasses.asdict(message)
    if isinstance(message, dict):
        return message
    data = {
        key: value
        for key in ("type", "subtype", "result", "session_id", "content")
        if (value := getattr(message, key, None)) is not None
    }
    return data or repr(message)


def _message_preview(message: Any) -> str:
    result = getattr(message, "result", None)
    if isinstance(result, str) and result.strip():
        return result.strip()
    content = getattr(message, "content", None)
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _message_session_id(message: Any) -> str | None:
    session_id = getattr(message, "session_id", None)
    return session_id if isinstance(session_id, str) and session_id else None
