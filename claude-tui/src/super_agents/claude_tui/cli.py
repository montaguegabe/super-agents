from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import __version__
from .config import SMOKE_MODELS_TO_AVOID, build_claude_command, recommended_smoke_model
from .models import preview
from .storage import Store, sessions_to_json


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        args.command = "tui"
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="super-agents-claude-tui")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command")

    tui = sub.add_parser("tui", help="open the local Super Agents Claude TUI")
    tui.set_defaults(func=cmd_tui)

    doctor = sub.add_parser("doctor", help="check local runtime requirements")
    doctor.set_defaults(func=cmd_doctor)

    start = sub.add_parser("start", help="create a named Claude agent session")
    start.add_argument("name")
    start.add_argument("--cwd")
    start.add_argument("--agent-name")
    start.add_argument("--model")
    start.set_defaults(func=cmd_start)

    rename = sub.add_parser("rename", help="rename a session")
    rename.add_argument("name")
    rename.add_argument("new_name")
    rename.set_defaults(func=cmd_rename)

    sessions = sub.add_parser("sessions", help="list all local sessions")
    sessions.add_argument("--json", action="store_true")
    sessions.set_defaults(func=cmd_sessions)

    recent = sub.add_parser("recent", help="list recent local sessions")
    recent.add_argument("--json", action="store_true")
    recent.set_defaults(func=cmd_sessions)

    active = sub.add_parser("active", help="list active or waiting sessions")
    active.add_argument("--json", action="store_true")
    active.set_defaults(func=cmd_active)

    status = sub.add_parser("status", help="compact status for sessions")
    status.add_argument("name", nargs="?")
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=cmd_status)

    read = sub.add_parser("read", help="read one session log and recent turns")
    read.add_argument("name")
    read.add_argument("--lines", type=int, default=80)
    read.add_argument("--json", action="store_true")
    read.set_defaults(func=cmd_read)

    queue = sub.add_parser("queue-turn", help="queue a prompt for the TUI runtime to send when the session is idle")
    queue.add_argument("name")
    queue.add_argument("prompt")
    queue.add_argument("--mode", choices=["default", "plan"], default="default")
    queue.add_argument("--model")
    queue.set_defaults(func=cmd_queue_turn)

    progress = sub.add_parser("progress", help="show compact progress for one session")
    progress.add_argument("name")
    progress.add_argument("--json", action="store_true")
    progress.set_defaults(func=cmd_progress)

    send = sub.add_parser("send", help="start/send from a foreground one-shot PTY controller")
    send.add_argument("name")
    send.add_argument("prompt")
    send.add_argument("--wait", type=float, default=2.0)
    send.add_argument("--model")
    send.set_defaults(func=cmd_send)

    steer = sub.add_parser("steer", help="alias for send")
    steer.add_argument("name")
    steer.add_argument("prompt")
    steer.add_argument("--wait", type=float, default=2.0)
    steer.set_defaults(func=cmd_send)

    cancel = sub.add_parser("cancel", help="mark a session cancelled; live cancellation is available inside the TUI")
    cancel.add_argument("name")
    cancel.set_defaults(func=cmd_cancel)

    answer = sub.add_parser("answer-request", help="record an intended approval decision for a session")
    answer.add_argument("name")
    answer.add_argument("decision", choices=["accept", "decline", "cancel"])
    answer.set_defaults(func=cmd_answer_note)

    return parser


def cmd_tui(_args: argparse.Namespace) -> int:
    from .tui import run

    run()
    return 0


def cmd_doctor(_args: argparse.Namespace) -> int:
    command = build_claude_command()
    executable = command[0] if command else "claude"
    payload = {
        "python": sys.version.split()[0],
        "claudeCommand": command,
        "claudeFound": bool(shutil.which(executable) or Path(executable).exists()),
        "dataStore": str(Store().path),
        "recommendedSmokeModel": recommended_smoke_model(),
        "avoidForSmoke": list(SMOKE_MODELS_TO_AVOID),
    }
    print(json.dumps(payload, indent=2))
    if not payload["claudeFound"]:
        print("Claude CLI is not available; install it or set SUPER_AGENTS_CLAUDE_TUI_CMD.", file=sys.stderr)
        return 1
    return 0


def cmd_start(args: argparse.Namespace) -> int:
    store = Store()
    session = store.create_session(args.name, cwd=args.cwd, agent_name=args.agent_name, model=args.model)
    print(json.dumps(session.to_json(), indent=2))
    return 0


def cmd_rename(args: argparse.Namespace) -> int:
    store = Store()
    session = store.require_by_name(args.name)
    renamed = store.rename_session(session.id, args.new_name)
    print(json.dumps(renamed.to_json(), indent=2))
    return 0


def cmd_sessions(args: argparse.Namespace) -> int:
    store = Store()
    sessions = store.list_sessions()
    if args.json:
        print(json.dumps(sessions_to_json(sessions), indent=2))
        return 0
    print_sessions(sessions)
    return 0


def cmd_active(args: argparse.Namespace) -> int:
    store = Store()
    sessions = store.list_sessions(include_inactive=False)
    if args.json:
        print(json.dumps(sessions_to_json(sessions), indent=2))
        return 0
    print_sessions(sessions)
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    store = Store()
    sessions = [store.require_by_name(args.name)] if args.name else store.list_sessions(include_inactive=False)
    items = [compact_status(store, session) for session in sessions]
    if args.json:
        print(json.dumps(items, indent=2))
        return 0
    for item in items:
        print(f"{item['name']}: {item['status']} - {item.get('lastObservedState', 'unknown')}")
    return 0


def cmd_read(args: argparse.Namespace) -> int:
    store = Store()
    session = store.require_by_name(args.name)
    turns = store.list_turns(session.id, limit=20)
    log_tail = store.tail_log(session, lines=args.lines)
    if args.json:
        print(
            json.dumps(
                {
                    "session": session.to_json(),
                    "turns": [turn.to_json() for turn in turns],
                    "logTail": log_tail,
                },
                indent=2,
            )
        )
        return 0
    print(json.dumps(session.to_json(), indent=2))
    if turns:
        print("\nRecent turns:")
        for turn in turns:
            print(f"{turn.id} {turn.status}: {preview(turn.prompt)}")
    if log_tail:
        print("\nLog tail:")
        print("\n".join(log_tail))
    return 0


def cmd_queue_turn(args: argparse.Namespace) -> int:
    store = Store()
    session = store.require_by_name(args.name)
    turn = store.create_turn(session.id, args.prompt, status="queued", mode=args.mode, model=args.model)
    print(json.dumps({"queued": True, "turn": turn.to_json()}, indent=2))
    return 0


def cmd_progress(args: argparse.Namespace) -> int:
    store = Store()
    session = store.require_by_name(args.name)
    item = compact_status(store, session)
    if args.json:
        print(json.dumps(item, indent=2))
    else:
        print(f"{item['name']}: {item['status']} - {item.get('lastObservedState', 'unknown')}")
        if item.get("lastUsefulMessage"):
            print(item["lastUsefulMessage"])
    return 0


def cmd_send(args: argparse.Namespace) -> int:
    import time

    from .controller import Runtime

    store = Store()
    session = store.require_by_name(args.name)
    runtime = Runtime(store)
    turn = runtime.send(session, args.prompt, model=getattr(args, "model", None))
    if args.wait:
        time.sleep(args.wait)
    controller = runtime.controller_for(session)
    state = controller.observe()
    print(json.dumps({"turn": turn.to_json(), "observed": state.__dict__}, indent=2))
    runtime.shutdown()
    return 0


def cmd_cancel(args: argparse.Namespace) -> int:
    store = Store()
    session = store.require_by_name(args.name)
    updated = store.update_session(session.id, status="cancelled", active_turn_id=None)
    print(json.dumps(updated.to_json(), indent=2))
    return 0


def cmd_answer_note(args: argparse.Namespace) -> int:
    store = Store()
    session = store.require_by_name(args.name)
    store.create_turn(
        session.id,
        f"answer_request decision={args.decision}",
        status="queued",
        mode="approval",
    )
    print(json.dumps({"queued": True, "decision": args.decision, "session": session.name}, indent=2))
    return 0


def compact_status(store: Store, session) -> dict[str, object]:  # type: ignore[no-untyped-def]
    queued = store.queued_turns(session.id)
    return {
        "id": session.id,
        "name": session.name,
        "cwd": session.cwd,
        "status": session.status,
        "pid": session.pid,
        "activeTurnId": session.active_turn_id,
        "lastTurnId": session.last_turn_id,
        "lastObservedState": session.last_observed_state,
        "lastUsefulMessage": session.last_useful_message,
        "queueDepth": len(queued),
        "updatedAt": session.updated_at,
    }


def print_sessions(sessions) -> None:  # type: ignore[no-untyped-def]
    if not sessions:
        print("No sessions.")
        return
    for session in sessions:
        print(f"{session.name:24} {session.status:10} {session.cwd}")


if __name__ == "__main__":
    raise SystemExit(main())
