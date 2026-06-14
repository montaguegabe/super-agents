from __future__ import annotations

import errno
import os
import select
import shlex
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

from .ansi import TerminalObservation
from .detector import ObservedState, infer_state
from .models import Session, Turn
from .storage import Store
from .timeutil import iso_now


KEYS = {
    "enter": "\r",
    "return": "\r",
    "tab": "\t",
    "escape": "\x1b",
    "esc": "\x1b",
    "ctrl-c": "\x03",
    "ctrl-d": "\x04",
    "up": "\x1b[A",
    "down": "\x1b[B",
    "right": "\x1b[C",
    "left": "\x1b[D",
}


class ClaudeController:
    """Owns one interactive Claude TUI process through a PTY."""

    def __init__(self, store: Store, session: Session) -> None:
        self.store = store
        self.session = session
        self.process: subprocess.Popen[bytes] | None = None
        self.master_fd: int | None = None
        self.observation = TerminalObservation()
        self._reader: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self.last_state = ObservedState("unknown", "not started")

    def start(self) -> ObservedState:
        with self._lock:
            if self.process and self.process.poll() is None:
                return self.observe()
            if not self.session.command:
                return self._fail("No Claude command configured.")
            executable = self.session.command[0]
            if shutil.which(executable) is None and not Path(executable).exists():
                return self._fail(
                    f"Claude command not found: {executable}. Set SUPER_AGENTS_CLAUDE_TUI_CMD or install Claude CLI."
                )
            cwd = Path(self.session.cwd).expanduser()
            if not cwd.exists():
                return self._fail(f"Session cwd does not exist: {cwd}")

            master_fd, slave_fd = os.openpty()
            raw_log = open_path(self.session.raw_log_path, "ab")
            text_log = open_path(self.session.log_path, "a")
            text_log.write(f"\n[{iso_now()}] starting: {shlex.join(self.session.command)}\n")
            text_log.flush()
            try:
                self.process = subprocess.Popen(
                    self.session.command,
                    cwd=str(cwd),
                    stdin=slave_fd,
                    stdout=slave_fd,
                    stderr=slave_fd,
                    start_new_session=True,
                    close_fds=True,
                )
            finally:
                os.close(slave_fd)

            self.master_fd = master_fd
            self._stop.clear()
            self._reader = threading.Thread(
                target=self._read_loop,
                args=(raw_log, text_log),
                name=f"claude-reader-{self.session.name}",
                daemon=True,
            )
            self._reader.start()
            self.session = self.store.update_session(
                self.session.id,
                status="running",
                pid=self.process.pid,
                last_exit_code=None,
                last_observed_state="process running",
            )
            time.sleep(0.05)
            return self.observe()

    def send_prompt(self, prompt: str, *, mode: str | None = None, model: str | None = None) -> Turn:
        state = self.start()
        if state.status == "failed":
            raise RuntimeError(state.detail)
        state = self.wait_for_input(timeout=15.0)
        if state.wants_approval:
            raise RuntimeError(f"Claude is waiting for approval before it can receive a prompt: {state.detail}")
        if not state.wants_input:
            raise RuntimeError(f"Claude did not become ready for input: {state.detail}")
        turn = self.store.create_turn(self.session.id, prompt, status="running", mode=mode, model=model)
        self.write(prompt)
        self.key("enter")
        self.session = self.store.update_session(
            self.session.id,
            status="running",
            active_turn_id=turn.id,
            last_turn_id=turn.id,
        )
        return turn

    def write(self, text: str) -> None:
        if self.master_fd is None:
            raise RuntimeError("Session is not running")
        os.write(self.master_fd, text.encode("utf-8"))

    def key(self, key_name: str) -> None:
        value = KEYS.get(key_name.lower())
        if value is None:
            if len(key_name) == 1:
                value = key_name
            else:
                raise ValueError(f"Unknown key: {key_name}")
        self.write(value)

    def answer_request(self, decision: str) -> None:
        normalized = decision.lower()
        if normalized in {"accept", "approve", "yes", "y"}:
            self.key("y")
            self.key("enter")
            return
        if normalized in {"decline", "deny", "no", "n"}:
            self.key("n")
            self.key("enter")
            return
        if normalized == "cancel":
            self.key("ctrl-c")
            return
        raise ValueError("decision must be accept, decline, or cancel")

    def interrupt(self) -> None:
        active_turn_id = self.session.active_turn_id
        self.key("ctrl-c")
        if active_turn_id:
            self.store.update_turn(active_turn_id, status="cancelled", finished_at=iso_now())
        self.session = self.store.update_session(
            self.session.id,
            status="cancelled",
            active_turn_id=None,
            last_observed_state="interrupt sent",
        )

    def terminate(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=2)
        self._stop.set()
        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None
        self.session = self.store.update_session(self.session.id, status="cancelled", pid=None)

    def observe(self) -> ObservedState:
        running = bool(self.process and self.process.poll() is None)
        state = infer_state(self.observation.text(), process_running=running)
        if self.process and self.process.poll() is not None:
            exit_code = self.process.returncode
            state = ObservedState("completed" if exit_code == 0 else "failed", f"process exited with {exit_code}")
            self.session = self.store.update_session(
                self.session.id,
                status=state.status,
                pid=None,
                active_turn_id=None,
                last_observed_state=state.detail,
                last_exit_code=exit_code,
            )
        else:
            if state.wants_input and self.session.active_turn_id:
                self.store.update_turn(
                    self.session.active_turn_id,
                    status="completed",
                    finished_at=iso_now(),
                )
                self.session = self.store.update_session(self.session.id, active_turn_id=None)
            self.session = self.store.update_session(
                self.session.id,
                status=state.status,
                last_observed_state=state.detail,
            )
        self.last_state = state
        return state

    def wait_for_input(self, timeout: float = 10.0) -> ObservedState:
        deadline = time.time() + timeout
        state = self.observe()
        while time.time() < deadline:
            if state.status == "failed" or state.wants_input or state.wants_approval:
                return state
            time.sleep(0.2)
            state = self.observe()
        return state

    def drain_queued(self) -> Turn | None:
        state = self.observe()
        if state.status not in {"waiting", "completed", "unknown"}:
            return None
        queued = self.store.queued_turns(self.session.id)
        if not queued:
            return None
        turn = queued[0]
        state = self.start()
        if state.status == "failed":
            self.store.update_turn(turn.id, status="failed", last_error=state.detail)
            return None
        state = self.wait_for_input(timeout=2.0)
        if not state.wants_input:
            return None
        self.store.update_turn(turn.id, status="running", attempts=turn.attempts + 1)
        self.write(turn.prompt)
        self.key("enter")
        self.session = self.store.update_session(
            self.session.id,
            status="running",
            active_turn_id=turn.id,
            last_turn_id=turn.id,
        )
        return turn

    def _fail(self, message: str) -> ObservedState:
        append_log(self.session.log_path, f"[{iso_now()}] {message}\n")
        self.session = self.store.update_session(
            self.session.id,
            status="failed",
            pid=None,
            last_observed_state=message,
            last_exit_code=127,
        )
        self.last_state = ObservedState("failed", message)
        return self.last_state

    def _read_loop(self, raw_log, text_log) -> None:  # type: ignore[no-untyped-def]
        assert self.master_fd is not None
        try:
            while not self._stop.is_set():
                try:
                    ready, _, _ = select.select([self.master_fd], [], [], 0.2)
                except (OSError, ValueError):
                    break
                if not ready:
                    if self.process and self.process.poll() is not None:
                        break
                    continue
                try:
                    data = os.read(self.master_fd, 4096)
                except OSError as exc:
                    if exc.errno in {errno.EIO, errno.EBADF}:
                        break
                    raise
                if not data:
                    break
                raw_log.write(data)
                raw_log.flush()
                text = self.observation.feed(data)
                if text:
                    text_log.write(text)
                    text_log.flush()
        finally:
            raw_log.close()
            text_log.close()


class Runtime:
    """In-process manager for all live controllers owned by the TUI."""

    def __init__(self, store: Store | None = None) -> None:
        self.store = store or Store()
        self.controllers: dict[str, ClaudeController] = {}

    def controller_for(self, session: Session) -> ClaudeController:
        existing = self.controllers.get(session.id)
        if existing:
            existing.session = self.store.get_session(session.id)
            return existing
        controller = ClaudeController(self.store, session)
        self.controllers[session.id] = controller
        return controller

    def start(self, session: Session) -> ObservedState:
        return self.controller_for(session).start()

    def send(self, session: Session, prompt: str, *, mode: str | None = None, model: str | None = None) -> Turn:
        return self.controller_for(session).send_prompt(prompt, mode=mode, model=model)

    def key(self, session: Session, key_name: str) -> None:
        self.controller_for(session).key(key_name)

    def answer(self, session: Session, decision: str) -> None:
        self.controller_for(session).answer_request(decision)

    def cancel(self, session: Session) -> None:
        self.controller_for(session).interrupt()

    def shutdown(self) -> None:
        for controller in list(self.controllers.values()):
            if controller.process and controller.process.poll() is None:
                controller.terminate()


def open_path(path: str | None, mode: str):  # type: ignore[no-untyped-def]
    if not path:
        raise RuntimeError("missing log path")
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved.open(mode)


def append_log(path: str | None, text: str) -> None:
    if not path:
        return
    resolved = Path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("a", encoding="utf-8") as handle:
        handle.write(text)
