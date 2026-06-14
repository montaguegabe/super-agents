from __future__ import annotations

import curses
import time
from pathlib import Path

from .controller import Runtime
from .models import Session
from .storage import Store


HELP = "n new | l launch | p prompt | x key | a approve | d deny | c ctrl-c | r refresh | q quit"


class TuiApp:
    def __init__(self, stdscr) -> None:  # type: ignore[no-untyped-def]
        self.stdscr = stdscr
        self.store = Store()
        self.runtime = Runtime(self.store)
        self.sessions: list[Session] = []
        self.selected = 0
        self.message = ""

    def run(self) -> None:
        curses.curs_set(0)
        self.stdscr.nodelay(True)
        self.stdscr.keypad(True)
        self.refresh_sessions()
        last_tick = 0.0
        while True:
            now = time.time()
            if now - last_tick > 0.5:
                self.tick()
                last_tick = now
            self.draw()
            key = self.stdscr.getch()
            if key == -1:
                time.sleep(0.05)
                continue
            if not self.handle_key(key):
                break
        self.runtime.shutdown()

    def tick(self) -> None:
        for controller in list(self.runtime.controllers.values()):
            controller.observe()
            try:
                controller.drain_queued()
            except Exception as exc:
                self.message = f"queue error: {exc}"
        self.refresh_sessions(keep_selection=True)

    def refresh_sessions(self, keep_selection: bool = False) -> None:
        current_id = self.current().id if keep_selection and self.current() else None
        self.sessions = self.store.list_sessions()
        if current_id:
            for index, session in enumerate(self.sessions):
                if session.id == current_id:
                    self.selected = index
                    break
        self.selected = max(0, min(self.selected, max(0, len(self.sessions) - 1)))

    def current(self) -> Session | None:
        if not self.sessions:
            return None
        return self.sessions[self.selected]

    def handle_key(self, key: int) -> bool:
        if key in {ord("q"), 27}:
            return False
        if key in {curses.KEY_UP, ord("k")}:
            self.selected = max(0, self.selected - 1)
        elif key in {curses.KEY_DOWN, ord("j")}:
            self.selected = min(max(0, len(self.sessions) - 1), self.selected + 1)
        elif key == ord("r"):
            self.refresh_sessions()
        elif key == ord("n"):
            self.new_session()
        elif key == ord("l"):
            self.launch_current()
        elif key == ord("p"):
            self.prompt_current()
        elif key == ord("x"):
            self.key_current()
        elif key == ord("a"):
            self.answer_current("accept")
        elif key == ord("d"):
            self.answer_current("decline")
        elif key == ord("c"):
            self.cancel_current()
        return True

    def new_session(self) -> None:
        name = self.input("session name")
        if not name:
            return
        cwd = self.input("cwd", str(Path.cwd())) or str(Path.cwd())
        try:
            self.store.create_session(name, cwd=cwd)
            self.message = f"created {name}"
            self.refresh_sessions()
        except Exception as exc:
            self.message = str(exc)

    def launch_current(self) -> None:
        session = self.current()
        if not session:
            return
        state = self.runtime.start(session)
        self.message = f"{session.name}: {state.status} - {state.detail}"

    def prompt_current(self) -> None:
        session = self.current()
        if not session:
            return
        prompt = self.input("prompt")
        if not prompt:
            return
        try:
            turn = self.runtime.send(session, prompt)
            self.message = f"sent {turn.id}"
        except Exception as exc:
            self.message = str(exc)

    def key_current(self) -> None:
        session = self.current()
        if not session:
            return
        key = self.input("key", "enter")
        if not key:
            return
        try:
            self.runtime.key(session, key)
            self.message = f"sent key {key}"
        except Exception as exc:
            self.message = str(exc)

    def answer_current(self, decision: str) -> None:
        session = self.current()
        if not session:
            return
        try:
            self.runtime.answer(session, decision)
            self.message = f"answered {decision}"
        except Exception as exc:
            self.message = str(exc)

    def cancel_current(self) -> None:
        session = self.current()
        if not session:
            return
        try:
            self.runtime.cancel(session)
            self.message = "interrupt sent"
        except Exception as exc:
            self.message = str(exc)

    def draw(self) -> None:
        self.stdscr.erase()
        height, width = self.stdscr.getmaxyx()
        left_width = min(36, max(24, width // 3))
        self.safe_addstr(0, 0, "Super Agents Claude", curses.A_BOLD)
        self.safe_addstr(1, 0, HELP[: width - 1])
        self.safe_addstr(2, 0, self.message[: width - 1], curses.A_DIM)
        self.draw_sessions(4, 0, height - 5, left_width)
        self.draw_detail(4, left_width + 1, height - 5, width - left_width - 2)
        self.stdscr.refresh()

    def draw_sessions(self, top: int, left: int, height: int, width: int) -> None:
        self.safe_addstr(top, left, "Sessions", curses.A_BOLD)
        for row, session in enumerate(self.sessions[: max(0, height - 1)], start=top + 1):
            marker = ">" if row - top - 1 == self.selected else " "
            text = f"{marker} {session.name[:14]:14} {session.status[:8]:8}"
            attr = curses.A_REVERSE if marker == ">" else curses.A_NORMAL
            self.safe_addstr(row, left, text[: width - 1], attr)

    def draw_detail(self, top: int, left: int, height: int, width: int) -> None:
        session = self.current()
        if not session:
            self.safe_addstr(top, left, "No session. Press n to create one.")
            return
        lines = [
            f"{session.name} [{session.status}]",
            f"cwd: {session.cwd}",
            f"state: {session.last_observed_state or 'unknown'}",
            f"pid: {session.pid or '-'}",
            "",
        ]
        lines.extend(self.store.tail_log(session, lines=max(1, height - len(lines) - 1)))
        for offset, line in enumerate(lines[:height]):
            self.safe_addstr(top + offset, left, line[: width - 1])

    def input(self, label: str, default: str = "") -> str:
        curses.echo()
        self.stdscr.nodelay(False)
        height, width = self.stdscr.getmaxyx()
        prompt = f"{label}"
        if default:
            prompt += f" [{default}]"
        prompt += ": "
        self.safe_addstr(height - 1, 0, " " * (width - 1))
        self.safe_addstr(height - 1, 0, prompt[: width - 1])
        curses.curs_set(1)
        try:
            raw = self.stdscr.getstr(height - 1, min(len(prompt), width - 2), max(1, width - len(prompt) - 1))
        finally:
            curses.curs_set(0)
            curses.noecho()
            self.stdscr.nodelay(True)
        value = raw.decode("utf-8", errors="replace").strip()
        return value or default

    def safe_addstr(self, y: int, x: int, text: str, attr: int = curses.A_NORMAL) -> None:
        height, width = self.stdscr.getmaxyx()
        if y < 0 or y >= height or x < 0 or x >= width:
            return
        try:
            self.stdscr.addstr(y, x, text[: max(0, width - x - 1)], attr)
        except curses.error:
            pass


def run() -> None:
    curses.wrapper(lambda stdscr: TuiApp(stdscr).run())
