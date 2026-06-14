import os
import tempfile
import time
import unittest
from pathlib import Path

from super_agents.claude_tui.controller import Runtime
from super_agents.claude_tui.storage import Store


FAKE_CLAUDE = """\
import sys
print("Fake Claude ready >", flush=True)
for line in sys.stdin:
    text = line.strip()
    if text == "exit":
        print("bye", flush=True)
        break
    print("thinking", flush=True)
    print("echo:" + text, flush=True)
    print(">", flush=True)
"""

FAKE_INTERRUPT_CLAUDE = r"""\
import sys
import termios
import tty

fd = sys.stdin.fileno()
old = termios.tcgetattr(fd)
tty.setraw(fd)
try:
    print("Fake Claude ready >", flush=True)
    prompt = ""
    while True:
        ch = sys.stdin.read(1)
        if ch in ("\r", "\n"):
            if prompt:
                print("\nrunning long task", flush=True)
                while True:
                    key = sys.stdin.read(1)
                    if key == "\x03":
                        print("\nINTERRUPTED", flush=True)
                        print(">", flush=True)
                        prompt = ""
                        break
            continue
        prompt += ch
finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, old)
"""


class ControllerTests(unittest.TestCase):
    def test_send_prompt_to_fake_pty(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SUPER_AGENTS_CLAUDE_HOME"] = tmp
            script = Path(tmp) / "fake_claude.py"
            script.write_text(FAKE_CLAUDE, encoding="utf-8")
            store = Store(Path(tmp) / "state.sqlite3")
            session = store.create_session(
                "fake",
                cwd=tmp,
                command=["python3", str(script)],
            )
            runtime = Runtime(store)
            runtime.send(session, "hello")
            time.sleep(0.4)
            refreshed = store.require_by_name("fake")
            log = "\n".join(store.tail_log(refreshed, lines=20))
            self.assertIn("echo:hello", log)
            state = runtime.controller_for(refreshed).observe()
            self.assertIn(state.status, {"running", "waiting"})

    def test_follow_up_steering_uses_same_live_pty(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SUPER_AGENTS_CLAUDE_HOME"] = tmp
            script = Path(tmp) / "fake_claude.py"
            script.write_text(FAKE_CLAUDE, encoding="utf-8")
            store = Store(Path(tmp) / "state.sqlite3")
            session = store.create_session(
                "fake",
                cwd=tmp,
                command=["python3", str(script)],
            )
            runtime = Runtime(store)
            runtime.send(session, "first")
            time.sleep(0.3)
            refreshed = store.require_by_name("fake")
            runtime.send(refreshed, "second follow-up")
            time.sleep(0.3)

            log = "\n".join(store.tail_log(store.require_by_name("fake"), lines=40))
            self.assertIn("echo:first", log)
            self.assertIn("echo:second follow-up", log)
            turns = store.list_turns(session.id, limit=10)
            self.assertEqual(len(turns), 2)

    def test_interrupt_sends_ctrl_c_to_running_pty_and_marks_turn_cancelled(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SUPER_AGENTS_CLAUDE_HOME"] = tmp
            script = Path(tmp) / "fake_interrupt_claude.py"
            script.write_text(FAKE_INTERRUPT_CLAUDE, encoding="utf-8")
            store = Store(Path(tmp) / "state.sqlite3")
            session = store.create_session(
                "fake-interrupt",
                cwd=tmp,
                command=["python3", str(script)],
            )
            runtime = Runtime(store)
            turn = runtime.send(session, "long task")
            wait_for_log(store, store.require_by_name("fake-interrupt"), "running long task")

            runtime.cancel(store.require_by_name("fake-interrupt"))
            refreshed = store.require_by_name("fake-interrupt")
            self.assertEqual(store.get_turn(turn.id).status, "cancelled")
            self.assertIsNone(refreshed.active_turn_id)

            self.assertTrue(wait_for_log(store, refreshed, "INTERRUPTED"))
            state = runtime.controller_for(refreshed).observe()
            self.assertTrue(state.wants_input)
            self.assertEqual(state.status, "waiting")


def wait_for_log(store, session, needle, timeout=2.0):  # type: ignore[no-untyped-def]
    deadline = time.time() + timeout
    while time.time() < deadline:
        refreshed = store.get_session(session.id)
        log = "\n".join(store.tail_log(refreshed, lines=80))
        if needle in log:
            return True
        time.sleep(0.05)
    return False


if __name__ == "__main__":
    unittest.main()
