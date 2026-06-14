import os
import tempfile
import unittest
from pathlib import Path

from super_agents.claude_tui.storage import Store


class StorageTests(unittest.TestCase):
    def test_create_session_and_queue_turn(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["SUPER_AGENTS_CLAUDE_HOME"] = tmp
            store = Store(Path(tmp) / "state.sqlite3")
            session = store.create_session("demo", cwd=tmp, command=["python3"])
            self.assertEqual(session.name, "demo")
            self.assertEqual(session.command, ["python3"])

            turn = store.create_turn(session.id, "hello", status="queued")
            queued = store.queued_turns(session.id)
            self.assertEqual([item.id for item in queued], [turn.id])

            fetched = store.require_by_name("demo")
            self.assertEqual(fetched.last_turn_id, turn.id)
            self.assertIsNone(fetched.active_turn_id)


if __name__ == "__main__":
    unittest.main()
