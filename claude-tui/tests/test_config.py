import os
import unittest

from super_agents.claude_tui.config import DEFAULT_SMOKE_MODEL, recommended_smoke_model


class ConfigTests(unittest.TestCase):
    def test_default_smoke_model_is_not_fable(self):
        old = os.environ.pop("SUPER_AGENTS_CLAUDE_SMOKE_MODEL", None)
        try:
            self.assertEqual(recommended_smoke_model(), DEFAULT_SMOKE_MODEL)
            self.assertNotEqual(recommended_smoke_model().lower(), "fable")
        finally:
            if old is not None:
                os.environ["SUPER_AGENTS_CLAUDE_SMOKE_MODEL"] = old


if __name__ == "__main__":
    unittest.main()
