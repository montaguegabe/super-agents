import unittest

from super_agents.claude_tui.detector import infer_state


class RealLogDetectionTests(unittest.TestCase):
    def test_detects_completed_claude_screen_as_waiting(self):
        text = """
        ⏺SAC_SMOKE_OK
        ✢ Deliberating…
        ❯
        (3s·↓1tokens)
        ✻Crunched for 3s
        ← for agents
        """
        state = infer_state(text, process_running=True)
        self.assertEqual(state.status, "waiting")
        self.assertTrue(state.wants_input)

    def test_detects_alternate_completion_footer(self):
        text = """
        ⏺SAC_SMOKE_OK_2
        ✢ Honking…
        ❯
        ✻Sautéed for5s
        ← for agents
        """
        state = infer_state(text, process_running=True)
        self.assertEqual(state.status, "waiting")
        self.assertTrue(state.wants_input)

    def test_detects_overwritten_footer_with_agent_return(self):
        text = """
        ⏺SAC_SMOKE_OK_3
        ✽ Canoodling… (4s · ↓ 1 tokens)
        ❯
        ✻runched for
        ← for agents
        """
        state = infer_state(text, process_running=True)
        self.assertEqual(state.status, "waiting")
        self.assertTrue(state.wants_input)


if __name__ == "__main__":
    unittest.main()
