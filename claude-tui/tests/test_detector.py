import unittest

from super_agents.claude_tui.detector import infer_state


class DetectorTests(unittest.TestCase):
    def test_detects_approval_prompt(self):
        state = infer_state("Do you want to proceed? yes/no", process_running=True)
        self.assertEqual(state.status, "waiting")
        self.assertTrue(state.wants_approval)

    def test_detects_missing_cli(self):
        state = infer_state("zsh: command not found: claude", process_running=False)
        self.assertEqual(state.status, "failed")

    def test_exited_zero_is_completed(self):
        state = infer_state("done", process_running=False)
        self.assertEqual(state.status, "completed")

    def test_final_prompt_wins_over_old_busy_word(self):
        state = infer_state("thinking\necho:first\n>", process_running=True)
        self.assertEqual(state.status, "waiting")
        self.assertTrue(state.wants_input)


if __name__ == "__main__":
    unittest.main()
