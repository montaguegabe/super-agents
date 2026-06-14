import unittest

from super_agents.claude_tui.ansi import strip_ansi


class AnsiTests(unittest.TestCase):
    def test_strips_full_csi_sequence(self):
        self.assertEqual(strip_ansi("\x1b[6AHello\x1b[15GWorld"), "HelloWorld")

    def test_strips_title_and_save_restore(self):
        self.assertEqual(strip_ansi("\x1b7\x1b]0;title\x07Ready\x1b8"), "Ready")


if __name__ == "__main__":
    unittest.main()

