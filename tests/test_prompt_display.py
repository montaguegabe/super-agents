"""Read-only presentation of optional integration envelopes."""

import pytest

from super_agents.claude_prompts import user_prompt_for_display, user_prompt_for_title


@pytest.mark.parametrize(
    ("raw", "shown"),
    [
        ("[Openbase system note: onboarding is pending [agent only].]\n\n<voice>Tic tac toe</voice>", "Tic tac toe"),
        ("<openbase-claude-code-context>cwd</openbase-claude-code-context>\n"
         "[Openbase system note: hidden]\n<voice>Tic tac toe</voice>\n"
         "<system-reminder>hidden</system-reminder>", "Tic tac toe"),
        ("<system-reminder>hidden</system-reminder>\nplain request", "plain request"),
        ("plain request\n[Openbase system note: hidden]", "plain request"),
        ("<voice>  compare &lt;old&gt; &amp; &amp;lt;new&amp;gt;\nnext  </voice>",
         "  compare <old> & &lt;new&gt;\nnext  "),
        ("[Openbase system note: truncated", ""),
        ("<openbase-claude-code-context>truncated", ""),
        ("<system-reminder>truncated", ""),
        ("<voice>ship it</voice>\n<system-reminder>truncated", "ship it"),
        ("  plain request\nnext  ", "  plain request\nnext  "),
        ("Document <voice> tags and [Openbase system note: examples]", "Document <voice> tags and [Openbase system note: examples]"),
        ("<voice>nested</voice><voice>tag</voice>", "<voice>nested</voice><voice>tag</voice>"),
        ("<example>keep markup</example>", "<example>keep markup</example>"),
    ],
)
def test_user_prompt_display(raw: str, shown: str) -> None:
    assert user_prompt_for_display(raw) == shown
    assert user_prompt_for_title(raw) == shown.strip()


def test_title_trims_transport_boundary_whitespace() -> None:
    assert user_prompt_for_title("  <voice>Tic tac toe</voice> ") == "Tic tac toe"
