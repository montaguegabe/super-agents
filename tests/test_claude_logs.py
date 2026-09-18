from claude_agent_sdk import AssistantMessage, TextBlock, UserMessage

from super_agents.claude_logs import message_preview


def test_skill_result_cannot_replace_latest_assistant_answer():
    reply = AssistantMessage(content=[TextBlock(text="Both agents are working.")], model="test")
    skill = UserMessage(content=[TextBlock(text="Internal skill instructions and paths")],
                        tool_use_result={"success": True})
    latest = ""
    for message in (reply, skill):
        latest = message_preview(message) or latest
    assert latest == "Both agents are working."


def test_user_text_blocks_are_not_assistant_progress():
    assert message_preview(UserMessage(content=[TextBlock(text="User request")])) == ""


def test_nested_agent_output_waits_for_the_parent_answer():
    nested = AssistantMessage(content=[TextBlock(text="Internal delegated analysis")],
                              model="test", parent_tool_use_id="child-tool")
    assert message_preview(nested) == ""
    assert message_preview(AssistantMessage(content=[TextBlock(text="Here is the result.")],
                                           model="test")) == "Here is the result."
