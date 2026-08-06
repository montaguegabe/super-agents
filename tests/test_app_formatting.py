from super_agents.app_formatting import find_turn_useful_text, find_useful_text, turn_text_preview


def test_find_useful_text_ignores_metadata_only_agent_message() -> None:
    assert (
        find_useful_text(
            {
                "type": "agentMessage",
                "id": "item-1",
                "role": "assistant",
                "phase": "final",
                "status": "completed",
            }
        )
        is None
    )


def test_find_useful_text_ignores_created_at_metadata() -> None:
    assert (
        find_useful_text(
            {
                "turnId": "turn-1",
                "status": "cancelled",
                "createdAt": "2026-08-05T22:55:38.399Z",
                "updatedAt": "2026-08-05T22:55:38.399Z",
            }
        )
        is None
    )


def test_find_useful_text_skips_metadata_then_uses_real_text() -> None:
    assert (
        find_useful_text(
            [
                {
                    "type": "agentMessage",
                    "id": "item-1",
                    "phase": "final",
                },
                {
                    "type": "agentMessage",
                    "id": "item-2",
                    "text": "Here is the useful answer.",
                },
            ]
        )
        == "Here is the useful answer."
    )


def test_turn_text_preview_prefers_short_assistant_answer_over_user_prompt() -> None:
    turn = {
        "items": [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "What is the capital of Alaska?",
                        }
                    ],
                },
            },
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_text",
                            "text": "Juneau.",
                        }
                    ],
                },
            },
            {
                "type": "event_msg",
                "payload": {
                    "type": "task_complete",
                    "last_agent_message": "Juneau.",
                },
            },
        ]
    }

    assert find_turn_useful_text(turn) == "Juneau."
    assert turn_text_preview(turn) == "Juneau."


def test_turn_text_preview_does_not_return_user_only_prompt() -> None:
    turn = {
        "items": [
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Your answer was not actually spoken.",
                        }
                    ],
                },
            }
        ]
    }

    assert find_turn_useful_text(turn) is None
    assert turn_text_preview(turn) is None
