from __future__ import annotations

import json
from pathlib import Path

from super_agents.app_permissions import (
    normalize_permission_response,
    pop_shared_permission_decision,
    shared_permission_requests,
)


def test_shared_permission_requests_excludes_decided_requests(tmp_path: Path) -> None:
    approvals_path = tmp_path / "approvals.json"
    approvals_path.write_text(
        json.dumps(
            {
                "requests": {
                    "approval-1": {
                        "id": "approval-1",
                        "method": "exec/requestApproval",
                        "params": {},
                    },
                    "approval-2": {
                        "id": "approval-2",
                        "method": "mcpServer/elicitation/request",
                        "params": {},
                    },
                },
                "decisions": {
                    "approval-1": {
                        "decision": "accept",
                        "decidedAt": "2026-06-29T00:00:00.000Z",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    requests = shared_permission_requests(approvals_path)

    assert [request["id"] for request in requests] == ["approval-2"]


def test_pop_shared_permission_decision_returns_elicitation_action(tmp_path: Path) -> None:
    approvals_path = tmp_path / "approvals.json"
    approvals_path.write_text(
        json.dumps(
            {
                "requests": {
                    "elicitation-1": {
                        "id": "elicitation-1",
                        "method": "mcpServer/elicitation/request",
                        "params": {},
                    },
                    "approval-1": {
                        "id": "approval-1",
                        "method": "exec/requestApproval",
                        "params": {},
                    },
                },
                "decisions": {
                    "elicitation-1": {
                        "decision": "accept",
                        "decidedAt": "2026-06-29T00:00:00.000Z",
                    },
                    "approval-1": {
                        "decision": "decline",
                        "decidedAt": "2026-06-29T00:00:00.000Z",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    assert pop_shared_permission_decision("elicitation-1", approvals_path) == {
        "action": "accept",
        "content": None,
        "_meta": None,
    }
    assert pop_shared_permission_decision("approval-1", approvals_path) == {
        "decision": "decline"
    }


def test_normalize_permission_response_converts_elicitation_decision() -> None:
    request = {
        "id": "elicitation-1",
        "method": "mcpServer/elicitation/request",
        "params": {},
    }

    assert normalize_permission_response(request, {"decision": "accept"}) == {
        "action": "accept",
        "content": None,
        "_meta": None,
    }
    assert normalize_permission_response(
        {"id": "approval-1", "method": "exec/requestApproval", "params": {}},
        {"decision": "accept"},
    ) == {"decision": "accept"}
