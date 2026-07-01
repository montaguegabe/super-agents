from __future__ import annotations

import json
from pathlib import Path

from super_agents.app_permissions import shared_permission_requests


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
