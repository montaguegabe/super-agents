from __future__ import annotations

import json
import stat
from pathlib import Path

from super_agents.app_permissions import (
    normalize_permission_response,
    pop_shared_permission_decision,
    shared_permission_requests,
    write_permission_store,
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
    assert pop_shared_permission_decision("approval-1", approvals_path) == {"decision": "decline"}


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


def test_permission_store_has_restrictive_permissions(tmp_path: Path) -> None:
    store_dir = tmp_path / "approval-store"
    approvals_path = store_dir / "approvals.json"

    write_permission_store(approvals_path, {"requests": {}, "decisions": {}})

    assert stat.S_IMODE(store_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(approvals_path.stat().st_mode) == 0o600


def test_concurrent_store_mutations_do_not_lose_decisions(tmp_path: Path) -> None:
    """Unlocked read-modify-write cycles clobbered concurrently written
    decisions, stranding answered approvals; the store lock must serialize
    mutators across threads (and, via flock, across processes)."""
    import threading

    from super_agents.app_models import PendingServerRequest
    from super_agents.app_permissions import (
        record_shared_permission_request,
        write_shared_permission_decision,
    )

    approvals_path = tmp_path / "approvals.json"
    seeded = [f"seeded-{index}" for index in range(20)]
    for request_id in seeded:
        record_shared_permission_request(
            PendingServerRequest(
                id=request_id,
                method="exec/requestApproval",
                params={},
                received_at="2026-06-29T00:00:00.000Z",
            ),
            approvals_path,
        )

    def decide() -> None:
        for request_id in seeded:
            assert write_shared_permission_decision(request_id, "accept", approvals_path)

    def churn() -> None:
        for index in range(20):
            record_shared_permission_request(
                PendingServerRequest(
                    id=f"churn-{index}",
                    method="exec/requestApproval",
                    params={},
                    received_at="2026-06-29T00:00:00.000Z",
                ),
                approvals_path,
            )

    threads = [threading.Thread(target=decide), threading.Thread(target=churn)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    store = json.loads(approvals_path.read_text(encoding="utf-8"))
    assert set(store["decisions"]) == set(seeded)
    assert set(store["requests"]) == set(seeded) | {f"churn-{index}" for index in range(20)}
