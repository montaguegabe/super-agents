from __future__ import annotations

import json
from pathlib import Path

from super_agents.state import read_state_file, update_state_file


def test_state_updates_preserve_unknown_top_level_fields(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    path.write_text(
        json.dumps(
            {
                "sessions": {},
                "futureFeature": {"schema": 2, "items": ["keep-me"]},
            }
        ),
        encoding="utf-8",
    )

    state = read_state_file(path)
    assert state.extra_fields["futureFeature"] == {
        "schema": 2,
        "items": ["keep-me"],
    }

    update_state_file(path, lambda current: current.sessions.clear())

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted["futureFeature"] == {"schema": 2, "items": ["keep-me"]}
