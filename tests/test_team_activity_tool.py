import asyncio

from super_agents.local_coder_api import filter_team_activity
from super_agents.mcp_server import _team_activity_handler, build_tools


class _StubClient:
    def __getattr__(self, name):
        async def method(*args, **kwargs):
            return {}

        return method


def test_team_activity_tool_registered():
    tools = build_tools(_StubClient())
    tool = next(t for t in tools if t.name == "team_activity")
    assert tool.annotations["readOnlyHint"] is True
    assert "repo" in tool.input_schema["properties"]


def test_handler_degrades_when_unsupported(monkeypatch):
    async def fake_fetch():
        return {"supported": False, "error": "local server unreachable"}

    monkeypatch.setattr(
        "super_agents.local_coder_api.fetch_team_activity", fake_fetch
    )
    result = asyncio.run(_team_activity_handler({}))
    assert result["available"] is False
    assert "Proceed normally" in result["note"]


def test_filter_by_repo_and_online():
    data = {
        "supported": True,
        "members": [
            {
                "user": {"email": "a@b.c"},
                "online": True,
                "devices": [
                    {
                        "device_id": "d1",
                        "threads": [{"thread_id": "t", "repo": "cli"}],
                        "repos": [{"name": "cli", "changed_files": ["x.py"]}],
                    }
                ],
            },
            {"user": {"email": "off@b.c"}, "online": False, "devices": []},
        ],
    }
    filtered = filter_team_activity(data, repo="cli")
    assert len(filtered["members"]) == 1
    filtered_other = filter_team_activity(data, repo="ios")
    assert filtered_other["members"] == []
    with_offline = filter_team_activity(data, include_offline=True)
    assert len(with_offline["members"]) == 2
