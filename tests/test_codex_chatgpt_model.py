"""ChatGPT-account Codex sign-ins reject the legacy gpt-5.5 pin (HTTP 400)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from super_agents.app_client_turns import TurnLifecycleMixin
from super_agents.defaults import (
    account_default_model,
    codex_auth_mode,
    codex_model_rejected_by_account,
)

MODEL_LIST = {
    "data": [
        {"id": "gpt-6-luna", "isDefault": True, "displayName": "GPT-6-Luna"},
        {"id": "gpt-5.6-terra", "displayName": "GPT-5.6-Terra"},
    ],
    "nextCursor": None,
}


def _codex_home(tmp_path: Path, monkeypatch, auth_mode: str | None) -> None:
    home = tmp_path / "codex-home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("CODEX_HOME", str(home))
    if auth_mode is None:
        (home / "auth.json").unlink(missing_ok=True)
    else:
        (home / "auth.json").write_text(
            json.dumps({"auth_mode": auth_mode, "tokens": {"access_token": "x"}}),
            encoding="utf-8",
        )


def test_codex_auth_mode_reads_only_the_mode(tmp_path: Path, monkeypatch) -> None:
    _codex_home(tmp_path, monkeypatch, "ChatGPT")
    assert codex_auth_mode() == "chatgpt"
    _codex_home(tmp_path, monkeypatch, None)
    assert codex_auth_mode() is None


def test_only_chatgpt_sign_ins_on_the_codex_backend_reject_the_pin(tmp_path: Path, monkeypatch) -> None:
    _codex_home(tmp_path, monkeypatch, "chatgpt")
    assert codex_model_rejected_by_account("gpt-5.5", backend="codex")
    assert not codex_model_rejected_by_account("gpt-6-astra", backend="codex")
    assert not codex_model_rejected_by_account(None, backend="codex")
    assert not codex_model_rejected_by_account("gpt-5.5", backend="openbase_cloud_codex")
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")
    assert not codex_model_rejected_by_account("gpt-5.5")
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")
    assert codex_model_rejected_by_account("gpt-5.5")
    _codex_home(tmp_path, monkeypatch, "apikey")
    assert not codex_model_rejected_by_account("gpt-5.5", backend="codex")
    _codex_home(tmp_path, monkeypatch, None)
    assert not codex_model_rejected_by_account("gpt-5.5", backend="codex")


def test_account_default_model_prefers_the_flagged_entry() -> None:
    assert account_default_model(MODEL_LIST) == "gpt-6-luna"
    assert account_default_model({"data": [{"id": "gpt-5.6-terra"}, {"id": "gpt-6-luna"}]}) == "gpt-5.6-terra"
    assert account_default_model({"data": []}) is None
    assert account_default_model({}) is None


class _Client(TurnLifecycleMixin):
    backend = "codex"

    def __init__(self, model_list: dict) -> None:
        self.calls: list[str] = []
        self._model_list = model_list

    async def request(self, method: str, params=None, timeout_seconds: float = 30, *, context=None):
        self.calls.append(method)
        return self._model_list


@pytest.mark.asyncio
async def test_chatgpt_client_swaps_the_pin_for_the_listed_default(tmp_path: Path, monkeypatch) -> None:
    _codex_home(tmp_path, monkeypatch, "chatgpt")
    client = _Client(MODEL_LIST)
    assert await client.codex_model_for_account("gpt-5.5") == "gpt-6-luna"
    assert await client.codex_model_for_account("gpt-5.5") == "gpt-6-luna"
    assert client.calls == ["model/list"]
    assert await client.codex_model_for_account("gpt-6-astra") == "gpt-6-astra"


@pytest.mark.asyncio
async def test_pin_is_kept_when_the_list_offers_nothing_or_the_account_accepts_it(tmp_path: Path, monkeypatch) -> None:
    _codex_home(tmp_path, monkeypatch, "chatgpt")
    assert await _Client({"data": []}).codex_model_for_account("gpt-5.5") == "gpt-5.5"
    _codex_home(tmp_path, monkeypatch, "apikey")
    client = _Client(MODEL_LIST)
    assert await client.codex_model_for_account("gpt-5.5") == "gpt-5.5"
    assert client.calls == []
