"""ChatGPT-account Codex sign-ins reject the legacy gpt-5.5 pin (HTTP 400)."""

from __future__ import annotations

import json
from pathlib import Path

from super_agents.defaults import (
    CHATGPT_DEFAULT_CODEX_MODEL,
    codex_auth_mode,
    codex_model_for_account,
)


def _codex_home(tmp_path: Path, monkeypatch, auth_mode: str | None) -> None:
    home = tmp_path / "codex-home"
    home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(home))
    if auth_mode is not None:
        (home / "auth.json").write_text(
            json.dumps({"auth_mode": auth_mode, "tokens": {"access_token": "x"}}),
            encoding="utf-8",
        )


def test_codex_auth_mode_reads_only_the_mode(tmp_path: Path, monkeypatch) -> None:
    _codex_home(tmp_path, monkeypatch, "ChatGPT")
    assert codex_auth_mode() == "chatgpt"


def test_codex_auth_mode_without_auth_file(tmp_path: Path, monkeypatch) -> None:
    _codex_home(tmp_path, monkeypatch, None)
    assert codex_auth_mode() is None


def test_chatgpt_account_swaps_the_rejected_pin(tmp_path: Path, monkeypatch) -> None:
    _codex_home(tmp_path, monkeypatch, "chatgpt")
    assert codex_model_for_account("gpt-5.5", backend="codex") == CHATGPT_DEFAULT_CODEX_MODEL
    assert codex_model_for_account("gpt-6-astra", backend="codex") == "gpt-6-astra"
    assert codex_model_for_account(None, backend="codex") is None


def test_api_key_and_unknown_sign_ins_keep_the_pin(tmp_path: Path, monkeypatch) -> None:
    _codex_home(tmp_path, monkeypatch, "apikey")
    assert codex_model_for_account("gpt-5.5", backend="codex") == "gpt-5.5"
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "missing"))
    assert codex_model_for_account("gpt-5.5", backend="codex") == "gpt-5.5"


def test_openbase_cloud_codex_backend_is_untouched(tmp_path: Path, monkeypatch) -> None:
    _codex_home(tmp_path, monkeypatch, "chatgpt")
    assert codex_model_for_account("gpt-5.5", backend="openbase_cloud_codex") == "gpt-5.5"


def test_backend_defaults_to_the_configured_one(tmp_path: Path, monkeypatch) -> None:
    _codex_home(tmp_path, monkeypatch, "chatgpt")
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "claude_code")
    assert codex_model_for_account("gpt-5.5") == "gpt-5.5"
    monkeypatch.setenv("OPENBASE_CODING_BACKEND", "codex")
    assert codex_model_for_account("gpt-5.5") == CHATGPT_DEFAULT_CODEX_MODEL
