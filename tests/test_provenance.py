"""Code-provenance resolution order: env, .build-sha file, git, unknown.

No real git/R2 — monkeypatches the env and the build-sha path so each rung of
the resolution chain is exercised in isolation.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import training.provenance as prov

_ENV_KEYS = ("MOMENTARILY_CODE_SHA", "MOMENTARILY_PRODUCER", "MOMENTARILY_CODE_DIRTY")


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in _ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _no_git(*_args: str) -> str | None:
    raise AssertionError("git must not be consulted when a higher rung resolves")


def test_env_sha_is_authoritative(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("MOMENTARILY_CODE_SHA", "abc123")
    monkeypatch.setenv("MOMENTARILY_PRODUCER", "ci")
    monkeypatch.setenv("MOMENTARILY_CODE_DIRTY", "true")
    monkeypatch.setattr(prov, "_git", _no_git)

    assert prov.code_provenance() == {
        "code_sha": "abc123",
        "dirty": True,
        "producer": "ci",
    }


def test_build_sha_file_used_when_env_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)
    sha_file = tmp_path / ".build-sha"
    sha_file.write_text("filecommit\n")
    monkeypatch.setattr(prov, "_BUILD_SHA_FILE", sha_file)
    monkeypatch.setattr(prov, "_git", _no_git)

    p = prov.code_provenance()
    assert p["code_sha"] == "filecommit"
    assert p["producer"] == "local"
    assert p["dirty"] is None  # no MOMENTARILY_CODE_DIRTY set


def test_git_fallback_reports_dirty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setattr(prov, "_BUILD_SHA_FILE", tmp_path / "absent")

    def fake_git(*args: str) -> str | None:
        if args[:1] == ("rev-parse",):
            return "gitsha"
        if args[:1] == ("status",):
            return " M training/eval.py"  # non-empty porcelain == dirty
        return None

    monkeypatch.setattr(prov, "_git", fake_git)
    assert prov.code_provenance() == {
        "code_sha": "gitsha",
        "dirty": True,
        "producer": "local",
    }


def test_git_fallback_clean_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setattr(prov, "_BUILD_SHA_FILE", tmp_path / "absent")

    def fake_git(*args: str) -> str | None:
        return "gitsha" if args[:1] == ("rev-parse",) else ""

    monkeypatch.setattr(prov, "_git", fake_git)
    assert prov.code_provenance()["dirty"] is False


def test_unknown_when_nothing_resolves(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setattr(prov, "_BUILD_SHA_FILE", tmp_path / "absent")

    def no_sha(*_args: str) -> str | None:
        return None

    monkeypatch.setattr(prov, "_git", no_sha)
    assert prov.code_provenance() == {
        "code_sha": "unknown",
        "dirty": None,
        "producer": "local",
    }
