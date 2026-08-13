"""Code-provenance resolution order: env, git, .build-sha file, unknown.

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


def _no_sha(*_args: str) -> str | None:
    return None


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


def test_build_sha_file_used_when_git_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The container path: the image excludes .git, so the COPYed file is all
    there is."""
    _clear_env(monkeypatch)
    sha_file = tmp_path / ".build-sha"
    sha_file.write_text("filecommit\n")
    monkeypatch.setattr(prov, "_BUILD_SHA_FILE", sha_file)
    monkeypatch.setattr(prov, "_git", _no_sha)

    p = prov.code_provenance()
    assert p["code_sha"] == "filecommit"
    assert p["producer"] == "local"
    assert p["dirty"] is None  # no MOMENTARILY_CODE_DIRTY set


def test_live_git_tree_beats_a_leftover_build_sha_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`deploy:ci` writes .build-sha into the checkout and nothing removes it,
    so in a dev tree it is stale from the next commit onward. The tree that
    actually ran the code wins, and it is the only source that can report
    dirty."""
    _clear_env(monkeypatch)
    sha_file = tmp_path / ".build-sha"
    sha_file.write_text("stalecommit\n")
    monkeypatch.setattr(prov, "_BUILD_SHA_FILE", sha_file)

    def fake_git(*args: str) -> str | None:
        if args[:1] == ("rev-parse",):
            return "headcommit"
        return " M training/train_em.py"

    monkeypatch.setattr(prov, "_git", fake_git)
    assert prov.code_provenance() == {
        "code_sha": "headcommit",
        "dirty": True,
        "producer": "local",
    }


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
    monkeypatch.setattr(prov, "_git", _no_sha)
    assert prov.code_provenance() == {
        "code_sha": "unknown",
        "dirty": None,
        "producer": "local",
    }
