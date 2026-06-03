"""Tests for the local data collector — focused on freshness honesty."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import pytest

if TYPE_CHECKING:
    # At runtime ``collector/collector.py`` is importable as the top-level module
    # ``collector`` (see ``pythonpath`` in pyproject). For static analysis the
    # same file is reachable as the ``collector.collector`` submodule, which is
    # what gives pyright the real, fully-typed symbols.
    from collector import collector
else:
    import collector


@pytest.fixture(autouse=True)
def redirect_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the collector at a per-test temp directory (applied to every test)."""
    alerts = tmp_path / "alerts"
    ene = tmp_path / "ene"
    meta = tmp_path / "meta"
    for d in (alerts, ene, meta):
        d.mkdir(parents=True)
    monkeypatch.setattr(collector, "DATA_DIR", tmp_path)
    monkeypatch.setattr(collector, "ALERTS_DIR", alerts)
    monkeypatch.setattr(collector, "ENE_DIR", ene)
    monkeypatch.setattr(collector, "META_DIR", meta)
    monkeypatch.setattr(collector, "LAST_FETCHED_PATH", meta / "last_fetched.json")
    monkeypatch.setattr(collector, "POLL_LOG_PATH", meta / "poll_log.jsonl")


def _install_fetch(monkeypatch: pytest.MonkeyPatch, responses: dict[str, Any]) -> None:
    """Replace collector._fetch with a per-URL response map.

    A value that is a callable will be invoked (to raise) instead of returned.
    """

    def fake_fetch(url: str) -> Any:
        if url not in responses:
            raise AssertionError(f"unexpected fetch URL: {url}")
        value = responses[url]
        if callable(value):
            return value()
        return value

    monkeypatch.setattr(collector, "_fetch", fake_fetch)


def _raise_http_error() -> Any:
    raise httpx.ConnectError("simulated network failure")


def _today() -> str:
    """The collector's UTC date stamp, used to locate the daily output file.

    ``_utc_today`` is private to the collector but tests must derive the same
    filename the collector writes to, so we reach in deliberately here.
    """
    return collector._utc_today()  # pyright: ignore[reportPrivateUsage]


def test_v1_ene_sources_includes_all_three_feeds() -> None:
    """The explicit v1 source list is what the README and acceptance criteria promise."""
    sources = {source for _url, source in collector.ENE_SOURCES}
    assert sources == {
        "nyct/nyct_ene.json",
        "nyct/nyct_ene_upcoming.json",
        "nyct/nyct_ene_equipments.json",
    }


def test_ene_full_success_advances_freshness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All three feeds succeed → returns observed_at, writes one record per feed."""
    _install_fetch(
        monkeypatch,
        {url: {"payload": source} for url, source in collector.ENE_SOURCES},
    )
    observed = collector.collect_ene()
    assert observed > 0

    daily = tmp_path / "ene" / f"{_today()}.jsonl"
    lines = daily.read_text().strip().splitlines()
    assert len(lines) == 3
    feed_sources = {json.loads(line)["feed_source"] for line in lines}
    assert feed_sources == {source for _url, source in collector.ENE_SOURCES}


def test_ene_partial_failure_still_advances_freshness(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """One feed fails, two succeed → freshness advances; only successes are recorded."""
    urls = [url for url, _ in collector.ENE_SOURCES]
    _install_fetch(
        monkeypatch,
        {
            urls[0]: _raise_http_error,
            urls[1]: {"payload": "current"},
            urls[2]: {"payload": "equipments"},
        },
    )
    observed = collector.collect_ene()
    assert observed > 0

    daily = tmp_path / "ene" / f"{_today()}.jsonl"
    lines = daily.read_text().strip().splitlines()
    assert len(lines) == 2


def test_ene_total_failure_returns_zero(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every feed fails → returns 0 so caller leaves state["ene"] stale."""
    _install_fetch(
        monkeypatch,
        {url: _raise_http_error for url, _ in collector.ENE_SOURCES},
    )
    observed = collector.collect_ene()
    assert observed == 0

    # No daily file should have been created (no successful writes)
    daily = tmp_path / "ene" / f"{_today()}.jsonl"
    assert not daily.exists()


def test_ene_total_failure_is_logged_in_poll_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """All failures must be visible in the poll log for ops debugging."""
    _install_fetch(
        monkeypatch,
        {url: _raise_http_error for url, _ in collector.ENE_SOURCES},
    )
    collector.collect_ene()

    poll_log = tmp_path / "meta" / "poll_log.jsonl"
    entries = [json.loads(line) for line in poll_log.read_text().splitlines()]
    assert len(entries) == len(collector.ENE_SOURCES)
    assert all(e["status"] == "error" for e in entries)
    logged_feeds = {e["feed"] for e in entries}
    assert logged_feeds == {source for _url, source in collector.ENE_SOURCES}
