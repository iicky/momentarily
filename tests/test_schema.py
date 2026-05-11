"""Tests asserting the snapshot serializes to the documented JSON shape."""

from __future__ import annotations

import json

from momentarily.schema import SCHEMA_VERSION, Compat, Snapshot


def test_minimal_snapshot_serializes() -> None:
    snap = Snapshot(generated_at=1_700_000_000)
    payload = json.loads(snap.model_dump_json())
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["generated_at"] == 1_700_000_000
    assert payload["alerts"] == []
    assert payload["routes"] == {}
    assert payload["compat"]["subwaynow_routes"] == {}


def test_attribution_present() -> None:
    snap = Snapshot(generated_at=0)
    assert "Momentarily" in snap.attribution
    assert "MTA" in snap.attribution
    assert "Not affiliated" in snap.attribution


def test_compat_default_empty() -> None:
    compat = Compat()
    assert compat.subwaynow_routes == {}
