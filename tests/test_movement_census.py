"""Round-trip test for the movement census reader (training/eval.py).

Uses the same fake-client pattern as test_prune.py: a minimal S3Client
substitute that returns canned JSON bodies by key, so the reader's list →
fetch → parse pipeline is exercised without a real bucket.
"""

from __future__ import annotations

import json
from datetime import date
from typing import Any

from training.eval import (
    MovementCensusRecord,
    MovementCensusRow,
    load_movement_census,
)

# --- Fake S3 client ---------------------------------------------------


class _FakeClient:
    """Minimal S3Client substitute for _list_keys + get_object_bytes."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        prefix = kwargs.get("Prefix", "")
        matching = [k for k in self._objects if k.startswith(prefix)]
        return {"Contents": [{"Key": k} for k in matching]}

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        key = kwargs["Key"]
        if key not in self._objects:
            raise KeyError(key)
        return {"Body": _FakeBody(self._objects[key])}


class _FakeBody:
    def __init__(self, data: bytes) -> None:
        self._data = data

    def read(self) -> bytes:
        return self._data


# --- Fixtures ----------------------------------------------------------


def _census_body(
    observed_at: int,
    regimes: dict[str, dict[str, Any]],
) -> bytes:
    return json.dumps({"observed_at": observed_at, "regimes": regimes}).encode()


TICK_1 = 1_700_000_900
TICK_2 = 1_700_001_200

CENSUS_1 = _census_body(
    TICK_1,
    {
        "A": {"state": "normal", "open_state": "normal", "open_since": 1_700_000_000},
        "B": {
            "state": "disrupted",
            "open_state": "disrupted",
            "open_since": 1_700_000_300,
        },
        "C": {"state": "unknown", "open_state": None, "open_since": None},
    },
)

CENSUS_2 = _census_body(
    TICK_2,
    {
        "A": {"state": "normal", "open_state": "normal", "open_since": 1_700_000_000},
        "B": {"state": "normal", "open_state": "normal", "open_since": TICK_2},
        "C": {"state": "suspended", "open_state": "suspended", "open_since": TICK_2},
    },
)


# --- Tests -------------------------------------------------------------


def test_round_trip_two_ticks() -> None:
    """Two census objects on the same day load in order and parse correctly."""
    client = _FakeClient(
        {
            f"archive/movement_census/2023-11-14/{TICK_1}.json": CENSUS_1,
            f"archive/movement_census/2023-11-14/{TICK_2}.json": CENSUS_2,
        }
    )

    records = load_movement_census(
        client,  # type: ignore[arg-type]
        "b",
        date(2023, 11, 14),
        date(2023, 11, 14),
    )

    assert len(records) == 2
    r1, r2 = sorted(records, key=lambda r: r.observed_at)

    assert r1.observed_at == TICK_1
    assert r1.regimes["A"] == MovementCensusRow("normal", "normal", 1_700_000_000)
    assert r1.regimes["B"] == MovementCensusRow("disrupted", "disrupted", 1_700_000_300)
    assert r1.regimes["C"] == MovementCensusRow("unknown", None, None)

    assert r2.observed_at == TICK_2
    assert r2.regimes["B"] == MovementCensusRow("normal", "normal", TICK_2)
    assert r2.regimes["C"] == MovementCensusRow("suspended", "suspended", TICK_2)


def test_empty_day_before_feature() -> None:
    """A date range with no census keys returns an empty list (no crash)."""
    client = _FakeClient({})
    records = load_movement_census(
        client,  # type: ignore[arg-type]
        "b",
        date(2023, 11, 10),
        date(2023, 11, 12),
    )
    assert records == []


def test_from_json_parses_record() -> None:
    raw = json.loads(CENSUS_1)
    record = MovementCensusRecord.from_json(raw)
    assert record.observed_at == TICK_1
    assert set(record.regimes) == {"A", "B", "C"}
    assert record.regimes["C"].state == "unknown"
    assert record.regimes["C"].open_state is None
    assert record.regimes["C"].open_since is None
