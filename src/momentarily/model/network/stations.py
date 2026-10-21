"""Canonical snapshots of the MTA Subway Stations dataset (data.ny.gov 39hk-dx4f).

The dataset is the only public source for station complexes, per-side
direction labels ("Manhattan", "Outbound"), borough, and ADA status. It is
live and unversioned, so a network build never reads the URL: it reads a
snapshot file written here. The file is canonical JSON (rows sorted by
gtfs_stop_id, keys sorted, no insignificant whitespace), so its blake3 digest
identifies the exact data a build used and a rebuild is byte-identical.

Socrata's ``:@computed_region_*`` columns are dropped: they are geo-joins the
portal recomputes on its own schedule, not source data, and would change the
digest without the stations changing.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import blake3
import httpx

STATIONS_URL = "https://data.ny.gov/resource/39hk-dx4f.json"
STATIONS_QUERY = {"$limit": "5000"}
FETCH_TIMEOUT = httpx.Timeout(30.0)


class NonCanonicalSnapshotError(ValueError):
    """The file's bytes are not the canonical form, so its digest would not
    identify its content."""

    def __init__(self) -> None:
        super().__init__(
            "stations snapshot is not canonical; rewrite it with snapshot-stations"
        )


@dataclass(frozen=True)
class StationRow:
    gtfs_stop_id: str
    complex_id: str | None
    stop_name: str
    borough: str | None
    ada: str | None
    north_direction_label: str | None
    south_direction_label: str | None


@dataclass(frozen=True)
class StationsSnapshot:
    digest: str
    retrieved_at: int
    rows: tuple[StationRow, ...]


def canonical_bytes(retrieved_at: int, rows: list[dict[str, Any]]) -> bytes:
    """The one serialisation a snapshot may have."""
    clean = [
        {k: v for k, v in row.items() if not k.startswith(":@")}
        for row in sorted(rows, key=lambda r: str(r.get("gtfs_stop_id", "")))
    ]
    doc = {"dataset": STATIONS_URL, "retrieved_at": retrieved_at, "rows": clean}
    text = json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return (text + "\n").encode()


def digest_of(data: bytes) -> str:
    return blake3.blake3(data).hexdigest()


def fetch_rows() -> list[dict[str, Any]]:
    with httpx.Client(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
        response = client.get(STATIONS_URL, params=STATIONS_QUERY)
        response.raise_for_status()
        return cast("list[dict[str, Any]]", response.json())


def write_snapshot(
    out_dir: Path, rows: list[dict[str, Any]], retrieved_at: int
) -> Path:
    data = canonical_bytes(retrieved_at, rows)
    path = out_dir / f"stations-{digest_of(data)[:16]}.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def snapshot_now(out_dir: Path) -> Path:
    return write_snapshot(out_dir, fetch_rows(), int(time.time()))


def _opt(row: dict[str, Any], key: str) -> str | None:
    value = row.get(key)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def parse_snapshot(data: bytes) -> StationsSnapshot:
    """Read a snapshot file, refusing one that is not in canonical form (its
    digest would then not identify its content)."""
    doc = cast("dict[str, Any]", json.loads(data))
    retrieved_at = int(doc["retrieved_at"])
    raw_rows = cast("list[dict[str, Any]]", doc["rows"])
    if canonical_bytes(retrieved_at, raw_rows) != data:
        raise NonCanonicalSnapshotError
    rows = tuple(
        StationRow(
            gtfs_stop_id=str(r["gtfs_stop_id"]).strip(),
            complex_id=_opt(r, "complex_id"),
            stop_name=str(r["stop_name"]).strip(),
            borough=_opt(r, "borough"),
            ada=_opt(r, "ada"),
            north_direction_label=_opt(r, "north_direction_label"),
            south_direction_label=_opt(r, "south_direction_label"),
        )
        for r in raw_rows
    )
    return StationsSnapshot(
        digest=digest_of(data), retrieved_at=retrieved_at, rows=rows
    )
