"""Build the committed network fixtures the Worker validates against the schemas.

tests/fixtures/model/gtfs_mini/ (a hand-written GTFS feed, kept as text so it
reviews as a diff) is zipped deterministically and built with
tests/fixtures/model/stations.json and direction_overrides.json into:

  * network_s1.json        the NetworkSnapshot
  * network_index_s1.json  the index.json a build of it writes
  * day_feed_map_s1.json   a day->feed map covering verified, unstamped, and
                           partly stamped days

worker/test/model_network.test.ts validates each against schema/model/s1/ and
reads the network through the generated TS types; tests/test_model_network.py
fails if any is stale.

Run:  uv run python -m scripts.gen_model_network_fixture
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from momentarily.model.network.build import (
    CodeStamp,
    Inputs,
    build_network,
    index_entry,
    load_overrides,
)
from momentarily.model.network.contracts import (
    SCHEMA_ID,
    DayFeedMap,
    NetworkIndex,
    NetworkSnapshot,
)
from momentarily.model.network.day_feed import resolve_day
from momentarily.model.network.stations import parse_snapshot
from training.traversal_archive import DayProvenance, FeedDigestRange

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "model"
GTFS_DIR = FIXTURE_DIR / "gtfs_mini"
STATIONS_PATH = FIXTURE_DIR / "stations.json"
OVERRIDES_PATH = FIXTURE_DIR / "direction_overrides.json"
FIXTURE_PATH = FIXTURE_DIR / "network_s1.json"
INDEX_FIXTURE_PATH = FIXTURE_DIR / "network_index_s1.json"
DAY_MAP_FIXTURE_PATH = FIXTURE_DIR / "day_feed_map_s1.json"
FIXTURE_CODE = CodeStamp(sha="fixture", dirty=None)


def zip_dir(directory: Path) -> bytes:
    """Stored (uncompressed), sorted, fixed-timestamp zip: identical bytes on
    every machine, so the fixture's feed_id is stable."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
        for path in sorted(directory.glob("*.txt")):
            info = zipfile.ZipInfo(path.name, date_time=(2026, 6, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            zf.writestr(info, path.read_bytes())
    return buf.getvalue()


def fixture_inputs() -> Inputs:
    overrides, digest = load_overrides(OVERRIDES_PATH.read_bytes())
    return Inputs(
        gtfs=zip_dir(GTFS_DIR),
        stations=parse_snapshot(STATIONS_PATH.read_bytes()),
        direction_overrides=overrides,
        overrides_digest=digest,
        first_observed=None,
    )


def build_fixture() -> NetworkSnapshot:
    return build_network(fixture_inputs(), stamp=FIXTURE_CODE)


def _dump(doc: object) -> str:
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def _day(feed_version: str, **kw: object) -> DayProvenance:
    return DayProvenance(
        schema=1,
        extractor=1,
        feed_version=feed_version,
        n_rows=0,
        n_source_objects=0,
        source_manifest="",
        code_sha=None,
        written_at=0,
        **kw,  # pyright: ignore[reportArgumentType]
    )


def render_fixtures() -> dict[Path, str]:
    snap = build_fixture()
    feed_id = snap.feed.feed_id
    index = NetworkIndex(schema=SCHEMA_ID, feeds={feed_id: index_entry(snap)})
    version = snap.feed.feed_version or ""
    days = {
        "2026-06-02": _day(version, feed_digest=None),
        "2026-06-03": _day(
            version,
            feed_digest=None,
            feed_digests=[FeedDigestRange(digest=feed_id, first_ts=1, last_ts=2)],
        ),
        "2026-06-04": _day(version, feed_digest=feed_id),
    }
    day_map = DayFeedMap(days={d: resolve_day(p, index) for d, p in days.items()})
    return {
        FIXTURE_PATH: _dump(snap.model_dump(mode="json")),
        INDEX_FIXTURE_PATH: _dump(index.model_dump(mode="json")),
        DAY_MAP_FIXTURE_PATH: _dump(day_map.model_dump(mode="json")),
    }


def main() -> int:
    for path, text in render_fixtures().items():
        path.write_text(text)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
