"""Build the committed network fixture the Worker validates against the schema.

tests/fixtures/model/gtfs_mini/ (a hand-written GTFS feed, kept as text so it
reviews as a diff) is zipped deterministically and built with
tests/fixtures/model/stations.json into tests/fixtures/model/network_s1.json.
worker/test/model_network.test.ts validates that file against
schema/model/s1/network.schema.json and reads it through the generated TS
types; tests/test_model_network.py fails if it is stale.

Run:  uv run python -m scripts.gen_model_network_fixture
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from momentarily.model.network.build import CodeStamp, Inputs, build_network
from momentarily.model.network.contracts import NetworkSnapshot
from momentarily.model.network.stations import parse_snapshot

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "model"
GTFS_DIR = FIXTURE_DIR / "gtfs_mini"
STATIONS_PATH = FIXTURE_DIR / "stations.json"
FIXTURE_PATH = FIXTURE_DIR / "network_s1.json"
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
    return Inputs(
        gtfs=zip_dir(GTFS_DIR),
        stations=parse_snapshot(STATIONS_PATH.read_bytes()),
        direction_overrides={},
        overrides_digest=None,
    )


def build_fixture() -> NetworkSnapshot:
    return build_network(fixture_inputs(), stamp=FIXTURE_CODE)


def render_fixture() -> str:
    doc = build_fixture().model_dump(mode="json")
    return json.dumps(doc, indent=2, sort_keys=True) + "\n"


def main() -> int:
    FIXTURE_PATH.write_text(render_fixture())
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
