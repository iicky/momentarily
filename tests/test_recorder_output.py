"""Recorder output through the Pydantic contracts.

tests/fixtures/recorder_output.json is one CollectorTick and its
TickBundleIndex exactly as the Worker recorder writes them, with every outcome
a GTFS-RT fetch can have. worker/test/recorder.test.ts fails when the recorder
would write anything else; this runs the cross-field rules (timings, status
classes, storage, bundle membership) that the exported JSON Schema cannot
express and the Worker never checks at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

from momentarily.model.observations.contracts import CollectorTick, TickBundleIndex

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "recorder_output.json"


def test_recorder_output_satisfies_the_contracts() -> None:
    output = json.loads(FIXTURE_PATH.read_text())
    tick = CollectorTick.model_validate(output["collector_tick"])
    index = TickBundleIndex.model_validate(output["tick_bundle_index"])

    assert {f.outcome for f in tick.fetches} >= {
        "ok",
        "empty",
        "unparseable",
        "http_error",
        "timeout",
        "network_error",
    }
    bundled = [f for f in tick.fetches if f.payload_member is not None]
    assert index.tick_id == tick.tick_id
    assert {f.fetch_id for f in index.fetches} >= {f.fetch_id for f in bundled}
