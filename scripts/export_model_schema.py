"""Export the data-model JSON Schemas (schema/model/s1/) from Pydantic, and
the pinned parquet layouts of the observation and event records
(schema/model/s1/<layer>/parquet/<record>.txt).

Pydantic (src/momentarily/model/) is the source of truth. The TypeScript types
in shared/model/s1/ are generated from these files by
worker/scripts/model_types.ts, so the chain is Pydantic -> JSON Schema -> TS
with no hand-written mirror. tests/test_model_network.py fails on a stale
schema; tests/test_model_observations.py and tests/test_model_events.py on a
stale layout; worker/test/model_network.test.ts on stale generated TS.

Run:  uv run python -m scripts.export_model_schema
      (cd worker && bun run gen:model)
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from momentarily.model.events.columns import RECORDS
from momentarily.model.events.contracts import (
    CoverageInterval,
    Headway,
    OpenState,
    SegmentTraversal,
    StopEvent,
    SupplyObservation,
    TripMatch,
)
from momentarily.model.network.contracts import (
    DayFeedMap,
    NetworkIndex,
    NetworkSnapshot,
)
from momentarily.model.observations.columns import NORMALIZED, describe
from momentarily.model.observations.contracts import (
    AlertPresence,
    AlertVersion,
    CollectorTick,
    DerivedFileMetadata,
    EquipmentObservation,
    FeedHeaderObservation,
    LegacyCollectorHealth,
    LegacyFetchOutcome,
    LegacyRouteAlertState,
    TickBundleIndex,
    TripUpdateObservation,
    VehicleObservation,
)

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema" / "model" / "s1"

# Layer -> its derived record kinds, each with a pinned parquet layout under
# SCHEMA_DIR/<layer>/parquet/.
LAYOUTS: dict[str, tuple[tuple[str, type[BaseModel]], ...]] = {
    "observations": tuple(NORMALIZED.items()),
    "events": tuple(RECORDS.items()),
}

# Path under SCHEMA_DIR (without .schema.json) -> the model it exports. A
# subdirectory per layer after the first; the TS mirror follows the same tree.
DOCUMENTS: dict[str, type[BaseModel]] = {
    "network": NetworkSnapshot,
    "network_index": NetworkIndex,
    "day_feed_map": DayFeedMap,
    "observations/collector_tick": CollectorTick,
    "observations/tick_bundle_index": TickBundleIndex,
    "observations/vehicle_observation": VehicleObservation,
    "observations/trip_update_observation": TripUpdateObservation,
    "observations/feed_header_observation": FeedHeaderObservation,
    "observations/alert_presence": AlertPresence,
    "observations/alert_version": AlertVersion,
    "observations/equipment_observation": EquipmentObservation,
    "observations/legacy_route_alert_state": LegacyRouteAlertState,
    "observations/legacy_fetch_outcome": LegacyFetchOutcome,
    "observations/legacy_collector_health": LegacyCollectorHealth,
    "observations/derived_file_metadata": DerivedFileMetadata,
    "events/trip_match": TripMatch,
    "events/stop_event": StopEvent,
    "events/segment_traversal": SegmentTraversal,
    "events/headway": Headway,
    "events/open_state": OpenState,
    "events/coverage_interval": CoverageInterval,
    "events/supply_observation": SupplyObservation,
}


def render_schemas() -> dict[str, str]:
    """Relative path -> committed text: sorted keys + trailing newline."""
    return {
        f"{name}.schema.json": json.dumps(
            model.model_json_schema(), indent=2, sort_keys=True
        )
        + "\n"
        for name, model in DOCUMENTS.items()
    }


def committed_schemas() -> dict[str, str]:
    """Relative path -> text of every schema file on disk, subdirectories
    included, so a stray file anywhere under SCHEMA_DIR shows up."""
    return {
        p.relative_to(SCHEMA_DIR).as_posix(): p.read_text()
        for p in SCHEMA_DIR.rglob("*.schema.json")
    }


def layout_dir(layer: str) -> Path:
    return SCHEMA_DIR / layer / "parquet"


def render_layouts(layer: str) -> dict[str, str]:
    """File name -> the parquet layout of each derived record kind of a layer."""
    return {f"{record}.txt": describe(model) for record, model in LAYOUTS[layer]}


def committed_layouts(layer: str) -> dict[str, str]:
    return {p.name: p.read_text() for p in layout_dir(layer).glob("*")}


def main() -> int:
    rendered = render_schemas()
    for stale in committed_schemas().keys() - rendered.keys():
        (SCHEMA_DIR / stale).unlink()
    for name, text in rendered.items():
        path = SCHEMA_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {path}")
    for layer in LAYOUTS:
        directory = layout_dir(layer)
        directory.mkdir(parents=True, exist_ok=True)
        layouts = render_layouts(layer)
        for stale in committed_layouts(layer).keys() - layouts.keys():
            (directory / stale).unlink()
        for name, text in layouts.items():
            (directory / name).write_text(text)
            print(f"wrote {directory / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
