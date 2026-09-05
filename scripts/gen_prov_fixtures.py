"""Generate the models-page PROV-JSON reader fixtures.

The viz reader (viz/lib/prov.ts) parses the trainer's PROV-JSON sidecar into the
derivation chain the models page draws. Its test must run against the exact
bytes the real producer ships, not a hand-written approximation — so the
fixtures are emitted here by training.prov.build_trainer_run, the same
assembler the trainer runs, from representative recorded facts.

Two documents:
  prov_full.json      a complete run: GTFS feed (version + digest), trainer run
                      (code sha, dirty, start/end), input manifest (blake3 +
                      alert/vehicle key counts), and four published artifacts
                      with their feed/manifest derivation edges.
  prov_feedless.json  a run whose GTFS fetch failed: no feed entity, no
                      feed-derived edges, and a dirty flag the build could not
                      determine (None) — the reader must read that as null, not
                      false.

The two degraded-document cases the reader also handles — a fetch that fails and
a malformed body — are the absence or corruption of a document, which no
producer emits, so those are constructed in the test itself.

Deterministic: the emitter sorts keys and drops incidental whitespace, so an
empty diff means the facts below did not move.

Run:  uv run python -m scripts.gen_prov_fixtures
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.prov import (
    AgentFacts,
    ArtifactFacts,
    FeedFacts,
    ManifestFacts,
    build_trainer_run,
)

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "viz" / "tests" / "fixtures"

# Representative recorded facts for one run. The trained_at is the same epoch a
# params key and the snapshot's params.trained_at would carry.
TRAINED_AT = 1788229972
STARTED_AT = 1788229900
CODE_SHA = "9f3c1ab7d240e5c8"

_ARTIFACTS = [
    ArtifactFacts(
        "params", "state/params/v1788229972.json", derived_from_manifest=True
    ),
    ArtifactFacts(
        "scheduled_headway",
        "state/scheduled_headway/v1788229972.json",
        derived_from_feed=True,
    ),
    ArtifactFacts(
        "segment_params",
        "state/segment_params/v1788229972.json",
        derived_from_feed=True,
        derived_from_manifest=True,
    ),
    ArtifactFacts("service_baseline", "state/service_baseline/v1788229972.json"),
]


def _full() -> dict[str, Any]:
    doc = build_trainer_run(
        trained_at=TRAINED_AT,
        started_at=STARTED_AT,
        agent=AgentFacts(code_sha=CODE_SHA, dirty=True, producer="train_em"),
        manifest=ManifestFacts(
            blake3="7c" + "0" * 62,
            n_alert_keys=4032,
            n_vehicle_keys=4032,
            manifest_version=2,
        ),
        artifacts=_ARTIFACTS,
        feed=FeedFacts(
            version="20260807-H-rockaways-extension-removed",
            sha256="e3" + "b" * 62,
            start="2026-08-07",
            end="2026-09-04",
        ),
    )
    return doc.to_dict()


def _feedless() -> dict[str, Any]:
    doc = build_trainer_run(
        trained_at=TRAINED_AT,
        started_at=STARTED_AT,
        agent=AgentFacts(code_sha=CODE_SHA, dirty=None, producer="train_em"),
        manifest=ManifestFacts(
            blake3="7c" + "0" * 62,
            n_alert_keys=10,
            n_vehicle_keys=20,
            manifest_version=2,
        ),
        artifacts=[
            ArtifactFacts(
                "params", "state/params/v1788229972.json", derived_from_manifest=True
            )
        ],
        feed=None,
    )
    return doc.to_dict()


def main() -> int:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    (FIXTURE_DIR / "prov_full.json").write_text(json.dumps(_full(), indent=2) + "\n")
    (FIXTURE_DIR / "prov_feedless.json").write_text(
        json.dumps(_feedless(), indent=2) + "\n"
    )
    print(f"wrote prov_full.json + prov_feedless.json to {FIXTURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
