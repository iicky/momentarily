"""Generate the Python<->TypeScript severity-graded-condition parity fixture.

worker/src/mapping.ts severityTier / deriveGradedMtaState are hand-ports of
src/momentarily/mapping.py severity_tier and training/recovery_baseline.py
derive_graded_mta_state. That rule is the SAME definition the weekly review
grades the published condition against, and the Worker now publishes it as
route_status.condition — so a drift between the two languages would publish a
condition the review scores as wrong. This fixture pins them together.

The committed fixture carries, for a fixed set of MTA alert_type inputs, the
severity tier of each type and the graded state of each type combination at both
the canonical severe-only floor (2) and the legacy breadth floor (1). Both
languages replay it — tests/test_graded_state_parity.py and
worker/test/graded_state_parity.test.ts.

Run:  uv run python -m scripts.gen_graded_state_parity_fixture
"""

from __future__ import annotations

import json
from pathlib import Path

from momentarily.mapping import CANONICAL_SEVERITY_FLOOR, severity_tier
from training.recovery_baseline import derive_graded_mta_state

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "parity_graded_state.json"
)

# Every alert_type that resolves a non-trivial tier, plus the boundary cases the
# two coarse-status algorithms disagree on if ported naively (Slow Speeds,
# Part Suspended), planned work, an unknown type, and the directional-suspension
# prefix rule.
SEVERITY_TYPES: list[str] = [
    "Suspended",
    "No Trains",
    "No Uptown Service",
    "No Northbound Service",
    "Part Suspended",
    "Severe Delays",
    "Delays",
    "Some Delays",
    "Slow Speeds",
    "Service Change",
    "Trains Rerouted",
    "Reroute",
    "Stops Skipped",
    "Stations Skipped",
    "Local to Express",
    "Express to Local",
    "Reduced Service",
    "Boarding Change",
    "Cancellations",
    "Track Change",
    "Weather",
    "Station Notice",
    "Special Schedule",
    "Information",
    "Other",
    "Planned - Track Maintenance",
    "Planned Work",
    "Brand New Mystery Type",
]

# Route-tick alert_type combinations. Mirrors the disruptive_types the review
# feeds derive_graded_mta_state (non-planned list); planned types are included
# to prove they read tier 0 and never lift the grade.
STATE_COMBOS: list[list[str]] = [
    [],
    ["Delays"],
    ["Some Delays"],
    ["Slow Speeds"],
    ["Severe Delays"],
    ["Suspended"],
    ["No Trains"],
    ["No Uptown Service"],
    ["Part Suspended"],
    ["Trains Rerouted"],
    ["Delays", "Severe Delays"],
    ["Delays", "Slow Speeds"],
    ["Suspended", "Delays"],
    ["Suspended", "Severe Delays"],
    ["Severe Delays", "Trains Rerouted"],
    ["Planned - Track Maintenance"],
    ["Planned - Track Maintenance", "Delays"],
    ["Planned - Track Maintenance", "Severe Delays"],
    ["Station Notice"],
    ["Information", "Station Notice"],
    ["Brand New Mystery Type"],
    ["Brand New Mystery Type", "Severe Delays"],
]


def build_fixture() -> dict[str, object]:
    severity = [{"alert_type": at, "tier": severity_tier(at)} for at in SEVERITY_TYPES]
    graded: list[dict[str, object]] = []
    for floor in (CANONICAL_SEVERITY_FLOOR, 1):
        for combo in STATE_COMBOS:
            graded.append(
                {
                    "alert_types": combo,
                    "floor": floor,
                    "expected": derive_graded_mta_state(tuple(combo), floor=floor),
                }
            )
    return {
        "canonical_floor": CANONICAL_SEVERITY_FLOOR,
        "severity": severity,
        "graded": graded,
    }


def main() -> int:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(build_fixture(), indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
