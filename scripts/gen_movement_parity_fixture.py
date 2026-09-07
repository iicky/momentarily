"""Generate the Python<->TypeScript movement-classifier parity fixture.

training/load_r2.py's classify_direction is the source of truth for the
per-(route, direction) Beta-Binomial movement call — a single tick's advance
count, regularized toward the cell's own baseline p0 by CLASSIFY_PRIOR_STRENGTH
pseudo-trials, called disrupted only when the posterior sits at/under
DISRUPTED_RATIO * p0 AND the low advance count is significant against p0
(binomial lower tail <= CLASSIFY_ALPHA), and abstaining below MIN_MATCHED_TRIPS
matches. worker/src/movement_state.ts (classifyAdvance) and viz/lib/movement.ts
(classifyDirection) are hand-ports of exactly that arithmetic, and the live
published nowcast rides on them agreeing with the offline authority.

All three WILL silently drift the moment a Python constant moves —
DISRUPTED_RATIO, CLASSIFY_ALPHA, CLASSIFY_PRIOR_STRENGTH, MIN_MATCHED_TRIPS —
unless a shared fixture pins them together. Before this fixture the only pin
was a four-row (advanced, stalled, p0) -> label table hand-copied into three
test files; a constant change had to be caught by remembering to edit all
three by hand.

Like the dwell/regime fixtures the classifier is a pure function of scalar
inputs, so the fixture is a flat list of labelled cases: each names the
function under test, its exact inputs, and the label classify_direction
produces for them today. All three languages replay the same list and must
reproduce every label — tests/test_movement_parity.py (Python),
worker/test/movement_parity.test.ts, and viz/tests/movement_parity.test.ts.

A case's baseline is carried as a bare `p0` (or null for "no baseline cell").
Each consumer rebuilds its own cell from p0 the way the trainer does —
AdvanceBaseline(p0=p0, n=50, alpha=50*p0, beta=50*(1-p0)) — because
classify_direction reads only p0 off it; alpha/beta/n are there for the type.
worker/src/movement_state.ts exposes only classifyAdvance(advanced, stalled,
p0), which has no no-baseline path (that guard lives in the unexported
classifyDirection), so the Worker replay skips the null-p0 case; Python and viz
replay every case.

Run:  uv run python -m scripts.gen_movement_parity_fixture
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from training.load_r2 import AdvanceBaseline, classify_direction

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "parity_movement.json"
)

Case = dict[str, object]


def _baseline(p0: float | None) -> AdvanceBaseline | None:
    """The cell every consumer rebuilds from a case's p0. classify_direction
    reads only p0; alpha/beta/n complete the trainer's Beta prior shape."""
    if p0 is None:
        return None
    return AdvanceBaseline(p0=p0, n=50, alpha=50 * p0, beta=50 * (1.0 - p0))


def _case(label: str, advanced: int, stalled: int, p0: float | None) -> Case:
    inputs: Mapping[str, object] = {
        "advanced": advanced,
        "stalled": stalled,
        "p0": p0,
    }
    return {
        "label": label,
        "fn": "classify_direction",
        "inputs": inputs,
        "expected": classify_direction(advanced, stalled, _baseline(p0)),
    }


def _shared_table_cases() -> list[Case]:
    """The exact (advanced, stalled, p0) -> label table that was hand-copied
    into tests/test_load_r2.py, worker/test/movement_state.test.ts and
    viz/tests/movement.test.ts. Case math (prior_strength=8,
    disrupted_ratio=0.5, alpha=0.05):

      case1 p0=0.125 advanced=0 matched=8:  post=0.0625==0.5*p0 (<=);
            tail=0.875**8~=0.3436>alpha -> None. THE FIX: a short shuttle's
            degenerate baseline no longer misfires disrupted on an ordinary
            zero-advance tick.
      case2 p0=0.125 advanced=0 matched=25: post~=0.0303<=0.0625;
            tail=0.875**25~=0.0356<=alpha -> disrupted. The same degenerate
            baseline still fires once there's enough evidence.
      case3 p0=0.55  advanced=0 matched=17: post=0.176<=0.275;
            tail=0.45**17~=1.2e-6<=alpha -> disrupted (a healthy trunk frozen).
      case4 p0=0.55  advanced=8 matched=17: post=0.496>0.275 -> normal
            (posterior clears the cutoff outright, no significance test needed).
    """
    return [
        _case("case1_shuttle_false_positive_now_abstains", 0, 8, 0.125),
        _case("case2_sustained_shuttle_freeze_still_fires", 0, 25, 0.125),
        _case("case3_trunk_freeze_still_fires", 0, 17, 0.55),
        _case("case4_normal_above_ratio", 8, 9, 0.55),
    ]


def _edge_cases() -> list[Case]:
    """Cells the four-row table doesn't reach, each pinning a distinct branch.

    n=0            no matched trips at all -> the matched<MIN_MATCHED_TRIPS
                   floor short-circuits before any posterior is formed.
    below_min      matched=2 (<3) -> same floor, with a nonzero count.
    prior_only     matched=3, all stalled, healthy trunk: post=(8*0.9)/(8+3)
                   =0.6545>0.45 -> normal. Three observations can't outvote
                   the prior, so a decisive stall is still called normal —
                   the prior-strength knob's whole job.
    shuttle_normal p0=0.1, advanced=1/10: post=1.8/18=0.10>0.05 -> normal.
                   Raw frac 0.10 would trip a single global 0.25 cutoff; the
                   baseline-relative rule doesn't.
    degenerate     p0 at the P0_FLOOR floor (1e-3): even 10 straight stalls
                   can't reach significance (tail=0.999**10~=0.99>alpha), so
                   the near-zero baseline abstains rather than firing on its
                   own normal zero-advance noise.
    no_baseline    p0 null -> no cell to judge against. Python/viz return
                   None; the Worker replay skips it (classifyAdvance has no
                   null-baseline path).
    """
    return [
        _case("edge_n_zero_no_matched_trips", 0, 0, 0.9),
        _case("edge_below_min_matched", 1, 1, 0.9),
        _case("edge_prior_only_stall_stays_normal", 0, 3, 0.9),
        _case("edge_shuttle_normal_debiased", 1, 9, 0.1),
        _case("edge_degenerate_baseline_abstains", 0, 10, 0.001),
        _case("edge_no_baseline_cell", 8, 1, None),
    ]


def build_fixture() -> dict[str, object]:
    cases: list[Case] = []
    cases += _shared_table_cases()
    cases += _edge_cases()
    return {"cases": cases}


def main() -> int:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(build_fixture(), indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
