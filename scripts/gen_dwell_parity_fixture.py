"""Generate the Python<->TypeScript dwell-mixture parity fixture.

training/dwell.py fits a point mass ("atom") at one publisher tick mixed with
a log-logistic tail left-truncated there, and worker/src/dwell.ts +
viz/lib/dwell.ts are hand-ports of the same math. All three WILL drift unless
something pins them together — across the mixture cutover, the dwell_cdf
lower-boundary bugfix (a strict `<` and largest-index-at-a-repeated-knot,
instead of the old `<=` that zeroed the CDF for the one-tick majority), and
the legacy curve+tail path a params.json without atom_p/atom_sec must keep
hitting unchanged.

Unlike the sequential HMM/regime fixtures, dwell math is a set of pure
functions of scalar/array inputs, so the fixture is a flat list of labelled
cases: each one names the function under test, its exact inputs, and the
output training.dwell produces for them today. All three languages replay
the same list and must reproduce every value to absolute tolerance 1e-9 —
tests/test_dwell_parity.py (Python), worker/test/dwell_parity.test.ts, and
viz/tests/dwell_parity.test.ts. viz/lib/dwell.ts does not expose a
remaining-quantile function, so the viz test consumes every case except the
conditional_remaining_quantile ones.

`atom` cases are stored as `{"p": ..., "sec": ...}` (or null) rather than a
[atom_p, atom_sec] tuple — both TS ports take the atom as an object with
those exact keys, and this fixture is meant to be used as-is on that side.

Run:  uv run python -m scripts.gen_dwell_parity_fixture
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from training.dwell import (
    conditional_remaining_quantile,
    dwell_cdf,
    mixture_quantile,
    mixture_survival,
    p_leave_by,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "parity_dwell.json"
)

Case = dict[str, object]
Atom = tuple[float, float]

# --- Atom cell -----------------------------------------------------------
#
# 70.4% of disrupted episodes clear in exactly one tick; the rest tail off
# log-logistically, conditional on T > 1 tick. These are the same constants
# used everywhere below an "atom cell" is called for.
ATOM_P = 0.688
ATOM_SEC = 300
ATOM_SHAPE = 1.4
ATOM_SCALE = 1500.0
ATOM_TAIL_LL = [ATOM_SHAPE, ATOM_SCALE]

# curve_sec the trainer would still publish for an atom cell: mostly flat at
# one tick (14 of 21 knots, per the contract), rising after. Every atom-aware
# consumer must ignore this in favor of the closed form — it's included in
# the atom p_leave_by / conditional_remaining_quantile cases only because the
# functions take curve_sec positionally regardless, and so a consumer that
# starts reading it again silently would go untested elsewhere, not here.
ATOM_CURVE_SEC = [300] * 14 + [400, 550, 700, 900, 1200, 1600, 2100]

ATOM_ELAPSED_SEC = [0, 150, 300, 301, 900, 7200]
HORIZONS_SEC = [1800, 3600, 7200]

# --- No-atom cell ----------------------------------------------------------
#
# The pre-mixture shape: strictly increasing curve_sec + an unconditional
# log-logistic tail. Exercises the untouched legacy path so the mixture
# rollout can't silently regress a params.json that never sets atom_p/sec.
NO_ATOM_CURVE_SEC = [
    300,
    350,
    420,
    500,
    600,
    720,
    850,
    1000,
    1150,
    1300,
    1500,
    1700,
    1900,
    2100,
    2400,
    2700,
    3000,
    3400,
    3800,
    4300,
    5000,
]
NO_ATOM_TAIL_LL = [1.7, 900.0]


def _case(label: str, fn: str, inputs: Mapping[str, object], expected: object) -> Case:
    return {"label": label, "fn": fn, "inputs": inputs, "expected": expected}


def _atom_json(atom: Atom | None) -> dict[str, float] | None:
    return None if atom is None else {"p": atom[0], "sec": atom[1]}


def _dwell_cdf_cases() -> list[Case]:
    curve = NO_ATOM_CURVE_SEC
    mid_x = (curve[10] + curve[11]) / 2.0
    strictly_increasing = [
        ("dwell_cdf_below_first_knot", curve, float(curve[0] - 50)),
        ("dwell_cdf_at_first_knot", curve, float(curve[0])),
        ("dwell_cdf_at_interior_knot", curve, float(curve[10])),
        ("dwell_cdf_between_knots", curve, mid_x),
        ("dwell_cdf_above_last_knot", curve, float(curve[-1] + 100)),
    ]

    # ATOM_CURVE_SEC's first 14 knots are a flat run at one tick — the exact
    # shape that used to grade P=0 at x=300. flat[13]=300, flat[14]=400.
    flat = ATOM_CURVE_SEC
    between_post_flat = (flat[14] + flat[15]) / 2.0
    repeated_knot = [
        ("dwell_cdf_repeated_knot_at_flat_run", flat, float(flat[0])),
        ("dwell_cdf_repeated_knot_just_above_flat_run", flat, 350.0),
        ("dwell_cdf_repeated_knot_between_post_flat_knots", flat, between_post_flat),
        ("dwell_cdf_repeated_knot_at_last_knot", flat, float(flat[-1])),
        ("dwell_cdf_repeated_knot_above_last_knot", flat, float(flat[-1] + 100)),
    ]

    out: list[Case] = []
    for label, curve_sec, x in strictly_increasing + repeated_knot:
        out.append(
            _case(
                label,
                "dwell_cdf",
                {"curve_sec": curve_sec, "x": x},
                dwell_cdf(curve_sec, x),
            )
        )
    return out


def _atom_boundary_case() -> Case:
    """F(atom_sec) == atom_p exactly — the whole point of the closed form.

    mixture_survival(atom_sec, ...) divides S_ll(atom_sec) by itself, so the
    ratio is exactly 1.0 and S(atom_sec) == 1 - atom_p to the bit. Recorded
    here as its own case (not just folded into the elapsed=300 p_leave_by
    grid below) so every consumer's test can assert the invariant directly
    against `1 - expected == atom_p`, not merely replay whatever value
    training.dwell happens to produce.
    """
    t = float(ATOM_SEC)
    inputs = {
        "t": t,
        "shape": ATOM_SHAPE,
        "scale": ATOM_SCALE,
        "atom_p": ATOM_P,
        "atom_sec": t,
    }
    expected = mixture_survival(t, ATOM_SHAPE, ATOM_SCALE, ATOM_P, t)
    return _case(
        "atom_boundary_survival_at_atom_sec", "mixture_survival", inputs, expected
    )


def _atom_p_leave_by_cases() -> list[Case]:
    atom = (ATOM_P, float(ATOM_SEC))
    out: list[Case] = []
    for elapsed in ATOM_ELAPSED_SEC:
        for horizon in HORIZONS_SEC:
            label = f"atom_p_leave_by_elapsed_{elapsed}_horizon_{horizon}"
            inputs = {
                "curve_sec": ATOM_CURVE_SEC,
                "elapsed_sec": float(elapsed),
                "horizon_sec": float(horizon),
                "tail_ll": ATOM_TAIL_LL,
                "atom": _atom_json(atom),
            }
            expected = p_leave_by(
                ATOM_CURVE_SEC,
                float(elapsed),
                float(horizon),
                tail_ll=ATOM_TAIL_LL,
                atom=atom,
            )
            out.append(_case(label, "p_leave_by", inputs, expected))
    return out


def _no_atom_p_leave_by_cases() -> list[Case]:
    curve = NO_ATOM_CURVE_SEC
    specs: list[tuple[str, float, float, list[float] | None]] = [
        ("no_atom_p_leave_by_inside_curve", 0.0, 1800.0, NO_ATOM_TAIL_LL),
        (
            "no_atom_p_leave_by_near_curve_end",
            float(curve[-1] - 50),
            1800.0,
            NO_ATOM_TAIL_LL,
        ),
        (
            "no_atom_p_leave_by_past_curve_with_tail",
            float(curve[-1] + 500),
            3600.0,
            NO_ATOM_TAIL_LL,
        ),
        (
            "no_atom_p_leave_by_past_curve_without_tail",
            float(curve[-1] + 500),
            3600.0,
            None,
        ),
    ]
    out: list[Case] = []
    for label, elapsed, horizon, tail_ll in specs:
        inputs = {
            "curve_sec": curve,
            "elapsed_sec": elapsed,
            "horizon_sec": horizon,
            "tail_ll": tail_ll,
            "atom": None,
        }
        expected = p_leave_by(curve, elapsed, horizon, tail_ll=tail_ll, atom=None)
        out.append(_case(label, "p_leave_by", inputs, expected))
    return out


def _invalid_atom_fallback_cases() -> list[Case]:
    """An atom that is present but not USABLE must fall back to the curve path.

    These are the cases the fixture was missing, and the omission mattered: every
    other atom case here carries a valid atom, so all three languages agreed on
    the happy path while the Worker was skipping validation entirely and would
    have computed a closed form off a degenerate parameter instead of falling
    back. A negative case is the only thing that pins a fallback.

    Each expected value below is the LEGACY curve result, because that is what a
    correct implementation produces when it refuses the mixture. Both entry
    points are covered: the bug was in pLeaveBy AND conditionalRecovery, so
    pinning only one would let the other regress on its own.
    """
    curve = ATOM_CURVE_SEC
    elapsed, horizon = 0.0, 1800.0
    specs: list[tuple[str, Atom | None, list[float] | None]] = [
        # atom_p at or outside the open interval (0, 1): no mass left to
        # normalise the tail against, and p=1 divides by zero on inversion.
        ("invalid_atom_p_zero", (0.0, float(ATOM_SEC)), ATOM_TAIL_LL),
        ("invalid_atom_p_one", (1.0, float(ATOM_SEC)), ATOM_TAIL_LL),
        ("invalid_atom_p_negative", (-0.2, float(ATOM_SEC)), ATOM_TAIL_LL),
        ("invalid_atom_p_above_one", (1.4, float(ATOM_SEC)), ATOM_TAIL_LL),
        # A non-positive atom location would place the point mass at or before
        # t=0 and then apply it at every elapsed value.
        ("invalid_atom_sec_zero", (ATOM_P, 0.0), ATOM_TAIL_LL),
        ("invalid_atom_sec_negative", (ATOM_P, -300.0), ATOM_TAIL_LL),
        # A degenerate tail makes the log-logistic survival constant, so the
        # "mixture" would flatten instead of erroring.
        ("invalid_atom_tail_shape_zero", (ATOM_P, float(ATOM_SEC)), [0.0, ATOM_SCALE]),
        ("invalid_atom_tail_scale_zero", (ATOM_P, float(ATOM_SEC)), [ATOM_SHAPE, 0.0]),
        # An atom with no tail at all has nothing to spend its remaining mass on.
        ("invalid_atom_without_tail", (ATOM_P, float(ATOM_SEC)), None),
    ]
    out: list[Case] = []
    for label, atom, tail_ll in specs:
        p_inputs = {
            "curve_sec": curve,
            "elapsed_sec": elapsed,
            "horizon_sec": horizon,
            "tail_ll": tail_ll,
            "atom": _atom_json(atom),
        }
        out.append(
            _case(
                label,
                "p_leave_by",
                p_inputs,
                p_leave_by(curve, elapsed, horizon, tail_ll=tail_ll, atom=atom),
            )
        )
        # Same rejection, through the remaining-time path. The Worker routes
        # these via conditionalRecovery, which had the identical missing guard.
        q_inputs = {
            "curve_sec": curve,
            "elapsed_sec": elapsed,
            "q": 0.5,
            "tail_ll": tail_ll,
            "atom": _atom_json(atom),
        }
        out.append(
            _case(
                f"{label}_remaining_quantile",
                "conditional_remaining_quantile",
                q_inputs,
                conditional_remaining_quantile(
                    curve, elapsed, 0.5, tail_ll=tail_ll, atom=atom
                ),
            )
        )
    return out


def _mixture_quantile_cases() -> list[Case]:
    specs = [
        ("mixture_quantile_u_below_atom_p", ATOM_P - 0.1),
        ("mixture_quantile_u_at_atom_p", ATOM_P),
        ("mixture_quantile_u_above_atom_p", ATOM_P + 0.1),
    ]
    out: list[Case] = []
    for label, u in specs:
        inputs = {
            "u": u,
            "shape": ATOM_SHAPE,
            "scale": ATOM_SCALE,
            "atom_p": ATOM_P,
            "atom_sec": float(ATOM_SEC),
        }
        expected = mixture_quantile(u, ATOM_SHAPE, ATOM_SCALE, ATOM_P, float(ATOM_SEC))
        out.append(_case(label, "mixture_quantile", inputs, expected))
    return out


def _conditional_remaining_quantile_cases() -> list[Case]:
    out: list[Case] = []
    atom = (ATOM_P, float(ATOM_SEC))
    # Past the atom (elapsed=900 > atom_sec) so the three quantiles actually
    # differ; at elapsed < atom_sec every q <= atom_p collapses to atom_sec,
    # which the atom_p_leave_by grid above already exercises at elapsed=0/150.
    for q in (0.25, 0.5, 0.75):
        label = f"atom_conditional_remaining_quantile_q_{q}"
        inputs = {
            "curve_sec": ATOM_CURVE_SEC,
            "elapsed_sec": 900.0,
            "q": q,
            "tail_ll": ATOM_TAIL_LL,
            "atom": _atom_json(atom),
        }
        expected = conditional_remaining_quantile(
            ATOM_CURVE_SEC, 900.0, q, tail_ll=ATOM_TAIL_LL, atom=atom
        )
        out.append(_case(label, "conditional_remaining_quantile", inputs, expected))

    for q in (0.25, 0.5, 0.75):
        label = f"no_atom_conditional_remaining_quantile_q_{q}"
        inputs = {
            "curve_sec": NO_ATOM_CURVE_SEC,
            "elapsed_sec": 0.0,
            "q": q,
            "tail_ll": NO_ATOM_TAIL_LL,
            "atom": None,
        }
        expected = conditional_remaining_quantile(
            NO_ATOM_CURVE_SEC, 0.0, q, tail_ll=NO_ATOM_TAIL_LL, atom=None
        )
        out.append(_case(label, "conditional_remaining_quantile", inputs, expected))
    return out


def build_fixture() -> dict[str, object]:
    cases: list[Case] = []
    cases += _dwell_cdf_cases()
    cases.append(_atom_boundary_case())
    cases += _atom_p_leave_by_cases()
    cases += _no_atom_p_leave_by_cases()
    cases += _invalid_atom_fallback_cases()
    cases += _mixture_quantile_cases()
    cases += _conditional_remaining_quantile_cases()
    return {"cases": cases}


def main() -> int:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(build_fixture(), indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
