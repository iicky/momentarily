"""Generate the Python<->TypeScript HMM parity fixture.

The Worker (worker/src/hmm.ts) is a hand-port of the inference half of
src/momentarily/hmm.py. The two WILL drift unless something pins them together.
This fixture is that pin: a canonical (params, observation sequence) plus the
forward-filter outputs Python produces for it. Both languages re-run the filter
over the fixture and assert they reproduce the recorded outputs —
tests/test_parity.py (Python side) and worker/test/parity_python.test.ts (TS).

Only the inference path is covered. fit_em lives in Python alone (the Worker
never trains), so there is nothing cross-language to pin for it.

Run:  uv run python -m scripts.gen_parity_fixture
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from momentarily.hmm import (
    EmissionParams,
    FilterState,
    HMMParams,
    Observation,
    expected_dwell_ticks,
    forward_step,
    initial_published_state,
    project_forward,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "parity_forward.json"
)

TICK_SECONDS = 300
START_NOW = 1_700_000_000

# Hand-authored, non-degenerate params. Deliberately not an fit_em output —
# we want the fixture stable and readable, and the cross-language contract is
# the forward filter, not the trainer.
PARAMS = HMMParams(
    transition=(
        (0.94, 0.05, 0.01),
        (0.10, 0.85, 0.05),
        (0.03, 0.12, 0.85),
    ),
    initial=(0.90, 0.08, 0.02),
    emissions=EmissionParams(
        poisson_lambda=(0.3, 4.0, 12.0),
        gamma_alpha=(1.0, 3.0, 6.0),
        gamma_beta=(2.0, 0.4, 0.2),
        bernoulli_p=(0.001, 0.05, 0.95),
        bernoulli_p_delays=(0.02, 0.6, 0.35),
        bernoulli_p_service_change=(0.02, 0.6, 0.4),
        bernoulli_p_planned=(0.05, 0.6, 0.35),
    ),
)


def _quiet() -> Observation:
    return Observation(
        alert_count=0,
        severity_sum=0,
        has_suspended_alert=False,
        has_delays=False,
        has_service_change=False,
        has_planned=False,
        tod_bin=1,
    )


def _delays() -> Observation:
    return Observation(
        alert_count=6,
        severity_sum=30,
        has_suspended_alert=False,
        has_delays=True,
        has_service_change=False,
        has_planned=False,
        tod_bin=1,
    )


def _suspended() -> Observation:
    return Observation(
        alert_count=14,
        severity_sum=75,
        has_suspended_alert=True,
        has_delays=False,
        has_service_change=False,
        has_planned=False,
        tod_bin=2,
    )


# Quiet -> delays -> suspended -> feed gap (None) -> suspended -> recovery.
# Exercises argmax changes, hysteresis promotion, and the obs=None branch.
OBSERVATIONS: list[Observation | None] = [
    _quiet(),
    _quiet(),
    _delays(),
    _delays(),
    _delays(),
    _suspended(),
    _suspended(),
    None,
    _suspended(),
    _quiet(),
    _quiet(),
    _quiet(),
]


def build_fixture() -> dict[str, object]:
    initial_filter = FilterState(
        probabilities=PARAMS.initial,
        regime_entered_at=START_NOW,
        last_updated_at=START_NOW,
    )
    state = initial_filter
    published = initial_published_state(initial_filter)

    steps: list[dict[str, object]] = []
    for i, obs in enumerate(OBSERVATIONS):
        now = START_NOW + (i + 1) * TICK_SECONDS
        state, published = forward_step(state, published, obs, PARAMS, now)
        steps.append(
            {
                "filter": {
                    "probabilities": list(state.probabilities),
                    "regime_entered_at": state.regime_entered_at,
                    "last_updated_at": state.last_updated_at,
                },
                "published": {
                    "label": published.label,
                    "pending_state": published.pending_state,
                    "pending_streak": published.pending_streak,
                    "last_updated_at": published.last_updated_at,
                },
            }
        )

    median, q25, q75 = expected_dwell_ticks(state, PARAMS)

    return {
        "description": (
            "Canonical Python<->TS parity fixture for the HMM forward filter. "
            "Regenerate with: uv run python -m scripts.gen_parity_fixture"
        ),
        "tick_seconds": TICK_SECONDS,
        "start_now": START_NOW,
        "params": {
            "transition": [list(row) for row in PARAMS.transition],
            "initial": list(PARAMS.initial),
            "emissions": asdict(PARAMS.emissions),
        },
        "initial_filter": {
            "probabilities": list(initial_filter.probabilities),
            "regime_entered_at": initial_filter.regime_entered_at,
            "last_updated_at": initial_filter.last_updated_at,
        },
        "observations": [None if o is None else asdict(o) for o in OBSERVATIONS],
        "expected_steps": steps,
        "expected_projections": {
            "10": list(project_forward(state, PARAMS, 10)),
            "50": list(project_forward(state, PARAMS, 50)),
            "200": list(project_forward(state, PARAMS, 200)),
        },
        "expected_dwell": {"median": median, "q25": q25, "q75": q75},
    }


def main() -> int:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(build_fixture(), indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
