"""Generate the Python<->TypeScript recovery-climatology parity fixture.

The recovery climatology conditions realized incident durations on how long a
disruption has already run: keep the durations longer than the elapsed time,
subtract it, and read the empirical p25/p50/p75 of what remains (or abstain when
fewer than min_samples outlast the elapsed time). Both the trainer/serve path
(training/recovery_baseline.py) and the Worker (worker/src/recovery.ts) run this
same math; this fixture pins them together.

The Python side is authoritative: expected values are computed by calling
training.recovery_baseline, so the fixture cannot drift from the reference. The
Worker half replays it in worker/test/recovery_parity.test.ts and the Python
half in tests/test_recovery_parity.py.

Regenerate with: uv run python -m scripts.gen_recovery_parity_fixture
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from training.recovery_baseline import (
    conditional_remaining_quantiles,
    empirical_quantile,
)

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "parity_recovery.json"
)

Case = dict[str, object]


def _q_case(label: str, values: Sequence[float], q: float) -> Case:
    return {
        "label": label,
        "fn": "empirical_quantile",
        "inputs": {"sorted": list(values), "q": q},
        "expected": empirical_quantile(values, q),
    }


def _cond_case(
    label: str, samples: Sequence[float], elapsed_min: float, min_samples: int
) -> Case:
    rq = conditional_remaining_quantiles(samples, elapsed_min, min_samples=min_samples)
    expected: object = (
        None if rq is None else {"p25": rq.p25, "p50": rq.p50, "p75": rq.p75, "n": rq.n}
    )
    return {
        "label": label,
        "fn": "conditional_remaining_quantiles",
        "inputs": {
            "samples_min": list(samples),
            "elapsed_min": elapsed_min,
            "min_samples": min_samples,
        },
        "expected": expected,
    }


def _empirical_quantile_cases() -> list[Case]:
    uniform = [10.0, 20.0, 30.0, 40.0]
    out: list[Case] = [
        _q_case("single element", [5.0], 0.5),
        _q_case("uniform q0", uniform, 0.0),
        _q_case("uniform q25 interpolated", uniform, 0.25),
        _q_case("uniform q50", uniform, 0.5),
        _q_case("uniform q75 interpolated", uniform, 0.75),
        _q_case("uniform q1", uniform, 1.0),
        _q_case("nonuniform median lands on a knot", [1.0, 2.0, 4.0, 8.0, 16.0], 0.5),
        _q_case("nonuniform interpolated", [1.0, 2.0, 4.0, 8.0, 16.0], 0.3),
    ]
    return out


def _conditional_cases() -> list[Case]:
    # Ten incident durations in minutes (multiples of the 5-min grid), ascending.
    samples = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 45.0, 60.0, 90.0, 120.0]
    out: list[Case] = [
        _cond_case("elapsed 0 keeps every sample", samples, 0.0, 5),
        _cond_case("fractional elapse conditions the tail", samples, 12.5, 5),
        _cond_case("boundary: exactly min_samples remain", samples, 25.0, 5),
        _cond_case("outlived: fewer than min_samples remain", samples, 90.0, 5),
        _cond_case("outlived: elapsed past the longest sample", samples, 200.0, 3),
        _cond_case("min_samples 8 with a large elapsed abstains", samples, 45.0, 8),
        _cond_case("min_samples 1 keeps a lone survivor", samples, 90.0, 1),
    ]
    return out


def build_fixture() -> dict[str, object]:
    cases: list[Case] = []
    cases.extend(_empirical_quantile_cases())
    cases.extend(_conditional_cases())
    return {"cases": cases}


def main() -> int:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(build_fixture(), indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
