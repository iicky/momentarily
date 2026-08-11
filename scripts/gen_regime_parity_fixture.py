"""Generate the Python<->TypeScript regime-clock parity fixture.

worker/src/regime.ts is a hand-port of training/regime.py. They will drift
unless something pins them together, and drift here is expensive in a specific
way: the trainer fits dwell curves on regimes it segmented offline, and the
Worker projects those curves over regimes it segmented online. Disagree on
where a regime starts and the curve describes an episode that never happened.

The fixture is a canonical tick sequence plus the entries and changes Python
produces for it. Both languages replay it — tests/test_regime.py and
worker/test/regime_parity.test.ts.

Run:  uv run python -m scripts.gen_regime_parity_fixture
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from training.regime import advance_regimes

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "parity_regime.json"
)

TICK_SECONDS = 300
START = 1_700_000_000

DEBOUNCE_TICKS = 2
MAX_IDLE_SEC = 3600


def _t(i: int) -> int:
    return START + i * TICK_SECONDS


# Mixed route-scope and segment-scope keys in one run — the clock is key-
# agnostic, and running both through one fixture is what proves it.
#
# Covers: cold-start entry, a single-tick blip that must NOT commit, a
# two-tick run that must commit back-dated to the blip's first tick, an
# abstention that holds a regime open while resetting the candidate run, a
# state that changes twice in a row, and an idle eviction past MAX_IDLE_SEC.
TICKS: list[tuple[int, dict[str, str]]] = [
    # 0: cold start — both cells open immediately, no debounce.
    (_t(0), {"A": "normal", "Q|north|Q05N": "normal"}),
    (_t(1), {"A": "normal", "Q|north|Q05N": "normal"}),
    # 2: one-tick blip on A. Must stay pending, not commit.
    (_t(2), {"A": "disrupted", "Q|north|Q05N": "normal"}),
    # 3: A recovers before the debounce completes — the blip is discarded.
    (_t(3), {"A": "normal", "Q|north|Q05N": "disrupted"}),
    # 4: segment's second disrupted tick commits, back-dated to t(3).
    (_t(4), {"A": "disrupted", "Q|north|Q05N": "disrupted"}),
    # 5: A abstains (absent). The pending run resets; the regime stays normal.
    (_t(5), {"Q|north|Q05N": "disrupted"}),
    # 6-7: A now runs disrupted twice cleanly and commits at t(6).
    (_t(6), {"A": "disrupted", "Q|north|Q05N": "disrupted"}),
    (_t(7), {"A": "disrupted", "Q|north|Q05N": "normal"}),
    # 8: A goes straight to suspended (second consecutive non-disrupted call
    # would be needed to commit); segment's second normal commits at t(7).
    (_t(8), {"A": "suspended", "Q|north|Q05N": "normal"}),
    (_t(9), {"A": "suspended", "Q|north|Q05N": "normal"}),
    # 10: a brand-new cell appears mid-window.
    (_t(10), {"A": "suspended", "Q|north|Q05N": "normal", "G|south|G22S": "disrupted"}),
    # Long gap: A and the segment are unobserved past MAX_IDLE_SEC and evict.
    (_t(10) + MAX_IDLE_SEC + TICK_SECONDS, {"G|south|G22S": "disrupted"}),
]


def build_fixture() -> dict[str, object]:
    entries: dict[str, object] = {}
    steps: list[dict[str, object]] = []
    state = {}
    for observed_at, calls in TICKS:
        state, changes = advance_regimes(
            state,
            calls,
            observed_at,
            debounce_ticks=DEBOUNCE_TICKS,
            max_idle_sec=MAX_IDLE_SEC,
        )
        steps.append(
            {
                "observed_at": observed_at,
                "observed": calls,
                "entries": {k: asdict(v) for k, v in sorted(state.items())},
                "changes": [asdict(c) for c in changes],
            }
        )
    entries = {k: asdict(v) for k, v in sorted(state.items())}
    return {
        "debounce_ticks": DEBOUNCE_TICKS,
        "max_idle_sec": MAX_IDLE_SEC,
        "steps": steps,
        "final_entries": entries,
    }


def main() -> int:
    FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE_PATH.write_text(json.dumps(build_fixture(), indent=2) + "\n")
    print(f"wrote {FIXTURE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
