"""Unmapped-alert_type input-drift signal (training/drift.py).

Synthetic prediction stubs — no R2. Only route + primary_alert_type matter.
"""

from __future__ import annotations

from dataclasses import dataclass

from training.drift import unmapped_alert_type_drift


@dataclass
class _Pred:
    route: str
    primary_alert_type: str | None


def test_all_mapped_is_zero_drift():
    preds = [
        _Pred("1", "Delays"),
        _Pred("A", "Suspended"),
        _Pred("F", "Planned - Work"),
    ]
    d = unmapped_alert_type_drift(preds)
    assert d["unmapped_rate"] == 0.0
    assert d["unmapped_types"] == {}
    assert d["by_route"] == {}
    assert d["n_typed_ticks"] == 3


def test_unmapped_types_are_named_and_rated():
    preds = [
        _Pred("1", "Delays"),  # known
        _Pred("1", "Meteor Strike"),  # unknown
        _Pred("A", "Meteor Strike"),  # unknown
        _Pred("A", "Gremlin Incursion"),  # unknown
    ]
    d = unmapped_alert_type_drift(preds)
    assert d["unmapped_rate"] == 0.75  # 3 of 4 typed ticks
    # Most frequent offender first; each is a mapping row to add.
    assert list(d["unmapped_types"]) == ["Meteor Strike", "Gremlin Incursion"]
    assert d["unmapped_types"]["Meteor Strike"] == 2
    assert d["by_route"] == {"1": 0.5, "A": 1.0}


def test_null_alert_type_excluded_from_denominator():
    preds = [_Pred("1", None), _Pred("1", None), _Pred("1", "Nonsense Type")]
    d = unmapped_alert_type_drift(preds)
    assert d["n_typed_ticks"] == 1  # the two None ticks don't count
    assert d["unmapped_rate"] == 1.0


def test_empty_is_safe():
    d = unmapped_alert_type_drift([])
    assert d["unmapped_rate"] == 0.0
    assert d["n_typed_ticks"] == 0


def test_passthrough_families_are_known():
    # "Planned -*" and "No <Dir> Service" are recognized by is_known_alert_type.
    preds = [
        _Pred("1", "Planned - Stations Skipped"),
        _Pred("1", "No Northbound Service"),
    ]
    d = unmapped_alert_type_drift(preds)
    assert d["unmapped_rate"] == 0.0


def test_hmm_excluded_types_are_not_drift():
    # Extra Service / No Scheduled Service are deliberately HMM-ignored — they
    # must not count as unmapped (or even land in the denominator), else the
    # panel cries wolf on handled types. Only the real unknown counts.
    preds = [
        _Pred("C", "Extra Service"),
        _Pred("C", "No Scheduled Service"),
        _Pred("C", "Meteor Strike"),
    ]
    d = unmapped_alert_type_drift(preds)
    assert d["n_typed_ticks"] == 1
    assert d["unmapped_rate"] == 1.0
    assert d["unmapped_types"] == {"Meteor Strike": 1}


# --- emission-channel distribution drift ------------------------------------

from momentarily.hmm import Observation  # noqa: E402
from training.drift import (  # noqa: E402
    build_input_profile,
    emission_channel_drift,
)
from training.load import TickObservation  # noqa: E402


def _tick(
    route: str,
    tod: int,
    count: int,
    *,
    susp: bool = False,
    delays: bool = False,
    sc: bool = False,
    planned: bool = False,
) -> TickObservation:
    return TickObservation(
        route_id=route,
        tick=0,
        observation=Observation(
            alert_count=count,
            severity_sum=0,
            has_suspended_alert=susp,
            has_delays=delays,
            has_service_change=sc,
            has_planned=planned,
            tod_bin=tod,
        ),
    )


def test_build_input_profile_bins_and_rates():
    ticks = [
        _tick("1", 0, 0),
        _tick("1", 0, 2, delays=True),
        _tick("1", 0, 7, susp=True),  # 7 -> "6+" bin (index 5)
        _tick("1", 0, 4),  # 4 -> "4-5" bin (index 4)
    ]
    cell = build_input_profile(ticks)["1"]["0"]
    assert cell["n"] == 4
    assert cell["hist"] == [1, 0, 1, 0, 1, 1]
    assert cell["flags"]["delays"] == 0.25
    assert cell["flags"]["suspended"] == 0.25
    assert cell["flags"]["planned"] == 0.0


def test_emission_drift_identical_is_zero():
    prof = build_input_profile([_tick("1", 0, 2) for _ in range(40)])
    d = emission_channel_drift(prof, prof, min_n=30)
    assert d["by_route"]["1"]["max_alert_count_psi"] == 0.0
    assert d["routes_drifted"] == []
    assert d["cells_scored"] == 1


def test_emission_drift_thin_cells_skipped():
    prof = build_input_profile([_tick("1", 0, 2) for _ in range(5)])  # < min_n
    d = emission_channel_drift(prof, prof, min_n=30)
    assert d["cells_scored"] == 0
    assert d["cells_skipped_thin"] == 1
    assert d["by_route"] == {}


def test_emission_drift_distribution_shift_is_significant():
    ref = build_input_profile([_tick("1", 0, 0) for _ in range(50)])  # always quiet
    cur = build_input_profile(
        [_tick("1", 0, 6, susp=True) for _ in range(50)]
    )  # always busy + suspended
    d = emission_channel_drift(ref, cur, min_n=30)
    cell = d["by_route"]["1"]
    assert cell["max_alert_count_psi"] > 0.25
    assert cell["significant"] is True
    assert cell["max_flag_delta"] == 1.0
    assert cell["max_flag_delta_channel"] == "suspended"
    assert d["routes_drifted"] == ["1"]


def test_emission_drift_empty_reference_is_unavailable_shape():
    d = emission_channel_drift({}, build_input_profile([_tick("1", 0, 2)]))
    assert d["by_route"] == {}
    assert d["routes_drifted"] == []
