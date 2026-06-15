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
    preds = [_Pred("1", "Delays"), _Pred("A", "Suspended"), _Pred("F", "Planned - Work")]
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
    preds = [_Pred("1", "Planned - Stations Skipped"), _Pred("1", "No Northbound Service")]
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
