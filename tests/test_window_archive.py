"""The long-lived announced-window archive (training/window_archive.py).

This is the ANSWER KEY half of the derived storage. It outlives
`archive/alerts/` (pruned at 90 days), so once those are gone a stored window
cannot be regenerated or checked — the same one-way door the traversal archive
faces, and the same defences apply.
"""

from __future__ import annotations

import gzip
import json
from datetime import date
from typing import Any

import pytest

from training.planned_work import Window, windows_from_alerts
from training.traversal_archive import DayProvenance
from training.window_archive import (
    PARSER_VERSION,
    SCHEMA_VERSION,
    WindowReadResult,
    decode_day,
    encode_day,
    key_for,
)


def _prov(*, extractor: int = PARSER_VERSION) -> DayProvenance:
    return DayProvenance(
        schema=SCHEMA_VERSION,
        extractor=extractor,
        feed_version=None,
        feed_digest=None,
        n_rows=1,
        n_source_objects=1,
        source_manifest="x",
        code_sha="deadbeef",
        written_at=T0,
    )


T0 = 1_786_552_200
HOUR = 3600


def _alert(
    *,
    alert_type: str = "Planned - Part Suspended",
    routes: list[str] | None = None,
    stops: list[str] | None = None,
    periods: list[tuple[int, int]] | None = None,
) -> dict[str, Any]:
    """One archived alert version, in the shape the live feed publishes."""
    base = routes if routes is not None else ["J"]
    entities: list[dict[str, Any]] = [{"route_id": r} for r in base]
    entities += [
        {"route_id": base[0], "stop_id": s}
        for s in (stops if stops is not None else ["J20", "J21"])
    ]
    return {
        "alert": {
            "alert": {
                "transit_realtime.mercury_alert": {"alert_type": alert_type},
                "informed_entity": entities,
                "active_period": [
                    {"start": s, "end": e}
                    for s, e in (periods or [(T0, T0 + 4 * HOUR)])
                ],
            }
        }
    }


def _window(start: int = T0, alert_type: str = "Planned - Part Suspended") -> Window:
    return Window(
        alert_type=alert_type,
        routes=frozenset({"J"}),
        stops=frozenset({"J20", "J21"}),
        start=start,
        end=start + 4 * HOUR,
    )


def _doc(windows: list[Window] | None = None) -> dict[str, Any]:
    blob = encode_day(windows or [_window()], source_keys=["k"])
    return json.loads(gzip.decompress(blob))


def test_a_day_of_windows_round_trips_including_the_named_stops():
    """Routes and stops are frozensets serialized as sorted lists; losing or
    reordering them would silently change which hops a grade calls affected."""
    windows = [_window(), _window(start=T0 + 86400, alert_type="Planned - Reroute")]
    got, prov = decode_day(encode_day(windows, source_keys=["a", "b"]))
    assert set(got) == set(windows)
    assert prov.schema == SCHEMA_VERSION
    assert prov.extractor == PARSER_VERSION
    assert (prov.n_rows, prov.n_source_objects) == (2, 2)


def test_an_unreadable_schema_raises_instead_of_parsing_partially():
    """The alert versions that could have checked a mis-parse are pruned."""
    doc = _doc()
    doc["provenance"]["schema"] = SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match="schema"):
        decode_day(gzip.compress(json.dumps(doc).encode()))


def test_reordered_fields_are_rejected_rather_than_silently_transposed():
    """Positional rows: a reorder would swap routes with stops and still parse."""
    doc = _doc()
    doc["fields"] = list(reversed(doc["fields"]))
    with pytest.raises(ValueError, match="field order"):
        decode_day(gzip.compress(json.dumps(doc).encode()))


def test_missing_provenance_counts_raise_rather_than_defaulting_to_zero():
    """A renamed field must not decode as "0 windows from 0 sources"."""
    doc = _doc()
    del doc["provenance"]["n_source_objects"]
    with pytest.raises(ValueError, match="missing"):
        decode_day(gzip.compress(json.dumps(doc).encode()))


def test_keys_are_one_object_per_day_under_a_date_segment():
    assert key_for(date(2026, 8, 12)) == "archive/windows/2026-08-12/windows.json.gz"


def test_a_span_crossing_a_parser_change_reports_itself_as_mixed():
    """Two answer keys built by different parsers are not one answer key, and
    after the alerts prune nothing else could reveal it."""
    mixed = WindowReadResult(
        windows=[],
        provenance={
            date(2026, 8, 12): _prov(extractor=1),
            date(2026, 8, 13): _prov(extractor=2),
        },
    )
    assert not mixed.homogeneous
    same = WindowReadResult(
        windows=[],
        provenance={date(2026, 8, 12): _prov(), date(2026, 8, 13): _prov()},
    )
    assert same.homogeneous


def test_the_parser_semantics_are_pinned_so_a_change_cannot_pass_silently():
    """THE GUARD ON PARSER_VERSION.

    `planned_work.windows_from_alerts` decides which alert types count, how the
    named stations are scoped, and how active periods become windows. If any of
    that changes, every answer key already written disagrees with every one
    written afterwards — and the alert versions that could re-derive the old days
    are pruned at 90 days. No downstream grade would fail; they would all keep
    producing plausible numbers against a subtly different answer key.

    When this fails, the change is real. Bump PARSER_VERSION and re-derive
    whatever alert history still survives; do NOT edit the expectation.
    """
    got = windows_from_alerts([_alert()])
    assert len(got) == 1
    w = got[0]
    assert w.alert_type == "Planned - Part Suspended"
    assert w.routes == frozenset({"J"})
    # Parent-station scoping: the alert names J20/J21 and the trace reports
    # directional platforms, so the stored form must stay parent ids.
    assert w.stops == frozenset({"J20", "J21"})
    assert (w.start, w.end) == (T0, T0 + 4 * HOUR)
    assert w.gradeable is True

    # A second active period on one alert is a second window, not one merged span.
    two = windows_from_alerts(
        [_alert(periods=[(T0, T0 + HOUR), (T0 + 86400, T0 + 86400 + HOUR)])]
    )
    assert len(two) == 2

    # A type a traversal measure cannot see must still parse, and must declare
    # itself ungradeable rather than being dropped and counted as a miss.
    boarding = windows_from_alerts([_alert(alert_type="Boarding Change")])
    assert len(boarding) == 1
    assert boarding[0].gradeable is False

    assert PARSER_VERSION == 1, (
        "windows_from_alerts semantics changed above; bump PARSER_VERSION and "
        "re-derive surviving alert history rather than editing the expectation"
    )


def test_the_stored_form_survives_a_round_trip_through_the_parser():
    """End to end: what the parser produces is exactly what a reader gets back,
    because a grade run years from now uses the decoded form, not the parser."""
    parsed = windows_from_alerts([_alert(), _alert(alert_type="Planned - Reroute")])
    got, _prov = decode_day(encode_day(parsed, source_keys=["k"]))
    assert set(got) == set(parsed)
    assert all(w.gradeable for w in got if w.alert_type != "Boarding Change")


def test_gradeability_survives_the_round_trip_because_it_derives_from_the_type():
    """`Window.gradeable` is a property computed from `alert_type`, not a stored
    field, so it is preserved by storing the type — verified rather than assumed,
    since an ungradeable window later read as gradeable would be counted as a
    detection miss against work the measure cannot see.

    What it DOES depend on is `planned_work.SERVICE_CHANGING`. That set is part of
    the parser's semantics, so a change to it must bump PARSER_VERSION — which the
    golden test above enforces by pinning both a gradeable and an ungradeable
    type.
    """
    ungradeable = _window(alert_type="Boarding Change")
    gradeable = _window(alert_type="Planned - Part Suspended")
    assert (ungradeable.gradeable, gradeable.gradeable) == (False, True)

    rows, _prov_out = decode_day(
        encode_day([ungradeable, gradeable], source_keys=["k"])
    )
    by_type = {w.alert_type: w for w in rows}
    assert by_type["Boarding Change"].gradeable is False
    assert by_type["Planned - Part Suspended"].gradeable is True
    assert set(rows) == {ungradeable, gradeable}
