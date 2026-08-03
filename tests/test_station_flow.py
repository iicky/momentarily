"""Per-station service-flow roll-up (training/station_flow.py)."""

from __future__ import annotations

from training.reliability import SegmentScore
from training.station_flow import station_flow, station_id


def _seg(frm: str, to: str, deficit: float, route: str = "F") -> SegmentScore:
    return SegmentScore(
        route=route,
        direction="south",
        from_stop=frm,
        to_stop=to,
        deficit=deficit,
        p0=0.9,
        observed_rate=0.9 * (1 - deficit),
        n=500,
        share=0.9,
        is_terminal_like=False,
        percentile=0.5,
    )


def test_station_id_strips_direction_suffix():
    assert station_id("A09N") == "A09"
    assert station_id("A09S") == "A09"
    assert station_id("A09") == "A09"  # no suffix left untouched
    assert station_id("") == ""


def test_segment_is_incident_to_both_endpoint_stations():
    flows = {f.station: f for f in station_flow([_seg("A09S", "A10S", 0.1)])}
    assert set(flows) == {"A09", "A10"}


def test_station_degraded_by_worst_incident_segment():
    # A08->A09 is fine; A09->A10 is stalled. Station A09 (touched by both) reads
    # degraded off the worst incident segment.
    flows = {
        f.station: f
        for f in station_flow([_seg("A08S", "A09S", 0.1), _seg("A09S", "A10S", 0.8)])
    }
    assert flows["A09"].status == "degraded"
    assert flows["A09"].worst_deficit == 0.8
    assert flows["A09"].worst_segment == ("A09S", "A10S")
    assert flows["A09"].n_segments == 2
    assert flows["A08"].status == "flowing"


def test_station_flowing_when_all_incident_below_threshold():
    flows = {f.station: f for f in station_flow([_seg("A09S", "A10S", 0.2)])}
    assert flows["A09"].status == "flowing"


def test_routes_aggregated_and_sorted_worst_first():
    flows = station_flow(
        [
            _seg("A09S", "A10S", 0.2, route="F"),
            _seg("A09S", "A10S", 0.9, route="G"),  # same station, worse, other route
            _seg("B01S", "B02S", 0.6, route="F"),
        ]
    )
    a09 = next(f for f in flows if f.station == "A09")
    assert a09.routes == ["F", "G"]
    assert flows[0].worst_deficit >= flows[-1].worst_deficit  # sorted worst first
