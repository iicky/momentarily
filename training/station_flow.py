"""Per-station service-flow roll-up from the segment reliability model (vhh.8).

The rider question is "is MY station moving", not "is the A line disrupted". A
station's service flow is the throughput of the segments that touch it: a segment
(route, direction, from_stop -> to_stop) is incident to both its endpoint stations,
so a station is flowing when the segments entering and leaving it are advancing near
their own normal, and degraded when one of them has stalled below it.

This rolls the segment reliability scores (training.reliability.SegmentScore) up to
the station, keyed on the GTFS stop id with its N/S direction suffix stripped, so
both directions of a stop share one station verdict. It is the same deficit-vs-own-
normal signal, aggregated — the scorecard, the segment call, and the station surface
never disagree about what "normal" is.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass

from training.reliability import SegmentScore

# A station reads degraded when its worst incident segment ran at or below this much
# of its own normal advance rate over the window (i.e. deficit at/above this).
DEGRADED_DEFICIT = 0.5


def station_id(stop_id: str) -> str:
    """Collapse a directional stop id to its station: strip a trailing N/S suffix
    ('A09N' -> 'A09'); leave ids without one untouched."""
    return stop_id[:-1] if stop_id and stop_id[-1] in ("N", "S") else stop_id


@dataclass(frozen=True)
class StationFlow:
    station: str
    status: str  # "flowing" | "degraded"
    worst_deficit: float
    worst_segment: tuple[str, str] | None  # (from_stop, to_stop) of the worst segment
    routes: list[str]  # routes whose segments touch this station
    n_segments: int  # incident segments considered


def station_flow(
    segments: Iterable[SegmentScore],
    *,
    degraded_deficit: float = DEGRADED_DEFICIT,
) -> list[StationFlow]:
    """Roll segment reliability scores up to per-station service flow. Each segment
    is incident to both endpoint stations; a station is degraded when its worst
    incident segment's deficit is at or above degraded_deficit. Sorted worst first."""
    incident: dict[str, list[SegmentScore]] = {}
    for seg in segments:
        for sid in {station_id(seg.from_stop), station_id(seg.to_stop)}:
            incident.setdefault(sid, []).append(seg)

    out: list[StationFlow] = []
    for sid, segs in incident.items():
        worst = max(segs, key=lambda s: s.deficit)
        out.append(
            StationFlow(
                station=sid,
                status="degraded" if worst.deficit >= degraded_deficit else "flowing",
                worst_deficit=worst.deficit,
                worst_segment=(worst.from_stop, worst.to_stop),
                routes=sorted({s.route for s in segs}),
                n_segments=len(segs),
            )
        )
    out.sort(key=lambda s: s.worst_deficit, reverse=True)
    return out


def station_flow_json(segments: Iterable[SegmentScore]) -> list[dict[str, object]]:
    """Serialize the station flow roll-up for summary.json (viz Models page)."""
    return [asdict(s) for s in station_flow(segments)]
