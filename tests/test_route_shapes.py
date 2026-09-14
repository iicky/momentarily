"""Route shape extraction from GTFS static shapes.txt."""

from __future__ import annotations

from tests.conftest import make_gtfs_zip
from training.gtfs_static import (
    rdp,
    read_shapes,
    route_shapes,
    shapes_to_json,
)

# Two trips on route A, shape_id "shA_N" northbound, "shA_S" southbound.
# One trip on route 1, shape_id "sh1_N" northbound.
TRIPS = [
    "A,ASP26GEN-1038-Weekday-00_063400_A..N,Weekday,Inwood,1,shA_N",
    "A,ASP26GEN-1038-Weekday-00_064500_A..S,Weekday,Far Rockaway,0,shA_S",
    "1,1SP26GEN-1038-Weekday-00_070000_1..N,Weekday,Van Cortlandt,1,sh1_N",
]

STOP_TIMES = [
    # A northbound trip
    "ASP26GEN-1038-Weekday-00_063400_A..N,A02N,06:34:00,06:34:00,1",
    "ASP26GEN-1038-Weekday-00_063400_A..N,A05N,06:38:00,06:38:00,2",
    # A southbound trip
    "ASP26GEN-1038-Weekday-00_064500_A..S,A05S,06:45:00,06:45:00,1",
    "ASP26GEN-1038-Weekday-00_064500_A..S,A02S,06:49:00,06:49:00,2",
    # 1 northbound trip
    "1SP26GEN-1038-Weekday-00_070000_1..N,101N,07:00:00,07:00:00,1",
    "1SP26GEN-1038-Weekday-00_070000_1..N,103N,07:05:00,07:05:00,2",
]

SHAPES = [
    # shA_N: 4 points, a gentle curve
    "shA_N,40.700000,-74.000000,1",
    "shA_N,40.710000,-73.995000,2",
    "shA_N,40.720000,-73.990000,3",
    "shA_N,40.730000,-73.985000,4",
    # shA_S: 3 points
    "shA_S,40.730000,-73.985000,1",
    "shA_S,40.710000,-73.995000,2",
    "shA_S,40.700000,-74.000000,3",
    # sh1_N: 2 points
    "sh1_N,40.800000,-73.900000,1",
    "sh1_N,40.810000,-73.895000,2",
]


def test_read_shapes() -> None:
    zf = make_gtfs_zip(TRIPS, STOP_TIMES, shapes_rows=SHAPES)
    raw = read_shapes(zf)
    assert set(raw) == {"shA_N", "shA_S", "sh1_N"}
    # shA_N: 4 points, ordered by sequence
    assert len(raw["shA_N"]) == 4
    assert raw["shA_N"][0] == (40.7, -74.0)
    assert raw["shA_N"][-1] == (40.73, -73.985)


def test_read_shapes_absent() -> None:
    """No shapes.txt → empty dict, not a crash."""
    zf = make_gtfs_zip(TRIPS, STOP_TIMES)
    assert read_shapes(zf) == {}


def test_route_shapes_keys_and_direction() -> None:
    zf = make_gtfs_zip(TRIPS, STOP_TIMES, shapes_rows=SHAPES)
    shapes = route_shapes(zf)
    assert ("A", "north") in shapes
    assert ("A", "south") in shapes
    assert ("1", "north") in shapes
    assert len(shapes) == 3


def test_route_shapes_longest_wins() -> None:
    """When two shapes map to the same (route, direction), keep the longest."""
    extra_trips = [
        *TRIPS,
        # A second northbound A trip using a shorter shape
        "A,ASP26GEN-1038-Weekday-00_090000_A..N,Weekday,Inwood,1,shA_N_short",
    ]
    extra_stop_times = [
        *STOP_TIMES,
        "ASP26GEN-1038-Weekday-00_090000_A..N,A02N,09:00:00,09:00:00,1",
        "ASP26GEN-1038-Weekday-00_090000_A..N,A05N,09:04:00,09:04:00,2",
    ]
    # Long shape: 4 points with a >tolerance bend at point 2 so RDP keeps it.
    # Short shape: 2 points with DIFFERENT endpoints so we can tell which won.
    long_shape = [
        "shA_N_long,40.700000,-74.000000,1",
        "shA_N_long,40.710000,-73.980000,2",  # ~0.015° off the straight line
        "shA_N_long,40.720000,-73.990000,3",
        "shA_N_long,40.730000,-73.985000,4",
    ]
    short_shape = [
        "shA_N_short,40.750000,-73.970000,1",
        "shA_N_short,40.760000,-73.965000,2",
    ]
    # Replace shA_N in SHAPES with the two competing shapes
    base_shapes = [s for s in SHAPES if not s.startswith("shA_N,")]
    all_shapes = [*base_shapes, *long_shape, *short_shape]
    # Remap both trips to the two competing shape ids
    remapped_trips = [
        t.replace(",shA_N", ",shA_N_long") if "shA_N" in t and "short" not in t else t
        for t in extra_trips
    ]
    zf = make_gtfs_zip(remapped_trips, extra_stop_times, shapes_rows=all_shapes)
    shapes = route_shapes(zf)
    pts = shapes[("A", "north")]
    # The long shape starts at (40.7, -74.0); the short starts at (40.75, -73.97).
    # Assert the long shape's start to prove it was selected.
    assert pts[0] == (40.7, -74.0)
    # The bend point at (40.71, -73.98) deviates well past tolerance and survives RDP.
    assert (40.71, -73.98) in pts
    assert len(pts) >= 3


def test_shapes_to_json() -> None:
    zf = make_gtfs_zip(TRIPS, STOP_TIMES, shapes_rows=SHAPES)
    shapes = route_shapes(zf)
    js = shapes_to_json(shapes)
    assert "A|north" in js
    assert "A|south" in js
    assert "1|north" in js
    coords = js["A|north"]["coordinates"]
    assert len(coords) == len(shapes[("A", "north")])
    # Each coordinate is [lat, lon]
    assert len(coords[0]) == 2


def test_rdp_identity_short() -> None:
    """Fewer than 3 points are returned unchanged."""
    pts = [(0.0, 0.0), (1.0, 1.0)]
    assert rdp(pts, 0.1) == pts
    assert rdp([], 0.1) == []


def test_rdp_simplifies_collinear() -> None:
    """Points on a straight line collapse to the endpoints."""
    pts = [(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)]
    result = rdp(pts, 0.01)
    assert result == [(0.0, 0.0), (1.0, 1.0)]


def test_rdp_preserves_deviation() -> None:
    """A point deviating more than tolerance is kept."""
    pts = [(0.0, 0.0), (0.5, 0.1), (1.0, 0.0)]
    result = rdp(pts, 0.001)
    assert len(result) == 3
