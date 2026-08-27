"""Generate the committed station-facts sidecar the Station page renders.

Joins our ~496 subway stations (NYS Open Data 39hk-dx4f, the same coordinate
source the viz `/api/stations` route and the Worker read) to two reference
bodies and writes ONE static JSON, `viz/public/station_facts.json`, committed
to the repo. It is a build-time artifact, never a live runtime dependency: the
Station page fetches the committed file exactly the way the map fetches
`diagram.json`, with no network call of its own.

Sources
-------
* Wikidata (`P16 = Q7733`, "part of the New York City Subway") — opening date
  (P1619), a Commons photo (P18, carried with its license/attribution), and any
  heritage designation (P1435). Wikidata has no GTFS-id property, so a station
  item is tied to one of our GTFS stops through, in priority order:

    1. transitland_onestop  — the AUTHORITATIVE map. Each item carries one or
       more Transitland Onestop IDs (P11109); Transitland resolves an Onestop ID
       to the exact GTFS stop_id of the feed it imported (the same NYCT feed our
       ids come from). Requires TRANSITLAND_API_KEY; used when set. This is the
       only tier that reads a real GTFS stop_id rather than inferring one.
    2. onestop_geohash      — FALLBACK. The Onestop ID embeds a 10-char geohash
       of Transitland's own per-platform coordinate (sub-metre against this feed
       — Transitland derived it from the same MTA stops). Decoded locally and
       nearest-matched. A spatial join on the Onestop ID's location, NOT a
       stop_id read: it is recorded and logged as a fallback.
    3. coordinate           — FALLBACK. Nearest item-level P625 coordinate, for
       items that carry no Onestop ID at all.

  OSM node ID (P11693) is a cross-check: the matched item's node coordinate is
  fetched from OpenStreetMap and compared to our stop. Every station carries its
  match method, whether it is authoritative, a confidence, the distance, and the
  cross-check result. Stations we cannot match are logged, never guessed.

* data.ny.gov — annual ridership rank per station from the MTA Subway Station
  Monthly Ridership feed (ak4z-sape), summed over the trailing twelve COMPLETE
  months. Rank is taken only across complexes that cover the full window, so a
  station open for part of it never distorts the ordering; a complex missing
  months carries ridership but no rank. Ridership is keyed by complex, so every
  GTFS stop of a complex shares its complex's total and rank.

  The MTA Permanent Art Catalog (4y8j-9pkd) is joined at the complex level by
  normalised station name and route overlap. It carries no image and no stable
  station id, so it ships as credits (artist, title, year, material, a link to
  the MTA collection page), never as a photo, and unmatched pieces are logged.

Determinism
-----------
No wall-clock stamp: output is sorted keys + trailing newline, so a re-run with
unchanged upstream data produces a byte-identical file and an empty diff. The
ridership window (real month bounds) is the only date carried, as provenance.

Run:  uv run python -m scripts.gen_station_facts
      TRANSITLAND_API_KEY=... uv run python -m scripts.gen_station_facts
      uv run python -m scripts.gen_station_facts --no-osm --no-art
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, cast
from urllib.parse import unquote

import httpx

FACTS_PATH = (
    Path(__file__).resolve().parent.parent / "viz" / "public" / "station_facts.json"
)

USER_AGENT = (
    "momentarily-station-facts/1.0 "
    "(https://momentarily.nyc; build script; contact via project repo)"
)

STATIONS_URL = "https://data.ny.gov/resource/39hk-dx4f.json"
RIDERSHIP_URL = "https://data.ny.gov/resource/ak4z-sape.json"
ART_URL = "https://data.ny.gov/resource/4y8j-9pkd.json"
SPARQL_URL = "https://query.wikidata.org/sparql"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
# Overpass mirrors, tried in order per batch so one rate-limiting doesn't blank
# the cross-check for a whole run.
OVERPASS_URLS = (
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
)
TRANSITLAND_STOPS_URL = "https://transit.land/api/v2/rest/stops"

# NYCT subway feed on Transitland; its imported stop_ids are our gtfs_stop_ids.
TRANSITLAND_FEED = "f-dr5r-nyctsubway"

# Match thresholds, metres. The Onestop geohash lands sub-metre on this feed, so
# 75 m only ever admits a correct platform while rejecting the two items whose
# Onestop set names a different complex; those fall to the item-coordinate tier,
# which is looser because P625 is a single rounded point per complex.
ONESTOP_MAX_M = 75.0
COORD_MAX_M = 200.0
# OSM node agrees with our stop if within this far. Nodes sit on a platform or
# an entrance, so the spread across a complex is real, not error.
OSM_AGREE_M = 250.0
# A spatial fallback that finds distinct items this close together cannot tell
# them apart (some share an identical geohash); it abstains and logs unmatched
# rather than guess. The authoritative Transitland tier has no such tie.
TIE_EPS_M = 2.0

_GEOHASH_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def haversine_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle metres between two (lat, lon) points."""
    r = 6_371_000.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * r * math.asin(math.sqrt(h))


def geohash_decode(gh: str) -> tuple[float, float]:
    """Centre (lat, lon) of a geohash cell."""
    lat: tuple[float, float] = (-90.0, 90.0)
    lon: tuple[float, float] = (-180.0, 180.0)
    even = True
    for ch in gh:
        cd = _GEOHASH_B32.index(ch)
        for mask in (16, 8, 4, 2, 1):
            if even:
                mid = (lon[0] + lon[1]) / 2
                lon = (mid, lon[1]) if cd & mask else (lon[0], mid)
            else:
                mid = (lat[0] + lat[1]) / 2
                lat = (mid, lat[1]) if cd & mask else (lat[0], mid)
            even = not even
    return (lat[0] + lat[1]) / 2, (lon[0] + lon[1]) / 2


_ONESTOP_GEOHASH = re.compile(r"^s-([0-9b-hjkmnp-z]+)-")


def onestop_latlon(onestop_id: str) -> tuple[float, float] | None:
    """The Transitland per-platform coordinate a stop Onestop ID embeds."""
    m = _ONESTOP_GEOHASH.match(onestop_id)
    return geohash_decode(m.group(1)) if m else None


def parse_point(wkt: str | None) -> tuple[float, float] | None:
    """(lat, lon) from a WKT `Point(lon lat)` literal."""
    if not wkt:
        return None
    m = re.match(r"Point\(([-\d.]+) ([-\d.]+)\)", wkt)
    return (float(m.group(2)), float(m.group(1))) if m else None


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class Station:
    gtfs_stop_id: str
    name: str
    lat: float
    lon: float
    complex_id: str | None
    routes: frozenset[str]


@dataclass
class WikiItem:
    qid: str
    label: str | None = None
    coord: tuple[float, float] | None = None
    onestops: set[str] = field(default_factory=set[str])
    osm_nodes: set[str] = field(default_factory=set[str])
    opened: set[str] = field(default_factory=set[str])
    heritage: set[str] = field(default_factory=set[str])
    images: set[str] = field(default_factory=set[str])


@dataclass
class Photo:
    title: str
    source: str
    image_url: str
    thumb_url: str | None
    artist: str | None
    license: str | None
    license_url: str | None
    credit: str | None
    attribution_required: bool


@dataclass
class ArtPiece:
    artist: str | None
    title: str | None
    year: str | None
    material: str | None
    link: str | None


@dataclass
class Match:
    qid: str
    method: str  # transitland_onestop | onestop_geohash | coordinate
    authoritative: bool
    confidence: float
    distance_m: float | None
    onestop_id: str | None
    osm_node: str | None
    osm_agrees: bool | None = None
    osm_distance_m: float | None = None


# --------------------------------------------------------------------------- #
# Fetchers
# --------------------------------------------------------------------------- #
def _client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=120.0)


def fetch_stations(client: httpx.Client) -> list[Station]:
    rows: list[dict[str, Any]] = (
        client.get(STATIONS_URL, params={"$limit": "2000"}).raise_for_status().json()
    )
    out: list[Station] = []
    for r in rows:
        try:
            lat = float(r["gtfs_latitude"])
            lon = float(r["gtfs_longitude"])
        except (KeyError, TypeError, ValueError):
            continue
        sid = r.get("gtfs_stop_id")
        if not sid:
            continue
        out.append(
            Station(
                gtfs_stop_id=sid,
                name=r.get("stop_name", sid),
                lat=lat,
                lon=lon,
                complex_id=r.get("complex_id"),
                routes=frozenset((r.get("daytime_routes") or "").split()),
            )
        )
    return out


_WD_QUERY = """
SELECT ?s ?sLabel ?coord ?onestop ?osm ?opened ?heritageLabel ?img WHERE {
  ?s wdt:P16 wd:Q7733 .
  OPTIONAL { ?s wdt:P625 ?coord }
  OPTIONAL { ?s wdt:P11109 ?onestop }
  OPTIONAL { ?s wdt:P11693 ?osm }
  OPTIONAL { ?s wdt:P1619 ?opened }
  OPTIONAL { ?s wdt:P1435 ?heritage . ?heritage rdfs:label ?heritageLabel .
             FILTER(LANG(?heritageLabel) = "en") }
  OPTIONAL { ?s wdt:P18 ?img }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "en" }
}
"""


def fetch_wikidata(client: httpx.Client) -> dict[str, WikiItem]:
    resp = client.post(
        SPARQL_URL,
        data={"query": _WD_QUERY, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
    ).raise_for_status()
    items: dict[str, WikiItem] = {}
    for row in resp.json()["results"]["bindings"]:
        qid = row["s"]["value"].rsplit("/", 1)[-1]
        it = items.setdefault(qid, WikiItem(qid=qid))
        if "sLabel" in row:
            it.label = row["sLabel"]["value"]
        if "coord" in row and it.coord is None:
            it.coord = parse_point(row["coord"]["value"])
        if "onestop" in row:
            it.onestops.add(row["onestop"]["value"])
        if "osm" in row:
            it.osm_nodes.add(row["osm"]["value"])
        if "opened" in row:
            it.opened.add(row["opened"]["value"])
        if "heritageLabel" in row:
            it.heritage.add(row["heritageLabel"]["value"])
        if "img" in row:
            it.images.add(row["img"]["value"])
    return items


def _latest_month(client: httpx.Client) -> date:
    row: list[dict[str, str]] = (
        client.get(RIDERSHIP_URL, params={"$select": "max(month) as m"})
        .raise_for_status()
        .json()
    )
    return date.fromisoformat(row[0]["m"][:10])


def window_start(latest: date) -> date:
    """First of the month twelve complete months back, inclusive."""
    y, m = latest.year, latest.month
    # 12 months inclusive of `latest` => start 11 months earlier.
    total = (y * 12 + (m - 1)) - 11
    return date(total // 12, total % 12 + 1, 1)


def fetch_ridership(
    client: httpx.Client,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """complex_id -> {rank, of, annual_entries}, and the window bounds.

    Rank spans only complexes covering all twelve months, so the ordering is a
    like-for-like comparison; a partial complex keeps its total but no rank.
    """
    latest = _latest_month(client)
    start = window_start(latest)
    rows: list[dict[str, str]] = (
        client.get(
            RIDERSHIP_URL,
            params={
                "$select": "station_complex_id, sum(ridership) as total, count(*) as months",
                "$where": (
                    f"month >= '{start.isoformat()}T00:00:00' "
                    f"and month <= '{latest.isoformat()}T00:00:00'"
                ),
                "$group": "station_complex_id",
                "$limit": "5000",
            },
        )
        .raise_for_status()
        .json()
    )

    complete = [r for r in rows if int(r["months"]) == 12]
    # Deterministic order: total desc, complex id asc as a stable tie-break so a
    # tie can't reshuffle ranks between runs.
    complete.sort(key=lambda r: (-float(r["total"]), r["station_complex_id"]))
    n = len(complete)
    ranked: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(complete, start=1):
        cid = r["station_complex_id"]
        ranked[cid] = {
            "rank": i,
            "of": n,
            "annual_entries": round(float(r["total"])),
        }
    window = {"start": start.strftime("%Y-%m"), "end": latest.strftime("%Y-%m")}
    return ranked, window


def _norm_name(s: str) -> str:
    s = s.lower().replace("&", "and")
    s = re.sub(r"[–—]", "-", s)
    s = re.sub(r"\bstreet\b", "st", s)
    s = re.sub(r"\bavenue\b", "av", s)
    s = re.sub(r"\broad\b", "rd", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def fetch_art(
    client: httpx.Client, stations: list[Station]
) -> tuple[dict[str, list[ArtPiece]], list[str]]:
    """gtfs_stop_id -> art pieces, and the names of pieces left unplaced.

    Subway pieces only. The catalog keys art by free-text station name + line,
    with no stable id, and a name like "23 St" is several different complexes.
    So a piece is attached only when name (narrowed by the catalog's line when it
    helps) resolves to exactly ONE complex, and then to every stop of that
    complex — art sits in a fare-controlled area, not on one platform. When the
    name spans complexes and the line can't single one out, the piece is left
    unplaced and logged rather than shown against a station it may not be in.
    """
    rows: list[dict[str, Any]] = (
        client.get(
            ART_URL,
            params={
                "$limit": "1000",
                "$order": "station_name, art_date, art_title, artist",
            },
        )
        .raise_for_status()
        .json()
    )

    by_norm: dict[str, list[Station]] = {}
    for st in stations:
        by_norm.setdefault(_norm_name(st.name), []).append(st)

    out: dict[str, list[ArtPiece]] = {}
    unplaced: list[str] = []
    for r in rows:
        if r.get("agency") != "NYCT":
            continue
        name = r.get("station_name", "")
        cands = by_norm.get(_norm_name(name), [])
        lines = frozenset(t for t in re.split(r"[,\s]+", r.get("line", "")) if t)
        # The catalog's line only disambiguates WHICH complex a shared name means
        # (e.g. one of the five "23 St"s). Once that resolves to a single complex,
        # attach to every same-name stop of it — art sits in the complex, not on
        # one platform — not just the platforms whose route matched the line.
        resolver = [s for s in cands if s.routes & lines] or cands
        complexes = {s.complex_id for s in resolver}
        if not resolver or len(complexes) != 1:
            unplaced.append(f"{name} ({r.get('line', '')})")
            continue
        target = next(iter(complexes))
        pool = [s for s in cands if s.complex_id == target]
        link: Any = r.get("art_image_link")
        art_link: str | None = None
        if isinstance(link, dict):
            url = cast("dict[str, Any]", link).get("url")
            art_link = url if isinstance(url, str) else None
        piece = ArtPiece(
            artist=r.get("artist"),
            title=r.get("art_title"),
            year=r.get("art_date"),
            material=r.get("art_material"),
            link=art_link,
        )
        for st in pool:
            out.setdefault(st.gtfs_stop_id, []).append(piece)
    return out, unplaced


def _extmeta(em: dict[str, Any], key: str) -> str | None:
    """One Commons extmetadata field, HTML stripped."""
    v = em.get(key, {}).get("value")
    return re.sub(r"<[^>]+>", "", v).strip() if isinstance(v, str) else None


def fetch_commons(client: httpx.Client, filepath_urls: set[str]) -> dict[str, Photo]:
    """Commons license + attribution for each P18 FilePath URL, keyed by URL."""
    titles: dict[str, str] = {}
    for url in filepath_urls:
        titles[url] = "File:" + unquote(url.rsplit("/", 1)[-1]).replace("_", " ")
    by_title = {t: u for u, t in titles.items()}
    photos: dict[str, Photo] = {}
    batch: list[str] = list(by_title)
    for i in range(0, len(batch), 50):
        chunk = batch[i : i + 50]
        resp = client.get(
            COMMONS_API,
            params={
                "action": "query",
                "format": "json",
                "prop": "imageinfo",
                "titles": "|".join(chunk),
                "iiprop": "url|extmetadata",
                "iiurlwidth": "960",
                "iiextmetadatafilter": (
                    "LicenseShortName|Artist|LicenseUrl|Credit|AttributionRequired"
                ),
            },
        ).raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for pg in pages.values():
            title = pg.get("title")
            info = pg.get("imageinfo")
            if not title or not info:
                continue
            src_url = by_title.get(title)
            if src_url is None:
                continue
            ii = info[0]
            em: dict[str, Any] = ii.get("extmetadata", {})

            thumb = ii.get("thumburl")
            if isinstance(thumb, str):
                thumb = thumb.split("?")[0]
            photos[src_url] = Photo(
                title=title.removeprefix("File:"),
                source=ii.get("descriptionurl", src_url),
                image_url=src_url,
                thumb_url=thumb,
                artist=_extmeta(em, "Artist"),
                license=_extmeta(em, "LicenseShortName"),
                license_url=_extmeta(em, "LicenseUrl"),
                credit=_extmeta(em, "Credit"),
                attribution_required=(
                    em.get("AttributionRequired", {}).get("value") == "true"
                ),
            )
    return photos


def fetch_osm_nodes(
    client: httpx.Client, node_ids: set[str]
) -> dict[str, tuple[float, float]]:
    """OSM node id -> (lat, lon) via Overpass.

    The cross-check is corroboration, but the committed asset should carry it
    stably, so each batch is retried across the mirror endpoints before giving
    up — a single mirror rate-limiting a run must not silently blank the field.
    Still best-effort: if every mirror fails the batch, those nodes stay absent.
    """
    ids = sorted(int(n) for n in node_ids if n.isdigit())
    if not ids:
        return {}
    coords: dict[str, tuple[float, float]] = {}
    for i in range(0, len(ids), 400):
        chunk = ids[i : i + 400]
        q = f"[out:json][timeout:90];node(id:{','.join(map(str, chunk))});out;"
        got = False
        for attempt, endpoint in enumerate(OVERPASS_URLS * 2):
            try:
                resp = client.post(endpoint, data={"data": q}, timeout=120.0)
                resp.raise_for_status()
                for el in resp.json().get("elements", []):
                    if el.get("type") == "node" and "lat" in el and "lon" in el:
                        coords[str(el["id"])] = (el["lat"], el["lon"])
                got = True
                break
            except (httpx.HTTPError, ValueError) as exc:
                print(
                    f"  OSM batch via {endpoint} failed ({exc}); retrying",
                    file=sys.stderr,
                )
                time.sleep(2.0 * (attempt + 1))
        if not got:
            print("  OSM cross-check batch gave up after all mirrors", file=sys.stderr)
    return coords


def parse_transitland_stops(payload: dict[str, Any]) -> dict[str, str]:
    """Onestop ID -> GTFS stop_id from a Transitland v2 /stops response.

    Reads the platform's own stop_id, collapsing the N/S directional suffix to
    the parent id our feed keys on. Pure so it can be tested without the API.
    """
    out: dict[str, str] = {}
    for stop in payload.get("stops", []):
        osid = stop.get("onestop_id")
        sid = stop.get("stop_id")
        if not osid or not sid:
            continue
        out[osid] = sid[:-1] if sid[-1:] in ("N", "S") else sid
    return out


def resolve_transitland(
    client: httpx.Client, onestop_ids: set[str], api_key: str
) -> dict[str, str]:
    """Authoritative Onestop ID -> GTFS stop_id map, from Transitland."""
    resolved: dict[str, str] = {}
    ids = sorted(onestop_ids)
    for i in range(0, len(ids), 50):
        chunk = ids[i : i + 50]
        resp = client.get(
            TRANSITLAND_STOPS_URL,
            params={
                "onestop_id": ",".join(chunk),
                "feed_onestop_id": TRANSITLAND_FEED,
                "limit": "1000",
                "apikey": api_key,
            },
        )
        resp.raise_for_status()
        resolved.update(parse_transitland_stops(resp.json()))
    return resolved


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #
def build_match_index(
    items: dict[str, WikiItem],
) -> tuple[list[tuple[float, float, str, str]], list[tuple[float, float, str]]]:
    """(onestop points: lat, lon, qid, onestop_id), (item points: lat, lon, qid)."""
    onestop_pts: list[tuple[float, float, str, str]] = []
    item_pts: list[tuple[float, float, str]] = []
    for it in items.values():
        for osid in it.onestops:
            ll = onestop_latlon(osid)
            if ll:
                onestop_pts.append((ll[0], ll[1], it.qid, osid))
        if it.coord:
            item_pts.append((it.coord[0], it.coord[1], it.qid))
    return onestop_pts, item_pts


def build_authoritative_index(
    onestop_pts: list[tuple[float, float, str, str]],
    transitland: dict[str, str],
) -> tuple[dict[str, tuple[str, str]], dict[str, set[str]]]:
    """gtfs_stop_id -> (qid, onestop_id) from the resolved Transitland map, plus
    the ambiguous stops to leave unmatched.

    A resolved Onestop ID names the exact GTFS stop_id, so this is the direct,
    authoritative join — keyed straight off our stop, no spatial step. When two
    DIFFERENT Wikidata items resolve to the same stop the join is ambiguous, so
    that stop is excluded and reported rather than guessed.
    """
    qids_per_stop: dict[str, dict[str, str]] = {}
    for _, _, qid, osid in onestop_pts:
        sid = transitland.get(osid)
        if sid is not None:
            qids_per_stop.setdefault(sid, {}).setdefault(qid, osid)
    index: dict[str, tuple[str, str]] = {}
    ambiguous: dict[str, set[str]] = {}
    for sid, qmap in qids_per_stop.items():
        if len(qmap) == 1:
            qid, osid = next(iter(qmap.items()))
            index[sid] = (qid, osid)
        else:
            ambiguous[sid] = set(qmap)
    return index, ambiguous


def _nearest_onestop(
    here: tuple[float, float],
    qid: str,
    osid: str,
    onestop_pts: list[tuple[float, float, str, str]],
) -> float | None:
    """Distance from our stop to the given item's Onestop point, for reporting."""
    best: float | None = None
    for lat, lon, q, o in onestop_pts:
        if q == qid and o == osid:
            d = haversine_m(here, (lat, lon))
            best = d if best is None else min(best, d)
    return round(best, 1) if best is not None else None


def match_station(
    st: Station,
    onestop_pts: list[tuple[float, float, str, str]],
    item_pts: list[tuple[float, float, str]],
    authoritative: dict[str, tuple[str, str]],
    ambiguous: set[str],
) -> Match | None:
    here = (st.lat, st.lon)

    # An ambiguous authoritative join is left unmatched, never guessed.
    if st.gtfs_stop_id in ambiguous:
        return None

    # Tier 1: authoritative Transitland map — a direct gtfs_stop_id lookup, no
    # spatial search, so a nearer wrong platform can never hide the real item.
    auth = authoritative.get(st.gtfs_stop_id)
    if auth is not None:
        qid, osid = auth
        d = _nearest_onestop(here, qid, osid, onestop_pts)
        return Match(qid, "transitland_onestop", True, 1.0, d, osid, None)

    # Tier 2 (fallback): nearest Onestop-ID platform point (decoded geohash).
    # Two different items can carry the SAME geohash (e.g. both Aqueduct stations
    # encode dr5rrg7ttu), so coordinates tie with no way to tell them apart. The
    # authoritative tier above resolves that (each Onestop ID maps to its own
    # stop_id); this spatial fallback cannot, so when distinct items tie within
    # TIE_EPS_M it abstains and the stop is logged unmatched rather than guessed.
    os_cands = [
        (haversine_m(here, (lat, lon)), qid, osid)
        for lat, lon, qid, osid in onestop_pts
    ]
    os_cands = [c for c in os_cands if c[0] <= ONESTOP_MAX_M]
    if os_cands:
        floor = min(c[0] for c in os_cands)
        tied = [c for c in os_cands if c[0] <= floor + TIE_EPS_M]
        if len({qid for _, qid, _ in tied}) > 1:
            return None
        d, qid, osid = min(tied, key=lambda c: (c[0], c[1], c[2]))
        conf = round(max(0.0, 1.0 - d / ONESTOP_MAX_M), 3)
        return Match(qid, "onestop_geohash", False, conf, round(d, 1), osid, None)

    # Tier 3 (fallback): nearest item-level coordinate (P625). Like the geohash
    # tier, if distinct items tie within TIE_EPS_M it abstains and logs unmatched
    # rather than break the tie by fiat.
    it_cands = [(haversine_m(here, (lat, lon)), qid) for lat, lon, qid in item_pts]
    it_cands = [c for c in it_cands if c[0] <= COORD_MAX_M]
    if it_cands:
        floor = min(c[0] for c in it_cands)
        tied = [c for c in it_cands if c[0] <= floor + TIE_EPS_M]
        if len({qid for _, qid in tied}) > 1:
            return None
        d, qid = min(tied, key=lambda c: (c[0], c[1]))
        conf = round(max(0.0, 1.0 - d / COORD_MAX_M), 3)
        return Match(qid, "coordinate", False, conf, round(d, 1), None, None)
    return None


# --------------------------------------------------------------------------- #
# Assembly
# --------------------------------------------------------------------------- #
def opened_year(opened: set[str]) -> tuple[int | None, str | None]:
    """Earliest opening year + its ISO date from the P1619 values."""
    best: tuple[int, str] | None = None
    for raw in opened:
        m = re.match(r"(-?\d{1,4})-(\d\d)-(\d\d)", raw)
        if not m:
            continue
        year = int(m.group(1))
        iso = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
        if best is None or year < best[0]:
            best = (year, iso)
    return (best[0], best[1]) if best else (None, None)


def photo_dict(p: Photo) -> dict[str, Any]:
    return {
        "title": p.title,
        "source": p.source,
        "image_url": p.image_url,
        "thumb_url": p.thumb_url,
        "artist": p.artist,
        "license": p.license,
        "license_url": p.license_url,
        "credit": p.credit,
        "attribution_required": p.attribution_required,
    }


def art_dict(a: ArtPiece) -> dict[str, Any]:
    return {
        "artist": a.artist,
        "title": a.title,
        "year": a.year,
        "material": a.material,
        "link": a.link,
    }


def match_dict(m: Match) -> dict[str, Any]:
    return {
        "method": m.method,
        "authoritative": m.authoritative,
        "confidence": m.confidence,
        "distance_m": m.distance_m,
        "onestop_id": m.onestop_id,
        "osm_node": m.osm_node,
        "osm_agrees": m.osm_agrees,
        "osm_distance_m": m.osm_distance_m,
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    with _client() as client:
        print("fetching stations (39hk-dx4f)…", file=sys.stderr)
        stations = fetch_stations(client)
        print(f"  {len(stations)} stations", file=sys.stderr)

        print("fetching Wikidata (P16=Q7733)…", file=sys.stderr)
        items = fetch_wikidata(client)
        print(f"  {len(items)} items", file=sys.stderr)

        print("fetching ridership (ak4z-sape)…", file=sys.stderr)
        ridership, window = fetch_ridership(client)
        print(
            f"  {len(ridership)} complexes ranked over {window['start']}..{window['end']}",
            file=sys.stderr,
        )

        art: dict[str, list[ArtPiece]] = {}
        art_unplaced: list[str] = []
        if not args.no_art:
            print("fetching art (4y8j-9pkd)…", file=sys.stderr)
            art, art_unplaced = fetch_art(client, stations)
            print(
                f"  {sum(len(v) for v in art.values())} pieces on "
                f"{len(art)} stations, {len(art_unplaced)} unplaced",
                file=sys.stderr,
            )
            if art_unplaced:
                print(
                    "  unplaced art (name spans complexes, line can't resolve): "
                    + "; ".join(sorted(set(art_unplaced))),
                    file=sys.stderr,
                )

        onestop_pts, item_pts = build_match_index(items)

        transitland: dict[str, str] = {}
        key = os.environ.get("TRANSITLAND_API_KEY")
        if key:
            print(
                "resolving Onestop IDs via Transitland (authoritative)…",
                file=sys.stderr,
            )
            all_os = {osid for _, _, _, osid in onestop_pts}
            transitland = resolve_transitland(client, all_os, key)
            print(f"  {len(transitland)} Onestop IDs resolved", file=sys.stderr)
        else:
            print(
                "  TRANSITLAND_API_KEY unset — no authoritative stop_id map; "
                "Onestop matches recorded as onestop_geohash (fallback)",
                file=sys.stderr,
            )

        authoritative, ambiguous = build_authoritative_index(onestop_pts, transitland)
        if ambiguous:
            print(
                f"  {len(ambiguous)} stop(s) with an ambiguous authoritative join "
                "left unmatched: "
                + ", ".join(
                    f"{s}({'/'.join(sorted(q))})" for s, q in sorted(ambiguous.items())
                ),
                file=sys.stderr,
            )

        matches: dict[str, Match] = {}
        for st in stations:
            m = match_station(st, onestop_pts, item_pts, authoritative, set(ambiguous))
            if m:
                it = items[m.qid]
                m.osm_node = min(it.osm_nodes) if it.osm_nodes else None
                matches[st.gtfs_stop_id] = m

        # OSM cross-check for the matched items' nodes.
        if not args.no_osm:
            wanted = {m.osm_node for m in matches.values() if m.osm_node}
            if wanted:
                print(f"cross-checking {len(wanted)} OSM nodes…", file=sys.stderr)
                osm = fetch_osm_nodes(client, wanted)
                by_id = {s.gtfs_stop_id: s for s in stations}
                for sid, m in matches.items():
                    if m.osm_node and m.osm_node in osm:
                        st = by_id[sid]
                        d = haversine_m((st.lat, st.lon), osm[m.osm_node])
                        m.osm_distance_m = round(d, 1)
                        m.osm_agrees = d <= OSM_AGREE_M

        # Photos for matched items only.
        wanted_imgs = {img for m in matches.values() for img in items[m.qid].images}
        photos: dict[str, Photo] = {}
        if wanted_imgs:
            print(
                f"fetching Commons metadata for {len(wanted_imgs)} images…",
                file=sys.stderr,
            )
            photos = fetch_commons(client, wanted_imgs)

    # ----- assemble -----
    out_stations: dict[str, dict[str, Any]] = {}
    method_tally: dict[str, int] = {}
    for st in stations:
        m = matches.get(st.gtfs_stop_id)
        rid = ridership.get(st.complex_id) if st.complex_id else None
        pieces = art.get(st.gtfs_stop_id)
        if m is None and rid is None and not pieces:
            continue

        entry: dict[str, Any] = {}
        if m:
            method_tally[m.method] = method_tally.get(m.method, 0) + 1
            it = items[m.qid]
            year, iso = opened_year(it.opened)
            entry["wikidata_qid"] = m.qid
            entry["wikidata_name"] = it.label
            entry["opened_year"] = year
            entry["opened_date"] = iso
            entry["heritage"] = sorted(it.heritage) if it.heritage else []
            photo = next(
                (photos[img] for img in sorted(it.images) if img in photos), None
            )
            entry["photo"] = photo_dict(photo) if photo else None
            entry["match"] = match_dict(m)
        if pieces:
            entry["art"] = [art_dict(a) for a in pieces]
        if rid:
            entry["ridership"] = {**rid, "complex_id": st.complex_id}
        out_stations[st.gtfs_stop_id] = entry

    unmatched = [st.gtfs_stop_id for st in stations if st.gtfs_stop_id not in matches]
    print(
        f"\nmatched {len(matches)}/{len(stations)} to Wikidata "
        f"({method_tally}); {len(unmatched)} unmatched",
        file=sys.stderr,
    )
    if unmatched:
        print("  unmatched: " + ", ".join(sorted(unmatched)), file=sys.stderr)

    return {
        "sources": {
            "stations": "data.ny.gov 39hk-dx4f",
            "wikidata": "wdt:P16 wd:Q7733",
            "ridership": "data.ny.gov ak4z-sape",
            "art": None if args.no_art else "data.ny.gov 4y8j-9pkd",
        },
        "ridership_window": window,
        "counts": {
            "stations": len(stations),
            "wikidata_items": len(items),
            "matched": len(matches),
            "by_method": method_tally,
            "unmatched": len(unmatched),
            "art_pieces": sum(len(v) for v in art.values()),
            "art_unplaced": len(art_unplaced),
        },
        "stations": out_stations,
    }


def render(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=FACTS_PATH)
    parser.add_argument(
        "--no-osm", action="store_true", help="skip the OSM cross-check"
    )
    parser.add_argument("--no-art", action="store_true", help="skip the art catalog")
    args = parser.parse_args()

    payload = build(args)
    text = render(payload)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(f"wrote {args.out} — {len(payload['stations'])} stations with facts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
