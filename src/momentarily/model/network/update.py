"""Keep the day->feed map and the network builds current from recorded evidence.

The traversal archive that the map was backfilled from is not the source going
forward. Each new UTC day is resolved from two records that the live pipeline
keeps:

* the recorder's ``gtfs_static`` HEAD in every ``CollectorTick``
  (``model/observations/s1/ticks/<day>/``): the ETag the static feed URL served
  at that minute;
* the static-feed capture records (``archive/gtfs/by_etag/<etag>.json``, the
  digest of the bytes a GET returned under that ETag, written only after
  ``archive/gtfs/<digest>.zip`` is stored).

The join is ETag to digest through ``by_etag``, accepted only when the zip it
names is held. A day is verified when every gap between successful HEADs,
counting from midnight to the first and from the last to the next midnight, is
at most ``MAX_GAP_S``, every ETag resolves, and they all resolve to one feed.
Anything less is recorded unverified with the reason; a single feed seen on
only part of the day is kept as ``inferred_feed_id``, the same lead the
backfill gives a partly stamped day.

``update`` appends days after the map's last day and never rewrites one, then
builds every feed the map names that has no build yet, with the newest
stations snapshot already in the output. Writes go through ``publish``, which
never replaces an object with different bytes, except that ``index.json`` and
``day_feed_map.json`` may be replaced by a strict extension of themselves.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import Any, Literal, cast

from pydantic import ValidationError

from momentarily.gtfs.static import read_version
from momentarily.model.network.build import (
    CodeStamp,
    Inputs,
    NetworkBuildError,
    build_network,
    first_observed,
    index_entry,
    network_digest,
    serialize,
)
from momentarily.model.network.contracts import (
    SCHEMA_ID,
    DayFeed,
    DayFeedMap,
    IndexEntry,
    NetworkIndex,
    NetworkSnapshot,
)
from momentarily.model.network.stations import StationsSnapshot, parse_snapshot
from momentarily.model.observations.contracts import CollectorTick
from momentarily.model.observations.derive.storage import Storage
from momentarily.model.observations.keys import PREFIX as OBSERVATIONS_PREFIX

NETWORK_PREFIX = "model/network/s1/"
INDEX_KEY = f"{NETWORK_PREFIX}index.json"
DAY_FEED_MAP_KEY = f"{NETWORK_PREFIX}day_feed_map.json"
STATIONS_PREFIX = f"{NETWORK_PREFIX}stations-"
TICKS_PREFIX = f"{OBSERVATIONS_PREFIX}ticks/"
GTFS_PREFIX = "archive/gtfs/"
BY_ETAG_PREFIX = f"{GTFS_PREFIX}by_etag/"

# The recorder runs every minute; on 2026-10-21 (03:57 to 13:30 UTC) 573 of
# 578 gaps between successful HEADs were 60 s and the rest were retries
# (0 s). A quarter hour without evidence is far outside that, and is where a
# day stops counting as covered.
MAX_GAP_S = 15 * 60
DAY_S = 86_400

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ZIP_KEY = re.compile(rf"{re.escape(GTFS_PREFIX)}([0-9a-f]{{64}})\.zip")
_WEAK_ETAG = re.compile(r"^\s*[Ww]/")

NO_TICKS = "no recorder ticks for this day"


def dump_json(doc: object) -> bytes:
    """The layer's JSON form: sorted keys, two-space indent, final newline."""
    return (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode()


def _day_start(day: str) -> int:
    return int(datetime.fromisoformat(day).replace(tzinfo=UTC).timestamp())


def _etag_stem(etag: str) -> str:
    """The ``by_etag`` key stem: the header with ``W/`` and quoting removed,
    as both capture writers name it."""
    stripped = etag.strip()
    if stripped[:2] in ("W/", "w/"):
        stripped = stripped[2:]
    return stripped.strip('"')


@dataclass(frozen=True)
class HeadSeen:
    """One successful HEAD with a strong ETag, at its tick's scheduled_at."""

    at: int
    etag: str


@dataclass(frozen=True)
class DayEvidence:
    """What the recorder saw of the static feed on one UTC day."""

    n_ticks: int
    n_unreadable: int
    heads: list[HeadSeen]


def read_evidence(store: Storage, day: str) -> DayEvidence:
    """Every tick filed under ``day``. A tick that does not parse, or whose
    HEAD failed or carried no strong ETag, is no evidence: its minute counts
    as a gap."""
    keys = store.list(f"{TICKS_PREFIX}{day}/")
    heads: list[HeadSeen] = []
    unreadable = 0
    for key in keys:
        try:
            tick = CollectorTick.model_validate_json(store.get(key))
        except ValidationError:
            unreadable += 1
            continue
        for fetch in tick.fetches:
            etag = fetch.http_etag
            if (
                fetch.source == "gtfs_static"
                and fetch.outcome == "ok"
                and etag is not None
                and not _WEAK_ETAG.match(etag)
            ):
                heads.append(HeadSeen(at=tick.scheduled_at, etag=etag))
    return DayEvidence(n_ticks=len(keys), n_unreadable=unreadable, heads=heads)


def get_present(store: Storage, key: str) -> bytes | None:
    """The bytes of ``key``, or None when it is absent. Listed first, so an
    absent key costs one listing rather than a store's not-found retries."""
    if key not in store.list(key):
        return None
    return store.get(key)


def held_feeds(store: Storage) -> set[str]:
    """Digests of every captured zip under ``archive/gtfs/``."""
    held: set[str] = set()
    for key in store.list(GTFS_PREFIX):
        m = _ZIP_KEY.fullmatch(key)
        if m:
            held.add(m.group(1))
    return held


def resolve_etag(store: Storage, etag: str, held: set[str]) -> str | None:
    """The feed captured under ``etag``, or None when there is no capture
    record or the zip it names is not held."""
    data = get_present(store, f"{BY_ETAG_PREFIX}{_etag_stem(etag)}.json")
    if data is None:
        return None
    raw: Any = json.loads(data)
    digest = (
        cast("dict[str, Any]", raw).get("digest") if isinstance(raw, dict) else None
    )
    if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
        return None
    return digest if digest in held else None


@dataclass(frozen=True)
class _Run:
    feed: str
    first: int
    last: int


def _runs(heads: list[HeadSeen], feeds: dict[str, str]) -> list[_Run]:
    """Spans of one feed, split where the feed changes or evidence lapses."""
    runs: list[_Run] = []
    for h in heads:
        feed = feeds[h.etag]
        prev = runs[-1] if runs else None
        if prev is not None and prev.feed == feed and h.at - prev.last <= MAX_GAP_S:
            runs[-1] = _Run(feed, prev.first, h.at)
        else:
            runs.append(_Run(feed, h.at, h.at))
    return runs


def resolve_recorded_day(
    day: str,
    evidence: DayEvidence,
    feeds: dict[str, str | None],
    versions: dict[str, str | None],
) -> DayFeed:
    """One UTC day from its recorder evidence. ``feeds`` maps each ETag seen to
    its captured feed (None when uncaptured); ``versions`` gives each feed's
    ``feed_version``."""

    def unverified(reason: str, feed: str | None = None) -> DayFeed:
        return DayFeed(
            feed_id=None,
            verified=False,
            reason=reason,
            feed_version=None if feed is None else versions[feed],
            inferred_feed_id=feed,
        )

    if evidence.n_ticks == 0:
        return unverified(NO_TICKS)
    if not evidence.heads:
        return unverified(
            f"no successful gtfs_static HEAD with a strong ETag in "
            f"{evidence.n_ticks} ticks ({evidence.n_unreadable} unreadable)"
        )
    uncaptured = sorted({h.etag for h in evidence.heads if feeds[h.etag] is None})
    if uncaptured:
        return unverified(f"no usable captured feed for ETag {', '.join(uncaptured)}")
    resolved = {etag: feed for etag, feed in feeds.items() if feed is not None}
    heads = sorted(evidence.heads, key=lambda h: h.at)
    runs = _runs(heads, resolved)
    spans = ", ".join(f"{r.feed[:8]} {r.first}..{r.last}" for r in runs)
    distinct = {r.feed for r in runs}
    if len(distinct) > 1:
        return unverified(f"day spans more than one feed ({spans})")
    (feed,) = distinct
    start = _day_start(day)
    edges = [start, *(h.at for h in heads), start + DAY_S]
    if max(b - a for a, b in pairwise(edges)) > MAX_GAP_S:
        return unverified(
            f"recorder evidence covers only part of the day ({spans})", feed
        )
    return DayFeed(
        feed_id=feed,
        verified=True,
        reason=None,
        feed_version=versions[feed],
        inferred_feed_id=None,
    )


# --- publishing ----------------------------------------------------------------


Action = Literal["put", "unchanged"]


class PublishConflictError(RuntimeError):
    """Objects that would have replaced different bytes; nothing was written."""

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        super().__init__(f"refusing to replace different bytes at {', '.join(keys)}")


def _extends(old: bytes, new: bytes, field_name: str) -> bool:
    """Every entry of ``old``'s ``field_name`` object is in ``new``, unchanged,
    and nothing else at the top level differs."""
    try:
        o: Any = json.loads(old)
        n: Any = json.loads(new)
    except ValueError:
        return False
    if not isinstance(o, dict) or not isinstance(n, dict):
        return False
    od, nd = cast("dict[str, Any]", o), cast("dict[str, Any]", n)
    if {k: v for k, v in od.items() if k != field_name} != {
        k: v for k, v in nd.items() if k != field_name
    }:
        return False
    oe, ne = od.get(field_name), nd.get(field_name)
    if not isinstance(oe, dict) or not isinstance(ne, dict):
        return False
    old_entries, new_entries = cast("dict[str, Any]", oe), cast("dict[str, Any]", ne)
    return all(k in new_entries and new_entries[k] == v for k, v in old_entries.items())


_GROWING = {INDEX_KEY: "feeds", DAY_FEED_MAP_KEY: "days"}


def publish(
    store: Storage, objects: list[tuple[str, bytes]]
) -> list[tuple[str, Action]]:
    """Write ``objects`` in order, all or nothing. A key already holding the
    same bytes is left alone; one holding different bytes is a conflict, unless
    it is the index or the day map and the new document extends it. Every
    check runs before the first write."""
    plan: list[tuple[str, bytes, Action]] = []
    conflicts: list[str] = []
    for key, data in objects:
        existing = get_present(store, key)
        if existing is None:
            plan.append((key, data, "put"))
            continue
        if existing == data:
            plan.append((key, data, "unchanged"))
        elif key in _GROWING and _extends(existing, data, _GROWING[key]):
            plan.append((key, data, "put"))
        else:
            conflicts.append(key)
    if conflicts:
        raise PublishConflictError(conflicts)
    for key, data, action in plan:
        if action == "put":
            store.put(key, data)
    return [(key, action) for key, _, action in plan]


# --- the update ----------------------------------------------------------------


@dataclass
class UpdateReport:
    days: dict[str, DayFeed] = field(default_factory=dict[str, DayFeed])
    built: dict[str, str] = field(default_factory=dict[str, str])
    build_errors: dict[str, list[str]] = field(default_factory=dict[str, list[str]])
    published: list[tuple[str, Action]] = field(
        default_factory=list[tuple[str, Action]]
    )


def _feed_bytes(store: Storage, feed: str) -> bytes:
    data = store.get(f"{GTFS_PREFIX}{feed}.zip")
    got = hashlib.sha256(data).hexdigest()
    if got != feed:
        raise ValueError(f"{GTFS_PREFIX}{feed}.zip hashes to {got}")
    return data


def _version_of(gtfs: bytes) -> str | None:
    with zipfile.ZipFile(io.BytesIO(gtfs)) as zf:
        version = read_version(zf).version
    return None if version == "unknown" else version


def current_stations(store: Storage) -> tuple[str, StationsSnapshot]:
    """The newest stations snapshot in the output, by its capture time."""
    found = [
        (snap.retrieved_at, key, snap)
        for key in store.list(STATIONS_PREFIX)
        if key.endswith(".json")
        for snap in (parse_snapshot(store.get(key)),)
    ]
    if not found:
        raise FileNotFoundError(f"no stations snapshot under {STATIONS_PREFIX}")
    _, key, snap = max(found, key=lambda t: (t[0], t[1]))
    return key, snap


def _days_after(last: str, through: str) -> list[str]:
    d, end = date.fromisoformat(last) + timedelta(days=1), date.fromisoformat(through)
    days: list[str] = []
    while d <= end:
        days.append(d.isoformat())
        d += timedelta(days=1)
    return days


def _build(
    out: Storage,
    gtfs: bytes,
    stations: StationsSnapshot,
    map_bytes: bytes,
    stamp: CodeStamp,
) -> tuple[str, bytes | None, IndexEntry]:
    """Build one feed. Returns the build key, the bytes to write (None when an
    existing file already holds the same network) and its index entry."""
    snapshot = build_network(
        Inputs(
            gtfs=gtfs,
            stations=stations,
            direction_overrides={},
            overrides_digest=None,
            first_observed=first_observed(map_bytes, gtfs),
        ),
        stamp=stamp,
    )
    key = f"{NETWORK_PREFIX}{snapshot.build_id}.json.gz"
    data = serialize(snapshot)
    existing_bytes = get_present(out, key)
    if existing_bytes is None:
        return key, data, index_entry(snapshot)
    if existing_bytes == data:
        return key, None, index_entry(snapshot)
    # A file left by an earlier run under another commit: keep it when it
    # holds the same network; anything else is a conflict for publish.
    try:
        existing = NetworkSnapshot.model_validate_json(gzip.decompress(existing_bytes))
    except (OSError, EOFError, ValueError, zlib.error):
        return key, data, index_entry(snapshot)
    if network_digest(existing) == network_digest(snapshot):
        return key, None, index_entry(existing)
    return key, data, index_entry(snapshot)


def update(
    source: Storage, out: Storage, through: str, stamp: CodeStamp
) -> UpdateReport:
    """Extend the day map through ``through``, build new feeds, publish.

    ``source`` holds the recorder ticks and ``archive/gtfs/``; ``out`` holds
    ``model/network/s1/``. In R2 they are the same bucket."""
    report = UpdateReport()
    day_map = DayFeedMap.model_validate_json(out.get(DAY_FEED_MAP_KEY))
    index_bytes = get_present(out, INDEX_KEY)
    index = (
        NetworkIndex(schema=SCHEMA_ID, feeds={})
        if index_bytes is None
        else NetworkIndex.model_validate_json(index_bytes)
    )

    held = held_feeds(source)
    etag_feeds: dict[str, str | None] = {}
    versions: dict[str, str | None] = {
        fid: entry.feed_version for fid, entry in index.feeds.items()
    }
    gtfs_cache: dict[str, bytes] = {}

    def gtfs_of(feed: str) -> bytes:
        if feed not in gtfs_cache:
            gtfs_cache[feed] = _feed_bytes(source, feed)
        return gtfs_cache[feed]

    def usable(feed: str | None) -> str | None:
        """``feed`` when its zip reads; else None, with the error reported."""
        if feed is None or feed in versions:
            return feed
        if feed in report.build_errors:
            return None
        try:
            versions[feed] = _version_of(gtfs_of(feed))
        except Exception as exc:
            report.build_errors[feed] = [f"{type(exc).__name__}: {exc}"]
            return None
        return feed

    for day in _days_after(max(day_map.days), through):
        evidence = read_evidence(source, day)
        for h in evidence.heads:
            if h.etag not in etag_feeds:
                etag_feeds[h.etag] = usable(resolve_etag(source, h.etag, held))
        report.days[day] = resolve_recorded_day(day, evidence, etag_feeds, versions)

    new_map = DayFeedMap(days={**day_map.days, **report.days})
    map_bytes = dump_json(new_map.model_dump(mode="json"))

    named = sorted(
        {
            f
            for d in new_map.days.values()
            for f in (d.feed_id, d.inferred_feed_id)
            if f is not None
        }
    )
    missing = [f for f in named if f not in index.feeds]
    builds: list[tuple[str, bytes]] = []
    feeds = dict(index.feeds)
    if missing:
        _, stations = current_stations(out)
        for feed in missing:
            if feed not in held:
                report.build_errors[feed] = [f"no {GTFS_PREFIX}{feed}.zip"]
                continue
            # Any failure stays with its feed: the map is still extended and
            # published, the step exits 1, and the next run retries the build.
            try:
                key, data, entry = _build(
                    out, gtfs_of(feed), stations, map_bytes, stamp
                )
            except NetworkBuildError as exc:
                report.build_errors[feed] = exc.problems
                continue
            except Exception as exc:
                report.build_errors[feed] = [f"{type(exc).__name__}: {exc}"]
                continue
            report.built[feed] = entry.build_id
            feeds[feed] = entry
            if data is not None:
                builds.append((key, data))

    new_index = NetworkIndex(schema=SCHEMA_ID, feeds=feeds)
    objects = [
        *builds,
        (INDEX_KEY, dump_json(new_index.model_dump(mode="json"))),
        (DAY_FEED_MAP_KEY, map_bytes),
    ]
    report.published = publish(out, objects)
    return report


def yesterday_utc(now: datetime) -> str:
    return (now.astimezone(UTC).date() - timedelta(days=1)).isoformat()


def print_report(report: UpdateReport) -> int:
    """One line per new day, build and object key; exit status 1 when a feed
    could not be built."""
    for day, entry in report.days.items():
        if entry.verified:
            print(f"day {day} verified {entry.feed_id}")
        else:
            lead = entry.inferred_feed_id or "none"
            print(f"day {day} unverified lead={lead}: {entry.reason}")
    for feed, build_id in report.built.items():
        print(f"build {build_id} feed={feed}")
    for feed, problems in report.build_errors.items():
        print(f"build failed feed={feed}: {'; '.join(problems)}")
    for key, action in report.published:
        print(f"{action} {key}")
    return 1 if report.build_errors else 0


def run(source: Storage, out: Storage, through: str, stamp: CodeStamp) -> int:
    """``update`` and its printed report; exit status 1 on a publish conflict
    (nothing written) or a feed that could not be built."""
    try:
        report = update(source, out, through, stamp)
    except PublishConflictError as err:
        print(err)
        return 1
    return print_report(report)
