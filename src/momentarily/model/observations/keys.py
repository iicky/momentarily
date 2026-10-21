"""R2 key layout of the observations layer.

Every key sits under ``model/observations/s1/`` and is partitioned by UTC
time, never by service day (layer 3 assigns service days):

    ticks/<YYYY-MM-DD>/<HH>/<scheduled_at>-<started_at_ms>.json      CollectorTick
    raw/gtfsrt/<YYYY-MM-DD>/<HH>/<scheduled_at>-<started_at_ms>.bin  tick bundle
    raw/blob/<YYYY-MM-DD>/<sha256>.<json|pb|zip>.gz                  deduplicated blob
    derived/<record>/<YYYY-MM-DD>/<HH>.parquet                       hourly records
    derived/<record>/<YYYY-MM-DD>.parquet                            daily records

Ticks and bundles are filed under the UTC hour of ``scheduled_at`` (the cron
minute), so the directory follows from the file name. A blob is filed under
the UTC day of the fetch that stored it; the same bytes fetched on another day
get another copy. Hourly derived files hold the records whose ``observed_at``
falls in that hour.

Two keys outside this prefix are referenced: the static GTFS archive
(``archive/gtfs/<sha256>.zip``) and the legacy predictions stream
(``v1/predictions/<YYYY-MM-DD>/<ts>.jsonl``).

Each key type formats with ``.key`` and parses with ``parse_key``;
``parse_key(k.key) == k`` for every valid key.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Literal, get_args

PREFIX = "model/observations/s1/"

HourlyRecord = Literal[
    "vehicle", "trip_update", "feed_header", "alert_presence", "equipment"
]
DailyRecord = Literal[
    "alert_version",
    "legacy_route_alert_state",
    "legacy_fetch_outcome",
    "legacy_collector_health",
]
BlobExt = Literal["json", "pb", "zip"]

HOURLY_RECORDS: tuple[HourlyRecord, ...] = get_args(HourlyRecord)
DAILY_RECORDS: tuple[DailyRecord, ...] = get_args(DailyRecord)
BLOB_EXTS: tuple[BlobExt, ...] = get_args(BlobExt)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_DAY = r"\d{4}-\d{2}-\d{2}"


class KeyLayoutError(ValueError):
    """A key, or the parts of one, that the layout does not allow."""


def _utc(instant: int) -> datetime:
    try:
        return datetime.fromtimestamp(instant, UTC)
    except (OverflowError, OSError, ValueError) as err:
        raise KeyLayoutError(f"instant {instant} is out of range") from err


def utc_day(instant: int) -> str:
    """ISO date of the UTC day containing ``instant``. Raises
    ``KeyLayoutError`` for an instant no calendar can hold."""
    return _utc(instant).date().isoformat()


def _utc_hour(instant: int) -> str:
    return f"{_utc(instant).hour:02d}"


def _check_day(day: str) -> None:
    try:
        ok = date.fromisoformat(day).isoformat() == day
    except ValueError:
        ok = False
    if not ok:
        raise KeyLayoutError(f"not an ISO UTC date: {day!r}")


def _check_sha(sha256: str) -> None:
    if not _SHA256.fullmatch(sha256):
        raise KeyLayoutError(f"not a lowercase sha256 hex digest: {sha256!r}")


def _check_instant(name: str, value: int) -> None:
    if value < 0:
        raise KeyLayoutError(f"{name} must be >= 0, got {value}")


@dataclass(frozen=True)
class _Attempt:
    scheduled_at: int
    started_at_ms: int

    def __post_init__(self) -> None:
        _check_instant("scheduled_at", self.scheduled_at)
        _check_instant("started_at_ms", self.started_at_ms)

    def _path(self, kind: str, ext: str) -> str:
        at = self.scheduled_at
        return (
            f"{PREFIX}{kind}/{utc_day(at)}/{_utc_hour(at)}/"
            f"{at}-{self.started_at_ms}.{ext}"
        )


@dataclass(frozen=True)
class TickKey(_Attempt):
    """The ``CollectorTick`` record of one Worker invocation."""

    @property
    def key(self) -> str:
        return self._path("ticks", "json")


@dataclass(frozen=True)
class BundleKey(_Attempt):
    """One tick's GTFS-RT bundle: a JSON index member, then one gzip member
    per feed."""

    @property
    def key(self) -> str:
        return self._path("raw/gtfsrt", "bin")


@dataclass(frozen=True)
class BlobKey:
    """Gzipped payload bytes, deduplicated within one UTC day."""

    day: str
    sha256: str
    ext: BlobExt

    def __post_init__(self) -> None:
        _check_day(self.day)
        _check_sha(self.sha256)
        if self.ext not in BLOB_EXTS:
            raise KeyLayoutError(f"blob extension {self.ext!r} not in {BLOB_EXTS}")

    @classmethod
    def for_fetch(cls, observed_at: int, sha256: str, ext: BlobExt) -> BlobKey:
        return cls(utc_day(observed_at), sha256, ext)

    @property
    def key(self) -> str:
        return f"{PREFIX}raw/blob/{self.day}/{self.sha256}.{self.ext}.gz"


@dataclass(frozen=True)
class DerivedHourKey:
    """Records of one kind observed in ``[hour_start, hour_start + 3600)``."""

    record: HourlyRecord
    hour_start: int

    def __post_init__(self) -> None:
        if self.record not in HOURLY_RECORDS:
            raise KeyLayoutError(f"{self.record!r} is not an hourly record")
        _check_instant("hour_start", self.hour_start)
        if self.hour_start % 3600:
            raise KeyLayoutError(f"hour_start {self.hour_start} is not on the hour")

    @property
    def key(self) -> str:
        at = self.hour_start
        return f"{PREFIX}derived/{self.record}/{utc_day(at)}/{_utc_hour(at)}.parquet"


@dataclass(frozen=True)
class DerivedDayKey:
    """Records of one kind built for one UTC day."""

    record: DailyRecord
    day: str

    def __post_init__(self) -> None:
        if self.record not in DAILY_RECORDS:
            raise KeyLayoutError(f"{self.record!r} is not a daily record")
        _check_day(self.day)

    @property
    def key(self) -> str:
        return f"{PREFIX}derived/{self.record}/{self.day}.parquet"


@dataclass(frozen=True)
class GtfsArchiveKey:
    """A static GTFS zip in the existing ``archive/gtfs/`` store."""

    sha256: str

    def __post_init__(self) -> None:
        _check_sha(self.sha256)

    @property
    def key(self) -> str:
        return f"archive/gtfs/{self.sha256}.zip"


@dataclass(frozen=True)
class LegacyPredictionsKey:
    """One tick of the legacy ``v1/predictions`` stream."""

    observed_at: int

    def __post_init__(self) -> None:
        _check_instant("observed_at", self.observed_at)

    @property
    def key(self) -> str:
        return f"v1/predictions/{utc_day(self.observed_at)}/{self.observed_at}.jsonl"


ObservationKey = (
    TickKey
    | BundleKey
    | BlobKey
    | DerivedHourKey
    | DerivedDayKey
    | GtfsArchiveKey
    | LegacyPredictionsKey
)

_P = re.escape(PREFIX)
_ATTEMPT = rf"(?P<day>{_DAY})/(?P<hour>\d{{2}})/(?P<sched>\d+)-(?P<ms>\d+)"
_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tick", re.compile(rf"{_P}ticks/{_ATTEMPT}\.json")),
    ("bundle", re.compile(rf"{_P}raw/gtfsrt/{_ATTEMPT}\.bin")),
    (
        "blob",
        re.compile(
            rf"{_P}raw/blob/(?P<day>{_DAY})/(?P<sha>[0-9a-f]{{64}})\.(?P<ext>\w+)\.gz"
        ),
    ),
    (
        "hour",
        re.compile(
            rf"{_P}derived/(?P<rec>\w+)/(?P<day>{_DAY})/(?P<hour>\d{{2}})\.parquet"
        ),
    ),
    ("daily", re.compile(rf"{_P}derived/(?P<rec>\w+)/(?P<day>{_DAY})\.parquet")),
    ("gtfs", re.compile(r"archive/gtfs/(?P<sha>[0-9a-f]{64})\.zip")),
    ("legacy", re.compile(rf"v1/predictions/(?P<day>{_DAY})/(?P<ts>\d+)\.jsonl")),
)


_EXT_BY_NAME: dict[str, BlobExt] = {e: e for e in BLOB_EXTS}
_HOURLY_BY_NAME: dict[str, HourlyRecord] = {r: r for r in HOURLY_RECORDS}
_DAILY_BY_NAME: dict[str, DailyRecord] = {r: r for r in DAILY_RECORDS}


def _build(kind: str, m: re.Match[str]) -> ObservationKey:
    g = m.groupdict()
    match kind:
        case "tick":
            return TickKey(int(g["sched"]), int(g["ms"]))
        case "bundle":
            return BundleKey(int(g["sched"]), int(g["ms"]))
        case "blob":
            ext = _EXT_BY_NAME.get(g["ext"])
            if ext is None:
                raise KeyLayoutError(f"blob extension {g['ext']!r} not in {BLOB_EXTS}")
            return BlobKey(g["day"], g["sha"], ext)
        case "hour":
            rec = _HOURLY_BY_NAME.get(g["rec"])
            if rec is None:
                raise KeyLayoutError(f"{g['rec']!r} is not an hourly record")
            _check_day(g["day"])
            start = datetime.fromisoformat(f"{g['day']}T{g['hour']}:00:00+00:00")
            return DerivedHourKey(rec, int(start.timestamp()))
        case "daily":
            daily = _DAILY_BY_NAME.get(g["rec"])
            if daily is None:
                raise KeyLayoutError(f"{g['rec']!r} is not a daily record")
            return DerivedDayKey(daily, g["day"])
        case "gtfs":
            return GtfsArchiveKey(g["sha"])
        case _:
            return LegacyPredictionsKey(int(g["ts"]))


def parse_key(key: str) -> ObservationKey:
    """The typed key ``key`` names. Raises ``KeyLayoutError`` for any key the
    layout does not produce, including a directory that disagrees with the
    file name (``ticks/<day>/<hour>/`` must be the hour of ``scheduled_at``)."""
    for kind, pattern in _PATTERNS:
        m = pattern.fullmatch(key)
        if m is None:
            continue
        try:
            parsed = _build(kind, m)
        except ValueError as err:
            raise KeyLayoutError(f"{key!r}: {err}") from err
        if parsed.key != key:
            raise KeyLayoutError(f"{key!r} is not canonical; expected {parsed.key!r}")
        return parsed
    raise KeyLayoutError(f"{key!r} is not an observations-layer key")
