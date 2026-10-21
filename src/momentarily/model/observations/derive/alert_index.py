"""The AlertVersion index of one UTC day, built from AlertPresence records.

Evidence is a presence: the full alert set of one fetch whose outcome was
``ok`` or ``empty`` and whose payload decoded. A failed, unstored or
undecodable fetch is no evidence, and neither is a minute with no tick.

**Window.** Day ``D``'s index is built over ``[D 00:00 - CARRY_SECONDS,
D+1 00:00)``: the day plus a carry before it, nothing after.

**Rows.** A version (``source``, ``alert_id``, ``content_digest``) gets one
row per *run*: a maximal stretch of sightings with no evidence lacking it in
between. A version that leaves and comes back is two rows, told apart by
``first_seen``; ``reappearing`` counts such versions. ``D``'s file holds every
run with a sighting in ``D``, and every run whose ``gone_by`` falls in ``D``
(its last sighting was in the carry), so a run that ends at midnight is
closed in ``D``. A run seen and gone within the carry belongs to ``D-1``.

* ``first_seen`` and ``last_seen`` are the run's first and last sightings.
  ``absent_before`` is the latest evidence before ``first_seen``, null when
  there is none in the window (present at the window's first evidence).
  ``gone_by`` is the earliest evidence after ``last_seen``, null when there
  is none (present at the window's last evidence).
* ``gone_kind`` is ``superseded`` when the ``gone_by`` presence holds the
  same ``alert_id`` with another digest, else ``removed``.
* ``unobserved``: the cron minute of an evidence fetch is
  ``scheduled_at // 60``. Two consecutive evidence fetches of the source in
  ``[first_seen, gone_by]`` more than one minute apart leave the seconds
  strictly between them unobserved. When ``gone_by`` is null and the last
  evidence falls before the window's last minute, the rest of the window is
  unobserved too.

A run that spans midnight is in both days' files. ``D``'s row ends with
``gone_by`` null; ``D+1``'s row starts in its carry, so it has the true
``absent_before`` when the run started within the carry, and otherwise
starts with ``absent_before`` null, overlapping ``D``'s row in the carry hour.
The one edge no file closes: no evidence at all for longer than the carry
across midnight. Then ``D``'s row ends with ``gone_by`` null and a trailing
``unobserved`` gap, and ``D+1``'s window holds none of the run's sightings.

``numbered`` alert ids are per-poll sequence numbers (doc §3.5), so nearly
every ``numbered`` version is seen once: ``first_seen == last_seen``,
``gone_by`` the next poll, ``gone_kind`` ``removed``. No identity is
inferred across ids.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date
from itertools import pairwise

from pydantic import JsonValue

from momentarily.model.conventions import Interval
from momentarily.model.observations.contracts import (
    AlertPresence,
    AlertSource,
    AlertVersion,
    GoneKind,
)

CARRY_SECONDS = 3600
_DAY = 86400
_MERCURY = "transit_realtime.mercury_alert"


def day_start(day: str) -> int:
    return (date.fromisoformat(day) - date(1970, 1, 1)).days * _DAY


def window(day: str) -> Interval:
    start = day_start(day)
    return Interval(start=start - CARRY_SECONDS, end=start + _DAY)


@dataclass(frozen=True)
class Evidence:
    """One presence and the cron minute of the fetch it came from."""

    scheduled_at: int
    presence: AlertPresence

    @property
    def minute(self) -> int:
        return self.scheduled_at // 60

    @property
    def at(self) -> int:
        return self.presence.observed_at


@dataclass
class IndexStats:
    versions: int = 0
    reappearing: int = 0


def _published(entity: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    alert = entity.get("alert")
    mercury = alert.get(_MERCURY) if isinstance(alert, dict) else None
    return mercury if isinstance(mercury, dict) else {}


def _gaps(
    seq: list[Evidence], lo: int, hi: int, open_end: bool, win: Interval
) -> list[Interval]:
    """Unobserved spans between ``seq[lo]`` and ``seq[hi]``, plus the tail of
    the window when ``open_end``."""
    found: list[Interval] = []
    for a, b in zip(seq[lo:hi], seq[lo + 1 : hi + 1], strict=True):
        if b.minute - a.minute > 1 and b.at > a.at + 1:
            found.append(Interval(start=a.at + 1, end=b.at))
    last = seq[hi]
    if open_end and (win.end - 1) // 60 > last.minute and win.end > last.at + 1:
        found.append(Interval(start=last.at + 1, end=win.end))
    return found


def _runs(ats: list[int], where: list[int]) -> list[list[int]]:
    """``where`` (sighting indices into a sequence observed at ``ats``) split
    where some evidence lies strictly between two sightings in time."""
    runs: list[list[int]] = [[where[0]]]
    for i, j in pairwise(where):
        if bisect_right(ats, ats[i]) < bisect_left(ats, ats[j]):
            runs.append([j])
        else:
            runs[-1].append(j)
    return runs


def _source_versions(
    day: str,
    source: AlertSource,
    seq: list[Evidence],
    entities: Mapping[tuple[str, str], dict[str, JsonValue]],
    stats: IndexStats,
) -> list[AlertVersion]:
    win = window(day)
    start = day_start(day)
    ats = [e.at for e in seq]
    seen: dict[tuple[str, str], list[int]] = {}
    for i, e in enumerate(seq):
        for entry in e.presence.entries:
            seen.setdefault((entry.alert_id, entry.content_digest), []).append(i)
    out: list[AlertVersion] = []
    for (alert_id, digest), where in sorted(seen.items()):
        runs = _runs(ats, where)
        stats.reappearing += len(runs) > 1
        for run in runs:
            first, last = seq[run[0]], seq[run[-1]]
            before = bisect_left(ats, first.at)
            after = bisect_right(ats, last.at)
            gone = seq[after] if after < len(seq) else None
            if not (
                any(start <= ats[i] for i in run)
                or (gone is not None and gone.at >= start)
            ):
                continue  # seen and gone within the carry: the previous day's
            kind: GoneKind | None = None
            if gone is not None:
                kind = (
                    "superseded"
                    if any(
                        x.alert_id == alert_id and x.content_digest != digest
                        for x in gone.presence.entries
                    )
                    else "removed"
                )
            hi = after if gone is not None else len(seq) - 1
            entity = entities[source, digest]
            published = _published(entity)
            out.append(
                AlertVersion.model_validate(
                    {
                        "lineage": first.presence.lineage,
                        "source": source,
                        "alert_id": alert_id,
                        "content_digest": digest,
                        "entity": entity,
                        "alert_type": published.get("alert_type"),
                        "created_at": published.get("created_at"),
                        "updated_at": published.get("updated_at"),
                        "absent_before": ats[before - 1] if before > 0 else None,
                        "first_seen": first.at,
                        "last_seen": last.at,
                        "gone_by": gone.at if gone is not None else None,
                        "gone_kind": kind,
                        "unobserved": _gaps(seq, run[0], hi, gone is None, win),
                        "window": win,
                    }
                )
            )
    return out


def alert_versions(
    day: str,
    evidence: Iterable[Evidence],
    entities: Mapping[tuple[str, str], dict[str, JsonValue]],
) -> tuple[list[AlertVersion], IndexStats]:
    """The versions of ``day``'s index. ``evidence`` may hold presences from
    anywhere; only those observed inside the window count. ``entities`` maps
    ``(source, content_digest)`` to the verbatim entity."""
    win = window(day)
    by_source: dict[AlertSource, list[Evidence]] = {}
    for e in evidence:
        if win.start <= e.at < win.end:
            by_source.setdefault(e.presence.source, []).append(e)
    stats = IndexStats()
    out: list[AlertVersion] = []
    for source in sorted(by_source):
        seq = sorted(by_source[source], key=lambda e: (e.at, e.presence.fetch_id))
        out += _source_versions(day, source, seq, entities, stats)
    stats.versions = len(out)
    return out, stats
