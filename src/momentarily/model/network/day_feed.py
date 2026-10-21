"""Which static feed each archived traversal day was collected against.

The answer comes only from the day's own ``DayProvenance``. A day is verified
when it carries one stamped ``feed_digest``. Days collected before digests
were stamped, or whose stamp covers only part of the day (``feed_digests``
ranges), are unverified; for those, ``inferred_feed_id`` names the built feed
whose ``feed_version`` matches the day's version, as a lead rather than a fact.
"""

from __future__ import annotations

from pathlib import Path

from momentarily.archive.traversals import DayProvenance, decode_day
from momentarily.model.network.contracts import DayFeed, DayFeedMap, NetworkIndex

DAY_OBJECT = "traversals.json.gz"


def resolve_day(prov: DayProvenance, index: NetworkIndex) -> DayFeed:
    if prov.feed_digest is not None:
        return DayFeed(
            feed_id=prov.feed_digest,
            verified=True,
            reason=None,
            feed_version=prov.feed_version,
            inferred_feed_id=None,
        )
    if prov.feed_digests:
        digests = sorted({r.digest for r in prov.feed_digests})
        spans = ", ".join(
            f"{r.digest[:8]} {r.first_ts}..{r.last_ts}" for r in prov.feed_digests
        )
        reason = f"digest stamped on only part of the day ({spans})"
        lead = digests[0] if len(digests) == 1 else None
        if len(digests) > 1:
            reason = f"day spans more than one feed ({spans})"
        return DayFeed(
            feed_id=None,
            verified=False,
            reason=reason,
            feed_version=prov.feed_version,
            inferred_feed_id=lead,
        )
    matches = sorted(
        fid
        for fid, entry in index.feeds.items()
        if prov.feed_version is not None and entry.feed_version == prov.feed_version
    )
    return DayFeed(
        feed_id=None,
        verified=False,
        reason="no feed digest stamped (collected before digest capture)",
        feed_version=prov.feed_version,
        inferred_feed_id=matches[0] if len(matches) == 1 else None,
    )


def day_feed_map(traversals_dir: Path, index: NetworkIndex) -> DayFeedMap:
    days: dict[str, DayFeed] = {}
    for day_dir in sorted(p for p in traversals_dir.iterdir() if p.is_dir()):
        blob = day_dir / DAY_OBJECT
        if not blob.is_file():
            days[day_dir.name] = DayFeed(
                feed_id=None,
                verified=False,
                reason=f"no {DAY_OBJECT} for this day",
                feed_version=None,
                inferred_feed_id=None,
            )
            continue
        _, prov = decode_day(blob.read_bytes())
        days[day_dir.name] = resolve_day(prov, index)
    return DayFeedMap(days=days)
