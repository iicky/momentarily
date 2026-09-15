"""Measure the poll latency the Worker's alerts-fetch cadence adds to the
published condition, straight from the archive.

For every alert-version object in a trailing window, latency = the first tick
that carried the version (the object BODY's `observed_at`) minus the alert's own
mercury `updated_at`. Reported split by ONSET (first version of an alert id) vs
update, and for the severe tier (severity_tier >= CANONICAL_SEVERITY_FLOOR) the
review grades on. Also reports `updated_at mod TICK` — uniform means the feed
posts continuously and a gated poll adds ~U(0, cadence); clustered means the
feed is on a clock and polling faster buys nothing. Clears carry no updated_at,
so their latency is bounded from the liveness stream's inter-tick gap and the
severe-episode version cadence gauges whether that bound is identifiable.

One-off diagnostic (not wired into any workflow); re-run when the fetch cadence
or the 5-minute pipeline gate is reconsidered, or to detect feed-behaviour
drift. Reads archive/alerts/ and archive/alerts_liveness/; writes nothing.

    uv run python -m training.alert_poll_latency --days 35 [--json]
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import Counter, defaultdict
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from typing import Any, cast

from momentarily.mapping import CANONICAL_SEVERITY_FLOOR, severity_tier
from training.load_r2 import fetch_objects, list_alert_keys, list_keys
from training.r2_client import load_config, make_client

TICK_SECONDS = 300  # the gated alerts-fetch cadence (index.ts isFiveMinuteBoundary)

# One archived alert version: (id, updated_at, observed_at, tier, active_period_start)
Rec = tuple[str, int, int, int, "int | None"]


def _quantile(xs: list[int], p: float) -> int | None:
    if not xs:
        return None
    xs = sorted(xs)
    i = min(len(xs) - 1, round(p * (len(xs) - 1)))
    return xs[i]


def _dist(xs: list[int]) -> dict[str, float | int | None]:
    if not xs:
        return {"n": 0}
    return {
        "n": len(xs),
        "mean": round(st.mean(xs), 1),
        "median": st.median(xs),
        "p90": _quantile(xs, 0.90),
        "p95": _quantile(xs, 0.95),
        "max": max(xs),
        "min": min(xs),
    }


def _mod_hist(xs: list[int], bin_s: int = 30) -> dict[str, int]:
    h: Counter[int] = Counter((x // bin_s) * bin_s for x in xs)
    return {str(k): h[k] for k in sorted(h)}


def measure(days: int) -> dict[str, Any]:
    cfg = load_config()
    client = make_client(cfg)
    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)

    keys = list_alert_keys(client, cfg.bucket, start, end)
    bodies = fetch_objects(client, cfg.bucket, keys)

    # (id, updated_at, observed_at, tier, active_period_start)
    recs: list[Rec] = []
    for b in bodies:
        obs = b.get("observed_at")
        ent = cast("dict[str, Any]", b.get("alert") or {})
        eid = ent.get("id")
        alert = cast("dict[str, Any]", ent.get("alert") or {})
        mer = cast("dict[str, Any]", alert.get("transit_realtime.mercury_alert") or {})
        ua = mer.get("updated_at")
        if not (isinstance(obs, int) and isinstance(ua, int) and isinstance(eid, str)):
            continue
        ap_start: int | None = None
        for ap in cast("list[dict[str, Any]]", alert.get("active_period") or []):
            s = ap.get("start")
            if isinstance(s, int):
                ap_start = s if ap_start is None else min(ap_start, s)
        at = mer.get("alert_type")
        at = at if isinstance(at, str) else None
        recs.append((eid, ua, obs, severity_tier(at), ap_start))

    versions_by_id: dict[str, list[Rec]] = defaultdict(list)
    for r in recs:
        versions_by_id[r[0]].append(r)
    # ONSET = the alert id's FIRST-OBSERVED version (archive write order), i.e.
    # the smallest observed_at, tie-broken by updated_at. Not min(updated_at):
    # a backdated Mercury revision can carry a lower updated_at than a version
    # seen earlier, and keying on updated_at would mislabel it as the onset.
    onset_key: dict[str, tuple[int, int]] = {
        i: min((v[2], v[1]) for v in vs) for i, vs in versions_by_id.items()
    }
    # SEVERE ONSET = each id's FIRST version that reaches the severe tier (again
    # by observed order), NOT the id-first version filtered to severe. An alert
    # that opens non-severe (e.g. Delays) and later ESCALATES to Severe Delays
    # crosses into severe on an UPDATE version, and that escalation IS the severe
    # onset the detection-latency metric wants — keying on the id-first version
    # would miscount it as a severe update and drop the escalation onset.
    severe_onset_key: dict[str, tuple[int, int]] = {}
    for i, vs in versions_by_id.items():
        sev = [(v[2], v[1]) for v in vs if v[3] >= CANONICAL_SEVERITY_FLOOR]
        if sev:
            severe_onset_key[i] = min(sev)

    lat_all: list[int] = []
    lat_onset: list[int] = []
    lat_update: list[int] = []
    lat_severe: list[int] = []
    lat_severe_onset: list[int] = []
    lat_severe_update: list[int] = []
    mod_all: list[int] = []
    mod_severe: list[int] = []
    ap_lat_severe_onset: list[int] = []
    tier_counts: Counter[int] = Counter()
    for eid, ua, obs, tier, aps in recs:
        lat = obs - ua
        lat_all.append(lat)
        mod_all.append(ua % TICK_SECONDS)
        tier_counts[tier] += 1
        is_onset = (obs, ua) == onset_key[eid]
        (lat_onset if is_onset else lat_update).append(lat)
        if tier >= CANONICAL_SEVERITY_FLOOR:
            lat_severe.append(lat)
            mod_severe.append(ua % TICK_SECONDS)
            if (obs, ua) == severe_onset_key.get(eid):
                lat_severe_onset.append(lat)
                if aps is not None:
                    ap_lat_severe_onset.append(obs - aps)
            else:
                lat_severe_update.append(lat)

    # Successful-fetch cadence -> the clear-DISCOVERY floor. archiveAlertsLiveness
    # writes one record per fetch ATTEMPT (success AND fail), keyed on execution
    # wall-clock, so raw key spacing mixes retries and failed fetches into the
    # gap. Fetch the bodies, keep outcome == "success", and collapse each to its
    # 5-min scheduled boundary (floor to 300; boundary jitter is < 60s) so a
    # boundary whose fetch only failed shows up as a widened gap, not a fetch.
    # The body is `{observed_at, outcome, fetched_at}` (worker/src/archive.ts
    # archiveAlertsLiveness writes `{observed_at: observedAt, ...liveness}`), so
    # observed_at is the attempt's execution wall-clock and is always present —
    # use it, NOT fetched_at, which on failure carries the last-good time.
    lv_keys: list[str] = []
    for k in list_keys(client, cfg.bucket, "archive/alerts_liveness/"):
        base = k.rsplit("/", 1)[-1].removesuffix(".json")
        try:
            t = int(base)
        except ValueError:
            continue
        if start <= datetime.fromtimestamp(t, UTC).date() <= end:
            lv_keys.append(k)
    lv_attempts = 0
    lv_success = 0
    success_boundaries: set[int] = set()
    for lv in fetch_objects(client, cfg.bucket, lv_keys):
        lv_attempts += 1
        if lv.get("outcome") == "success":
            lv_success += 1
            oa = lv.get("observed_at")
            if isinstance(oa, int):
                success_boundaries.add((oa // TICK_SECONDS) * TICK_SECONDS)
    boundaries = sorted(success_boundaries)
    gaps = [b - a for a, b in pairwise(boundaries)]

    # Severe-episode version cadence: can last-observed stand in for last-present?
    severe_ids = {
        i
        for i, vs in versions_by_id.items()
        if any(v[3] >= CANONICAL_SEVERITY_FLOOR for v in vs)
    }
    intra_gaps: list[int] = []
    spans: list[int] = []
    single_version = 0
    for i in severe_ids:
        obs_list = sorted({v[2] for v in versions_by_id[i]})
        if len(obs_list) == 1:
            single_version += 1
        intra_gaps.extend(b - a for a, b in pairwise(obs_list))
        spans.append(obs_list[-1] - obs_list[0])

    return {
        "window": {"start": str(start), "end": str(end), "days": days},
        "n_version_objects": len(recs),
        "n_alert_ids": len(versions_by_id),
        "tier_counts": {str(k): v for k, v in sorted(tier_counts.items())},
        "latency_seconds": {
            "note": "observed_at (body) - mercury updated_at; observed_at is the "
            "gated tick, so this is the poll latency the gate adds",
            "all": _dist(lat_all),
            "onset": _dist(lat_onset),
            "update": _dist(lat_update),
            f"severe_tier>={CANONICAL_SEVERITY_FLOOR}": _dist(lat_severe),
            "severe_onset": _dist(lat_severe_onset),  # first severe version per id
            "severe_update": _dist(lat_severe_update),  # later severe versions
        },
        "active_period_start_lat_severe_onset_sec": _dist(ap_lat_severe_onset),
        "updated_at_mod_tick": {
            "tick_seconds": TICK_SECONDS,
            "all": _dist(mod_all),
            "severe": _dist(mod_severe),
            "hist_all_30s": _mod_hist(mod_all),
            "hist_severe_30s": _mod_hist(mod_severe),
        },
        "liveness": {
            "note": "successful fetches only, collapsed to 5-min scheduled "
            "boundaries; gap is the successful-feed cadence / clear-discovery floor",
            "n_attempts": lv_attempts,
            "n_success": lv_success,
            "n_success_boundaries": len(boundaries),
            "median_gap_sec": st.median(gaps) if gaps else None,
            "gap_hist_sec": {str(k): v for k, v in sorted(Counter(gaps).items())},
        },
        "clears": {
            "n_severe_ids": len(severe_ids),
            "single_version_severe": single_version,
            "intra_episode_version_gap_sec": _dist(intra_gaps),
            "episode_span_sec": _dist(spans),
            "identifiability": "clear carries no updated_at; floor = liveness gap. "
            "Presence is reconstructable from the version stream "
            "only if intra-episode version gaps ~= the tick "
            "cadence; a high single-version-severe share means "
            "the exact clear boundary is under-identified.",
        },
    }


def _print_text(s: dict[str, Any]) -> None:
    w = s["window"]
    print(
        f"window {w['start']}..{w['end']} ({w['days']}d)  "
        f"{s['n_version_objects']} versions / {s['n_alert_ids']} ids  "
        f"tiers={s['tier_counts']}"
    )
    print("\nlatency_seconds (observed_at - mercury updated_at):")
    lat = s["latency_seconds"]
    for k in (
        "all",
        "onset",
        "update",
        f"severe_tier>={CANONICAL_SEVERITY_FLOOR}",
        "severe_onset",
        "severe_update",
    ):
        d = lat[k]
        if d.get("n"):
            print(
                f"  {k:>16}: n={d['n']:>5} mean={d['mean']:>7} median={d['median']:>6} "
                f"p90={d['p90']:>5} p95={d['p95']:>6} max={d['max']:>7}"
            )
    m = s["updated_at_mod_tick"]
    print(
        f"\nupdated_at mod {m['tick_seconds']}s (uniform => continuous posting): "
        f"all median={m['all'].get('median')} severe median={m['severe'].get('median')}"
    )
    print(f"  30s-bin hist all: {m['hist_all_30s']}")
    lv = s["liveness"]
    print(
        f"\nliveness: {lv['n_success']}/{lv['n_attempts']} successful fetches, "
        f"{lv['n_success_boundaries']} boundaries, median gap {lv['median_gap_sec']}s "
        f"(successful-feed cadence / clear-discovery floor)"
    )
    c = s["clears"]
    print(
        f"clears: {c['single_version_severe']}/{c['n_severe_ids']} severe ids "
        f"single-version; intra-episode version gap median "
        f"{c['intra_episode_version_gap_sec'].get('median')}s => "
        "per-episode clear boundary under-identified"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--days", type=int, default=35, help="trailing window to read (default 35)"
    )
    ap.add_argument("--json", action="store_true", help="print the summary as JSON")
    args = ap.parse_args()
    summary = measure(args.days)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        _print_text(summary)


if __name__ == "__main__":
    main()
