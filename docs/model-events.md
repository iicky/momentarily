# Data model, layer 3: events (`events/s1`) — scoping proposal

Status: decided on 2026-10-21 (§9). Slice 1, the match index, is built
(§10, "As built (slice 1)"); nothing else is. The build plan (§10) starts
with the match index and two data gates; deferred items are listed under
Deferred.

Layer 3 turns layer-2 observations into facts about trains. It answers
questions such as these: which scheduled trip a realtime train is, which
service day it belongs to, when it reached and left each stop (as a bound,
not a point guess), how long each segment and dwell took, the headway at each
stop, which scheduled trips never ran, which trains left their pattern, and
which minutes we actually observed. It keeps no judgement about whether
service is good. That belongs to layers 4 (normal) and 5 (rider impact).

All measurements here were made on 2026-10-20 and 2026-10-21 with throwaway
scripts kept outside the repo (`<work>/step3_scope/`). The prototype behind
them is a measurement tool, not the proposed code. Inputs:

- **Legacy lineage.** The layer-2 VehicleObservations backfilled from three
  trace days: 2026-08-12 (from 16:00 UTC, 450 invocations), 2026-09-15 and
  2026-10-19 (full days). The networks are `2faf6350456df7f6` (08-12) and
  `606d9f640f5aad67` (later days). The old pipeline's `archive/traversals/`
  covers the same days up to 10-18.
- **Raw lineage.** These samples were decoded with the repo decoder's
  `parse_feed`:
  - `step2`: the step-2 scoping sample, 40 polls of every feed 15 s apart,
    2026-10-20 14:36–14:46 UTC;
  - `live`: 8 GTFS-RT feeds every 30 s, 2026-10-21 02:38–04:55 UTC, from
    22:38 to 00:55 New York time, so it crosses the local midnight. All
    2,200 fetches in that span returned HTTP 200. Six feeds were frozen for
    about 75 minutes of it (§5);
  - the committed `recorded_hour` fixture, run through the derive job into a
    scratch directory.

  Figures marked "at 60 s" keep every second (`live`) or fourth (`step2`)
  poll, which matches the production cadence.

`[inference]` marks a claim that was not measured.

## 1. What layer 3 must answer

Read from the product back. Layer 6 publishes what layer 5 decides, so it
asks layer 3 nothing directly.

| Consumer question | Layer-3 answer | Granularity | Latency |
|---|---|---|---|
| L5: is this line or stopping pattern moving now? | `OpenState` open runs: last stop event of each train, the current elapsed gap since it (a finite interval at the evaluation time), and the still-open eventual duration (right-censored) | per run, per pattern segment | live, ≤ 2 min |
| L5: is a train overdue at this stop? (90% of rider-felt episodes are gap-only, per the rider-truth audit) | `OpenState` open headways: time since the last **stopping** train (service evidence) and, separately, since any train passed (physical), per (stop, route, direction) and per (stop, pattern), initialized from scheduled service, with coverage | per stop × pattern | live, ≤ 2 min |
| L5: is this a feed outage or a suspension? | coverage per feed from transport evidence only (failed fetch, missing invocation, stale header), and, separately, observed supply per feed and route against the schedule, with a multi-feed suspect-outage annotation | per feed × fetch | live, ≤ 2 min |
| L5: is this train rerouted or running express/local? | off-pattern and skipped-stop events, unscheduled runs | per run | live, ≤ 2 min |
| L5: when will it be over? (ETA of recovery) | the same events, and their history, for layer 4's recovery normals | per segment | nightly |
| L4: normal segment run time and dwell | SegmentTraversal and StopEvent dwell, as interval-censored bounds for fitting; numbered and L only. For the B division and SIR, stop-to-stop (dwell included) only (§3.1) | per pattern segment | nightly |
| L4: normal headway | Headway between consecutive passings, at two scopes: per (stop, route, direction) and per (stop, pattern) | per stop, per pattern | nightly |
| L4: normal supply | ScheduledTripOutcome: scheduled, observed, never listed, not covered | per trip | nightly |
| L4/L5: excluding bad data | CoverageInterval, old-timestamp tails (censored, not deleted), placeholder rows that were removed | per feed × interval | both |
| ETA grading (`eta_grade.py`) | StopEvent arrival bounds | per stop | nightly |

Layers 4 and 5 therefore need these things from layer 3:

- segment run times, dwells and headways, each tagged with the pattern,
  service date and time of day, and each a bound with an optional reported
  point;
- the open, right-censored versions of the same values for live use;
- scheduled trips that never appeared, together with coverage, so that a
  trip missing during an outage is never read as a cancellation.

### 1.1 What layer 3 replaces in the live pipeline

| Code | What it does today | What layer 3 replaces | What it gets wrong |
|---|---|---|---|
| `training/trace.py` | Splits the trace into runs per trip id (`TRIP_GAP_SECONDS` 600). An arrival is the first stopped row at a stop, timed by `vehicle_ts` with no clock guard (`_row_time`). The departure is the first later row not stopped there. Consecutive arrivals become a traversal, with `n_hops` from `stop_seq` differences: `exact` when 1, `interval` when more, `right` for an open end | StopEvent, SegmentTraversal, dwell | Point times, not bounds. B-division departures are the next arrival (§3.1). A "next"-family train never seen stopped at a stop leaves no arrival, so the hop becomes an `interval` row [inference: this is why the holdup found 14–22% `n_hops` > 1 on the 1, 3 and 6]. Placeholders are not removed (the trace has no assignment). Arrivals with no `stop_seq` and backward spans are dropped silently. `to_dwell_samples` drops every `interval` row |
| `training/traversal_archive.py` | One `archive/traversals/` object per UTC day of trace, with no lead-in | the per-service-day event files | A run crossing 00:00 UTC (20:00 ET) is split across two files, and its state is lost at the boundary. Nothing is keyed to a service day or a scheduled trip |
| `src/momentarily/archive/traversals.py` | The `Traversal` row codec (`at`, `seconds` arrival to arrival, `moving_seconds`, `n_hops`, `censoring`) | the contracts in §6 | No trip match, no pattern, no lineage, no coverage. `route_id` is folded (`6X` → `6`) |
| `src/momentarily/gtfs/static.py` `service_dates` | Service day = the nearer of the local date and the local date − 1, by the trip id's origin | the matched trip's ServiceDate (§2.4); `service_dates` stays only as the fallback for unmatched runs | **The midnight wrap.** Realtime ids carry after-midnight origins as `000100` (static `144100` = 24:01), so a run seen at 00:05 lands on the new calendar day. Measured: 80 (09-15) and 76 (10-19) matched runs a day disagree with the matched trip's day; 58 and 47 of them are wrapped ids. `Timetable.day_for` does not help (holdup: 5,139 disagreements) |
| `worker/src/vehicles.ts` (`deriveRouteMovementMetric`), `regime.ts`, `movement_state.ts` | Per tick: per-route counts of vehicles, stopped and moving trains, and cross-tick `advanced_n` / `stalled_n` with `from>to` transitions, feeding the debounced route regimes | the live subset of §7 (open runs, stop events, coverage) | `stalled_n` counts a trip whose `stop_id` did not change between ticks. For B-division trains (`stop_id` = last stop reached) that is also a train in motion between stations. All ticks count, including placeholders and frozen feeds. Routes are folded, and nothing is per pattern |
| `training/eta_grade.py` | Grades archived ETAs against trace arrivals and passings (`LEAD_IN` 20 min across UTC midnight) | StopEvent arrival bounds as the truth | Truth is a point (`vehicle_ts`), and a prediction counts as `never_seen` with no check of feed freshness (`fresh_feeds` is not read) |

## 2. Trip matching

### 2.1 What each lineage gives the matcher

| Signal | Raw lineage | Legacy lineage (trace) | Use |
|---|---|---|---|
| `trip_id` (`<origin>_<route>..<dir><path>`) | verbatim | verbatim | origin time, route field, direction, path code |
| `route_id` as published (`6X`, `SS`, `W`) | yes | **`Unknown`**. Layer 2 keeps no route at all, and the trace's folded route lives only in the archive JSON | tier 5 and the miss classes |
| `start_date` | yes | `Unknown` | check on the service day (§2.4) |
| NYCT `is_assigned` | yes (absent on unassigned B-division trips) | `Unknown` | removes placeholders |
| NYCT `train_id` | yes | `Unknown` | ties runs together across id changes |
| `current_status` | 0/1/2/absent | stopped or `Unknown` | stop-event semantics (§3.1) |
| `vehicle_timestamp` | yes | yes | reported event time; placeholder and frozen-position tests |
| TripUpdate stop list | yes | no | "listed" evidence (§4). It is **not** departure evidence for the B division (§3.1) |
| `schedule_relationship` | yes, but only `l` sends it, and only 0 (scheduled) was seen | no | recorded. ADDED or CANCELED would be used if they ever appear |
| `trip_replacement_period` | yes | no | §4 |

### 2.2 Before matching: what is not a train

1. **Placeholders.** On six feeds, every TripUpdate that is not yet
   dispatched comes with a VehiclePosition. That position has no status and
   a timestamp in the future. These rows are 34% of all rows: 33.8% on
   09-15, 34.5% on 10-19 and 33.0% on 08-12. They must never become
   movement.
   - Raw rule: `is_assigned` is not `true`. For SIR, which flags every trip
     assigned, the rule is "no status and a timestamp more than 60 s ahead".
   - Legacy rule: not stopped and `vehicle_timestamp > observed_at`. For
     the "last" family (B division, not SIR), any row that is not stopped,
     because assigned B-division trains always report STOPPED_AT (§3.1).
   - The legacy rule checked against the raw rule (at 60 s):
     - `step2`: of 2,235 raw placeholders it catches 2,183 and misses 52,
       and it flags 1 of 4,369 other rows;
     - `live`, fresh polls only: it catches 6,512, misses 399 and flags 3.

     The misses are B-division rows with `is_assigned` absent but a
     STOPPED_AT status. They may be trains that NYCT has not flagged
     [inference]. A legacy run built only from such rows is kept and
     matched like any other.
   - The B-division clause matters only when a feed is frozen. While the
     six feeds served one payload for about 75 minutes (§5), unassigned
     trips kept their scheduled departure as their timestamp, so it fell
     into the past. The pure timestamp rule then missed 9,915 placeholder
     rows (30 s polls). On
     the legacy days the clause changes 13 rows (09-15).
   - 338 trip ids on 09-15 appear only as placeholders. They are evidence
     that the trip was listed (§4), not that it ran.
2. **Frozen positions.** 6.3% (09-15) and 7.0% (10-19) of the remaining
   rows have a `vehicle_timestamp` more than 300 s old. In one example, a G
   trip "stopped" at A42S for more than 80 minutes with its timestamp
   frozen. Not every old timestamp is a ghost.
   - On 09-15, 6,480 of the 32,527 such rows (20%) come before a later
     stop change of the same run, so the train did move on. These rows are
     a held train, a delay that layer 5 must see.
   - The other 26,047 are a trailing block after which the run never
     advances (on 10-19: 6,621 and 28,146). A trailing block is
     **unidentified**, not a proven ghost. A terminal hold, a withdrawal or
     the end of the input window produce the same pattern.

   Rule:
   - Nothing is deleted. The arrival observed at the block's stop is kept.
   - Its departure is right-censored: `departure = Unknown`, flag
     `liveness_uncertain`. Old-timestamp rows add no further events,
     because they carry no new position.
   - The run's `end_state` says why it ended:
     - `left_feed`: the trip id left the feed;
     - `stale_tail`: it was still present with an old timestamp when it
       left or when the window ended;
     - `window_end`: it was still live at the end of the input window.
   - An old-timestamp block followed by movement is a stop with a long
     dwell, flagged `held_old_timestamp`.
   - Live, an open trailing block is `possibly_stale`. Because batch never
     deletes or retro-classifies the tail, the nightly build of a prefix
     equals the live state at the end of that prefix (§10, prefix replay).
   - Runs whose every row is old (14 on 09-15, 152 on 10-19) keep their
     TripMatch with `stale_only = true` and emit no StopEvents.

   The matching figures in §2.3 were measured with the whole trailing block
   removed. Keeping its first row changes them only where that row is the
   run's only evidence for a stop [not re-measured]. The bound widths in §3
   were measured with all old rows dropped.
3. **Duplicate invocations.** Two invocations ran 5 s apart on 10-19, and
   1,117 observation pairs are less than 30 s apart. Both observations are
   kept. They only narrow a bound and never add an event.

### 2.3 Algorithm

A **run** is one realtime `trip_id` seen with no gap longer than 30 minutes.
Splitting at 600 s, 1,800 s or 4 h gives 9,069, 9,053 and 9,045 runs on
09-15, so the gap barely matters. 30 minutes keeps a reused id apart and
tolerates a feed outage.

Matching uses the network-holdup tiers. The first tier that hits wins. Every
tier also needs the candidate service day to be active, with candidates
D−1, D and D+1 around the local date of the first observation.

| Tier | Rule | Time check |
|---|---|---|
| T1 exact id | (origin, route field, direction, path) equal to a stripped static id. Also try origin + 24 h (`000100` → `144100`) | ≤ 2 h at the first common stop |
| T2 path prefix | one path is a prefix of the other (realtime paths are often truncated or empty) | ≤ 2 h |
| T3 line code | same 2-character line code, other suffix | ≤ 2 h |
| T4 path differs | same origin, route field and direction | ≤ 30 min |
| T5 drift | no id hit. Same route and direction, observed stops in a pattern's order, the nearest active trip at the first common stop | ≤ 15 min |
| miss | `unscheduled`, `service_day_mismatch`, `path_code_differs`, `route_field_differs`, `unknown_route`, `unparseable_id` | — |

A run seen at only one stop is flagged `stationary` whatever its outcome
(GS trips match at T1 and stay `stationary`). It yields no traversal.

Scheduled-trip identity and observed-pattern fitting are **separate
results**. Every run carries `compatible_pattern_ids`: the patterns of its
route and direction whose stop order contains its observed platform stops.
It is computed from observations alone, before and independently of
matching, so a live run has a pattern attribution even when the
scheduled trip is unknown. A matched trip fixes `pattern_id`. Otherwise
the run is `pattern_unique`, `pattern_ambiguous` or `off_pattern` (no
compatible pattern).

Measured on raw runs at 60 s, where live matching has only tiers 1–4
(§7):

| Sample | Runs | No tier 1–4 match | of which: one compatible pattern | 2+ patterns (ambiguous) | none (off pattern) |
|---|---|---|---|---|---|
| `live`, fresh polls | 845 | 262 (31.0%) | 60 | 137 (2–6+ patterns) | 65 |
| `step2` | 504 | 101 (20.0%) | 4 | 96 | 1 |

On the legacy days 12.3% (09-15) and 16.8% (10-19) of runs have no tier 1–4
match. Ambiguity is carried, never resolved by a guess (§4: headways,
`OpenState`).

The proposal changes the holdup matcher in four places:

- **One-to-one assignment.** Today each run is matched independently. On
  09-15, 456 scheduled trips were claimed by two or more runs, and 547 on
  10-19. 334 of the 456 involve tier 5, and 194 involve runs that overlap in
  time. Proposed: tiers 1–4 first; then tier 5 as a **minimum-cost,
  maximum-cardinality assignment** per (route, direction, service day)
  over the remaining scheduled trips. Cost = |offset| at the first common
  stop, only candidates within 15 min, ties broken by trip id. A run that
  gets no trip becomes `unscheduled`, with its nearest candidate kept. Its
  stop events still use its observed pattern fit (`pattern_attribution`).
  - Measured with whole trailing old-timestamp blocks removed first (§2.2
    now keeps their arrival, which was not re-measured). The independent
    match rates are 97.7% (09-15), 96.9% (10-19) and 96.5% (08-12,
    partial).
    - Greedy (smallest offset first) assigns 660 of 908 tier-5 runs on
      09-15, 892 of 1,197 on 10-19 and 449 of 518 on 08-12. Match rates:
      94.99%, 93.43% and 94.71%.
    - Min-cost (`scipy` `linear_sum_assignment`) assigns 679, 910 and
      469, for 95.20%, 93.64% and 95.24%. It gives 145, 136 and 147 runs
      a different trip from greedy.
    - **With retained tails** (§2.2 as now specified: a trailing block's
      arrival row kept, later old rows dropped, stale-only runs kept),
      min-cost gives 95.19% (09-15) and 92.37% (10-19), and greedy 94.99%
      and 92.19%. 10-19 is lower because its 152 stale-only runs now take
      part. These are the proposal's figures.
  - After min-cost assignment with retained tails, 435 of 9,053 (09-15)
    and 674 of 8,832 (10-19) runs stay unmatched.
  - **Match rate is not accuracy.** No ground truth exists for which
    scheduled trip a drifted train is. A min-cost assignment can give a
    late train a later departure's trip. One-to-one assignment fixes the
    scheduled-trip outcome count, but it does not deduplicate physical
    trains: a losing run is still a run. The matching slice is therefore
    graded with a withheld-input evaluation and an owner spot-check (§10).
  - The tier-5 runs that lose are not shown to be extra trains. Under
    min-cost assignment 229 (09-15) and 287 (10-19) lose; greedy leaves
    248 and 305. For 152 of the 229 and 175 of the 287, the run holding
    the loser's nearest candidate trip does not overlap it in time. They
    may be drifted copies of trains already counted. Their class is open.
  - Matching ignores order. Among adjacent same-pattern pairs at a stop,
    the order is inverted against the schedule in 4.7% of pairs that
    involve a tier-5 run, against 0.9% for id-only pairs (a rough proxy, not
    re-measured here). An order constraint is an open refinement.
  - Id tiers also double-claim: 130 (09-15) and 105 (10-19) runs hold a
    trip that another id-tier run already holds. Only 2 of 124 such groups
    (09-15) share an id. The usual case is one train split in two by a
    path-code change, e.g. `118650_1..N17R` followed within 60 s by
    `118650_1..N`. Proposed: join consecutive runs with the same origin,
    route field and direction when one ends and the next starts within
    120 s at the same or an adjacent stop. Raw lineage can confirm this
    with `train_id`, legacy cannot [not measured]. A run that stays
    double-claimed carries `shares_scheduled_trip = true`.
- **Placeholders are removed before runs are formed; trailing
  old-timestamp blocks are kept as censored tails** (§2.2).
- **One trip id at two stops in one poll.** This happens in 18 runs on
  09-15 (591 polls, 17 of them on the 7), 17 on 10-19 and 5 on 08-12. It
  produced the degenerate `after == by` events the prototype emitted: 510
  arrivals, 531 departures and 240 passages on 09-15. These are two
  vehicles sharing a trip id [inference].
  - Raw: split by `train_id`, then by stop continuity. `entity_index` is
    a position in one payload and is not stable across payloads, so it is
    never used for identity.
  - Legacy: split by stop continuity (two interleaved stop tracks).
  - A run that cannot be split is flagged `duplicate_trip_id` and emits no
    events for the shared polls.
- **Raw lineage: tighter candidates.** Use `start_date` as a check on the
  matched day (§2.4), and `train_id` to join runs of one train.

**Identity.**
- `run_id = f"{lineage}:{source}:{trip_id}@{first_observed_at}#{track}"`.
- `track` is 0 for an unsplit run. For split tracks it is the ordinal of
  the tracks sorted by (first observed stop's pattern index, first stop
  id, first observation instant, then the track's full observed sequence
  of (observed_at, stop_id, vehicle_timestamp) compared lexicographically).
  The last key breaks the tie between two trains that appear at the same
  stop in the same poll and then diverge. It depends only on content,
  never on entity order in a payload. Identical sampled positions do not
  prove one physical train. Two tracks with identical full sequences are
  merged only when their published identities are compatible: equal
  `train_id` (raw), or no published identity at all (legacy). Merging
  indistinguishable legacy tracks is observational deduplication, not proof
  of one physical train. The merged run is flagged `deduplicated_by_position`. If the
  `train_id`s differ, both tracks are kept, ordered by `train_id` as the
  final key, and each carries `identity_uncertain_with = [other run_id]`.
- A joined run (split-run join above) keeps the earliest member's
  `run_id` and lists `member_run_ids`.
- Live ids are provisional: the nightly build may split or join
  differently. The nightly TripMatch lists every live id it absorbed in
  `supersedes_run_ids`.
- Events key on their occurrence within the run (`StopEvent.seq`), so a
  run that revisits a stop after an order violation keeps distinct keys.
  Traversals key on (`run_id`, `from_seq`, `to_seq`), not on pattern
  index.

### 2.4 Service day

A run's ServiceDate is that of the scheduled trip it matched: the candidate
day whose active service holds the trip, nearest in time at the first common
stop. When the origin matches only after adding 24 h (a wrapped id), the
service day is the previous one.

- The old rule, `service_dates(at, trip_id)[0]` (the nearest of the local
  date and the local date − 1 by the id's origin, with no wrap), picks a
  different day for 80 matched runs on 09-15 and 76 on 10-19, with runs
  matched independently and old-timestamp rows kept (the holdup definition). On
  09-15 these are 58 wrapped ids, 20 tier-5 runs and 2 tier-2 runs. On
  10-19 they are 47 wrapped, 28 tier-5 and 1 tier-2. This matches the
  holdup audit's 1.06% over 68 days. On the proposal's definition
  (one-to-one, trailing old-timestamp blocks removed) the counts are 66 and 58, in the
  table below. Fewer tier-5 runs are matched there.
- An **unmatched** run with a parseable id gets its ServiceDate from the
  old rule, `service_dates(first_at, trip_id)[0]`, marked
  `service_date_basis = "id_origin"`. An id that does not parse falls back
  to a clock rule (04:00 New York cut, `"clock_rule"`). Scored against the
  matched runs' true days (one-to-one, trailing old-timestamp blocks removed):

  | Rule | 09-15 wrong (all) | 09-15 wrong (first seen 00–04 ET) | 10-19 wrong (all) | 10-19 wrong (00–04 ET) |
  |---|---|---|---|---|
  | `service_dates` | 66 of 8,605 (0.8%) | 64 of 530 (12%) | 58 of 8,128 (0.7%) | 57 of 529 (11%) |
  | clock rule | 438 of 8,605 | 436 of 530 (82%) | 447 of 8,128 | 446 of 529 (84%) |

  The clock rule is wrong because most trains seen 00:00–04:00 belong to
  the new calendar date. The `service_dates` residual is the 24:00 wrap: a
  realtime `000100` at 00:05 is the same instant as 24:01 of the previous
  day, so the id alone cannot decide. Only a scheduled-trip match can. The
  11–12% is measured on **matched** runs used as controls, not on
  unmatched runs, for which there is no ground truth. So an unmatched run
  first seen 00:00–04:00 carries `service_date_ambiguous = true` and the
  alternative date, not a calibrated error rate.
- **Raw `start_date` is not the service date.** It is the calendar date on
  which the trip starts. Across local midnight (`live`, fresh polls, 754
  matched runs):
  - 680 trips scheduled before 24:00 carry the service date;
  - 57 trips scheduled at 24:00 or later carry the service date + 1, with
    both wrapped and unwrapped realtime ids. Realtime ids also appear
    unwrapped (`144050_6..N01X016`);
  - 17 disagree in other ways: tier-5 matches (including runs whose
    realtime origin is after midnight but that matched a pre-midnight
    trip), plus tier-2 and tier-4 cases.

  So `start_date` equals the service date for 683 of 754 runs here, and a
  pipeline that took it as the service date would misfile 9% of runs
  around midnight. Layer 3 checks it instead: expected `start_date` = the
  local date of the matched trip's first scheduled departure. A run that
  disagrees is flagged `start_date_mismatch`. On the daytime `step2`
  sample, all 501 matched runs agree.
- **DST.** GTFS seconds are placed with `instant_of(service_date, s, tz)`,
  which counts from noon − 12 h. On 2026-11-01 that origin is 01:00 EDT, not
  midnight. No measured day crosses a DST change [not verified].

### 2.5 Measured

Legacy lineage. 08-12 is a partial day. The table matches runs
independently, as the holdup audit did, with old-timestamp rows kept. §2.3
gives the one-to-one rates with trailing old-timestamp blocks removed,
which are the proposal's figures.

| UTC day | Rows | Real rows | Runs | Matched | Id-anchored (T1–T4) | T5 | Unmatched (moving / stationary) |
|---|---|---|---|---|---|---|---|
| 2026-08-12 | 312,743 | 209,429 | 3,788 | 96.3% | 82.4% | 525 | 126 / 15 |
| 2026-09-15 | 777,446 | 514,653 | 9,053 | 97.7% | 87.7% | 909 | 140 / 65 |
| 2026-10-19 | 755,488 | 494,711 | 8,832 | 96.9% | 82.0% | 1,318 | 263 / 10 |

Tiers on 09-15: T1 6,089, T2 1,673, T3 98, T4 79, T5 909, miss 205. The
misses are 130 `unscheduled` moving and 65 `unscheduled` stationary, plus
7 `service_day_mismatch`, 2 `path_code_differs` and 1
`route_field_differs`. On 10-19 most misses are also `unscheduled` (245).

**How legacy lineage degrades matching.** Layer 2 keeps no route for legacy
rows. Two routes were tried: the trip id's route field (all that layer 2
offers) and the trace's folded route (read from the archive JSON). They
change the outcome of 3, 10 and 5 runs a day. The route matters only in
tier 5 and the miss classes. The route a rider sees comes from the matched
ScheduledTrip, not from the realtime row. That matters: `W` trains carry `N`
in the trip id (134,414 holdup rows), and `5` trains carry `2`. So an
unmatched legacy run has only its trip-id route field, and it must say so
(`route_basis = "trip_id_field"`).

Raw lineage:

| Sample | Runs | Matched | Id-anchored | Same polls reduced to legacy fields |
|---|---|---|---|---|
| `step2` at 60 s (10:36 ET, Tue) | 504 | 99.4% | 80.0% | 510 runs; all 504 raw runs got the same match, and 6 extra runs are unassigned rows the legacy rule keeps |
| `live` at 60 s, fresh polls (22:38–00:55 ET) | 845 | 89.2% | 69.0% | 864 runs, 89.0% matched; 841 of the 845 raw runs got the same match, 2 a different one and 2 none; 19 extra runs are unassigned rows the legacy rule keeps |

The late-night misses are 71 `unscheduled` runs, most of them on E (18),
D (17), 6 (13), Q (10) and A (8). The legacy days match 95.7% (09-15) and
94.2% (10-19) of runs first seen in the same 22:30–01:00 window, so this
night had more unscheduled running [inference: overnight planned work; the
alerts were not checked].

`train_id` was constant within all but 2 of 845 live runs and 1 of 504
`step2` runs. In `live`, 17 pairs of consecutive runs of one train id lie
within 120 s of each other, and the frozen-feed gap splits trains into such
runs. That is what `train_id` is for: linking the runs of one train across
a gap.

## 3. Stop events and segment traversals

### 3.1 What a vehicle row means, per feed

These were measured poll to poll on `step2` (15 s), with each vehicle joined
to its own trip update in the same payload.

| Feed family | `stop_id` is | Changes on | Status | `vehicle_timestamp` | Departure visible from |
|---|---|---|---|---|---|
| **"last"**: `ace`, `bdfm`, `g`, `jz`, `nqrw`, `si` | the last stop reached | arrival at the next stop | always STOPPED_AT for assigned trains | on a stop change, it **equals** the trip update's arrival time at the new stop (617 of 617 changes where the TU lists that stop first) | **not observable**; bounded only by the next arrival (below) |
| **"next"**: `numbered`, `l` | the current or next stop | departure (`STOPPED_AT X` → absent or `IN_TRANSIT_TO Y`) | `numbered`: INCOMING_AT / STOPPED_AT / absent. `l`: STOPPED_AT / IN_TRANSIT_TO | `numbered`: within the bracket on 609 of 628 changes. `l`: equals the poll (median age 1 s), so it is useless as an event time | the vehicle row itself |

**B-division trip updates are not a departure signal.** Taking "the trip
update stops listing s" as the departure was tested on `step2` and fails:
- 1,004 such drops (`ace`, `bdfm`, `g`, `jz`, `nqrw`; SIR was not in this
  probe) fall 32–47 s (p1–p99) after the stop's predicted time. The lag is
  the same whether the vehicle is still at the stop (400 drops) or already
  at the next one (604).
- The drop correlates with the vehicle's next arrival at only 0.12.
- Predicted arrival equals predicted departure in all 9,408 first-stop
  entries.

That is a time-based prune of the prediction list, not an observed
departure. B-division departure, dwell and run time are therefore bounded
only by the next arrival, in raw lineage as well as legacy. A different
B-division departure signal, such as `vehicle_timestamp` behaviour while
a train stands at a stop, is untested [inference].

So "first observation at a stop" means different physical events per feed.
For the B division it is an arrival. For the numbered and L feeds it is the
departure from the previous stop. The old pipeline's arrival, the first
*stopped* row at a stop (`trace.py` `_RunState`), is right for both
families. Its departure, the first row no longer stopped there, is right
for "next" but is the arrival at the next stop for "last". For B-division
trains it therefore folds dwell into run time, and its `moving_seconds`
lower bound is about zero. Layer 3 inherits this limit for the B division;
it only stops pretending otherwise.

### 3.2 Bound rules

Every event is a **bound** `(after, by]` in integer UTC seconds, exclusive
below and inclusive above: the event happened after the last observation
that did not show it and no later than the first that did. These are
**feed-supported** bounds, statements about what the feed said and when we
polled, not guaranteed physical truth. A **reported** point (the feed's
own time) is kept beside the bound when the feed supplies one.

**Service evidence.** Each StopEvent says how strongly it shows passenger
service at the stop:
- `stopped`: the train was observed stopped there. In the "next" family
  that is a STOPPED_AT row at s. In the "last" family the feed reported s as
  the last stop reached, a feed statement that the train stopped at s
  [inference: not checked against express running].
- `reported_not_stopped`: "next" family only. The feed named s as the
  current or next stop, but never showed the train stopped there (13–14%
  of "next"-family stop blocks). It is physical passage. Service is
  unknown.
- `inferred_passage`: a pattern stop between two observed stops that the
  feed never named (1.5% of on-pattern events). The train went past or
  through s, or around it on a reroute. Service is unknown.

Only `stopped` is evidence of passenger service. The other two never
close a rider-service gap (§4).

- **Arrival** (`stopped` only).
  - "last" family: `(previous observation, first observation at s]`.
    Reported: `vehicle_timestamp` of that row, which equals the TU arrival.
  - "next" family: `(last observation at s not stopped, first STOPPED_AT
    observation at s]`. Reported: `vehicle_timestamp` (numbered only).
  - **First stop of a run.** No earlier observation exists, so the
    arrival is left-censored: `after = null`, `by` = first observation,
    flag `first_of_run`. On 09-15 there are 6,463 such arrivals (4,549 B
    division, 1,914 numbered and L), 3% of stop blocks. They are excluded
    from the widths in §3.3.
- **Departure.**
  - "next" family: `(last observation at s, first observation past s]`.
  - "last" family, raw and legacy: only `[arrival.after, arrival at the
    next stop]`, with `departure_basis = "next_arrival"` (§3.1).
- **Passage** (`reported_not_stopped`, `inferred_passage`):
  `(last observation showing the train before s, first observation showing
  it past s]`.
- **Reported time earlier than the bracket.** Set
  `after = min(after, reported − 1)` and flag `reported_widened_after`,
  including when `reported == after`. When `after` is null (left-censored),
  no widening applies: `reported` is kept if `reported ≤ by`. This keeps the integer exclusive
  lower endpoint and puts `reported` inside `(after, by]`. A reported time
  after `by` is dropped and flagged `reported_after_bound`. An explicit
  inclusive-endpoint flag was rejected because it would give every
  consumer two interval conventions to handle.
- **Dwell** = departure − arrival: `lo = max(0, dep.after − arr.by)`,
  `hi = dep.by − arr.after`. For the "last" family it is not separable
  from run time and is emitted as `Unknown`.
- **Segment run time** = departure(from) → arrival(to), with
  `lo = max(0, to.after − from.by)` and `hi = to.by − from.after`. The old
  pipeline's stop-to-stop time (arrival → arrival, dwell included) is also
  emitted, for continuity with `archive/traversals`. For the "last" family,
  stop-to-stop is the only meaningful duration.
- **Censoring propagation.**
  - A duration whose **start** event has `after = null` has a known lower
    bound and no upper bound: `lo = max(0, end.after − start.by)`,
    `hi = null`, `censoring = "right"`. For example, a start ≤ 10 and an
    end in (19, 20] give a duration ≥ 9.
  - A duration whose **end** is still open (`OpenState`) is also
    right-censored. No duration in layer 3 is left-censored, so
    `censoring` is `"none"` or `"right"`.
  - Lower bounds are always clamped to 0.
  - `uncovered_s` is null when `after` is null, because there is no
    interval to measure.
- **Degenerate events** (`after == by`) come only from one trip id at two
  stops in one poll (§2.3). Splitting removes them. An unsplit run emits
  nothing for those polls, so `after < by` stays a strict invariant.

### 3.3 Measured bound widths

Legacy lineage, p50 / p90 / p99 in seconds, with the share ≤ 120 s:

| Bound | 08-12 | 09-15 | 10-19 |
|---|---|---|---|
| arrival, "last" (B div) | 60 / 60 / 180 · 98.8% | 60 / 60 / 180 · 98.9% (n 110,475) | 60 / 61 / 180 · 98.9% |
| arrival, "next" (numbered, L) | 60 / 60 / 60 · 100% | 60 / 60 / 67 · 99.8% (n 83,721) | 60 / 60 / 120 · 99.4% |
| departure, "next" | 60 / 60 / 60 · 99.8% | 60 / 60 / 69 · 99.6% | 60 / 61 / 178 · 98.8% |
| departure, "last" (next arrival only) | 180 / 300 / 660 · 26.0% | 180 / 300 / 600 · 25.4% | 180 / 300 / 600 · 23.6% |
| dwell, "next" | 120 / 120 / 120 · 99.8% | 120 / 120 / 129 · 97.0% | 120 / 121 / 238 · 88.5% |
| segment run time, "next" | 120 / 120 · 99.7% | 120 / 120 · 97.7% | 120 / 120 · 92.0% |
| segment run time, "last" (next arrival only) | 180 / 300 · 25.9% | 180 / 300 · 25.2% | 180 / 300 · 23.4% |
| passage at an unreported stop | 60 / 180 (last), 60 / 60 (next) | 60 / 180, 60 / 60 | 60 / 360, 60 / 297 |

The share of arrival bounds ≤ 60 s is 96–98% on 09-15 and 89–90% on 10-19.
Raw lineage at 60 s gives arrival bounds of 60 / 60 / 60 (`step2`) and
60 / 60 / 61 (`live`, fresh polls) for both families.

**Per event, duration bounds are mostly uninformative.** A 120 s-wide
interval around a 60–120 s inter-station run or a short dwell [inference:
typical lengths, not measured here] usually has
a lower end of 0:

| Share with lower bound = 0 | 08-12 | 09-15 |
|---|---|---|
| run time, "next" | 81.6% | 81.3% |
| dwell, "next" | 92.2% | 91.7% |
| run time, "last" (next arrival only) | 99.7% | 99.7% |

So the median numbered-line dwell is "somewhere in [0, 120] s", and a
midpoint is meaningless. Layer 3 still emits the bounds, because they are
exact statements of what was observed. Layer 4 must estimate run-time and
dwell distributions with **interval-censored** methods across many
events (e.g. a censored-likelihood or Turnbull estimator per segment and
time band), never from per-event midpoints. Whether 60 s polling is enough
for layer 4 is therefore open. It depends on the estimator's precision,
which this scoping did not measure, so layer-2 decision 8 should be
revisited once layer 4 exists.

**Reported points.** On 09-15 the reported time fell inside the arrival
bracket for 99.0% of B-division arrivals and 97.1% of numbered arrivals.
When it fell before the bracket, it did so by a median of 5 s (B) and 2 s
(numbered), with p99 87 s and 58 s: the feed published the event late. It
fell after the bracket for 0.4% (B), never for numbered. In `live` (fresh
polls) it fell inside for 98.4% (B) and 97.6% (numbered). The rule for an
earlier reported time is in §3.2.

### 3.4 The awkward cases

| Case | Measured (09-15) | Handling |
|---|---|---|
| Pattern stops never reported between two polls | 3,259 stops (1.5% of on-pattern events) | `inferred_passage` events, bound by the surrounding observations; no service evidence (§3.2) |
| "next"-family stop blocks never seen stopped | 13,258 | `reported_not_stopped`, with a passage bound and no arrival; no service evidence |
| Realtime-only stop ids (R60N, R65S, D23S, H19N, …: 46 ids) | 717 stop blocks | no StopEvent. Merged through using layer 1's realtime-stop alias table when it exists (holdup recommendation 1); the segment's `hidden_realtime_stops` counts them |
| Off-pattern stops (reroute, express/local switch) | 656 blocks; 3.6% of matched runs in the holdup were off-pattern and 5.5% skip-only | the StopEvent has `pattern_index = null`. The run gets `deviation` (`consistent`, `skips_only`, `off_pattern`, `order_violation`, …) and its `compatible_pattern_ids` |
| Stop order going backwards | 818 blocks | event emitted, flagged `order_violation`; no traversal across it |
| Duplicate invocations in a minute | 34 pairs (10-19: 1,117) | both kept. Bounds use the nearest observations on each side |
| Feed outage inside a bracket | §5.1 | the bound simply widens. A bound that spans an uncovered interval carries `uncovered_s` > 0, so layer 4 can drop it. Low or zero train counts never count as uncovered |
| Short trips (GS, 90 s) | 459 GS runs on 10-19, each 1–2 observations at the origin | `matched` at T1 and flagged `stationary`, with no traversal. GS supply is observable: 477 of 478 scheduled GS trips on 09-15 are observed. The holdup audit's "GS is absent" (21 runs in 68 days) is an artifact of traversals needing two stopped stops. GS movement is not observable |
| Old-timestamp positions | 32,015 rows of matched runs with an old `vehicle_timestamp`; 26,047 of all such rows are trailing | the §2.2 rule: the arrival is kept, the tail is censored (`liveness_uncertain`, `end_state = stale_tail`). An old-timestamp block followed by movement is a stop flagged `held_old_timestamp` |
| One trip id at two stops in one poll | 18 runs, 591 polls (17 runs on the 7) | split, else flagged `duplicate_trip_id` with no events for those polls (§2.3) |

### 3.5 Holdout against the old pipeline

**Arrival times.** The prototype's arrival bounds were compared with
`archive/traversals` (`at` = old arrival at `from_stop`), joined on (trip
id, stop). The old `at` is the same `vehicle_timestamp` that layer 3
keeps as `reported`, and reported points fall inside the bracket 97–99% of
the time (§3.3). So this check mostly tests that the two pipelines pair
the same rows. It is not an independent check of the times.

| Day | Old rows | Old `at` inside the new bracket | Within 60 s of it | Further | No new event at that (trip, stop) |
|---|---|---|---|---|---|
| 2026-08-12 | 76,597 | 90.9% | 3.1% | 1.0% | 5.1% (3,317 trip unmatched, 568 off pattern or frozen) |
| 2026-09-15 | 193,718 | 94.6% | 1.9% | 0.9% | 2.6% (3,797 trip unmatched, 1,148 off pattern or frozen) |

**Durations.** For old `exact` traversals, the old `seconds` (arrival to
arrival) falls inside the new stop-to-stop interval for 103,140 of 103,305
B-division hops and 69,382 of 69,568 numbered and L hops (09-15). On 08-12
the figures are 38,817 of 38,949 and 27,188 of 27,229. The old
`moving_seconds` falls inside the new numbered and L run-time interval for
46,333 of 46,481 (09-15). Old exact rows with no new pair are 7,970 of
180,843 (4.4%, 09-15) and 5,041 of 71,219 (7.1%, 08-12, a partial day).
This is again a consistency check, not truth.

**Per-day gate for the backfill.** Days without a hand check are built
with this gate:
- arrival containment ≥ 90%;
- duration containment ≥ 99%;
- no-pair share ≤ 8%.

A day that fails is reported and left out of layer 4 fits until it is
inspected.

The new events that the old archive lacks (09-15) are 15,653 stop events
and 2,387 passages. These are mostly "next"-family passages and runs'
first stops, which the old pipeline drops because it needs a stopped row
[inference].

## 4. Headways and absence

- **Headway.** Adjacency is computed separately at two scopes, ordering
  passings by the middle of their bounds:
  - `route`: consecutive passings of any observed train of the route at a
    (stop, direction);
  - `pattern`: consecutive passings of trains of one pattern at a (stop,
    pattern), with other patterns' trains ignored.

  The pattern scope is not a filter on the route pairs. Filtering would
  drop the P→P headway whenever a Q train passes in between (P–Q–P), which
  is exactly the branch-level gap that the rider-truth audit found
  invisible at line level in 36% of branch episodes. 18–19% of adjacent
  route-scope pairs mix two patterns, and each such pair hides a
  same-pattern pair that only the pattern scope sees.

  Each headway is built at two **evidence levels**:
  - `service`: only `stopped` events (departure for "next", arrival for
    "last", whose departure is not observable, §3.1). This is the
    rider-service series. A `reported_not_stopped` or `inferred_passage`
    event never closes or resets it, so a local suspension that trains
    pass through stays a growing gap.
  - `physical`: any event at the stop, passages included. It shows that
    trains moved through the stop, for layer 5 to tell a suspension from
    an outage.

  Each record carries both trains' runs, patterns, passing bounds and
  service evidence.

  **Pattern attribution for headways.** A run counts toward a pattern
  scope when its `pattern_id` is fixed, or when it has exactly one
  compatible pattern. An ambiguous run (2+ compatible patterns, §2.3)
  counts toward each compatible pattern with `pattern_attribution =
  "ambiguous"`. Pattern-scope adjacency is built **twice**, as two
  separate series (`ambiguity_set`):
  - `certain`: only runs with `scheduled_trip` or `unique_fit`
    attribution;
  - `with_ambiguous`: certain runs plus ambiguous ones.

  The `certain` series gives the upper bound on a pattern gap and
  `with_ambiguous` the lower bound. Filtering one series cannot rebuild
  the other, because adjacency changes when trains are added or removed,
  so both are stored. `OpenState` keeps one clock per series. An
  off-pattern run counts at route scope only.
  - Route scope, measured (all evidence, before this split): 206,597
    records on 09-15 and 196,747 on 10-19, with a bound width of p50 120 s,
    p90 120 s and p99 180–420 s.
  - The pattern scope and the service level were not measured separately.
    The record counts are at least the same-pattern route pairs, and at
    most the route counts, minus passages [inference].
- **Open gaps** (`OpenState`, live and at a nightly window end).
  - **Initialization.** For each (stop, route, direction) and each
    (stop, pattern) with scheduled service in force (from layer 1, either
    of the overlapping service days), the open gap starts (`since`) at the
    last observed service passing. When none exists, it starts at the
    later of the first scheduled passing of that scope at that stop and
    the start of coverage.
  - When no passing has been observed since service began, the record is
    `kind = "scheduled_no_observation"` with `last_event = null`. It means
    "service should exist, and no train has been seen". So an entirely
    absent pattern is visible from its first scheduled minute.
  - **Endpoint propagation.** At evaluation time `t`, a last passing
    bounded by `(after, by]` (integer seconds, so in `[after + 1, by]`)
    gives a **current elapsed gap** in `[t − by, t − after − 1]`. Example:
    at `t = 200`, a passing in `(100, 160]` gives `[40, 99]`. Event-time
    uncertainty is kept even without an outage. A left-censored last
    passing (`after = null`) gives `hi = null`.
  - **Across outages.** An uncovered interval does not erase the last
    passing or restart the clock. A train may have passed unseen during
    the outage, so the time since the last observed passing is only an
    upper bound on the real gap. When an uncovered interval ended at `e`
    and may overlap the last-passing bound, `lo = max(0, t − max(by, e))`
    for a finite `by`, and `lo = 0` while the feed is still uncovered.
    `hi` stays `t − after − 1`.
  - The current elapsed gap is a **finite interval at `t`**
    (`OpenState.current_gap`). The eventual headway that will close it is
    a separate value (`OpenState.eventual`): a `DurationBound` with
    `censoring = "right"`, `hi = null` and `lo = current_gap.lo`.
  - **Overnight closure.** A headway whose span contains a period with no
    scheduled service at that scope and stop is flagged
    `spans_no_service`. Layer 4 excludes it from normals, and the first
    train after the closure starts a fresh series.
- **Cross-day ownership.** A headway belongs to the follower's service
  date. Its leader may belong to the previous service date
  (`leader_service_date` is kept). Day D's build window starts at
  D 00:00 local − 2 h (§6). When no leader passing lies in that window
  (for example a suspension of more than 2 hours across midnight), the
  build carries in a **checkpoint** `OpenState` that D−1's build writes
  at D's replay start (`as_of` = D 00:00 local − 2 h). It never uses D−1's
  final state: D−1's window runs to D 08:00, so that state can hold
  passings later than D's follower. A carried-in leader must have
  `last_event.by ≤ as_of ≤` the follower's passing, or it is rejected.
  When no valid checkpoint exists, the headway has `leader_unknown = true`,
  null leader fields, and `headway.hi = null`. The midnight and DST
  crossings are covered by the headways-slice tests.
- **Scheduled trips never observed.** Scheduled trips fully inside each day
  (30-minute margins), against runs matched independently (before
  one-to-one, which moves some trips from observed to not observed):

  | Day | In window | Observed | Not observed, route running | Of which an unmatched run is nearby | Not observed, route absent |
  |---|---|---|---|---|---|
  | 09-15 | 7,640 | 93.5% | 488 (6.4%) | 101 | 6 |
  | 10-19 | 7,389 | 94.6% | 397 (5.4%) | 136 | 3 |

  By route on 09-15, the highest shares are B 25%, 7 12%, 2 11%, J 11% and
  W 10%. On 10-19 they are Z 17% (12 trips), 5 12%, N 12%, A 10% and 7 10%.
  "Not observed"
  is not "cancelled". 101 (09-15) and 136 (10-19) of these trips have an
  unmatched moving run of the same route and direction within 30 min
  [inference: a reroute or drifted id, not checked run by run]. The
  contract therefore keeps four outcomes apart:
  - `observed` (matched by a run);
  - `listed_not_run`: the trip appeared in the feed but never moved. Raw:
    any trip update. Legacy: B-division placeholder rows only, with
    numbered and L `Unknown`;
  - `not_listed`: never in the feed, while the route's feed was covered for
    the trip's scheduled span. A trip in an uncovered span is
    `not_covered`, never a cancellation.
- **Trip replacement periods.** Raw headers say the feed replaces the
  schedule until about +60 min (B division, SIR), +30 min (L) or "now"
  (numbered, whose end equals the header timestamp). In principle a
  scheduled trip missing from the feed inside that period is cancelled. In
  practice realtime ids drift.
  - In `step2`, matching on the exact id key (origin, route field,
    direction) found no trip update for 157 of 483 scheduled trips
    departing in the sample or the hour after it.
  - Matching the listed trips by the run tiers instead (their predicted
    times at their first stops) leaves 14 of 277 scheduled departures in
    the sample and the following 30 minutes unlisted (8 of them on the 6,
    which runs reroute codes). 184 of 277 were listed and not yet
    assigned.

  So `not_listed` is decided only after time-based matching, and only
  inside the replacement period. Measured, it is 5% of scheduled
  departures on a weekday morning, an upper bound on cancellations
  [inference].
- **Added trips** are the runs left unmatched after one-to-one assignment:
  435 of 9,053 (09-15) and 674 of 8,832 (10-19) with retained tails,
  stationary and stale-only runs included.
  They carry `outcome = unscheduled` with the nearest scheduled candidate.
  Section 2.3 explains why they are not all extra trains.

## 5. Coverage and supply

Downstream must tell "no data" from "no trains". These are two different
records:

- **Coverage** says whether we have fresh data from a feed. It is judged
  per feed from transport evidence only: the fetch, the payload and the
  header.
- **Supply** says how many trains that data shows, per feed and route,
  against the schedule.

A route, or a whole single-route feed (`g`, `l`, `si`), going to zero while
its header advances is **observed absence of trains**: supply evidence for
layer 5, never a coverage gap. Train counts never make an interval
uncovered.

An HTTP 200 does not mean data:

- On 2026-10-21 from 03:22 to 04:37 UTC (23:22–00:37 ET), `ace`, `bdfm`,
  `g`, `jz`, `nqrw` and `si` returned byte-identical payloads with HTTP 200
  for about 75 minutes. The header was stuck at 03:22:05–03:22:10 until
  04:37:17, and each feed served about 150 identical payloads in a row.
  `l` and `numbered` stayed live.
- Read as data, every B-division and SIR train stood still at its stop for
  75 minutes, and the unassigned trips aged into the past (§2.2).
- The legacy trace would have recorded the same rows every minute, with
  `fresh_feeds` saying success.

### 5.1 Coverage (transport evidence, per feed)

Coverage is keyed to fetch instants (`observed_at`), not wall-clock
minutes, so dispatch jitter is never a gap:

```text
covered(feed) = every stretch between two consecutive good fetches of the
                feed that are at most 120 s apart, with no failed fetch
                inside it
```

Any other stretch is uncovered. Its state is `fetch_failed` when a failed
fetch lies inside it (whatever its length), otherwise `no_invocation`.
What counts as a good fetch differs by lineage:

- **Raw.** `FeedFetch.outcome == "ok"`, the payload decoded, and
  `header_timestamp` advanced since the previous good fetch. A header that
  stands still for more than 120 s makes the interval `stale_header`, from
  the stuck header time onward. The 03:22 episode is a direct test case:
  `header_timestamp` and `payload_sha256` are both in `FeedFetch`.
- **Legacy.** `LegacyFetchOutcome(archive/trace, source).succeeded`. The
  trace keeps no header, so staleness can only be suspected, never proven.
  - Three or more consecutive trace objects whose **non-empty** per-feed
    row set (trip, stop, `vehicle_timestamp`, placeholders excluded) is
    identical give `suspect_stale`, a separate state.
  - Empty sets never count: three fresh empty polls are a feed with no
    trains, which is supply evidence (§5.2).
  - Identical non-empty sets can also be genuinely held trains, so
    `suspect_stale` is not authoritative. Layer 4 may exclude its bounds
    (`suspect_s` on each bound). Nothing treats it as a transport failure,
    and it never turns a suspension into missing data.
  - Legacy days are history only, so no live display depends on it.
  - Measured with every trace object that holds at least one vehicle row
    included, zero rows for a feed counted, and no row floor: no episode
    of 3 or more identical sets on any feed on 08-12, 09-15 or 10-19. The
    script enumerates polls from vehicle rows, so an invocation whose
    trace object holds no rows at all is not seen. Such fully empty
    invocations are untested; slice 3 covers them with a synthetic case.
  - Empty polls are rare: 1 `numbered` poll on 09-15 (the failed fetch)
    and 1 `g` poll on 10-19.
  - The same test on the `live` sample reduced to legacy fields finds the
    03:22 freeze on five feeds as one 75-poll episode each. On thin `si`
    it finds the freeze as five episodes of 5–31 polls, all inside the
    freeze window, because SIR placeholders age as the timestamps stand
    still.

Measured on the legacy days, keyed to polls:
- 09-15 has one 180 s gap at 11:04 ET (the 2 missing trace minutes, so
  `no_invocation`) and 1 failed `numbered` fetch (`fetch_failed`);
- 10-19 has one 99 s gap at 15:49 ET (a late invocation), which is covered;
- 08-12 has none.

These are exactly the minutes the earlier count-based rule flagged on the
unaltered days. They are transport events, now in their own states, and
no count-based flag remains.

Live latency: a failed fetch is known at once, and a stale header or a
missing invocation after 120 s, within the ≤ 2 min that §1 needs.

**Feed of a row.** Legacy rows carry no `source`. Layer 3 maps the trip
id's route field to a feed through a fixed table (`RouteFeed`, versioned
with the code):
- `ace`: A C E H;
- `bdfm`: B D F FX M FS;
- `g`: G;
- `jz`: J Z;
- `nqrw`: N Q R W;
- `l`: L;
- `numbered`: 1–7, 6X, 7X, GS;
- `si`: SI, SS.

Legacy StopEvents carry this as `source` with
`source_basis = "route_feed_table"`, so `uncovered_s` and `not_covered` can
be computed per feed.

### 5.2 Supply (observed trains against schedule)

Per good fetch and per (feed, route), layer 3 records real vehicles (after
§2.2) and scheduled trips in service at that instant. Scheduled trips come
from layer 1, from both service days that overlap the instant. Vehicles
are counted twice: `vehicles` (certain: fresh positions) and
`vehicles_stale_tail` (uncertain: runs in a stale tail, which may be a held
train or a ghost). An unidentified held train is never turned into a
definite absence. A vehicle counts toward its **matched** route: Z trains
carry `J` in the trip id, so route-level supply by trip-id field would show
the Z at 0 all day (122 of 122 Z-scheduled polls on both days). Unmatched
runs fall back to the trip-id field. Raw lineage adds the number of trips
listed in trip updates.

These records are evidence for layer 5, not coverage. The tests:

- **10-19 J/jz.** J ran at 0.40–0.70 of schedule for 91 polls
  (20:00–21:49 ET), and the `jz` feed never fell below 30% of schedule.
  Supply records it. Coverage stays `covered`.
- **Simulated single-route zero inside a live feed** (A in `ace`, J in
  `jz`, 6 in `numbered`, and the whole single-route `g`; 20 and 50 min, 3
  start times, 09-15 and 10-19): coverage stays `covered`, and the
  multi-feed flag below never fires (0% of polls).

**Multi-feed suspect outage (deferred, see Deferred).** One signal is kept for the rider-truth
audit's system-wide collapses (≥ 3 feed groups at once): a
`suspect_producer_outage` annotation. It is set when 3 or more feeds are
each below 30% of scheduled supply (where at least 5 trips are scheduled)
in the same poll, for 2 or more consecutive polls.
- On simulated zeros of 3 feeds (`ace`, `bdfm`, `nqrw`) and of all feeds,
  20 and 50 min, it fires on 100% of the polls.
- On the unaltered days it fires on 0 polls.
- It is an annotation on supply, not an uncovered state. Suppressing a
  gap as "no data" also needs transport evidence (§5.1), or an owner
  decision (§9, decision 10).
- It is live after 2 polls, within 2 min. It needs per-feed and per-route
  scheduled supply, which the nightly match index (slice 1) provides.

**Planned work and weekends.** The baseline is the static timetable.
- A planned suspension or reroute that the static feed does not encode
  therefore shows as low supply on the affected routes. That is correct as
  observation, and layer 5 explains it by joining planned-work windows.
- A rerouted train keeps its route id, so it stays in its route's feed.
- Weekend and holiday schedules come from the layer-1 calendar.
- Both measured days are weekdays (Tue 09-15, Mon 10-19). Weekends, nights
  with heavy planned work, and planned-work days are **unverified**. On
  such nights the multi-feed annotation could fire with no outage
  [inference], which is why it is not a coverage state.

Count-based coverage rules were tested and rejected:
- the ±10-minute median found 5% and 2% of the minutes of simulated 20- and
  50-minute whole-feed collapses;
- a scheduled-supply ratio plus a 5-minute zero rule found 100%, but it
  would also have marked possible real single-route suspensions (the G and
  L collapses in the rider-truth audit) as missing data.

Coverage is stored as intervals (`CoverageInterval`) with a state
(`covered`, `fetch_failed`, `no_invocation`, `stale_header`), a source and
a lineage. A layer-3 bound, headway or ScheduledTripOutcome records the
uncovered seconds it overlaps.

## 6. Draft contracts

The conventions are layer 1's: `Model` (frozen, closed, no defaults),
`Instant`, `ServiceDate`, `Unknown`, `Interval`, `Provenance`. Schema id
`events/s1`. Pydantic is the source of truth, under
`momentarily.model.events`. JSON Schema goes to `schema/model/s1/events/`
and TypeScript to `shared/model/s1/events/`.

```python
SCHEMA_ID = "events/s1"
Lineage = Literal["raw", "legacy"]
Family = Literal["last", "next"]  # §3.1, from the source feed (raw) or RouteFeed (legacy)
SourceBasis = Literal["published", "route_feed_table"]

class EventBound(Model):
    """Feed-supported bound: the event happened in (after, by], integer UTC
    seconds, exclusive below. after null = left-censored. after < by, strictly.
    reported, when kept, satisfies after < reported <= by (§3.2 widening)."""
    after: Instant | None
    by: Instant
    reported: Instant | None
    uncovered_s: int | None = Field(description="Seconds of (after, by] not covered; null when after is null.")
    suspect_s: int | None = Field(description="Seconds of (after, by] in legacy suspect_stale; null when after is null.")

class DurationBound(Model):
    """Interval-censored duration; consumers fit distributions, never midpoints."""
    lo: int = Field(ge=0, description="Clamped at 0.")
    hi: int | None = Field(description="null when censoring is right.")
    censoring: Literal["none", "right"]   # left-censored starts give right-censored durations (§3.2)
    reported: int | None
    uncovered_s: int | None

class GapInterval(Model):
    """Current elapsed gap at an evaluation instant; integer seconds, both ends inclusive.
    hi null only when the last passing is left-censored, or for scheduled_no_observation
    (lo = as_of − since)."""
    lo: int = Field(ge=0)
    hi: int | None

ServiceEvidence = Literal["stopped", "reported_not_stopped", "inferred_passage"]
PatternAttribution = Literal["scheduled_trip", "unique_fit", "ambiguous", "off_pattern"]

class TripMatch(Model):
    """One realtime run and what it is."""
    run_id: str = Field(description="f'{lineage}:{source}:{trip_id}@{first_observed_at}#{track}' (§2.3).")
    track: int = Field(ge=0)
    member_run_ids: list[str]                # runs joined into this one (split-run join)
    supersedes_run_ids: list[str]            # provisional live ids absorbed by this nightly run
    lineage: Lineage
    source: GtfsRtSource
    source_basis: SourceBasis
    realtime_trip_id: str
    route_published: str | None | Unknown    # legacy: Unknown
    start_date_published: str | None | Unknown
    nyct_train_ids: list[str] | Unknown
    first_observed_at: Instant
    last_observed_at: Instant
    n_observations: int
    n_placeholder_rows: int
    n_old_timestamp_rows: int
    stale_only: bool                         # every row has an old vehicle_timestamp
    outcome: Literal["matched", "unscheduled", "unparseable"]
    stationary: bool                         # seen at one stop only; no traversal
    flags: list[Literal["duplicate_trip_id", "start_date_mismatch", "joined_split_run",
                        "service_date_ambiguous", "deduplicated_by_position"]]
    tier: Literal["exact_id", "path_prefix", "line_code", "path_differs", "drift"] | None
    miss_class: str | None
    network_build_id: str
    scheduled_trip_id: str | None
    nearest_scheduled_trip_id: str | None    # for unscheduled: the trip it nearly matched
    compatible_pattern_ids: list[str]        # fitted from observations alone, sorted
    pattern_id: str | None                   # matched trip's pattern, or the unique fit
    pattern_attribution: PatternAttribution
    shares_scheduled_trip: bool              # another id-tier run holds the same trip
    identity_uncertain_with: list[str]       # runs with identical positions but a different train_id (§2.3)
    route_id: str | None                     # network route of the matched trip
    route_basis: Literal["scheduled_trip", "published", "trip_id_field"]
    service_date: ServiceDate
    service_date_basis: Literal["scheduled_trip", "id_origin", "clock_rule"]
    alternative_service_date: ServiceDate | None   # when service_date_ambiguous
    origin_wrapped: bool
    offset_s: int | None                     # observed − scheduled at the first common stop
    deviation: Literal["consistent", "skips_only", "extras_only", "skips_and_extras",
                       "off_pattern", "order_violation"] | None
    end_state: Literal["left_feed", "stale_tail", "window_end"]

class StopEvent(Model):
    run_id: str
    seq: int = Field(ge=0, description="Occurrence of this event within the run.")
    service_date: ServiceDate
    lineage: Lineage
    source: GtfsRtSource
    source_basis: SourceBasis
    family: Family
    stop_id: str                             # platform
    pattern_id: str | None
    pattern_index: int | None                # null when off pattern or ambiguous
    pattern_attribution: PatternAttribution
    service_evidence: ServiceEvidence
    arrival: EventBound | None               # stopped only
    departure: EventBound | None | Unknown   # stopped only; Unknown: censored tail / run end
    passage: EventBound | None               # reported_not_stopped and inferred_passage only
    departure_basis: Literal["vehicle", "next_arrival"] | None
    dwell: DurationBound | Unknown | None    # Unknown for the "last" family (§3.2)
    flags: list[Literal["order_violation", "reported_after_bound", "reported_widened_after",
                        "held_old_timestamp", "first_of_run", "liveness_uncertain"]]

class SegmentTraversal(Model):
    run_id: str
    from_seq: int                            # StopEvent.seq occurrences, not pattern indices
    to_seq: int
    service_date: ServiceDate
    lineage: Lineage
    source: GtfsRtSource
    pattern_id: str
    from_index: int
    to_index: int                            # > from_index; > from+1 only across unreported stops
    segment_ids: list[str]                   # layer-1 segments covered
    hidden_realtime_stops: int
    from_event: EventBound                   # departure ("next") or arrival ("last") at from
    to_arrival: EventBound
    run_time: DurationBound | Unknown        # departure(from) → arrival(to); Unknown for "last"
    stop_to_stop: DurationBound              # arrival(from) → arrival(to), old-traversal comparable
    run_time_basis: Literal["vehicle", "next_arrival"]

class Headway(Model):
    service_date: ServiceDate                # the follower's
    leader_service_date: ServiceDate | None  # null when leader_unknown (§4 carry-in)
    leader_unknown: bool
    lineage: Lineage
    stop_id: str
    route_id: str
    direction: Literal["N", "S"]
    scope: Literal["route", "pattern"]       # adjacency computed per scope, never filtered
    pattern_id: str | None                   # the pattern for scope "pattern"; null for "route"
    evidence: Literal["service", "physical"] # service: stopped events only (§4)
    leader_run_id: str | None
    follower_run_id: str
    leader_pattern_attribution: PatternAttribution | None
    follower_pattern_attribution: PatternAttribution
    leader_seq: int | None                   # StopEvent occurrence of the leader at this stop
    follower_seq: int                        # StopEvent occurrence of the follower (revisits differ)
    ambiguity_set: Literal["certain", "with_ambiguous"] | None   # pattern scope only (§4)
    leader_passing: EventBound | None
    follower_passing: EventBound
    leader_service_evidence: ServiceEvidence | None
    follower_service_evidence: ServiceEvidence
    basis: Literal["departure", "arrival", "passage"]
    headway: DurationBound
    spans_no_service: bool                   # scheduled closure inside the span (§4)

class OpenState(Model):
    """Right-censored state at one instant (live store, §7; also the
    window_end tail of a nightly file). Never a closed record."""
    as_of: Instant
    lineage: Lineage
    kind: Literal["open_run", "open_headway", "scheduled_no_observation"]
    run_id: str | None                       # open_run
    compatible_pattern_ids: list[str]        # open_run: live pattern fit (§2.3)
    last_event: EventBound | None            # null only for scheduled_no_observation
    since: Instant                           # start of the open interval (§4 initialization)
    stop_id: str
    route_id: str
    direction: Literal["N", "S"]
    scope: Literal["route", "pattern"] | None
    pattern_id: str | None
    evidence: Literal["service", "physical"] | None
    ambiguity_set: Literal["certain", "with_ambiguous"] | None   # pattern scope: one clock per series
    current_gap: GapInterval                 # finite at as_of: [as_of − by, as_of − after − 1], outage-adjusted lo (§4)
    eventual: DurationBound                  # the still-open headway: censoring = "right", hi = null
    flags: list[Literal["possibly_stale", "uncovered_since", "ambiguous_patterns_only"]]

class ScheduledTripOutcome(Model):
    service_date: ServiceDate
    lineage: Lineage
    scheduled_trip_id: str
    network_build_id: str
    route_id: str
    pattern_id: str
    scheduled_start: Instant
    scheduled_end: Instant
    outcome: Literal["observed", "listed_not_run", "not_listed", "not_covered"] | Unknown
    run_ids: list[str]
    uncovered_s: int

class CoverageInterval(Model):
    """Transport evidence only. Train counts never make an interval uncovered.
    suspect_stale (legacy) is uncertain evidence, not a transport failure."""
    source: GtfsRtSource
    lineage: Lineage
    interval: Interval
    state: Literal["covered", "fetch_failed", "no_invocation", "stale_header",
                   "suspect_stale"] | Unknown
    evidence_count: int                      # fetches (raw) or trace objects (legacy) inside

class SupplyObservation(Model):
    """Observed trains against schedule at one good fetch, per feed and route."""
    observed_at: Instant
    lineage: Lineage
    source: GtfsRtSource
    source_basis: SourceBasis
    route_id: str                            # network route: matched trip's route, else trip-id field
    route_basis: Literal["scheduled_trip", "trip_id_field"]
    vehicles: int                            # certain: real vehicles after §2.2, no placeholders, fresh positions
    vehicles_stale_tail: int                 # uncertain: runs in a stale_tail block (held train or ghost)
    scheduled_in_service: int                # layer-1 trips in service at observed_at
    listed_trips: int | Unknown              # raw: trip updates for the route; legacy: Unknown
    suspect_producer_outage: bool            # heuristic; interpretation deferred (§5.2)
```

Keys and rules:

- **Natural keys.**

  | Record | Key |
  |---|---|
  | TripMatch | `run_id` |
  | StopEvent | (`run_id`, `seq`) |
  | SegmentTraversal | (`run_id`, `from_seq`, `to_seq`) |
  | Headway | (`lineage`, `scope`, `evidence`, `ambiguity_set`, `stop_id`, `route_id`, `direction`, `pattern_id`, `follower_run_id`, `follower_seq`) |
  | OpenState | (`as_of`, `kind`, `run_id` or (`stop_id`, `route_id`, `direction`, `scope`, `pattern_id`, `evidence`, `ambiguity_set`)) |
  | ScheduledTripOutcome | (`lineage`, `service_date`, `scheduled_trip_id`) |
  | CoverageInterval | (`source`, `lineage`, `interval.start`) |

- **Lineage is carried end to end.** A record built from legacy rows says
  `legacy`, so a consumer filters on one field. A run is never built from
  both lineages. While both exist (§8), both are built and filed apart,
  under `events/s1/<lineage>/…`. Raw is preferred for any day whose raw
  coverage is complete.
- **Partitioning: by ServiceDate.**
  `events/s1/<lineage>/<record>/<service_date>.parquet`, and coverage by UTC day.
  Every consumer (normals by day type and time of day, supply against the
  timetable) thinks in service days. A UTC-hour partition would split
  every run that crosses 20:00 New York time.
  - Service day D needs observations from D 00:00 local minus 2 h to
    D + 1 08:00 local, because trips run to 28:xx GTFS time. It can be
    final once layer 2 has derived the UTC day holding D + 1 08:00 local.
    With the 05:00 UTC nightly derive, that is the second night after D,
    about 25 hours after D ends.
  - Unmatched runs are filed by their `id_origin` date (§2.4).
- **Network build per day.** Each service day uses the build of the feed
  that served it: `day_feed_map` `feed_id` when verified, else its
  inferred feed. The choice (`network_build_id`, `verified`) goes into
  provenance. The two current builds are content-identical apart from
  supplement trip ids.
- **Live updates** do not write these files (§7). The live state is a
  rolling store that is replaced by the nightly final.
- **Determinism.** Inputs are the layer-2 files of the window (named with
  digests in `Provenance.inputs`), the network build, and code.
  `built_at` = the latest `observed_at` used. The writer and overwrite rule
  are layer 2's. The tier-5 assignment is solved over inputs sorted by
  `run_id` and scheduled trip id, so a rebuild gives the same bytes.

## 7. Where it runs

| Option | Cost (measured or derived) | Pros | Cons |
|---|---|---|---|
| **A. Nightly Python batch** from layer-2 parquet | Prototype, one legacy day, one core: load 2.2 s, runs + matching 1.0 s, events 2.2 s, headways + absence + coverage about 3 s, so **under 10 s/day**. The min-cost tier-5 step with candidate search ran inside a 4 s whole-script run per day [the step alone was not timed]. Raw days add the trip-update read: 0.19 s for the fixture's 11 invocations, about 25 s a day [inference]. Output: StopEvents for 09-15 are 217,176 rows, 1.06 MB as zstd parquet without provenance | one implementation, rebuildable, matches layers 1–2 | final service day lands about 25 h after it ends; nothing live |
| **B. Live incremental in the Worker (TS)** | Per tick: about 660 vehicles, each updating its run's open block [inference: well under the 39–55 ms the tick already costs]. State: open runs (about 600) plus a per-service-day match index that Python builds nightly (subset of the 2.9 MB gz network) [inference on size] | ≤ 1 min latency. The Worker already computes movement (`vehicles.ts`, `movement_state.ts`) | a second implementation that needs parity tests. Tier 5 needs the timetable in the Worker |
| C. Python every minute in a Container | Container cold start and per-minute cost [not measured] | one language | goes against ADR 0001 (Worker for live, Container weekly); new always-on cost |

**Recommendation: A now, then B for a narrow live subset.**

- Python is the source of truth. It writes the final per-service-day files
  nightly, after the layer-2 derive.
- The Worker gets a small incremental layer 3 for what layer 5 needs
  live:
  - open runs with tier 1–4 matching against a nightly match index;
  - **live pattern fitting** for every run: `compatible_pattern_ids` from
    the pattern stop lists in the match index, carried as unique,
    ambiguous or off pattern (§2.3). Without this, the 31% of live runs
    with no id match would leave pattern gaps open during exactly the
    reroutes layer 5 must explain;
  - latest stop events with bounds and service evidence;
  - `OpenState` records at both scopes and both evidence levels,
    initialized from scheduled service (§4);
  - coverage from its own `FeedFetch` records.
- A parity fixture keeps the two in line. The `recorded_hour` replay is
  fed to both, and the TS output for the open window must equal Python's
  events for the same polls.
- Tier 5, one-to-one assignment and ScheduledTripOutcome stay nightly.
  Live runs with no id match are shown with their pattern fit as
  `unmatched (pending)`, not as added trips.

## 8. Backfill

| What | Legacy (trace, from 2026-08-12) | Raw (from the recorder's first day) |
|---|---|---|
| TripMatch | yes, about 92–95% matched one-to-one with retained tails (95.2% on 09-15, 92.4% on 10-19). No `start_date`, route, train id or assignment (`Unknown`). Placeholders by the legacy rule (98% recall on `step2`, 94% on fresh `live` polls) | yes, all fields |
| ServiceDate | from the matched trip; `id_origin` rule when unmatched, flagged `service_date_ambiguous` for runs first seen 00:00–04:00 (the rule is wrong for 11–12% of matched control runs there; unmatched error has no ground truth, §2.4) | same, cross-checked with `start_date` |
| Arrivals | bounds ≤ 120 s for 99%, ≤ 60 s for 89–98%, and a reported time | same |
| Departures, dwell, run time: numbered and L | 60 s / 120 s bounds, mostly with lower end 0 (§3.3) | same |
| Departures, dwell, run time: B division and SIR | **only via the next arrival**: 180–300 s wide, dwell `Unknown`, stop-to-stop only | same: trip updates are not a departure signal (§3.1) |
| Headways | yes (arrival-based for the B division) | yes |
| ScheduledTripOutcome | `observed`, and `listed_not_run` for the B division. Numbered and L `listed_not_run` and `not_listed` are `Unknown` (unassigned trips were never in their vehicle feed) | all outcomes |
| Coverage | per feed from trace objects; staleness only as `suspect_stale` (identical non-empty row sets; none found on the three days) | full, with header-based staleness |
| Cost | 68 legacy days × < 10 s ≈ 10 min | — |

Lost for good on legacy days: 6X/7X/FX for unmatched runs, NYCT
assignment (52 of 2,235 placeholders on `step2` and 399 of 6,911 on `live`
slip through), `train_id` (needed to rejoin split runs), trip-update
evidence of listed or cancelled numbered and L trips, and a direct
staleness signal. B-division dwell is not lost relative to raw: neither
lineage observes it.

## 9. Decisions

The owner accepted all ten recommendations on 2026-10-21. Each is marked
**decided**, with the owner's scoping changes recorded where they apply.

1. **Where the live part runs. Decided:** §7, A now. Nightly Python is the
   source of truth. The narrow TS incremental in the Worker (B) is
   deferred (see Deferred).
2. **Partitioning. Decided:** by ServiceDate, final about 25 h
   after the day ends. The alternative, UTC hour like layer 2, makes live
   appends easy but splits runs.
3. **Point or bound downstream. Decided:** layer 3 emits bounds plus
   the feed's reported time, and widens `after` when the feed reports
   earlier. For **event times** (arrivals, headway ends), layer 4 may use
   `reported` when it lies within the bound (97–99% of arrivals). For
   **durations** (run time, dwell), layer 4 must fit interval-censored
   distributions across events and never use per-event midpoints: 81–92%
   of numbered-line duration bounds start at 0 (§3.3).
4. **B-division dwell and run time. Decided.** These cannot be observed in either
   lineage: trip updates prune by predicted time, not on departure (§3.1).
   Recommendation: emit dwell and run time as `Unknown` for the "last"
   family, and give layer 4 stop-to-stop times only, which include dwell.
   Alternative: look for another departure signal, such as
   `vehicle_timestamp` while a train stands at a stop. That would need
   raw data and an independent check.
5. **One-to-one tier-5 assignment. Decided:** min-cost
   maximum-cardinality. The reason is narrow: it stops two runs from
   claiming one scheduled trip. It does not deduplicate physical trains,
   because a losing run stays a run. Rider-gap attribution comes from the
   observed pattern fit (§2.3, §4), not from the scheduled trip. So the
   decision mainly affects each run's service day and its lateness against
   the schedule. With retained tails (§2.2), the match rate falls from
   97.7% to 95.2% (09-15) and from 96.9% to 92.4% (10-19); greedy gives
   95.0% and 92.2%. A match rate is not accuracy (§2.3). Open: what the
   losers are, and an order constraint. The matching slice grades it with
   a withheld-input evaluation (§10).
6. **Thresholds. Decided:**
   - run gap 30 min (the result is insensitive between 10 min and 4 h);
   - old-timestamp row at more than 300 s of `vehicle_timestamp` age; a
     trailing block becomes a censored `stale_tail`, never deleted;
   - placeholder rule as in §2.2;
   - stale header at more than 120 s without advancing (raw); legacy
     `suspect_stale` at 3 identical consecutive non-empty row sets;
   - coverage gap at more than 120 s between good fetches.

   (`suspect_producer_outage` is deferred.)

   Recommendation: adopt them as named constants with a version, and
   revisit after 30 days of raw data.
7. **Unmatched service date. Decided:** `service_dates` by the id's
   origin (`service_date_basis = "id_origin"`). It is 99.2–99.3% right on
   matched runs used as controls, but 11–12% wrong for runs first seen 00:00–04:00, because
   of the 24:00 wrap. The clock rule is 82–84% wrong there and is kept only
   for unparseable ids.
8. **Trip replacement periods as cancellation evidence. Decided:**
   only to bound the window for `not_listed`, after time-based matching.
   Never cancel a trip from an id that is missing.
9. **Realtime-only stop ids. Decided:** add the alias table to
   layer 1 (holdup recommendation 1) before the stop-events slice. Until then, layer 3
   records them on the run and merges through them.
10. **Missing data versus no trains in the live product. Decided.** Layer 3 keeps
    them apart (§5):
    - "no current data" only for transport evidence: `stale_header`,
      `fetch_failed` or `no_invocation` on the line's feed, as in the
      03:22 episode;
    - zero or low supply with a fresh, advancing header is shown as what
      it is, no trains, and layer 5 classifies it (suspension, planned
      work, delay). It is never "no data";
    - the `suspect_producer_outage` annotation is deferred (see Deferred).
      Nothing suppresses a gap without transport evidence.

## 10. Step 3 build plan

Slices in order; each depends on the one before unless noted. Two gates
come first: Gate A before the contracts slice, and Gate B before the
headways slice. The identity (§2.3), censoring (§3.2) and open-gap
bootstrap (§4) rules were accepted with the decisions on 2026-10-21, and
Gate A is the last check before the schema is generated from them.

1. **Match index.** It does not wait for Gate A. A per-service-day index
   built from a network build:
   - stripped trip keys, origins with the 24:00 wrap;
   - per (pattern, stop) sorted scheduled times;
   - pattern stop lists for pattern fitting;
   - scheduled supply per route and minute (for SupplyObservation and
     `scheduled_no_observation`);
   - active services;
   - the `RouteFeed` table.

   The build is chosen per day from `day_feed_map`. Acceptance:
   - lookups agree with the prototype's `Net` on both builds;
   - the index is byte-identical on rebuild;
   - its size per day is reported.

#### As built (slice 1)

The index is `momentarily.model.events.match_index`. It is internal: it is
not a public contract and has no JSON Schema, TypeScript or parquet layout.

- **Command.** `python -m momentarily.model.events match-index --network
  <layer-1 dir> --date YYYY-MM-DD [--date ...] [--build-id ID] --out <dir>
  [--overwrite]` builds the index of each date from a local layer-1 output
  directory. It prints one JSON line per date (key, the built index's
  bytes, entry counts, and the outcome: `written`, `unchanged`, `replaced`
  or `conflict`). Exit status 1 means a build failed or a file conflicts.
- **Build choice.** Without `--build-id`, the build is the feed that
  `day_feed_map` gives for the UTC day with the same date: `feed_id` when
  verified, else `inferred_feed_id` (`build_basis` `verified`, `inferred`
  or `explicit`). A service day runs from about 04:00 UTC on its date to
  about 08:00 UTC the next day, so that UTC day holds most of it. A date the
  map does not cover fails unless a build is named. The snapshot is checked
  against the index's `content_digest` before use. A missing `index.json`,
  day map or build file is a build error (exit 1). Layer 1 now extends the
  map every night from recorder evidence (docs/model-network.md, "Keeping
  the map current"); a day it leaves without a feed (2026-10-19 and 10-20,
  before the recorder ran) still needs `--build-id`.
- **Key.** `model/events/s1/match_index/v<INDEX_VERSION>/<service_date>/<build_id>-<route_feed16>.json.gz`
  (`momentarily.model.events.keys`; `route_feed16` is the first 16 hex of
  the blake3 of `route_feed.json`). The index version, the build and the
  table are in the key, so a new version, a new table or a changed build
  choice gets a new key, not new bytes under the old one.
- **Stored-file rule** (the layer-2 derive job's): no file, write it;
  identical bytes, or bytes that differ only in the stored `code_sha` and
  `code_dirty`, keep it (`unchanged`); any other difference, or an
  unreadable file, is a `conflict`: the stored file is kept and the
  command exits 1, unless `--overwrite` replaces it (`replaced`).
- **Format.** `MatchIndexDocument` as canonical JSON (sorted keys,
  compact), gzipped with mtime 0. The same inputs and code give the same
  bytes; a second CLI run reports `unchanged`. `metadata` holds the key,
  `INDEX_VERSION`, the `route_feed.json` digest, service date, time zone,
  `day_origin` (`instant_of(date, 0)`), build, `build_basis`, day type,
  entry counts and layer 1's `Provenance` (`inputs`: the network
  `content_digest`, the `route_feed.json` digest and, when used, the day
  map's; `built_at`: the network's; `period_start`: the service date).
- **Contents.**
  - `trips`: every trip active that day, sorted by id, with the stripped
    key (static origin, route field, direction, path), block, route,
    pattern, service, first departure and last arrival.
  - `patterns`: every pattern of the build with its stop list, and for
    patterns that run, the trips by first arrival and an arrival matrix
    (per stop, per trip). Instants are `instant_of(service_date, s, tz)`.
  - `platforms`, `active_service_ids`, day type and exception flag.
  - `supply`: per route with 6X, 7X and FX folded into 6, 7 and F (riders
    see the 6; the patterns keep the variants apart), trips in service per
    epoch minute, from the minute of the first departure through the minute
    of the last arrival. This is the prototype's rule (`m13_supply.py`,
    `base_route`); `supply` folds the route it is asked for the same way.
  - `route_feed`: the `RouteFeed` table.
- **Lookups** (`MatchIndex`, built from a document in about 0.04 s for a
  weekday): `realtime_key` (a realtime origin `o` hits static origin `o`,
  then `o + 144000` flagged `wrapped`; a wrapped hit belongs to this
  service day, the calendar day before the observation), `static_key`,
  `origin_direction`, `arrival(trip, stop)`, `stop_times(pattern, stop)`
  (sorted instants and trip ids), `patterns_of(route, direction)` (6X, 7X
  and FX folded, as in the prototype), `pattern_stops`, `stop_position`,
  `platforms`, `supply`, `supply_by_feed` and `feed_of`.
  `candidate_service_dates(at, tz)` gives D−1, D and D+1. The index stores
  arrivals only; scheduled departures at intermediate stops are not in it.
- **`RouteFeed` source.** The table is measured, not assumed:
  `scripts/gen_route_feed.py` counts the distinct trip ids each feed
  published per route key in the layer-2 raw-lineage vehicle and
  trip-update records of 2026-10-21 03:57–13:30 UTC (545,889 records,
  Tuesday 10-20 23:57 EDT to Wednesday 09:30 EDT). A route key is a
  published `route_id` or a
  trip-id route field (`W` carries `N`, `Z` `J`, `6X` `6`, `SS` `SI`). The
  majority feed is the route's feed. The table agrees with §5.1 except
  that a feed can publish another feed's route: `ace` published 2 of 172 F
  trips and 1 of 140 R trips, overnight (02:00–05:00 New York time). Each
  entry keeps every feed's count, so a legacy
  row's feed is the route's usual feed, not a proven one. The build fails
  when an active route or trip-id route field has no entry. The window has
  no weekend; rerun the script on a longer window as days accumulate.
- **Agreement with `Net`.** Run on both full builds for 2026-07-03 (an
  added Saturday-service Friday), 08-26 and 08-27 (the last day of
  `2faf6350456df7f6` and the first of `606d9f640f5aad67` in the day map),
  09-07 (Labor Day, Sunday service), 09-08, 09-15 (Tue), 09-19 (Sat), 09-20
  (Sun), 10-19 and 10-31: all twelve lookup families (active services, id
  keys, wrapped keys, origin and direction, trip fields, arrivals, sorted
  stop times, patterns per route and direction, pattern stops, platforms,
  supply per route and minute, candidate days) are equal for every route,
  on both builds and all 10 dates. The committed test checks the same
  families by digest on a route subset (H, SI, Z, 7X, FX; 101 KB per
  build, gzipped).
- **Generator input.** `scripts/gen_route_feed.py` reads raw-lineage
  derived files only (`parquet.file_lineage`) and exits with a message on
  a legacy-backfilled day, whose rows have no `source`.
- **Wrap and DST.** Tested on a synthetic network on real 2026 DST
  nights, outside the builds' range (05-26 to 10-31): 25:30 on 10-31 and
  00:30 on 11-01 are both 05:30 UTC (the first 01:30), 26:30 on 10-31 is
  06:30 UTC (the second 01:30), 02:30 on 03-08 is 06:30 UTC (01:30 EST),
  and 26:30 on 03-07 is 07:30 UTC (03:30 EDT). A realtime `009000` seen
  at 01:31 on 03-08 hits the 25:30 trip of 03-07, 60 s off, and the same
  trip of 03-08 more than 12 h off.
- **Size per day** (bytes gzipped / uncompressed, trips, stop times):

  | Day | `2faf6350456df7f6` | `606d9f640f5aad67` |
  |---|---|---|
  | Weekday (09-15) | 637,569 / 5,176,289; 8,497; 238,724 | 637,590 / 5,179,827; 8,497; 238,724 |
  | Saturday (09-19) | 406,261 / 3,794,851; 6,186; 169,429 | 406,250 / 3,796,271; 6,186; 169,429 |
  | Sunday (09-20) | 376,730 / 3,443,280; 5,626; 154,778 | 376,735 / 3,444,468; 5,626; 154,778 |
  | Labor Day (09-07) | 378,259 / 3,444,217; 5,626; 155,037 | 378,274 / 3,445,697; 5,626; 155,037 |
  | Added Saturday (07-03) | 408,442 / 3,795,871; 6,186; 169,772 | 408,432 / 3,797,623; 6,186; 169,772 |

  Every index also holds 218 patterns, 992 platforms and 30 `RouteFeed`
  keys; supply covers 37,406 route-minutes on a weekday. The builds differ
  only in the block prefix of 3,385 trip ids (`X` added), so their indexes
  hold the same schedule under partly different ids.

**Gate A: recorded raw days.** Run the prototype on recorded raw data
from `<mirror>/model/observations/s1/` (ticks, `raw/gtfsrt` bundles,
`raw/blob`), derived through the layer-2 derive job:
- one full weekday, 2026-10-22, the first complete recorder day;
- one weekend day, 2026-10-24 or 2026-10-25;
- a planned-work day, if one falls in range.

Report per day:
- match rate (independent and min-cost, retained tails);
- tier mix and unmatched classes;
- bound widths per family;
- coverage states, with every uncovered stretch and its cause;
- service-day results, including `start_date` agreement and
  `service_date_ambiguous` counts.

The contracts are revised only where a result breaks them. Otherwise the
contracts slice proceeds unchanged.

2. **Contracts and generation** (`momentarily.model.events`, schema export,
   TS generation, fixtures, parquet layout pinned by version; deferred
   records and fields excluded, see Deferred). Acceptance:
   - layer-1/2-style tests fail on a stale schema, stale TS or a stray file;
   - ajv validates every fixture;
   - a contract with a default is rejected;
   - `EventBound` rejects `after >= by`, accepts `after = null`, and
     rejects a `reported` outside `(after, by]`;
   - `DurationBound` rejects `hi = null` with `censoring = "none"` and a
     negative `lo`;
   - `TripMatch` rejects `service_date_basis = "scheduled_trip"` without a
     scheduled trip;
   - a "last"-family `StopEvent` with a numeric dwell is rejected;
   - a `Headway` with `evidence = "service"` and a non-`stopped` end is
     rejected.
3. **Coverage** (CoverageInterval for both lineages, keyed to fetch
   instants; stale header; legacy `suspect_stale`). It needs no matching.
   Acceptance:
   - a scripted fixture with a failed fetch, a missing tick and a stale
     header gives the expected CoverageIntervals;
   - a 99 s gap between invocations stays `covered`;
   - a short excerpt of the 2026-10-21 03:22 UTC frozen-feed episode (a few
     polls before and during, committed as a recorder-format fixture)
     gives `stale_header` for the six frozen feeds and `covered` for `l`
     and `numbered`;
   - **legacy replay cases**:
     - three or more fresh **empty** trace objects for a feed stay
       `covered`;
     - a fully empty trace object (no rows on any feed, which the
       measurement script could not see) is a good fetch for every feed
       that `fresh_feeds` lists, and stays `covered`;
     - a thin feed (1–3 rows) whose rows change stays `covered`;
     - a stationary but fresh feed whose rows are identical gives
       `suspect_stale`, never `stale_header` or `fetch_failed`;
   - on 09-15, 10-19 and 08-12 the only uncovered stretches are the
     transport events in §5.1, and there is no `suspect_stale`.
4. **Runs, pattern fitting, trip matching and supply** (placeholders,
   censored stale tails, duplicate-trip-id split with tracks, split-run
   join, `compatible_pattern_ids`, tiers 1–5, min-cost one-to-one tier 5,
   service day, miss classes, SupplyObservation). Supply needs matched
   routes, so it comes here, not with coverage. Acceptance:
   - on 2026-09-15 and 2026-10-19 the min-cost match rate is within
     0.5 pt of 95.2% and 92.4%. These were measured with retained tails: a
     trailing old-timestamp block's first row (the arrival) kept, later
     old rows dropped, stale-only runs and stationary runs included;
   - no scheduled trip is held by two tier-5 runs;
   - **withheld-input evaluation**. Labels must come from evidence the
     matcher does not use. `train_id` is used for splitting and joining,
     so it is not independent, and a physical train id alone does not
     prove a scheduled-trip assignment.
     - On the Gate A raw days, the matcher runs with the trip id's origin
       and path, `start_time`, `start_date` and `train_id` **withheld**:
       the legacy-equivalent inputs. It keeps the route field and the
       direction.
     - Its scheduled-trip assignment and service day are graded against
       the trip each run's full raw descriptor identifies through tiers
       1–4 on origin, path and `start_date`.
     - Results are reported per conflict class (fragmentation, extra
       train, sustained lateness, overtaking, equal costs, competing
       service dates). The owner sets the threshold after seeing them;
   - **owner spot-check**: the owner reviews 10–15 hard cases chosen from
     the disagreements and from contested components, with their raw
     evidence;
   - **prefix replay**: replaying every prefix of the `recorded_hour`
     fixture (and of one legacy hour), with a restart mid-hour and with
     duplicate invocations delivered out of order, gives the same runs,
     tracks and censored tails as the full replay up to each prefix end;
   - **identity**: two trains sharing a trip id and first poll get
     distinct `run_id`s; a run that revisits a stop keeps distinct
     `(run_id, seq)` keys; a joined run lists its members;
   - **identity tie**: two synthetic trains with one trip id at the same
     stop in the same poll that then diverge get the same pair of
     `run_id`s under every shuffled entity order of every payload;
   - **identity merge**: two tracks with identical sampled positions and
     equal `train_id`s merge; with different `train_id`s they stay two
     runs, each marked `identity_uncertain_with` the other;
   - **pattern fitting**: synthetic trains that (a) serve one pattern,
     (b) deviate from all patterns, (c) fit two patterns ambiguously get
     `unique_fit`, `off_pattern` and `ambiguous` with the right
     `compatible_pattern_ids`;
   - **fresh-feed suspension**: a synthetic feed whose headers advance
     while one route (and, separately, a whole single-route feed such as
     `g`) has zero vehicles for 50 minutes stays `covered`, and its
     SupplyObservations show 0 against the schedule;
   - Z vehicles count toward route Z (matched route), not J;
   - synthetic cases:
     - a wrapped id (`000100` at 00:05) gets the previous service date;
     - a raw trip scheduled at 24:10 with `start_date` = service date + 1
       is not flagged;
     - a placeholder-only trip gives no run;
     - a trailing old-timestamp block keeps its arrival and gives
       `end_state = stale_tail` with a censored departure;
     - a run cut by the input window gives `window_end`;
     - an unmatched run seen at 00:30 gets its `id_origin` date with
       `service_date_ambiguous`;
     - **DST**: trips on 2026-11-01 and 2026-03-08 service days, including
       one past 24:00, get the instants that `instant_of` defines, and are
       matched on the right day;
   - a rebuild is byte-identical.
5. **Stop events, traversals and dwell** (per-family rules, service
   evidence, left-censored first stops, `uncovered_s` from slice 3).
   Acceptance:
   - invariants: `after < by`, a reported time inside or flagged, dwell
     `lo <= hi`, "last"-family dwell `Unknown`;
   - **synthetic endpoint tests**:
     - a reported time equal to `after` widens to `reported − 1`;
     - a reported time after `by` is dropped and flagged;
     - a negative duration lower bound clamps to 0;
     - a left-censored start gives a duration with `censoring = "right"`,
       `hi = null` and `lo = max(0, end.after − start.by)`, and null
       `uncovered_s`;
     - a left-censored arrival with a reported time keeps it without
       widening;
     - an unreported stop gives `inferred_passage`, a not-stopped
       "next"-family stop gives `reported_not_stopped`, and neither has an
       arrival;
   - **replay test**: the `recorded_hour` fixture through derive and
     layer 3 gives events whose canonical bytes match a pinned digest;
   - **holdout** on 08-12, 09-15 and every other overlapping day up to
     10-18: the per-day gate of §3.5 (arrival containment ≥ 90%,
     stop-to-stop containment ≥ 99%, no-pair ≤ 8%). Failures are listed by
     class. On 09-15 and 08-12 the results are at least the prototype's
     (94.6% and 90.9% arrivals; 99.8% and 99.7% durations). This gate
     mostly checks pairing, not truth (§3.5); the synthetic tests above
     carry the semantics.

**Gate B: can "last"-family feeds separate service from passage?** The
B-division and SIR feeds report STOPPED_AT on nearly every poll
(`docs/model-observations.md` §1.2), so "seen stopped" may not tell a stop
from a pass-through there. On the Gate A raw days, measure for `ace`,
`bdfm`, `g`, `jz`, `nqrw` and `si`:
- whether vehicles are ever reported at a stop that their trip update
  and matched pattern skip (an express passing a local station, a
  skip-stop reroute);
- whether any reported stop gets a STOPPED_AT status different from a
  stop the train serves.

If service evidence does not differ from physical passage on these
feeds, their events carry `service_evidence = "unknown"`. Rider-service
gaps there are then marked `service_not_separable`, never shown as a
service difference. The `stopped` value is kept only where Gate B shows
a difference. The contract gains `"unknown"` in `ServiceEvidence` and
the flag on Headway and OpenState.

6. **Headways and OpenState** (both scopes, both evidence levels, both
   ambiguity series, bootstrap from scheduled supply). Acceptance:
   - headways never cross a coverage gap without `uncovered_s > 0`;
   - a scripted P–Q–P passing sequence at one stop gives one route-scope
     pair P→Q, one route-scope pair Q→P, and one pattern-scope pair P→P
     spanning the Q passing;
   - a train that only passes (`inferred_passage` or
     `reported_not_stopped`) between two stopping trains does not split
     the service headway;
   - on Gate B's feeds, if it found no difference, service headways carry
     `service_not_separable`;
   - an ambiguous run counts toward each compatible pattern in the
     `with_ambiguous` series and not in `certain`. A P–A–P sequence (A
     ambiguous) gives P→A and A→P pairs in `with_ambiguous` and one P→P
     pair in `certain`. `OpenState` keeps both clocks;
   - a follower that revisits a stop gives two headways with distinct
     `follower_seq` keys;
   - across a 10-minute outage that ended at `e` after the last passing,
     `current_gap` is `[t − max(by, e), t − after − 1]`;
   - **outage overlapping the bound**: last passing in `(100, 160]`, an
     outage ending at `e = 120`, `t = 200` gives `lo = 40` (not 80);
   - **long cross-day absence**: a 3-hour suspension spanning midnight
     takes its leader from D−1's checkpoint `OpenState`, and with that
     missing gives `leader_unknown = true`;
   - **future-dated carry-in**: a checkpoint whose last passing is later
     than D's replay start, or than the follower, is rejected;
   - **endpoint propagation**: at `t = 200` with the last passing in
     `(100, 160]` and no outage, `current_gap` is `[40, 99]` and
     `eventual` is right-censored with `lo = 40`;
   - **an entirely absent pattern** produces `scheduled_no_observation`
     from its first scheduled passing;
   - **first service after an overnight closure** is not paired with the
     previous night's last train without `spans_no_service`;
   - **midnight and DST**: a headway whose leader is on D−1 is filed under
     D with `leader_service_date = D−1`, also across 2026-11-01;
   - `OpenState` at the end of the replay fixture equals the last passings.
7. **Nightly wiring and legacy backfill** (after the layer-2 derive, per
   service day, layer 2's report and overwrite rule; writes the checkpoint
   `OpenState` at the next day's replay start). Acceptance: all trace days
   back to 2026-08-12 build with exit status 0; two runs are
   byte-identical; the per-day report lists match rate, unmatched classes,
   coverage and the §3.5 gate.

## Deferred

The owner deferred these on 2026-10-21. The design text in §§4–7 stays as
a record. The pieces are out of the build plan, the acceptance tests and
the generated contracts until they are taken up again.

- **ScheduledTripOutcome cancellation classification** (`observed`,
  `listed_not_run`, `not_listed`, `not_covered`; §4, §6) and the use of
  trip replacement periods for it (decision 8). `scheduled_no_observation`
  stays: it initializes open gaps from the match index's scheduled
  supply, not from ScheduledTripOutcome.
- **The live Worker slice** (§7 option B: incremental layer 3, live
  pattern fitting in TS, the parity fixture). The live-oriented rules stay
  in the design so that prefix replay keeps batch equal to a future live
  build.
- **The multi-feed `suspect_producer_outage` annotation** (§5.2) and its
  SupplyObservation field. Coverage stays transport-only. Count drops stay
  plain supply observations.

## What was not verified

- The cause and frequency of upstream frozen feeds. One episode was seen
  (about 75 min, six feeds). Whether it recurs nightly, and whether the trace
  days hid such episodes, is not known beyond the three days checked.
- Overnight behaviour beyond 22:38–00:55 ET of one Tuesday night, weekend
  behaviour in raw lineage, and the share of `6X`/`7X`/`SS` trips in raw
  data.
- Physical event times. All bounds are knowledge brackets on what the feed
  said. The feed itself lags by seconds to a minute (§3.3), and nothing
  here was checked against the trains.
- Any B-division departure signal. The trip-update drop was shown not to
  be one (§3.1). Whether `vehicle_timestamp` while a train stands at a stop
  carries departure information is untested.
- Whether the unassigned B-division rows with a STOPPED_AT status (52 in
  `step2`, 399 in `live`) are trains.
- What the tier-5 runs that lose the assignment are, how accurate the
  assignment is (the hand-labelled conflict set is planned, not done), the
  split-run join rule (proposed, not measured), and whether an order
  constraint changes the assignment.
- How fast live pattern ambiguity resolves as a run accumulates stops.
  The 137 ambiguous live runs were counted over each run's whole observed
  life, not at each instant.
- The service and physical headway series and the pattern scope were
  specified, not measured. Whether the "last" family's reported stop is
  always a stop with doors open (and never an express passing a local
  stop) is not checked.
- Supply and coverage on weekends, holidays and planned-work days. Both
  measured days are weekdays. The rider-truth audit's single-route G and L
  collapses could not be replayed: only three trace days are mirrored
  locally, and the collapse dates were not checked against them. Whether
  real producer outages always freeze the header (so that transport
  evidence catches them) is shown by one episode only.
- The per-day backfill gate thresholds rest on two measured days (08-12,
  partial, and 09-15).
- A full raw day. Raw figures come from 10 minutes (`step2`) and about
  2 hours (`live`) of polls, not from the recorder.
- Worker CPU and memory for option B, the size of a per-day match index,
  and Container costs for option C.
- Cancellations: no ground truth separates a cancelled trip from an
  unmatched one.
