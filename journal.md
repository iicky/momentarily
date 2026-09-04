# Engineering Journal

Append-only. When you discover something non-obvious that cost real effort to
find — a footgun, a wrong assumption, a number nobody would have guessed —
add an entry at the end. Never edit or delete an existing entry, even after
it's superseded; if a later finding changes the picture, add a new entry and
say which older one it revises.

## Format

Each entry is a `##` heading with a date and a short slug, an `origin:` line,
then the observation body:

    ## YYYY-MM-DD — short-slug
    origin: self | agent | artifact

    What was tried, what was found, and the number that proves it.

**`origin` values:**

- `self` — a human found this directly (debugging, reading code, reviewing output)
- `agent` — an AI agent found this while doing unrelated work
- `artifact` — surfaced by a build, test run, log, or generated report, not by a person or agent reasoning about it

## The rule

**Negative results always get recorded, and they MUST carry the number that
made them negative.** "Tried X, regressed" is not an acceptable entry — there
is nothing in it to check, reproduce, or compare against later. "Tried X,
held-out log-likelihood dropped from -812.4 to -1043.9" is acceptable. If you
can't state the number, the investigation isn't finished — go get the number,
or write down exactly which number you'd need and why it's not available yet.

---

## 2026-08-09 — worked-example: em-prior-strength-floor

origin: self

`training/train_em.py` fits the per-route HMM transition matrix with a
Dirichlet prior anchored at `prior_strength` pseudo-counts (default 100.0).
Tried dropping it to 20 to let thin-data routes move further from the global
prior: held-out log-likelihood on the March incident set fell from -812.4 to
-1043.9 over the same 20 EM iterations, and two routes (G, FS) started
oscillating instead of converging monotonically. Reverted to 100.0. Below
~40 the Dirichlet counts stop dominating the per-route transition counts for
anything but the highest-volume routes, so thin routes overfit their own
sparse incident history instead of leaning on the pooled prior. If this needs
revisiting, the number to chase is the per-route incident count above which
`prior_strength=20` stops hurting — untested here.

## 2026-08-11 — pooled-normal-dwell coverage is transition-gated

origin: agent

The pooled log-logistic AFT for normal-regime dwell was built to lift the
elapsed-conditioning fix off the four routes that cleared
`MIN_SAMPLES_FOR_EMPIRICAL`. It does, but not as far as the design implies: a
`train_em --dry-run` on the current 14-day window fits normal cells for 11 of
28 routes. `pooled_dwell_cells` reads `dwell_samples_by_cell`, which derives a
still-open regime from a route's _last transition record_, so the 17 routes
with zero transitions in the window supply neither an event nor a censored
observation and never reach the pooling step — the steadiest routes, which are
the ones the pooling exists to serve, are still the ones excluded. Those routes
carry 62% of normal route-ticks.

Counterfactual over 50,866 predictions (08-05..08-11), pooled curves vs the
live geometric projection, graded against the model's own condition at t+H:

| horizon | covered subset (n=19,217)       | fleet (n=50,663)          |
| ------- | ------------------------------- | ------------------------- |
| 30 min  | Brier 0.01734 -> 0.00326 (-81%) | 0.01714 -> 0.01180 (-31%) |
| 60 min  | 0.03746 -> 0.00392 (-90%)       | 0.03895 -> 0.02623 (-33%) |
| 120 min | 0.06052 -> 0.00416 (-93%)       | 0.06380 -> 0.04242 (-34%) |

On the covered subset mean predicted `p_normal_in_120` moves 0.7582 -> 0.9870
against an actual base rate of 0.9968, so the sharpness bias really is closed
where a cell exists. The number to chase is the fleet column: it only moves if
open regimes are reconstructed from prediction rows rather than from transition
records.

## 2026-08-11 — open regimes from prediction rows: 11 -> 29 normal cells

origin: agent

Follows the entry above, which named the number to chase. `_open_regimes` read
each route's still-open regime off its _last transition record_, so a route with
no transition in the window supplied neither an event nor a censored
observation. Sourcing open regimes from the prediction stream instead — every
live route is written every tick, and the row already carries
`regime_entered_at` — closes it.

Counterfactual on the same 14-day window (104 transitions, 122,438 predictions):
pooled normal-dwell cells go **11 -> 29**, gaining 2, 4, 6X, 7, A, C, E, F, FX,
G, GS, H, J, M, Q, R, SI, W. The 18 gained routes had held one regime for a
median ~450h, which is exactly why they never appeared in the transition stream.

The live blast radius is larger than the 11 suggests. Published params
(`trained_at=1785774438`) carry a `normal` cell for **4 of 28** routes — 6, B,
FS, H — so the other 24 run the geometric fallback. That is the source of the
overconfidence visible on the site: the 1 train, normal for 88h, publishes
`p_normal_in_30min=0.8814` against an empirical base rate of 0.999. With a
fitted cell the same route reads 0.9987, and the 18 newly covered routes average
0.9997.

Two footguns found while wiring it. Predictions load by whole-day prefix, so a
backdated `--end` reads rows past the censoring boundary and reports the regime
open _now_ — `open_regimes_from_predictions` takes `window_end` and ignores
later rows. And `not_scheduled` must be dropped: it is not a regime the filter
dwells in, so banking scheduled downtime as a long healthy run would inflate the
very curve this corrects.

## 2026-08-11 — the forecast is anti-correlated with the arm consumers read

origin: agent

`training/eval.py` graded everything against `PredictionRecord.condition`, which
is the **alert-shadow**, not `published_condition`, the movement-primary arm the
site serves. `published_condition` was parsed into the record and never read.
`scorecard.py` had already migrated (7a15b1b); eval never did, so every headline
number described a model no consumer reads.

Grading the same `p_normal_in_H` forecast against each arm over 7 days:

| arm                                   | n      | Brier   | BSS vs persistence | AUC       |
| ------------------------------------- | ------ | ------- | ------------------ | --------- |
| shadow (`condition`), h30             | 55,390 | 0.01709 | -7.93              | **0.780** |
| movement (`published_condition`), h30 | 41,132 | 0.01605 | -16.37             | **0.145** |
| movement, h60                         | 40,807 | 0.03640 | -40.27             | 0.144     |
| movement, h120                        | 40,081 | 0.06059 | -55.48             | 0.142     |

AUC 0.142-0.145 is not weak, it is inverted: random is 0.5, and inverting the
forecast would score ~0.855. The mechanism is not subtle. Of the 29
movement-disrupted ticks in the window, **29/29 have shadow `condition ==
"normal"`** — `effectiveCondition` (worker/src/snapshot.ts) hard-returns `normal`
whenever `disruptiveAlertCount === 0`, so movement-disrupted-while-alerts-silent
is by construction the case the filter is most confident about. Mean
`p_normal_in_30min` is **0.9166 on movement-disrupted ticks vs 0.8780 on
movement-normal ticks**: the forecast is _more_ sure of health exactly where
movement says the line is stalled.

Thin — 29 disrupted ticks (6:12, R:7, G:3, GS:2, J:2, M:1, W:1, 5:1) — so treat
the magnitude as provisional. The sign is not provisional; it has a mechanism.

Also note truth-map semantics. The movement truth must be dense over gradeable
ticks with `truth_default=None`, not sparse-with-absent-means-normal: 16.6% of
ticks publish `unknown` and 8.1% `not_scheduled`, and scoring those as calm
would credit the forecast for 24.7% of the window it was never tested on.
Gradeable share is 75.3%.

## 2026-08-11 — the movement arm was blind for three weeks and nothing noticed

origin: agent

Census of `published_condition` over its whole life (shipped 2026-07-11 in
15e4b65, 262,459 prediction rows to 2026-08-11, 97.8% carrying the field):

| period                   | unknown share | gradeable share |
| ------------------------ | ------------- | --------------- |
| 2026-07-11               | 3.7%          | 92.2%           |
| 2026-07-12 .. 2026-08-02 | ~100%         | ~0%             |
| 2026-08-03               | 64.1%         | 34.9%           |
| 2026-08-04 .. 2026-08-11 | 15-17%        | 68-78%          |

For 22 consecutive days the site published `unknown` for essentially every
route on every tick and no check fired. Coverage returns exactly at the
`trained_at=1785774438` retrain (2026-08-03T16:27Z), so the cause is an empty
movement advance-baseline in the published params — the same condition
`train_em` now warns about ("movement-primary condition will publish 'unknown'
for every route"). The warning exists; nothing downstream treats a fleet-wide
`unknown` as an outage.

Consequence for planning: the usable movement history is **8 days**
(2026-08-04 onward), not the 31 days the ship date implies. Any claim fitted on
"a month of movement data" is fitted on eight days.

Route-level movement episode census over the full window (73 episodes, all
closed, 17 routes): median duration **10 min**, min 5 min, max 155 min; 9 routes
clear MIN_VOTER_EVENTS=3 (3:19, GS:8, 4:7, 6:6, G:6, R:6, 1:5, D:3, J:3).

That median is the headline. The alert-shadow's episodes run to hundreds of
hours (one F "Severe Delays" held 459h, recorded in episodes.py). Movement
disruptions are a different phenomenon an order of magnitude shorter, so the
32.9-min recovery MAE the shadow reports is not a baseline the movement arm
should be compared against — most movement stalls are over in two ticks.

## 2026-08-11 — alert corroboration of movement episodes: 15.4% confirmed, confirmed slice sits at lead=0

origin: agent

`training/escalation.py`'s old tick-based `alert_confirmed_rate`/`evaporated_rate`
couldn't validate against real archived data (the escalation cohort filter and
the published-archive/offline-recompute source lock-in left it untested).
Rebuilt on `training/episodes.py` episodes (`confirmation_rate` + `lead_time`,
with `n_unconfirmed` reported and never turned into a rate) and ran it via
`murk exec` against real R2 archives, 2026-08-07..2026-08-11 (published_archive
source — the movement arm is live now, see the entry above): 40,455
predictions, 259 movement-disrupted episodes, canonical severity_floor=2 alert
truth as the corroborating set, ±30/60 min window around each episode's onset.

`confirmation_rate = 40/259 = 0.1544`; `n_unconfirmed = 219` (84.6% — not
disconfirmed, per the metric's own definition). Of the 40 confirmed, median
lead time is **0.0 min**, IQR `[0.0, 0.0]`: confirmation is almost entirely a
same-tick alert already active, not a later alert catching up to a movement
call that led it. Combined, the data does not show a "movement leads, alerts
confirm later" pattern in this window — it shows a small minority of
movement-disrupted episodes overlapping an alert that was already posted, and
the rest (84.6%) with no alert record either way, consistent with the 29/29
alert-silent finding two entries above. Number to chase next: split the 219
unconfirmed episodes by duration — short blips (movement-classifier noise) vs.
long stalls (a real alert-feed gap) look identical in this rollup.

## 2026-08-11 — static GTFS topology fixes segment-graph fragmentation: 44/52 chains vs 8/53 observed

origin: agent

`training/segments.py`'s `canonical_adjacency` (observed cross-tick modal
successor, gated at `MIN_SHARE>=0.5` in the live `train_em.write_segment_params`)
fragments the line graph. Measured directly against a real 14-day R2
vehicle-archive window (2026-07-28..2026-08-11, `canonical_adjacency` +
connected-components over each (route,direction)'s edge set): 53
(route,direction) groups have any observed adjacency; only **8/53** reduce to
one connected component under the `MIN_SHARE>=0.5` gate live today (33/53
without that gate, using only `canonical_adjacency`'s own `min_advances=3`
existence floor — so the reliability-share gate alone drops 25 previously-
connected groups to fragmented). This is the same failure mode previously reported
("2/53, Q north 45 links/14 components") but a different archive window, not
independently reproduced at that exact figure.

New `training/gtfs_static.py` parses the real NYCT static feed
(`https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip`, verified against
`http://web.mta.info/developers/data/nyct/subway/google_transit.zip` — a 301 to
the identical object, same ETag/Content-Length) directly: 52 (route,direction)
groups, 1797 (route,direction,from_stop) keys, **131** from_stops with more than
one static successor (real branch/express splits — Rockaway Park Shuttle +
Lefferts Blvd on the A, Dyre Ave on the 2/5 — confirmed by inspecting the
disjoint components, not archive noise). **44/52** groups reduce to one
connected component on the dominant-successor graph; the remaining 8 are
genuine branch tails that never reconverge (A south: 3 components; 5 north, 2
south, E north, N north, W south, R north, R south: 2 components each).

Combining both real datasets: of the 1795 (route,direction,from_stop) segments
with both a static successor and an observed cross-tick baseline this window,
**276 (15.4%)** were excluded from `segment_params.json` by the live
`MIN_SHARE>=0.5` gate despite the static timetable naming a real successor for
them — the number `MIN_SHARE`-as-existence-gate was actually costing.

## 2026-08-11 — movement regime debounce: debounce=2 erases 76% of route episodes

origin: agent

`training/movement_backfill.py` reconstructs the movement-regime transition
stream offline through `training.regime.replay_regimes`, from two independent
sources — `published_condition` (route scope only) and `archive/vehicles`
(route + segment scope, the only source that reaches segment granularity) —
and swept `debounce_ticks` in {1, 2, 3} against real R2 archives over the
usable window (2026-08-04..2026-08-11, via `murk exec`).

Route scope, from `published_condition` (this archived window predates the
regime clock landing this session, so it is the raw per-tick classifier call,
not a debounced regime):

| debounce_ticks | n episodes | median (min) | min   | max   | routes ≥3 eps | survival vs 1 |
| -------------- | ---------- | ------------ | ----- | ----- | ------------- | ------------- |
| 1              | 17         | 5.0          | 4.95  | 34.92 | 3 of 6        | 100%          |
| 2              | 4          | 10.125       | 10.0  | 34.92 | 0 of 2        | 23.5%         |
| 3              | 1          | 49.95        | 49.95 | 49.95 | 0 of 1        | 5.9%          |

`debounce_ticks=2` (the shipped default, tuned for `derive_actual_recovery`'s
alert-count service metric, not movement) erases 13 of 17 episodes and every
episode under 10 minutes. The raw population's own median is 5.0 min — one
tick — so debounce=2 erases the median outright, not just a short tail;
debounce=3 leaves a single episode.

Segment scope, from `archive/vehicles` (the only source that reaches segment
keys; per-tick calls are already smoothed by a 6-tick/30-min trailing sum
before the regime clock runs at all — real archive median matched-trips per
segment leaf per tick is 1, ten times under `segments.py`'s
`MIN_MATCHED_TRIPS=3` floor):

| debounce_ticks | n episodes | median (min) | min  | max  | cells ≥3 eps | survival vs 1 |
| -------------- | ---------- | ------------ | ---- | ---- | ------------ | ------------- |
| 1              | 828        | 10.0         | 5.0  | 435  | 35 of 79     | 100%          |
| 2              | 411        | 30.0         | 10.0 | 765  | 26 of 59     | 49.6%         |
| 3              | 216        | 55.0         | 15.0 | 1090 | 25 of 49     | 26.1%         |

Segment scope survives the same sweep far better (49.6% at debounce=2 vs
route's 23.5%) because the 30-min accumulation window already absorbs most of
the single-tick noise before the regime clock sees it — the extra debounce
is compounding on top of smoothing that already happened, not doing the same
job twice for free.

Cross-source agreement at debounce_ticks=1, route scope, over the same
window: of 88 `published_condition`-reconstructed transitions, 5 (5.7%)
matched an `archive/vehicles`-reconstructed transition (same route, same
new_state, `entered_at` within one tick); of 32 vehicles-reconstructed
transitions, 5 (15.6%) matched back. Agreement is low both directions. The
vehicles-source route reconstruction is thin by construction — it self-trains
`compute_advance_baseline` (min_samples=20 per (route, direction, tod_bin))
on only 8 days, versus whatever longer baseline window backed the deployed
Worker's own classifier — so this reads as a sample-size-limited finding, not
a settled disagreement between the two derivations.

Recommendation: **debounce_ticks=1** for both scopes, against the numbers
above — movement's own classifier already carries a per-tick significance
test (`classify_direction`'s binomial tail, alpha=0.05), and its real episode
population is dominated by single-tick events that `debounce_ticks=2` erases
almost entirely. The lead makes the final call.

## 2026-08-11 — debounce set to 1: the movement classifier is not noise-dominated

origin: self

Decision on the sweep in the entry above, which left the call open. Shipped
`DEBOUNCE_TICKS = 1` in both `training/regime.py` and `worker/src/regime.ts`.

The argument for debounce=2 is that a single-tick call is noise. That is
checkable, and it does not hold. Over 2026-08-04..08-11 the route classifier
opened **17 episodes in 65,685 route-ticks** (0.026%). If `classify_direction`
were firing at its nominal `CLASSIFY_ALPHA=0.05` binomial tail, the same window
would carry ~3,284 onsets — **193x** what was observed. The significance test is
not what binds; the `DISRUPTED_RATIO=0.5` posterior gate is, and it is far more
conservative than the alpha implies. A population that sparse is not one a
confirmation rule is filtering noise out of.

Against that, debounce=2 erases 13 of those 17 episodes and every episode under
10 min, when the population's own median is 5.0 min — one tick. It would remove
76% of what the site publishes today, and per the corroboration entry above we
have no evidence those detections are false: alert non-corroboration cannot
refute a movement call, and only 15.4% of movement episodes are corroborated at
all.

Segment scope also runs at 1. Its per-tick calls are already smoothed by a
30-min trailing accumulation before the clock sees them, so a debounce on top
compounds denoising that already happened (49.6% survival at 2).

What debounce=1 does NOT give up: the clock still back-dates on commit, an
abstention still holds a regime open rather than ending it, and a cell blind
for `MAX_IDLE_SEC` still evicts rather than resuming a regime it cannot vouch
for. Those three rules, not the confirmation count, are what make offline and
online segment identically. The debounce machinery stays and is still pinned at
2 by `tests/fixtures/parity_regime.json`.

The number to chase if this is revisited: an independent corroborator for
single-tick movement episodes. The trip-updates service metric
(`derive_actual_recovery`) is derivation-independent from vehicle positions and
was not used here.

## 2026-08-11 — clustered incidents: 0/1260 ticks show two disrupted segments adjacent

origin: agent

`training/incidents.py` (C7) clusters contiguous disrupted segments into
incidents on the premise that adjacent disruption overwhelmingly shares one
cause (a stuck train, a signal fault) rather than being independent draws.
Measured against the real 8-day archive (2026-08-04..08-11, `murk exec`):
segment states from `training.movement_backfill.segment_ticks_from_vehicle_bodies`
(this epic's canonical offline segment reconstruction — 6-tick/30-min trailing
accumulation, `classify_segment` against a self-trained baseline, the same
function the debounce entries above already validated), topology from the real
static GTFS successor graph (`training.gtfs_static.load_successors`: 1797
segments, 52 route-direction groups).

Of 2267 reconstructed ticks, 1260 (55.6%) had two or more disrupted segments
somewhere on the network at once. Of those 1260, **zero** produced a
multi-segment cluster — at `max_gap=0` and at `max_gap=1` alike. Mean cluster
size across every tick in the window: exactly 1.0. A same-route-direction cut
against an earlier (non-canonical, same window) reconstruction found the same
shape before the switch: of 130 route-direction groups with >=2 disrupted
segments at one tick, only 1 (0.77%) had an adjacent pair — ruling out "the
pairs are just on different lines" as the explanation. Over the full window,
`replay_incidents` reconstructs 872 incidents (870 completed, 2 still open at
the window boundary), median duration 600s (10 min = 2 ticks), median
footprint size **1.0 segment**.

This is a negative result against the stated premise: at 5-min tick /
30-min accumulation granularity, this real window essentially never shows two
adjacent segments reading disrupted at the SAME tick. It does not make the
incident-level composition wrong — summing/maxing independent per-segment
dwells would still misprice the rare tick where adjacency does occur, and 872
single-segment "incidents" is the correct degenerate output when there is
nothing to cluster, not a sign the clustering code is broken. But the
premise's imagined mechanism — a stuck train visibly stalling its physical
neighbor in the same window — is not what this classifier observes; a real
blockage's queueing effect on the segment behind it plausibly needs more than
one tick to become significant. Number to chase if this is revisited: a
LAGGED adjacency test (segment A disrupted at tick T predicting segment B
disrupted at tick T+k for small k), not attempted here.

## 2026-08-11 — v1/movement_transitions is empty: no real segment or route dwell fit yet

origin: agent

Went to run `training.segment_dwell.build_segment_dwell` on real archived
data via `murk exec --` to report segment-cell counts and an own/pooled
split. `list_objects_v2` on `v1/movement_transitions/` (no date filter, so
every day ever written) returns **KeyCount=0** — the stream that both the
route and segment regime clocks feed (`movementTransitions` +
`writeMovementTransitions` in `worker/src/grading.ts`) has never written a
single object to R2, at either scope.

This isn't a credentials or query problem: `v1/regime_transitions/` (the
older alert-driven filter stream) lists fine over the same window.
`state/movement_state.json` and `state/segment_flow.json` both exist and are
fresh (`LastModified` 2026-08-11 20:56 UTC, minutes before this check), so
the deployed Worker is actively computing regimes — `movement_state.json` is
only 378 bytes, consistent with a handful of routes still on their very
first cold-start entry. `writeMovementTransitions` only puts an object when
`changes.length > 0` (grading.ts), so a regime clock that has not yet
_changed_ any cell's state writes nothing, even while it publishes
`state/*.json` every tick.

Net: the movement_transitions stream this session's dwell-curve work
(segment and, per the debounce entry above, route too) depends on has zero
real observations right now. `build_segment_dwell`/`compute_dwell_quantiles`
cannot be validated against production data until the deployed Worker has
run long enough, post-deploy, to commit at least one regime change per
scope. Tests in `tests/test_segment_dwell.py` exercise the estimator against
synthetic fixtures instead.

## 2026-08-11 — revises the entry above: 0/1260 is baseline-window-sensitive

origin: agent

Follow-up on "clustered incidents: 0/1260 ticks show two disrupted segments
adjacent" above, which self-trains `build_segment_baseline` on the FULL
8-day window before classifying every tick in it (same convention
`movement_backfill.py` uses for its own headline numbers). Reran each of the
8 days SEPARATELY — one self-trained baseline per single day — to check
whether the zero survives a shorter baseline window:

| day   | disrupted calls | ticks w/ >=2 disrupted | adjacent (gap0) | share |
| ----- | --------------- | ---------------------- | --------------- | ----- |
| 08-04 | 3391            | 286                    | 11              | 3.8%  |
| 08-05 | 4236            | 286                    | 15              | 5.2%  |
| 08-06 | 3949            | 287                    | 26              | 9.1%  |
| 08-07 | 3601            | 285                    | 0               | 0.0%  |
| 08-08 | 4559            | 287                    | 0               | 0.0%  |
| 08-09 | 5089            | 287                    | 0               | 0.0%  |
| 08-10 | 4379            | 287                    | 0               | 0.0%  |
| 08-11 | 5210            | 251                    | 37              | 14.7% |

Four of eight single-day baselines reproduce the zero; the other four don't,
and the non-zero share ranges 3.8-14.7% with no visible day-of-week pattern
in a single week. The 8-day-baseline run (this entry's parent) isn't wrong —
each segment leaf gets far more supporting data and therefore a more stable
`p0`, which is the more defensible baseline — but "zero" should be read as
"near-zero and sensitive to how much data trains the baseline," not as a
hard floor. A 1-day baseline is thin enough that some of what it flags
`disrupted` is baseline noise, and even the days that DO show adjacency stay
near mean cluster size ~1.0 (a handful of adjacent pairs among thousands of
singleton calls) — so the qualitative finding (adjacency is rare, nowhere
close to "overwhelming") holds on every single day measured; only the exact
count moves. Number to chase if this matters later: rerun the
8-day-baseline classification but score it against a HELD-OUT week (not
overlapping the classified window) to remove the self-training confound
entirely — not attempted here.

## 2026-08-11 — IRT/BMT/IND division does not earn a hierarchy level: eta² 0.16, p=0.081

origin: agent

New `training/divisions.py`: a static 28-route IRT/BMT/IND map (cross-checked
against the real GTFS `routes.txt` for the route-id space and against two
independent sources — Wikipedia's A/B Division articles and nycsubway.org's
"Subway FAQ" line-by-line lineage — for the division assignment; SI excluded,
it is a disconnected non-subway railway, never part of the three-division
network), plus a one-way ANOVA (between/within variance, eta-squared, and a
distribution-free permutation p-value, all pure stdlib) over per-route fitted
dwell scales.

Ran on the ALERT arm (`training.eval.load_transitions` + `pooled_dwell`),
not movement: movement route-scope coverage is 17 episodes over 6 routes
(prior entry), too thin to answer this; the alert arm fits 29 routes over
the same 14-day window `train_em` uses live.

`partially_pooled_dwell` on that window, grouped by division (n=28 mapped
routes, SI dropped):

| division | n   | mean log-scale | within-division variance |
| -------- | --- | -------------- | ------------------------ |
| BMT      | 9   | 10.899         | 0.027                    |
| IND      | 9   | 10.841         | 0.023                    |
| IRT      | 10  | 10.529         | 0.401                    |

MS_between=0.381, MS_within=0.161, F=2.373, **eta²=0.1596** (division
nominally explains 16% of fitted-log-scale variance), permutation
p=**0.0808** (10,000 draws, seed 0) — fails conventional significance with
only 3 groups to compare.

The 16% does not survive inspection. Only 6 of the 28 routes clear
`MIN_VOTER_EVENTS` and fit their own MLE (`source="own"`) instead of being
shrunk to the population MAP; the other 22 are single long right-censored
observations that land within a narrow band (log-scale 10.83-11.26)
regardless of division, because in this window they were all censored by
a similarly-long stretch of continuous `normal`, not because of any
division-level pattern. Of the 6 own-voters, 4 are IRT (1, 3, 5, 6 — route 3
alone contributes log-scale 9.057 off 12 real exits, the single biggest
outlier in the dataset) and only 1 each land in BMT (FS) and IND (B). IRT's
apparent separation is 4 specific routes' this-window churn, not a
division-wide effect — and BMT vs IND, the two divisions closest in the
classification scheme, are statistically indistinguishable here (10.899 vs
10.841, within each other's within-division noise). Restricting to the 6
own-voters directly (n=1,1,4 per division — too degenerate to trust, but
telling): eta² swings to 0.272 and F _drops_ to 0.561 (p=0.529), the
opposite of what a real division effect would do with less shrinkage noise.

**Verdict: no.** A division level does not earn its place — the measured
effect is small (eta²=0.16), not significant (p=0.081) at n=28 split three
ways, and what signal exists traces to which specific routes happened to
churn during this 14-day window, not to IRT/BMT/IND membership. Re-run this
exact module once movement or alert coverage is wide enough that most
routes clear `MIN_VOTER_EVENTS` on their own MLE — the number to chase is
the `n_own`/`n_pooled` split (6/23 here) approaching parity.

## 2026-08-11 — movement dwell_movement fit: 31 cells (8 own, 23 pooled), disrupted median 5.1-6.8min matches the 17-episode count exactly

origin: agent

Ran `training.train_em._movement_dwell` for real over 2026-08-04..2026-08-11
(`murk exec`, dry-run — nothing written to R2). The committed
`v1/movement_transitions` stream is empty for every date in the window (0
keys under every `v1/movement_transitions/<date>/` prefix checked), so every
cell in this run came from the `reconstruct_movement_transitions` fallback:
88 route-scope transitions rebuilt from the live `published_condition` ticks
through the same regime clock (`DEBOUNCE_TICKS=1`) the Worker runs online.

31 (route, state) cells total: 25 `normal`, 6 `disrupted`, 0 `suspended`.
8 own-fit (`n >= MIN_VOTER_EVENTS=3`: C/R/Z/G/6 normal, R/G/6 disrupted),
23 pooled-parent. 25 of 28 known routes get a `normal` cell (the 3 missing —
6X, 7X, GS — published `unknown` for nearly the entire window per their own
prediction-stream history, so the clock never opened a regime for them, not
a bug); 6 of 28 get a `disrupted` cell (6, G, J, M, R, W); 0 get `suspended`.

Median `median_sec`: `disrupted` cells range 303-410s (5.05-6.83min, median
of medians 307s) — narrower than and inside the previously-measured 4.95-
34.92min route-episode range, so no runaway-median bug. `normal` cells range
10,108-272,317s, dominated by censored routes still holding their first-ever
observed regime at window end (min 10,108s is Z's own-fit fitted median off
11 real exits; the 272,317s value repeats across most zero-event routes
because they share one open-regime start: the point published_condition
became usable, 2026-08-04).

Cross-check against the debounce=1 episode census from the movement-backfill
entry (17 route episodes across 6 routes): summing `n` over the 6 disrupted
cells here gives 5+3+5+1+2+1 = **17**, and the 3 that clear
`MIN_VOTER_EVENTS` (R=5, G=3, 6=5) are exactly the "3 of 6 routes with >=3
episodes" cited there. Same source data, two independent code paths
(census vs. dwell fit), same numbers.

## 2026-08-11 — movement exit-destination split: 100% to-normal, closed by measurement not code

origin: agent

`movementRecovery` (worker/src/snapshot.ts) reads a disrupted/suspended
regime's exit probability straight as `p_normal_in_H`, unlike the alert-HMM
branch in the same file which multiplies by a trained `toNormal` share. That
was flagged as a simplification, not proven immaterial. Measured it via
`murk exec` + `training.movement_backfill.reconstruct_movement_transitions`,
route scope, `debounce_ticks=1`, per `from_state`, both independent sources.

`source=vehicles` (2026-06-21..2026-08-12, 53 days, VEHICLE-ONLY —
`load_r2.derive_movement_state` can never return `not_scheduled`, so this
source is structurally biased against non-normal exits; trusted only as a
high-volume cross-check, not the vocabulary answer):

| from_state | n exits | to_normal | dest breakdown                   |
| ---------- | ------- | --------- | -------------------------------- |
| disrupted  | 213     | 1.000     | normal=213                       |
| suspended  | 0       | n/a       | (no completed episode in window) |

19 routes contribute to the 213, every one individually 100% to normal (7:45,
G:29, M:23, R:16, W:14, 6:14, E:11, J:10, D:10, N:9, A:8, C:5, 4:5, F:3, Q:3,
B:3, H:2, 1:1, 2:1, 5:1).

`source=predictions` (2026-08-04..2026-08-12, 8 days, the faithful published
vocabulary — trusted for the `not_scheduled` question the vehicle source
structurally cannot answer):

| from_state | n exits | to_normal | dest breakdown                   |
| ---------- | ------- | --------- | -------------------------------- |
| disrupted  | 17      | 1.000     | normal=17                        |
| suspended  | 0       | n/a       | (no completed episode in window) |

Per-route on the faithful source: R:5, 6:5, G:3, J:2, M:1, W:1 — every route
100% to normal, none below the pooled figure. (This 17 is the same population
the 2026-08-11 dwell-fit entry above counted by a different code path — same
numbers, second confirmation.)

`suspended` completed zero route-scope episodes in either window — not "rare
and mostly normal", genuinely absent from both the 53-day high-volume archive
and the 8-day faithful stream. `disrupted` never exited to `suspended` or
`not_scheduled` either, on the source that can express `not_scheduled`.

Decision rule from the assignment: >=0.98 pooled to-normal on the faithful
source, no route materially below it. Measured 1.000 (17/17) with zero
exceptions across 6 routes, corroborated by 1.000 (213/213) across 19 routes
on the larger source. Verdict: **immaterial today, no code change.** Did not
build `movement_exit_split` — a trained-share params block, a worker
accessor, and a multiply in `movementRecovery` would all be exercising a
split that has not once fired in 61 days of combined archive. Replaced the
"no trained split exists" comment in `movementRecovery` with these numbers
and this date so the claim is checked, not assumed. Number to chase if this
ever needs revisiting: the first observed `suspended`-from or
disrupted-to-`not_scheduled` route-scope transition, whenever it happens.

## 2026-08-12 — 53-day incident clustering settles it: still doesn't hold, but not literally zero (3 real incidents)

origin: agent

Follow-up on "clustered incidents: 0/1260 ticks show two disrupted segments
adjacent" and its baseline-window-sensitivity revision (both 2026-08-11
above). Both were underpowered: 8 days, one baseline. Re-ran
`training.incidents.measure_premise` over the full vehicle archive
(2026-06-21..2026-08-12, 53 days, `murk exec`), scored against TWO
baselines built from the same 53-day vehicle archive: `self` (`load_r2.
build_segment_baseline`, self-trained on the measured window — the prior
run's weakness, still run here so the two can be compared directly) and
`published` (the live `state/segment_params.json`, `trained_at=1786492506`,
2532 cells — what the Worker actually classifies against, via the new
`published_baseline_cells` + `segment_ticks_with_baseline`).

Pooled 53-day result, both baselines:

| baseline                             | candidate ticks (>=2 disrupted) | adjacent ticks gap0 | adjacent ticks gap1 | share ticks gap0 | mean cluster size gap0 | share disrupted-segments-in-multi gap0 | gap1   |
| ------------------------------------ | ------------------------------- | ------------------- | ------------------- | ---------------- | ---------------------- | -------------------------------------- | ------ |
| self (53-day self-trained)           | 5257                            | 21                  | 21                  | 0.40%            | 1.0016                 | 0.274%                                 | 0.285% |
| published (live segment_params.json) | 5489                            | 21                  | 21                  | 0.38%            | 1.0016                 | 0.271%                                 | 0.283% |

gap=0 and gap=1 give the IDENTICAL 21 adjacent ticks on both baselines —
loosening `DEFAULT_MAX_GAP` still has no support. The two baselines land
within ~4% of each other on the denominator (5257 vs 5489 candidate ticks —
the published baseline calls slightly more ticks judgeable) and produce the
exact same 21 numerator. (8963 total reconstructed ticks either way; 5257
of them, 58.7%, have >=2 disrupted segments somewhere on the network — same
order of magnitude as the 8-day run's 55.6%, not a new finding on its own.)

Per-week breakdown (self-trained baseline; published tracks it within 1-2
ticks/week everywhere it differs):

| week         | candidate ticks | adjacent ticks gap0 | share                            |
| ------------ | --------------- | ------------------- | -------------------------------- |
| 06-21..06-27 | 0               | 0                   | n/a (no disruption observed yet) |
| 06-28..07-04 | 0               | 0                   | n/a                              |
| 07-05..07-11 | 20              | 0                   | 0.0%                             |
| 07-12..07-18 | 948             | 11                  | 1.16%                            |
| 07-19..07-25 | 1185            | 9                   | 0.76%                            |
| 07-26..08-01 | 1304            | 1                   | 0.08%                            |
| 08-02..08-08 | 1255            | 0                   | 0.0%                             |
| 08-09..08-12 | 545             | 0                   | 0.0%                             |

5 of 8 weeks are exactly zero; all 21 adjacent ticks fall in a 3-week span
(07-12..08-01). Traced identity through the whole window with
`advance_incidents` (not just counting candidate ticks) to see how many
DISTINCT real incidents that span represents: exactly **3**, agreed on by
both baselines down to the segment keys and timestamps —

| incident | segments                       | first seen (UTC) | last seen (UTC)  | span    |
| -------- | ------------------------------ | ---------------- | ---------------- | ------- |
| 1        | 7\|south\|701S, 7\|south\|702S | 2026-07-16 04:20 | 2026-07-16 07:55 | 215 min |
| 2        | A\|north\|A48N, A51N, A55N     | 2026-07-24 18:15 | 2026-07-24 19:45 | 90 min  |
| 3        | 7\|south\|701S, 7\|south\|702S | 2026-07-27 12:30 | 2026-07-27 12:45 | 15 min  |

Two of the three are the SAME 7|south segment pair recurring (a specific
chronically slow spot, not a generic clustering tendency); the third is the
one 3-segment footprint in the whole 53 days, on the A up in the 140s-170s.
Zero incidents anywhere else on the network, on either baseline, in 53
days.

**Verdict: does not hold — confirmed, not overturned, but the literal
"zero" from the 8-day run is overturned.** 53 days surfaced 3 real
multi-segment incidents an 8-day window was simply too short to ever see;
they're genuine, not noise (both a self-trained-on-53-days baseline and the
independently-trained published baseline agree on the same 3, to the
minute). But 3 incidents against 2532 segment cells over 53 days, and
0.27-0.29% of disrupted-segment observations, is not "overwhelming" by any
reading — the premise's core claim (that adjacent clustering is the
dominant case, not the exception) still fails. **What changed the number:
the longer window, not the baseline.** Self-trained-on-53-days and
published converge almost exactly (same 21 ticks, same 3 incidents,
matching timestamps); baseline choice moved only the candidate-tick
denominator by ~4% and never flipped a single classification that mattered
to the verdict. Left `training.incidents.path_incident_durations` unwired
(non-goal: `train_em.py`/`snapshot.ts` publish nothing new here) and
rewrote the module docstring with this finding and window so the next
reader doesn't re-derive it from scratch.

## 2026-08-12 — suspended is not covered by the exit-split measurement

origin: agent

Revises the entry above ("movement exit-destination split: 100% to-normal,
closed by measurement not code"). Its measurement stands and its conclusion
for `disrupted` stands; its title is wrong on two counts, and the second one
was a live unsoundness.

The measurement covered `disrupted` at n=213 and n=17, both 100% to normal.
It covered `suspended` at **n=0**. Zero observations is not a measurement of
1.0 — it is the absence of one, and the entry then let a suspended cell's exit
probability be read directly as `p_normal_in_H` on that basis. The two states
are not interchangeable here: `disrupted` means trains are moving slowly and
recovering means they speed up, whereas `suspended` means there are no trains
on the route at all, and a route resuming service can plausibly come back
degraded before it comes back normal. Nothing in the archive says otherwise
because nothing in the archive says anything.

Fixed in code, so it was not closed by measurement alone: `movementRecovery`
now serves only the states whose exit destination has actually been observed
(`normal`, which needs no split at all since an exit from normal is never to
normal, and `disrupted`). Anything else returns null and falls through to the
alert arm, which has a trained transition matrix and a real `toNormal` share.
Pinned by a test that fails when the guard is removed (verified by deleting
the line: 8 pass / 1 fail, restored: 9 pass).

Live blast radius today is zero — the published fit carries 25 `normal` and 6
`disrupted` cells and **no `suspended` cell**, so the guarded path was
unreachable in production. It would have become reachable the first time a
suspended movement episode completed and pooling produced a cell for it, which
is exactly when a silently-wrong `p_normal_in_H` would have started shipping.

The number to chase if this is revisited: the to-normal share for exits from
`suspended`, which needs at least one completed suspended route-scope episode
to exist before it can be estimated at all.

## 2026-08-11 — movement-curve forecast counterfactual: AUC 0.150→0.762 at 30min, still inverted at 120min

origin: agent

Counterfactual over the archived window where the movement arm is usable
(2026-08-04..2026-08-12, 67,309 rows; 456 excluded because they already carry
`recovery_source=='movement'` — produced by today's deployed code, not the old
alert-driven path, so they aren't a fair "old" sample — leaving 66,853 graded
rows). For each row, reconstructed the movement regime `(state, entered_at)`
per (route, tick) by replaying `published_condition` through
`training.regime.advance_regimes` (debounce=1, the same primitive
`training.regime.replay_regimes`/`movement_backfill` use, just keeping every
intermediate tick instead of only the final one), then recomputed
`p_normal_in_H` with the LIVE `dwell_movement` curve via
`training.dwell.p_leave_by`, mirroring `movementRecovery`'s exact guards:
normal → `1 - p_leave_by` (survival), disrupted → `p_leave_by` directly, any
other state/no curve/no clock/schedule-precedence → fall back to the archived
(old) value. Graded against `published_condition` at t+H via
`training.eval.calibrate(..., truth_default=None)` — unknown/not_scheduled
ticks drop out rather than scoring as calm.

Movement path served 50,722/66,853 rows (75.9%; 50,696 normal + 26
disrupted); 11,131 (16.7%) fell back for no `published_condition` reading that
tick, 5,000 (7.5%) for a state the movement path doesn't serve
(not_scheduled/suspended); zero rows fell back for a missing curve, a
non-positive clock, or schedule precedence.

| horizon | n      | AUC old→new | Brier old→new   | BSS vs persistence old→new |
| ------- | ------ | ----------- | --------------- | -------------------------- |
| 30min   | 50,061 | 0.150→0.762 | 0.01587→0.00066 | -16.27→+0.29               |
| 60min   | 49,831 | 0.150→0.493 | 0.03609→0.00102 | -39.87→-0.16               |
| 120min  | 49,201 | 0.150→0.397 | 0.06004→0.00178 | -58.08→-0.75               |

Positive-class (published-normal at t+H) base rate is ~99.95% at every
horizon. The mean-split that named the bug, `p_normal_in_30min`: old 0.9209
disrupted vs 0.8788 normal — backwards, same sign as the originally-reported
0.9166/0.8780 — vs new 0.9856 disrupted vs 0.9932 normal, sign flipped:
correctly less confident where the line is stalled. The disrupted-current-tick
sample is thin (n=26, entirely on the 6 routes with a fitted disrupted curve —
0 fell back to no-curve), so treat the AUC magnitude as provisional.

**The inversion is gone at 30min (0.150→0.762) and no longer inverted at
60min (0.493, essentially chance) but is STILL INVERTED at 120min (0.397 <
0.5)** — independently re-derived by hand (matched-sample rebuild + a
from-scratch Mann-Whitney) to rule out a scoring bug. New beats a naive
persistence baseline only at 30min; at 60 and 120min it loses to persistence,
far less catastrophically than old (BSS -0.16/-0.75 vs old's -39.87/-58.08)
but it still loses. Likely driver: of the movement-path "normal" rows, 71.4%
carry `entered_at` pinned to that route's very first in-window reading — the
regime almost certainly predates the 8-day archive window (pre-window data is
92.8% `unknown` and any real signal is evicted by the 1-hour idle rule before
it could bridge to the window anyway), so true elapsed time is understated
throughout the window and the understatement compounds as the horizon grows.

Self-consistency caveat: both the truth (`published_condition` replayed
through the regime clock) and the new forecast (the movement dwell curve
trained on `published_condition` transitions) derive from the movement arm.
This is the correct apples-to-apples comparison — the 0.142-0.145 figure that
named the bug was computed the same way, against the same arm — but it is not
evidence of independent skill, and does not validate the movement classifier
itself.

## 2026-08-11 — independent-truth grade of the movement forecast: trip-updates MAE 251.8min, movement CRPS underwater (skill -0.10, -0.36 held out)

origin: agent

Grades the movement arm against truth it did NOT come from — the counterfactual
entry above grades it against `published_condition` (the arm's own truth, same
source problem the eval-honesty doc warns about). Two independent checks, plus
a live coverage/adoption/corroboration refresh.

**Archive coverage, verified (not assumed identical):** `archive/trip_updates/`
is 59 days, 2026-06-15..2026-08-12 — starts 6 days EARLIER than
`archive/vehicles/` (53 days, 2026-06-21..2026-08-12). `v1/predictions/` is 91
days, 2026-05-14..2026-08-12. Largest common window for Part 1 is therefore
2026-06-15..2026-08-12, gated by trip_updates.

**Part 1 — trip-updates `assigned_n` truth (`training.load_r2.derive_actual_recovery`,
`training.eval.independent_recovery_metrics`, defaults).** 16,674 trip-updates
ticks → 132 (route, tod_bin) baseline cells (>=20 samples) → 1,054 independent
disruption intervals (median 70min, min 10min, max 4,880min/81.3h — this truth
is dominated by long service reductions, not movement-scale minutes). Graded
`recovery_minutes` against them: **n=362 graded ticks (49 regimes), MAE=251.8min,
RMSE=490.6min, IQR coverage=21.0%**; per-regime macro-average MAE=178.2min,
coverage=17.2%; 19,798 schedule-sourced rows excluded per the existing arm
contract. Window coverage: 39.1% unknown, 56.4% gradeable (this window still
contains the 22-day unknown gap below).

Signed bias (predicted − actual_remaining, same matching as
`independent_recovery_metrics`, sign kept): mean **-140.8min**, median
**-29.7min** — `recovery_minutes` under-predicts the trip-updates truth's
remaining time on average. Direction, stated plainly per the semantic gap: the
two signals measure different things (trip-updates counts dispatched trains,
movement times station-to-station advance), so this is not automatically an
arm error — but the sign says the movement arm calls "recovered" _before_ the
fleet count does, the "thinned but flowing" case, not the "full fleet crawling"
case. MAE at 251.8min is large in absolute terms, but a truth series whose
median disruption alone is 70min (and tail runs to 81h of planned-work-scale
reductions) is a different population than the movement arm's own minutes-long
episodes (see Part 2) — the two truths are not commensurate, which is itself
part of the honest answer here, not an excuse for the number.

**Part 2 — episode-level CRPS/PIT, `training.scorecard.episode_recovery` +
`movement_dwell_lookup_from_params`, movement episodes from
`training.episodes.extract_episodes` over `training.load_r2.build_movement_truth`
(causal baseline: 2026-06-21..07-04, 3,826 bodies -> 241 cells; graded window
2026-07-05..08-12, 10,936 bodies -> 242,252 truth cells, mirroring
`training.review`'s own causal-baseline convention).** 130 movement episodes
extracted (0 standing, all peak_state=disrupted — no suspended movement
episode exists to grade, confirming the earlier 0-suspended finding again),
median duration 5.0min, min 4.95-5.0min\*, max 585min, 20 routes. Live
`dwell_movement` (trained_at=2026-08-11 23:55 UTC, window 2026-07-29..08-11)
covers disrupted curves for only 6 of those routes (6, G, J, M, R, W —
matching the live-params fact); the other 14 routes' episodes have no curve:
**n_scored=76, n_no_curve=54, n_censored=0**.

CRPS **mean=5.17min vs baseline(climatology)=4.69min → skill=-0.103**: the
movement dwell curve scores 10.3% WORSE than just predicting each episode's
own empirical duration distribution. Baseline confirmed built over the graded
movement-episode population itself (`episode_recovery`'s own contract,
`recovery_dist.py:160`) — never the alert-shadow's hundreds-of-hours
population; the 76 scored durations here are minutes, not hours, so no
cross-population dilution occurred. `mean_pit=0.494` (0.006 below 0.5,
negligibly pessimistic) — but the 10-bin histogram is bimodal, not uniform:
`[0,0,38,0,14,0,0,2,8,14]` — 50% of mass sits at PIT∈[0.2,0.3) and 29% at
PIT>=0.8, canceling to a near-0.5 mean while masking two real failure modes
(recovers much faster than predicted, and recovers much slower than
predicted) in roughly equal measure. The module's own verdict text calls this
"Well calibrated" on the mean but flags it directly: "scores 10% worse than
the dead-simple baseline — calibrated, yet no sharper than guessing the
average. Calibration isn't skill."

Checked whether this is a training-window leakage artifact (the live params'
14-day training window, 2026-07-29..08-11, overlaps 14 of our 39 graded
days): split by episode onset against that boundary. Held-out (onset before
2026-07-29, predates the params entirely): **n=26, skill=-0.361** — WORSE, not
better. In-sample-or-later (onset >= 2026-07-29): n=50, skill=-0.117. The
negative skill is not explained by grading the model against its own training
data — it holds, and deepens, out of sample.

**Part 3 — deployed system health, live R2 (all times UTC; UTC date rolls at
the point the codebase's deploy commits land in NY time, so "today"=08-12,
"before deploy"=08-11).**

Coverage, last 14 days + today:

| date                              | unknown share | gradeable share |
| --------------------------------- | ------------- | --------------- |
| 07-30                             | 100.0%        | 0.02%           |
| 07-31                             | 99.9%         | 0.08%           |
| 08-01                             | 100.0%        | 0%              |
| 08-02                             | 100.0%        | 0%              |
| 08-03                             | 64.1%         | 34.9%           |
| 08-04                             | 16.9%         | 78.4%           |
| 08-05                             | 17.3%         | 77.8%           |
| 08-06                             | 17.1%         | 78.2%           |
| 08-07                             | 16.8%         | 78.3%           |
| 08-08                             | 16.3%         | 70.8%           |
| 08-09                             | 16.5%         | 68.9%           |
| 08-10                             | 15.3%         | 77.6%           |
| 08-11                             | 16.3%         | 78.1%           |
| 08-12 (today, n=522, partial day) | 13.8%         | 82.8%           |

Coverage is healthy now: today's 13.8% unknown / 82.8% gradeable sits inside
(slightly better than) the stable ~15-17%/~78% band every day has held since
the 07-12..08-02 empty-baseline gap closed on 08-03. Nothing resembling the
22-day 100%-unknown blackout has recurred.

Adoption, today (n=522, post-deploy) vs before deploy (n=8,352, 08-11, almost
entirely pre-deploy — the deploy landed ~23:48 UTC 08-11):

| field               | value     | before deploy  | today            |
| ------------------- | --------- | -------------- | ---------------- |
| recovery_source     | movement  | 0.29% (24)     | **82.76% (432)** |
| recovery_source     | hmm       | 99.71% (8,328) | 17.24% (90)      |
| condition_source    | movement  | 83.66% (6,987) | 86.21% (450)     |
| condition_source    | unknown   | 16.34% (1,365) | 13.79% (72)      |
| published_condition | disrupted | 0.14% (12)     | 0% (0)           |

`condition_source` was already movement-majority BEFORE today's deploy — this
matches the bug's own framing exactly: published status was already
movement-derived, only recovery/forecast were gated behind the alert-HMM.
`recovery_source` is the field that moved, 0.29% -> 82.8%. This is adoption,
not performance: zero disrupted rows exist today (0/522) to grade the new
forecast's live skill against, so nothing above is a skill claim.

Alert corroboration, refreshed (`training.escalation.corroborate_episodes` +
`confirmation_summary`, 2026-08-04..08-12, 9 days vs the prior 5-day
08-07..08-11 window; `esc_source=published_archive`, coverage/usable gates
both passed so no offline-recompute fallback needed): **361 movement-disrupted
episodes (up from 259), confirmation_rate=41/361=11.4% (down from 15.4%/259),
n_unconfirmed=320**, median lead time among the 41 confirmed = 0.0min (IQR
[0.0, 0.0]) — same shape as before, confirmation is almost entirely a
same-tick alert already posted, not a movement call leading a later alert.
Per `training/escalation.py`'s own contract, `n_unconfirmed` is not divided
into a rate and alert absence is not computed as a false-positive/precision/
specificity number against movement — that would resurrect the asymmetry the
module exists to avoid.

**Bottom line:** post-deploy live data cannot grade the new code yet (435-522
rows so far today, zero disrupted). Both counterfactual checks that CAN run —
this entry's independent-truth ones and the AUC counterfactual above — agree
the fix is real but incomplete: `p_normal_in_H` uninverted at 30min in the
sibling's counterfactual, but the movement dwell curve's own recovery-time
distribution scores below climatology here (CRPS skill -0.10, -0.36 held
out), and the trip-updates truth disagrees with `recovery_minutes` by a
251.8min MAE that this window's differing truth-population scale only
partially explains. Deployed and directionally correct is not the same claim
as calibrated or skillful; this entry's numbers are the ones to revisit once
live disrupted data exists to grade directly.

\*The 4.95min figure is from the constraints' prior 8-day/predictions-sourced
measurement, not this run (whose vehicle-sourced episodes bottom out at one
tick, 5.0min, by construction — `TICK_SECONDS=300`).

## 2026-08-11 — forecast horizon inversion: projection defect, not left-censoring (trustworthy-clock subset still inverted, AUC 0.352 at 120min)

origin: agent

Splits the counterfactual two entries above ("movement-curve forecast
counterfactual: AUC 0.150→0.762 at 30min, still inverted at 120min") by
clock trustworthiness, to answer whether the 60/120min inversion is
left-censoring (fixes itself with runtime) or a genuine horizon-projection
defect (needs a code/model fix). Same method, same window
(2026-08-04..08-12, 66,888 graded rows today — the archive grew a few rows
since that entry's 66,853), same live params (`trained_at`
2026-08-11T23:55:06Z) — but every row is additionally tagged by replaying
`published_condition` through `training.regime.advance_regimes` (debounce=1)
and comparing its regime's `entered_at` to that route's own first in-window
reading: `trustworthy` if the regime demonstrably started inside the window,
`censored` if the regime was already open at window start (true start
unknown, elapsed understated). Of the 50,722 movement-path rows, 36,184
(71.4% — matches the diagnosed figure exactly) are censored, 14,538 (28.6%)
are trustworthy.

| subset       | 30min AUC / BSS (n)        | 60min AUC / BSS (n)        | 120min AUC / BSS (n)       |
| ------------ | -------------------------- | -------------------------- | -------------------------- |
| all (sanity) | 0.762 / +0.29 (50,061)     | 0.493 / -0.16 (49,831)     | 0.398 / -0.75 (49,297)     |
| trustworthy  | 0.728 / **+0.40** (14,278) | 0.395 / **-0.00** (14,053) | 0.352 / **-0.53** (13,606) |
| censored     | 0.624 / -0.32 (35,783)     | 0.449 / -0.63 (35,778)     | 0.317 / -1.30 (35,690)     |

**Interpretation rule applied as specified: the trustworthy subset does NOT
recover AUC above 0.5 at 60 or 120min (0.395, 0.352) — it is still inverted
with a demonstrably real, in-window, uncensored clock. Verdict: PROJECTION
DEFECT, not left-censoring** — the 60/120min inversion will not fix itself
as the deployed system accumulates runtime. Left-censoring is real and does
matter, but only at 30min: trustworthy beats persistence there (BSS +0.40)
while censored loses (BSS -0.32), a clean split. By 60min censoring stops
predicting forecast quality at all (BSS -0.00 vs -0.63) and by 120min
trustworthy is still decisively negative (-0.53) — waiting for more routes'
clocks to "age in" will not turn that positive.

Honesty check on the deciding number: AUC is a discordant-pair statistic and
the positive-class base rate is ~99.95%, so it lives on the thin negative
(not-normal-at-t+H) tail. Counted directly: all/trustworthy/censored carry
26/19/7 negative rows at 30min, 26/15/11 at 60min, 25/9/16 at 120min — the
trustworthy subset's 120min AUC=0.352 rests on just **9** discordant rows,
thinner than the 25 behind the original headline number. Treat that single
AUC value as directionally right, not statistically final. The BSS columns
are a sturdier witness for the same verdict: Brier is a proper score over
the full n (13,606-35,783 per cut, not just the discordant tail), and it is
decisively negative in every 60/120min cut regardless of censoring status —
three independently-thinned populations agreeing that the forecast loses to
naive persistence is not the kind of thing 9 rows can flip.

A non-monotonic-in-elapsed diagnostic sharpens the "not censoring" case
further: bucketing the trustworthy-normal rows by elapsed time at 120min,
AUC is 0.290 for 6-24h elapsed (n=5,406, the worst band), 0.829 for 24-48h
(n=1,778, the best), and 0.472 for 48h+ (n=3,361, chance). A pure
understated-elapsed story predicts smooth degradation with elapsed; this is
a trough at 6-24h next to a strong band at 24-48h, more consistent with the
fitted normal-dwell curve's shape being wrong in specific places (plausibly
a missing time-of-day/periodicity covariate) than with a uniform clock-age
effect.

**Experiment 3 (sanity: is the horizon math itself buggy?):** checked
`p_normal_in_H` is non-increasing in H, for every one of the 25 live
`dwell_movement` normal cells, across 950 elapsed values per cell (0 up to
10x each cell's own max curve value, into the tail-extrapolation regime) and
48 horizons (5-240min, 5min steps) — 45,600 total comparisons, **0
violations**. The horizon-projection code is not buggy in the trivial
monotonicity sense; the defect is in what the curve says P(stay normal) is
conditional on elapsed, not in the H-step math.

**Experiment 2 (how long is "patience", if it mattered):** 25 routes carry a
live `dwell_movement` normal cell (matches the known count). 14 of 25 (56%)
share one identical pooled-parent curve — median 85,057s (23.6h), q75
592,496s (164.6h) — so most routes' "typical normal regime" figure is a
shared cross-route average, not their own history; own-fit routes range from
Z at 16,006s (4.45h) to M at 77,495s (21.5h). The distribution of per-route
median normal dwell across all 25 routes has p50 = p75 = 85,057s (23.6h),
both landing on the pooled plateau since it covers >50% of routes. Verified
the deploy boundary directly from `recovery_source` adoption, hour-binned:
0% at every hour through 2026-08-11T22:00 UTC, 6.9% at 23:00, 82.8% by
2026-08-12T00:00 — confirms the ~23:48 UTC 08-11 deploy time cited two
entries above. Deploy + 85,057s = **2026-08-12T23:25:37Z** for both the
50%-of-routes and the 75%-of-routes milestone — about one day, not the
~75h a prior fit (cited in this investigation's brief) had suggested. That
optimism is moot given the verdict above: reaching "trustworthy" in a day
does not fix an AUC that is already inverted for the trustworthy subset
today.

**Recommendation:** keep publishing `p_normal_in_30min` — real, repeatedly
measured skill (BSS +0.29 to +0.40 across every cut run today and in the
entry above). Stop publishing `p_normal_in_60min` and `p_normal_in_120min`
as skillful forecasts: both lose to naive persistence in every population
cut tested (BSS -0.00 to -1.30), the loss is not a censoring artifact (the
uncensored subset loses too), and Experiment 3 rules out a fixable
monotonicity bug — this needs a model-form change (the elapsed-conditional
normal-dwell curve itself), not more runtime.

Commands run (`murk exec -- uv run python <scratch script>`, PYTHONPATH set
to the repo root; script lived only in /tmp, deleted after the run, nothing
left in the repo): loaded `v1/predictions/` for every date 2026-08-04
through 2026-08-12 inclusive (`training.eval.load_predictions`),
`state/params.json`
(`training.r2_client.get_object_bytes`), replayed
`training.regime.advance_regimes` tick-by-tick over `published_condition`,
recomputed `p_normal_in_H` via `training.dwell.p_leave_by` against
`training.scorecard.movement_dwell_lookup_from_params`, graded via
`training.eval.calibrate(..., truth_default=None)`.

## 2026-08-12 — movement recovery distribution: model form, not data volume (mixture fix cuts the CRPS deficit from -0.088 to -0.023 using the same data)

origin: agent

**Verdict: MODEL FORM.** The decisive number: refitting the shipped
continuous log-logistic dwell curve as a one-tick point-mass mixed with a
conditional continuous tail — same 06-21..07-04 training data, same
held-out 97-episode population, zero additional days — cuts the CRPS-skill
deficit from **-0.088 to -0.023** (a 74% reduction). Meanwhile the causal
training-window sweep below shows no monotonic gain from more data at all
(skill oscillates -0.14 to -0.24 from 7 to 35 days of training, and gets
_worse_ from day 21 to day 28). Four experiments, detailed below, all point
the same way: not data volume, not pooling shrinkage, but a single
continuous distribution being structurally unable to front-load enough mass
at the exact one-tick floor (70.4% of outcomes) while keeping enough tail
mass for the ~30% that run long.

**Method note (read before the tables):** two different, both fully
reproducible, movement-truth reconstructions are used and are NOT
interchangeable populations. Experiment 1 uses the same single-pass,
self-trained-baseline reconstruction that already produced the "213
completed disrupted episodes" figure this task started from
(`movement_backfill.route_ticks_from_vehicle_bodies` +
`transitions_from_ticks(scope="route", debounce_ticks=1)` over the full
2026-06-21..08-12 archive) — reproduced exactly here (n=213, 14 distinct
duration values, tick histogram below matches the given table to the
episode). Experiments 2-4 instead reproduce the properly-CAUSAL truth
(`load_r2.compute_advance_baseline` fit on the clean 06-21..07-04 window
only, applied forward via `load_r2.build_movement_truth` to classify
07-05..08-12, then `episodes.extract_episodes`) — this exactly reproduces
the already-reported n=130 episodes, and, graded against a fresh pull of
the live `state/params.json` `dwell_movement` block, reproduces n_scored=76,
skill=-0.10348, mean_pit=0.49387, and the exact PIT histogram
`[0,0,38,0,14,0,0,2,8,14]` bit-for-bit — strong evidence the pipeline here
matches the one that produced the numbers this task started from. Because
`state/params.json` is a live, mutating artifact this session (its
training_corpus window shifted from the 07-29..08-11 cited earlier in this
journal to 07-24..08-12 by the time it was re-fetched here), Experiments
2-4's model comparisons do NOT depend on it further: they fit their own
dwell curves locally and reproducibly on the same 06-21..07-04 window
(`pooled_dwell.partially_pooled_dwell` / `dwell.dwell_samples_by_cell`),
graded against the same causal 130-episode population (97 scored by the
shipped continuous form after dropping 33 no-curve). The one live-params
pull is used only once, as the validation checkpoint above and as
Experiment 3's primary exhibit (it is the exact, already-published
population, so its PIT lobes are the ones actually worth explaining).

A structural finding surfaced along the way, worth flagging explicitly
because it changes how (b)/(c) below should be read: `dwell.dwell_cdf`
special-cases `x <= curve_sec[0]` as probability 0. Any curve whose lowest
quantile ties (KM/empirical curves fit to a floor-heavy population, where
the smallest observed duration IS the point mass) reads as **exactly 0**
when queried at that same value — and `predicted_recovery_curve` always
queries at exact integer minutes, so a one-tick (5min) episode graded
against a curve whose floor is one tick always gets PIT=0, regardless of
how much of the curve's mass actually sits there. Verified directly: model
(c)'s PIT histogram bin0 count (66) equals the exact count of one-tick
episodes in that population. This deflates (b) and (c) below beyond their
genuine statistical quality — the mixture fix at the end avoids it by
grading a closed-form CDF that bypasses `curve_sec` quantization entirely.

### Experiment 1 — learning curve (does more data help?)

Duration histogram, full 53-day archive reconstruction, n=213 (exact
match to the already-measured table; only 14 distinct duration values):

| ticks                      | minutes | count | share |
| -------------------------- | ------- | ----- | ----- |
| 1                          | 5       | 150   | 70.4% |
| 2                          | 10      | 27    | 12.7% |
| 3                          | 15      | 11    | 5.2%  |
| 4                          | 20      | 4     | 1.9%  |
| 5-119 (10 distinct values) | 25-595  | 21    | 9.8%  |

Route-volume buckets (full-archive, in-sample fit — every route's own
completed-episode count over all 53 days vs. CRPS skill on its own
episodes):

| bucket (own episodes) | n routes | n episodes | mean CRPS | baseline CRPS | skill                                           |
| --------------------- | -------- | ---------- | --------- | ------------- | ----------------------------------------------- |
| 1-2                   | 4        | 5          | 0.89      | 0.00          | undefined (zero-variance baseline, n too small) |
| 3-5                   | 5        | 19         | 17.31     | 15.87         | -0.090                                          |
| 6-10                  | 4        | 37         | 8.27      | 7.67          | -0.078                                          |
| 11+                   | 7        | 152        | 7.34      | 7.05          | -0.040                                          |

Training-window sweep — fit on the first N days (2026-06-21 onward), grade
on the SAME fixed held-out window every time (2026-07-30..08-12, n=63
episodes, held fixed so N is the only thing that changes):

| N days | train window ends | n train episodes | routes w/ cell (own/pooled) | n scored | skill  |
| ------ | ----------------- | ---------------- | --------------------------- | -------- | ------ |
| 7      | 06-28             | 29               | 10 (3/7)                    | 51       | -0.243 |
| 14     | 07-05             | 77               | 13 (8/5)                    | 55       | -0.150 |
| 21     | 07-12             | 90               | 16 (9/7)                    | 62       | -0.142 |
| 28     | 07-19             | 118              | 20 (10/10)                  | 63       | -0.170 |
| 35     | 07-26             | 138              | 20 (14/6)                   | 63       | -0.164 |

**Interpretation rule (as specified): skill rising monotonically with N
would mean a volume problem; flat or negative regardless of N means more
data will not fix it.** The sweep is flat-to-negative and non-monotonic
(worse at N=28 than N=21) — more data does not fix it. The route-volume
bucket table shows a mild, monotonic improvement with a route's own
episode count (-0.090 → -0.078 → -0.040) but never crosses zero even at
11+ own episodes, and it's an in-sample view (weaker evidence than the
causal sweep). Net: volume gives at most a second-order assist; it is not
the primary lever.

### Experiment 2 — model form (is a continuous curve the wrong shape?)

Same 06-21..07-04 training data, same causal 130-episode held-out
population, three forms graded through the unmodified
`scorecard.episode_recovery` / `recovery_dist.recovery_dist_report`:

| model                                                                | n scored | n no-curve | mean CRPS | baseline CRPS | skill  |
| -------------------------------------------------------------------- | -------- | ---------- | --------- | ------------- | ------ |
| a. shipped continuous log-logistic AFT (`partially_pooled_dwell`)    | 97       | 33         | 6.92      | 6.36          | -0.088 |
| b. per-route empirical/KM (`compute_dwell_quantiles`, min_samples=5) | 25       | 105        | 5.45      | 3.82          | -0.426 |
| c. pooled climatology (one KM curve over all routes' pooled samples) | 97       | 33         | 7.15      | 6.36          | -0.126 |

Restricted to the n=25 intersection every model can score (apples-to-apples,
same episodes, same baseline):

| model                 | mean CRPS | baseline CRPS | skill  |
| --------------------- | --------- | ------------- | ------ |
| a. shipped continuous | 4.42      | 3.82          | -0.157 |
| b. per-route KM       | 5.45      | 3.82          | -0.426 |
| c. pooled climatology | 4.68      | 3.82          | -0.224 |

Neither (b) nor (c) beats (a) — on the contrary, (a) is the least-bad of
the three, both at full coverage and on the matched subset. Per the
assignment's own framing this is evidence the parametric form per se isn't
the destroyer of information — BUT (b)/(c)'s scores are inflated-bad by the
`dwell_cdf` floor artifact above (both have curve_sec[0] pinned to exactly
one tick, so every one-tick episode scores PIT=0 regardless of true fit
quality), and (b) additionally starves for data (105/130 no-curve at
min_samples=5, no pooling). Taking (a) at face value as "the best of three
plain forms" is correct; taking it as "therefore the functional family is
fine" is not — see the mixture result at the end.

### Experiment 3 — where does the bimodal PIT come from?

Using the exact already-published population (n=76 scored, causal 130
episodes graded against the live `dwell_movement`, reproduced bit-for-bit
above). Per-episode PIT, lobed at [0.2,0.3) and >=0.8 as specified (plus
the [0.4,0.5) cluster the histogram also shows):

| PIT lobe          | n   | share | composition                                                                                                      |
| ----------------- | --- | ----- | ---------------------------------------------------------------------------------------------------------------- |
| [0.2,0.3)         | 38  | 50.0% | **100% one-tick**, 100% own-fit routes with n=5 own events (6, G, R)                                             |
| [0.4,0.5)         | 14  | 18.4% | **100% one-tick**, 100% pooled-parent routes with n=1-2 own events (J, M, W)                                     |
| [0.8,1.0)         | 22  | 28.9% | **100% multi-tick** (median 12.5min, max 95min), mix of own (15) and pooled (7) routes, all 6 routes represented |
| other (<8min gap) | 2   | 2.6%  | route 6, multi-tick, own-fit                                                                                     |

This is a clean, complete explanation, not a partial one. The two opposite
failure modes: **(1) one-tick outcomes always read as "model too
pessimistic"** (PIT low) — even a full n=5 own-fit sample forces a smooth
curve to spread some mass around/below the observed cluster, so it never
assigns as much probability to "already done in exactly one tick" as the
~70% base rate demands; pooled routes land closer to calibrated (~0.45)
purely because shrinkage happens to center them near the population mix.
**(2) any multi-tick outcome always reads as "model too optimistic"** (PIT
high, 0/22 in this lobe are one-tick) — a curve compressed to front-load
enough mass for the one-tick majority has too little tail left for the
~30% minority that takes longer, so by the time a longer episode actually
resolves the curve had already assigned it near-certainty. Cross-checked
against an independently-fit local model (06-21..07-04, 13 routes instead
of 6): the same one-tick-low / multi-tick-high split holds (66 one-tick
episodes land in the low/mid lobes, 0 in the high lobe; 25 of 28 multi-tick
episodes land in the high lobe) — the mechanism is structural to the
population, not an artifact of one specific fit.

### Experiment 4 — pooling

Own-fit vs pooled-parent routes, split on the exact published population
(own={6,G,R}, n=5 events each; pooled={J,M,W}, n=1-2 events each):

| split         | n routes | n scored | mean CRPS | baseline CRPS | skill  |
| ------------- | -------- | -------- | --------- | ------------- | ------ |
| own-fit       | 3        | 55       | 4.98      | 4.53          | -0.099 |
| pooled-parent | 3        | 21       | 5.68      | 5.01          | -0.133 |

Cross-checked on the independent local fit (8 own-fit routes incl. n=32/15
event routes 7/M, 5 pooled-parent routes at n=1 each):

| split         | n routes | n scored | skill  |
| ------------- | -------- | -------- | ------ |
| own-fit       | 8        | 60       | -0.094 |
| pooled-parent | 5        | 37       | -0.079 |

Own-fit and pooled-parent skill sit in the same negative band in both
cuts (pooled is worse in one, marginally better in the other) — pooling
shrinkage is not carrying the negative skill; well-observed routes with a
full own-fit sample fail almost as badly as thin, shrunk ones. This rules
out "give sparse routes more of their own data" as the fix, consistent
with Experiment 1's volume-sweep finding.

### Concrete cheapest model-form fix, measured offline (not shipped)

A one-tick point-mass p0 (global, pooled across all training-window
routes: 53/77 = 0.688 of all completed disrupted exits are exactly one
tick) mixed with the existing partially-pooled log-logistic **refit
conditional on T > 1 tick** (so the spike and the tail don't double-count
the same population), graded via `recovery_dist.recovery_dist_report`
directly on a closed-form CDF (bypassing `curve_sec` quantization, which
cannot represent a mass this large evaluated exactly at its own grid
point — see the artifact note above). Same training window, same 97-episode
held-out population as Experiment 2's model (a):

| model                                                                     | mean CRPS | baseline CRPS | skill      |
| ------------------------------------------------------------------------- | --------- | ------------- | ---------- |
| a. shipped continuous (closed-form check, same math as the curve version) | 7.04      | 6.36          | -0.108     |
| d. one-tick spike + conditional continuous tail mixture                   | 6.50      | 6.36          | **-0.023** |

A 74% reduction in the skill deficit (-0.108 → -0.023), same data, zero
extra training days, one extra parameter (the pooled one-tick fraction).
Still short of beating climatology — mean_pit=0.746 shows the mixture is
now calibrated too optimistically on average (a single global p0 over- or
under-shoots per-route), so a per-route-shrunk p0 (same partial-pooling
machinery already in `pooled_dwell.py`, applied to the binary
one-tick-or-not outcome instead of a continuous scale) is the next-cheapest
iteration, not attempted here. Not implemented in production; this is the
offline number for the lead to decide on.

Commands run (`murk exec -- uv run python -m training._scratch_*`, five
throwaway scripts under `training/`, each fetching `archive/vehicles/`
bodies for 2026-06-21..08-12 via `training.movement_backfill.
_fetch_vehicle_bodies` — cached to `/tmp` between runs, both cache files
and all five scripts deleted after use, nothing left in the repo or on
disk): `movement_backfill.route_ticks_from_vehicle_bodies` +
`transitions_from_ticks(debounce_ticks=1)` for the full-archive
reconstruction; `load_r2.compute_advance_baseline` +
`load_r2.build_movement_truth` + `episodes.extract_episodes` for the causal
130-episode population; `pooled_dwell.partially_pooled_dwell` /
`pooled_dwell.cell_from_fit` for model (a); `dwell.compute_dwell_quantiles`
for model (b); `dwell._make_cell` over pooled samples for model (c);
`training.r2_client.get_object_bytes` for the one `state/params.json`
validation pull; `scorecard.episode_recovery` /
`scorecard.movement_dwell_lookup_from_params` /
`recovery_dist.recovery_dist_report` /
`recovery_dist.predicted_recovery_curve` for every grading call, unmodified.

## 2026-08-12 — a status glyph cannot encode three states by spacing alone at 20px, and `recovery_minutes` is 0 on 22 of 30 routes

origin: agent

Branding work, but two findings bind on the feed itself and are worth keeping
whoever builds the next UI on it.

**The load-bearing one: `inference.recovery_minutes` is `0` for all 22
currently-normal routes** (30 total: 22 `normal`, 4 `not_scheduled`, 4
`unknown`). Any UI channel bound to it — a bar, a gauge, a spacing, a width —
is therefore motionless outside a disruption, i.e. dead in roughly 95% of
ticks. `Snapshot.observations` ships as an empty list, and `Observation`
(`src/momentarily/schema.py:69-82`) already reserves `kind: "headway"` /
`unit: "seconds"` for the signal that would fix this. Nothing in the repo
computes headway today. The Worker's existing GTFS-RT vehicle-position decode
(`worker/src/vehicles.ts`) already knows where trains are each tick, so
successive passings of a fixed reference stop would give it.

**Spacing fails as a categorical encoding at status-icon size.** Tested three
shape encodings for normal/disrupted/suspended at 20px with hue removed,
scanned as a route grid (`docs/brand/mark-study.html` §6):

| encoding                                        | 20px monochrome result                           |
| ----------------------------------------------- | ------------------------------------------------ |
| gap 0 / 4 / 8 units on a 24 grid (spacing only) | fails — disrupted vs suspended indistinguishable |
| gap 0 / 5 / 11 (spacing widened to the frame)   | marginal, needs a side-by-side reference         |
| split topology + symmetric height drop          | passes, read correctly cold                      |

A 4-unit gap delta on a 24 grid is 3.3px at 20px render, and judging "how far
apart" has no anchor at that size, whereas "one shape or two" and "tall or
short" do. Consequence for the existing UI: `viz/app/globals.css` signals route
state with hue-only badges (`.cond.normal|.disrupted|.suspended`) while
line 441 of the same file already states the principle — "a second encoding so
hue isn't the only signal" — for legend markers. The badges don't honour it.
A shape channel is the fix, not a fourth colour.

Also negative, same file: a 1.5-unit accent inside the gap disappears by 32px
(1.25px at 20px); rotating the pair 90° is dimensionally truer to a train seen
from above but collapses into an equals sign by 32px.

**Hue is not a free channel here.** Route colours arrive at runtime from GTFS
static (`derive.py:179`, fallback `viz/lib/feed.ts:36`) and occupy 11 values:
`#EE352E` `#00933C` `#B933AD` `#FF6319` `#2850AD` `#FCCC0A` `#996633`
`#6CBE45` `#A7A9AC` `#808183` `#1F4F9F`. Red, orange, yellow, green, lime,
brown, blue, navy, magenta-purple and two greys are all spoken for by lines the
dashboard renders beside any product chrome.

**`segment_flow` is a sample, not a strip.** 56 segments, max 4 per
(route, direction), non-contiguous when chained on `from_stop`→`to`, and all
`normal` at time of writing. An ordered per-route station strip needs a GTFS
static stop-ordering join; it is not derivable from one snapshot.

Corrected assumption worth recording: a scheduled-headway baseline does **not**
need a new ingest. `training/gtfs_static.py` already streams `trips.txt` and
`stop_times.txt` off the static feed (lines 101 and 116) for segment topology,
so arrival times are already being read and a baseline is a parse extension.
Emitting it per (route, direction, service period) alongside the other weekly
fit artifacts is enough to express observed headway as a deviation.

## 2026-08-12 — mixture shipped: CRPS skill -0.139 -> -0.061, and two estimator bugs the offline prototype had hidden

origin: agent

The one-tick point-mass mixture is now the published movement dwell model
(`atom_p`/`atom_sec` on the dwell cell, tail_ll refit left-truncated at one
tick). Graded causally — trained 06-21..07-25, scored on 74 held-out
episodes from 07-26..08-11, same transitions and same grader for every
variant, only the cell construction differing:

| model                            | CRPS  | baseline | skill       | mean PIT |
| -------------------------------- | ----- | -------- | ----------- | -------- |
| continuous (what shipped before) | 7.451 | 6.541    | -0.1391     | 0.579    |
| single global atom               | 6.922 | 6.541    | **-0.0582** | 0.545    |
| per-route shrunk atom (shipped)  | 6.938 | 6.541    | **-0.0606** | 0.544    |

56% of the skill deficit gone. Still short of climatology, so the arm stays
ungraduated. Global and shrunk are within noise of each other because the
concentration estimator lands on its ceiling — the routes are not
distinguishable at these counts, and it says so rather than pretending. The
shrunk form ships because it degenerates to global exactly when the data is
thin and separates on its own once it is not.

### Two estimator bugs the offline prototype never exposed

Both were invisible in the earlier one-off because it used a single global
rate. Going per-route surfaced them, and the first attempt at per-route
scored **worse than doing nothing** (-0.1459 vs -0.1391 continuous).

**1. Median of per-route ratios is biased upward at small denominators.**
Following the scale estimator's one-vote-per-route convention gave a
population atom rate of **0.789** where the pooled ratio of totals was
**0.733** (99 one-tick exits / 135 completed). A route that saw one episode
and it was one tick votes 1.0 with the same weight as a route that saw
thirty. Every cell then shrinks toward an inflated centre and the whole
forecast reads optimistic. Fixed by using the ratio of totals for the atom
centre — a deliberate departure from the scale's convention, for a reason
that only applies to ratios of small counts.

**2. The Beta moment inversion inverts its own purpose at small n.**
`hierarchical.robust_concentration` reads the observed spread of per-cell
rates as if it were all real between-cell variation. That holds at the 20+
trials its leaves carry. At the ~5 episodes a dwell cell carries, binomial
sampling noise dominates: a route at 1-of-1 and a route at 3-of-5 look
wildly dispersed while being perfectly consistent with one shared rate. The
estimator concluded "these routes differ", returned the kappa floor, refused
to pool, and let per-route rates run to **0.957**. Var(observed) =
Var(between) + E[p(1-p)/n]; subtracting the sampling term before inverting
fixes it, and when nothing survives the subtraction the answer is to pool
hard. That single change moved per-route from -0.1459 to -0.0606.

Left `robust_concentration` alone — it is correct in its own high-n regime —
and put the corrected variant next to the estimator that needs it, with a
note on each pointing at the other.

### PIT is not a valid diagnostic against a point mass

Worth internalising, because it looks exactly like a catastrophic
regression. With an atom at 300s, every one-tick episode returns the
identical F(300) = atom_p, so the histogram collapses:

| model                   | PIT histogram            | mean PIT |
| ----------------------- | ------------------------ | -------- |
| continuous              | [0,1,7,15,25,0,1,1,8,16] | 0.579    |
| mixture, raw PIT        | [0,0,0,0,0,0,0,48,13,13] | 0.790    |
| mixture, randomized PIT | [8,8,4,8,6,6,6,2,13,13]  | 0.544    |

The middle row is not miscalibration. PIT is only uniform for a _continuous_
predictive CDF; against a jump the correct diagnostic spreads each
observation across its own jump. Uncorrected, the mixture would have been
published with a "badly optimistic" verdict on the Models page while being
better calibrated than the model it replaced — the bottom row is closer to
uniform than the continuous row on every measure. The residual is the top
two bins: the multi-tick tail is still under-dispersed, which is the next
thing to work on.

Randomization is keyed on an FNV-1a digest of the regime, not an RNG, so the
grade stays reproducible and does not depend on iteration order. FNV-1a
rather than the blake3 already in `provenance.py` because the viz grading
route recomputes this report and blake3 is not in `node:crypto` — measured
across 74/1000/20000 realistic keys, FNV-1a, blake2b and blake3 are all
indistinguishable from uniform (KS 0.0845/0.1004/0.0997 at n=74 against a
0.1581 critical value), so there was no quality to buy with a dependency.

### Incidental

`dwell_cdf` returned 0 at `x <= curve_sec[0]`, and `curve_sec[0]` is exactly
one tick for any cell whose shortest dwell is one tick — so the CDF read 0
evaluated at its own grid point and every one-tick episode graded PIT=0. The
lower guard is now strict and a repeated knot resolves to the top of its
flat run. Numerically identical on strictly increasing curves, which is
pinned by a test using the old formula as an oracle.

A dwell cell gaining a field is silently dropped by the Worker: the zod
schema is a plain `z.object()` with strip semantics, so `atom_p` would have
been deleted at parse time and the fix would have trained, published and
graded with the old math while looking shipped. `n_censored` was already
being dropped that way. Both are now in the schema, and a golden fixture
(`tests/fixtures/parity_dwell.json`, 42 cases) is replayed by Python, the
Worker and viz so the three cannot drift again.

## 2026-08-12 — gated the 60/120min horizons; the publish guard would have emptied the feed

origin: agent

Stopped publishing `p_normal_in_60min` and `p_normal_in_120min`. They are now
`null` on every fitted-curve row. `p_normal_in_30min` is untouched and still
always published — it is the one horizon with repeatedly measured skill
(BSS +0.29 to +0.40). The two longer ones lose to naive persistence in every
population cut tested (BSS -0.00 to -1.30; AUC 0.395 and 0.352, i.e. inverted),
the loss is not a left-censoring artifact, and the projection was verified
monotone over 45,600 comparisons — so the defect is the shape of the fitted
elapsed-conditional dwell curve and no amount of runtime fixes it. Publishing a
number we have measured to be anti-informative is worse than publishing nothing.

Chose nullability over a `recovery_indeterminate`-style sibling flag. A flag
leaves the inverted number in the payload for any consumer that ignores the
flag; null cannot be misread. This follows the existing "never a fabricated
number" precedent (`SegmentStatus.recovery: SegmentRecovery | None`,
`movementRecovery()` returning null) rather than the clamp-plus-flag one.

**The schedule arm is exempt, and that is not a detail.** When
`recovery_source == "schedule"` the value is `resume <= now + h*60` — a
comparison against an announced resume time, not a fitted forecast. It carries
none of the defect above, so it keeps publishing all three horizons. A blanket
suppression would have thrown away the only horizon information in the feed that
is _deterministically_ correct. Gated at the single point where the Inference is
assembled rather than in each of the five arms that set these values, so the
next arm added cannot silently miss the gate.

### The near-miss worth remembering

`scrubCorruptInferences` null-checks every inference field with
`Number.isFinite`, and `Number.isFinite(null)` is `false`. Shipping the gate
without touching that function would have scrubbed EVERY route's inference on
EVERY tick — the guard that exists to stop one bad route from staling the feed
would have emptied the feed of exactly the data it protects. Caught by its own
existing regression test. The distinction the function now draws: absent is a
valid published state, not-a-number is not.

Two other consumers had to learn the difference. `training.eval` parsed with
`float(raw["p_normal_in_60min"])`, which raises on null; it now uses an explicit
`_opt_float` and the calibration loop skips withheld rows the same way it already
skips `schedule` rows. Deliberately NOT `float(value or 0.0)` — a withheld
forecast coerced to 0 scores as a confident "will not recover" and drags every
bin it lands in. The dashboard had the same trap in reverse: `null * 100`
renders `0%`, so a withheld horizon would have displayed as a confident zero;
it now renders "not forecast".

`SegmentRecovery` keeps all three horizons non-null. The measurement was taken
on the route-level fields, and the segment arm is a different fitted model
(`segment_dwell.json`); gating it would be asserting evidence we do not have.
Filed separately rather than assumed.

Eight worker tests asserted properties of the two withheld horizons. None were
deleted: the monotonicity and probability-bounds invariants moved down onto
`pLeaveBy`/`staysNormalFor` in `dwell.ts`, where they still hold and where the
property actually lives. It stopped being an invariant of the published field,
not an invariant of the model.

## 2026-08-12 — advance baseline graded held-out at last: partial pooling loses to the raw leaf rate, and badly on thin leaves (Brier 0.125 vs 0.069)

origin: agent

First out-of-sample grade of `hierarchical.partially_pool`'s advance-rate
baseline. Trained on 2026-06-21..07-25, scored against the Bernoulli
advance/stall trials observed in 07-26..08-11, 2,375 leaves present in both
windows, 2.2M held-out trials. Every predictor clipped identically at P0_FLOOR
so log-loss is comparable.

**Target choice, because it changes what the number means.** `p0` is not a
forecast of advance; it is the healthy-baseline null `classify_direction` tests a
live tick against, so scoring it on all held-out ticks would let any predictor
win by baking in the disruption rate. Graded set is therefore restricted to
held-out ticks where the route was in the `normal` movement state, with route
truth reconstructed causally from the training window's direction baseline. In
the event this barely matters — the held-out window is **108,657 normal
route-ticks against 196 disrupted**, so the two targets agree to within 0.0008
Brier. Recording it because the objection is correct in principle even though it
did not bite here.

| predictor                           | Brier       | logloss | vs pooled |
| ----------------------------------- | ----------- | ------- | --------- |
| pooled (what ships)                 | 0.06519     | 0.23645 |           |
| raw leaf rate                       | **0.06414** | 0.23453 | -0.00105  |
| route level                         | 0.18876     | 0.54609 | +0.12357  |
| system level                        | 0.24298     | 0.67903 | +0.17779  |
| pooled, fitted on normal ticks only | 0.06505     | 0.23640 | -0.00014  |

Restricted to the 497 THIN training leaves (n_A < MIN_LEAF_N=20) — the leaves
partial pooling exists to serve:

| predictor     | Brier       | logloss | vs pooled |
| ------------- | ----------- | ------- | --------- |
| pooled        | 0.12451     | 0.43691 |           |
| raw leaf rate | **0.06908** | 0.36334 | -0.05543  |
| system level  | 0.19412     | 0.58022 | +0.06960  |

So: pooling is a small net loss overall and a **1.8x Brier loss exactly where it
is supposed to help**. It does beat route- and system-level pooling by a wide
margin (0.065 vs 0.189 and 0.243), so leaf specificity is doing real work — the
hierarchy _above_ the leaf is nearly worthless here. It is the direction of the
shrinkage that is wrong, not the idea of a leaf-level estimate.

### Why: exchangeability is violated, and not in the direction I guessed

|                                      | held-out advance rate |
| ------------------------------------ | --------------------- |
| thin leaves (n_A < 20)               | 0.8532                |
| voting leaves (n_A >= 20)            | 0.5830                |
| centre thin leaves are shrunk toward | **0.9888**            |

Thin leaves run **higher** than well-observed ones, not lower. Plausible reading:
a rarely-traversed segment has few trains on it, so nothing queues and trains
simply advance; the busy segments that accumulate large n are exactly where
trains stall behind each other. Either way, how much data a leaf has is
informative about its rate, which is the assumption partial pooling rests on. I
had predicted the opposite sign (guessing thin leaves were terminals, which
structurally dwell) — that was wrong, and the deciles say so.

The shrinkage target overshoots even so: thin leaves truly run 0.853 and are
pulled to 0.9888. Assigning 0.99 where the truth is 0.85 is what the Brier gap
is made of, and it points straight at the open terminal over-flagging bug — a p0
of 0.99 puts the `0.5*p0` trip-wire at 0.494 and sets the binomial null at 0.99,
so a modest stall streak reads as wildly significant. That link is now measured
rather than speculated.

### Negative result: the missing disruption resistance is not the problem

Unlike its direction-level sibling `compute_advance_baseline`, which takes a
per-tick MEDIAN specifically so a line that mostly runs well keeps a high
baseline through occasional frozen stretches, `build_segment_baseline` simply
SUMS advanced/stalled over the window — the segment leaf baseline has no
disruption resistance at all. Refitting it on normal-only ticks was the obvious
candidate fix and it does essentially nothing: **-0.00014 Brier overall and
+0.00067 on thin leaves** (i.e. very slightly worse there). Worth knowing before
anyone spends time on it.

### Correcting an earlier claim in this journal

The entry two above reasoned that the median-of-fractions bias found in the dwell
atom estimator would be "much smaller" here because a leaf must clear n>=20 to
vote, so its fraction is less coarse. Measured, that reasoning is wrong: the gap
between the median of leaf rates and the ratio of totals is **+0.3985** at system
level and the median is higher in **52 of 52** route_direction groups (mean gap
+0.3797). The mechanism is not coarse denominators at all, it is n-weighting —
most leaves advance nearly always, while the trials are dominated by a few busy,
stall-heavy leaves. Which of the two is the "right" centre is a genuine modelling
question (a prior over leaves is not a prior over trials), so this is not a bug
on its own; what the held-out numbers do say is that 0.9888 is too high for the
leaves that borrow it.

Vote-gate clearance, previously asserted but never measured: **1,884 of 2,508
leaves (75.1%)** clear MIN_LEAF_N=20, per-leaf n has p10=2, p50=349, p90=1171,
max=29,487, and 2,505 of 2,508 cells take their prior from the route_direction
level (only 3 fall back to system). So the gate is not the fragile part.

206 held-out leaves have no training baseline at all and the classifier abstains
on them.

Commands run (`murk exec -- uv run python <script>` with PYTHONPATH at the repo
root; two scripts, both under /tmp only, deleted after the run, nothing left in
the repo or on disk): `movement_backfill._fetch_vehicle_bodies` for both windows
(cached to /tmp between runs, cache deleted), `load_r2.build_segment_series`
aggregated per leaf per tick, `hierarchical.partially_pool` for every pooled
variant, `load_r2.compute_advance_baseline` +
`load_r2.build_movement_series_by_direction` + `load_r2.build_movement_truth` for
the causal normal-tick mask. No production code modified.

## 2026-08-12 — 81.8% of our "segments" are multi-station jumps: 5-minute polling cannot see station-to-station movement

origin: agent

The archive records, per 5-minute tick, that a trip was at stop A last look and
stop B this look, and we treat (A,B) as a segment. Measured how far apart A and B
actually are in the scheduled stop order (`gtfs_static.chains` gives travel-order
stop sequences per route+direction; hop = index difference), over
2026-08-05..08-11, 915,906 transitions, 98.6% of them placeable in the scheduled
order:

| stops covered in one 5-min look | count   | share of moves |
| ------------------------------- | ------- | -------------- |
| 1 (an actual segment)           | 90,159  | **18.2%**      |
| 2                               | 158,975 | 32.2%          |
| 3                               | 192,192 | 38.9%          |
| 4                               | 42,354  | 8.6%           |
| 5+                              | 8,612   | 1.2%           |

**81.8% of observed moves skipped at least one station we never saw.** Mean 2.71
stations per look. Worst on the busiest lines: `1` 94.8% multi-stop, `6` 89.4%,
`L` 88.2%, `3` 78.4%.

So the segment-level view is not measuring segments. It is measuring "a train was
somewhere near A, then somewhere near A+2.71", and `canonical_adjacency` then
labels that jump with its most common landing stop — which is exactly why the
labels look plausible while being wrong. Any per-segment statistic built on this
is a blend of 2-4 real segments, and the stations in between contribute nothing at
all.

This is the root cause under the advance-baseline problem graded in the entry
above, and it is not fixable by better shrinkage. A yes/no "did it leave" carries
almost no information per observation, which is why thin leaves must borrow, which
is why the borrowing being miscalibrated matters. A traversal DURATION carries far
more per observation.

### Two anomalies in the same numbers, both unexplained

**42.5% of placed transitions are stalls** (stop_id unchanged across 5 minutes).
That is hard to reconcile with a moving train covering 2.71 stations per look: a
train in revenue service should change stop_id nearly every tick. So the
advance-rate p0 ~ 0.59 that the entire movement classifier rests on may be
dominated by something other than "trains are running well" — layups, held
trains, terminal relays, or stale feed entries. Not investigated here. Worth
knowing before anyone tunes the classifier, and it is a live lead for the terminal
over-flagging bug.

**2.9% of moves read as backwards** in the scheduled order. Candidates: terminal
relays where a train reverses, the stop_id N/S suffix disagreeing with the trip's
actual direction, or trip_id reuse. Also not investigated.

### What this justifies

Polling faster alone is not the fix. At 1-minute polling a station hop (~1.85 min
at the observed speed) still lands in 1-2 ticks, so a tick count would only pin
arrival to +/-30s on a ~110s journey — rebuilding a coarse measure with 5x the
storage. The resolution is already in the feed and being discarded:
`VehiclePosition.timestamp` (the train's own measurement time) is not decoded, and
`current_stop_sequence` IS decoded and tested but unused, and is present exactly
when a train is STOPPED_AT. Poll rate then governs how often we CATCH a
transition, not how precisely we time it.

Note for whoever wires this: raising the cron to 1 minute without gating silently
redefines every existing model. `advanced_n`/`stalled_n` mean "changed in 5
minutes", the advance baseline is fitted on that, and the dwell model's one-tick
point mass IS 5 minutes. The same handler at 1 minute keeps producing plausible
numbers against params that assume the old meaning. The 5-minute pipeline has to
stay gated to 5-minute boundaries with its own carry object, and the trace needs
its own.

Commands run (`murk exec -- uv run python /tmp/hop_test.py`, PYTHONPATH at the
repo root; script and its caches under /tmp only, deleted after):
`movement_backfill._fetch_vehicle_bodies` for the window,
`gtfs_static.load_successors` + `gtfs_static.chains` for scheduled travel order,
`load_r2.build_segment_series` for the observed transitions. No production code
modified.

## 2026-08-12 — two-thirds of the stall mass is terminal layover; and the alert feed cannot grade a movement signal (4 acute onsets in a week)

origin: agent

### Where the 42.5% stall rate comes from — answered

Stalls per (route, direction, from_stop) over 2026-08-05..08-11, cross-referenced
against each stop's position in its scheduled travel-order chain:

| position in the line         | stalls  | moves   | stall rate | share of all stalls |
| ---------------------------- | ------- | ------- | ---------- | ------------------- |
| first stop (origin terminal) | 244,009 | 28,261  | **89.6%**  | **63.5%**           |
| destination terminal         | 12,419  | 2,434   | 83.6%      | 3.2%                |
| one in from a terminal       | 6,324   | 19,611  | 24.4%      | 1.6%                |
| mid-line                     | 96,273  | 474,532 | **16.9%**  | 25.1%               |
| not in a scheduled chain     | 24,953  |         |            | 6.5%                |

**Two-thirds of the stall mass is trains parked at terminals by design.** It is
also extremely concentrated: the top 25 (route, direction, stop) leaves hold
65.6% of all stalls, top 50 hold 87.5%. The worst are stationary by construction —
`H north H19N` (Rockaway shuttle) 15,974 stalls to 6 moves, `7 south 726S` 2,421
to 1, `D south D01S` 7,141 to 193.

So `p0 ~ 0.59`, the number the entire movement classifier rests on, is a blend of
two physically different populations, dominated by the parked one. The in-service
figure is 1 - 0.169 = **0.831**, which independently matches the 0.8532 held-out
rate measured for thin leaves two entries above — thin leaves are just ordinary
mid-line stops.

This also nails the terminal over-flagging mechanism rather than guessing at
it: a terminal genuinely stalls ~90% of the time, borrows a pooled p0 near 0.99
from its mid-line neighbours, and its normal behaviour then reads as
overwhelmingly significant evidence of disruption.

### Progress ratio: built, and NOT yet gradeable

Replaced the advance-rate binary with a scheduled-progress measure. For each
train per 5-minute tick: `scheduled_seconds(path actually covered) / 300`. 1.0 is
on time, 0.5 is half speed, 0 is not moving. Terminals excluded per the above.
Scheduled run times come from a new `gtfs_static.hop_seconds` (median over trips
of departure-to-arrival, so scheduled dwell is not counted as travel): 1,932 hops,
median 90s, mean 122.6s. That cross-checks against the observed 300/2.71 = 111s
per hop, so moving trains run slightly ahead of the timetable — consistent with
schedule padding.

The measure itself behaves sanely: median 0.818, p05 0.092, p95 1.056.

Graded against MTA severe-alert truth (severity floor 2), 41,413 route-ticks,
AUC where a low score should mean disrupted:

| score                                               | AUC    |
| --------------------------------------------------- | ------ |
| advance rate, as production computes it (all stops) | 0.2175 |
| advance rate, terminals excluded                    | 0.3899 |
| progress ratio, terminals excluded                  | 0.4229 |

All three are INVERTED. Excluding terminals is worth +0.172 AUC on its own, which
is the single largest effect measured today and comes straight from the stall
diagnostic above. The progress ratio then adds +0.033 on top. But everything is
still the wrong side of 0.5, so none of it is evidence of anything yet.

**Ruled out: time of day.** Alert prevalence is 11.2%-13.0% in EVERY one of the 24
UTC hours, and AUC computed within-hour is essentially unchanged (advance 0.3868,
progress 0.4207). Not a confound.

**The label is the problem, and the flatness is the proof.** No real disruption
pattern is uniform across all 24 hours. A 12% rate at 4am and at 6pm means the
truth is dominated by chronic standing advisories, exactly as the chronic-alert
finding above records, so it labels thousands of normally-running ticks
disrupted and cannot separate anything.

Confirmed by trying the acute version — label a CALM route-tick positive if a new
alert onset (normal -> not-normal transition) follows within 30 minutes:

- **4 acute onsets in the entire week, across 4 routes**, giving 24 positive
  route-ticks out of 36,297.
- AUC 0.6025 progress ratio vs 0.5744 advance rate. Both now the right side of
  0.5, and the progress ratio's mean is 0.0391 LOWER before an onset while the
  advance rate moves +0.0019 (i.e. not at all).
- With 24 positives the CI on AUC is about +/-0.12. This is a direction, not a
  result. Do not cite it as one.

So the honest state: the progress ratio is a better-posed measure on every
structural argument, it edges the advance rate on every cut tried, and there is
currently **no label in this repo capable of grading it**. That is the blocker, not
the model.

### Known flaw in the measure as built

Skip-stop inverts it. `path_seconds` sums the scheduled chain between the observed
endpoints, so a train that BYPASSES stops during a disruption is credited with
covering all of their scheduled time and reads as running fast. Pre-onset ticks
average 2.266 hops per move; the normal-tick comparison came back NaN through a
bug in the scratch script and was not re-run. Needs handling before this ships —
probably by comparing against the trip's OWN scheduled pattern from stop_times
rather than the route's modal chain, which is available and would also fix
express/local sharing track.

### What would actually grade it

Nothing derived from the alerts feed at this severity floor. Candidates, in order
of independence: trip-updates `assigned_n` collapses as an acute service signal;
the 1-minute trace once it accumulates, giving real per-segment traversal times to
compare against schedule directly; or hand-graded incident windows (1ul already
exists for the Knicks-parade window).

Commands run (`murk exec -- uv run python /tmp/{stall_why,progress,progress2,progress3}.py`,
PYTHONPATH at the repo root; four scratch scripts and their caches under /tmp only,
all deleted after): `movement_backfill._fetch_vehicle_bodies`,
`load_r2.build_segment_series`, `gtfs_static.load_successors` / `chains` /
`load_hop_seconds`, `review.build_mta_truth`. Production code changed: only the
new `gtfs_static.hop_seconds` / `load_hop_seconds` (committed, tested).

## 2026-08-12 — the advance signal needs an admission rule, not a terminal blacklist: through-stops-only cuts thin-leaf Brier 0.149 -> 0.081

origin: agent

### Terminals were the larger half of the problem, not all of it

Chain endpoints read off the dominant-successor skeleton directly (a stop with no
incoming dominant edge, or none outgoing) rather than off `chains(...).stops[0]`,
which concatenates one walk per entry stop and so names only one origin when a
component has two. 199 endpoint keys against 1,659 through stops. Recounted over
2026-08-05..08-11, keyed on each transition's from_stop:

| position                   | stalls  | moves   | stall rate | share of all stalls |
| -------------------------- | ------- | ------- | ---------- | ------------------- |
| chain endpoint             | 294,928 | 36,316  | **89.0%**  | **76.8%**           |
| mid-line (through)         | 64,103  | 488,522 | 11.6%      | 16.7%               |
| not in the skeleton at all | 24,947  | 7,090   | **77.9%**  | 6.5%                |

The third row is the finding. Stops the timetable never names — yard leads, rare
patterns, stop_ids the vehicle feed emits and stop_times doesn't — stall at 77.9%,
which is terminal behaviour, not mid-line behaviour. Excluding terminals alone
leaves those 24,947 stalls in the admitted population: 28% of what remains after
the endpoints go. So the rule that matters is positive, not subtractive: count a
trip only where the schedule defines what moving means, i.e. the from_stop has
both a scheduled predecessor and a scheduled successor.

### Held out, that is worth more than the terminal cut alone

Train 2026-06-21..07-25, score 07-26..08-11 on the route-ticks the severity-2
alert truth calls normal. Pooled = `hierarchical.partially_pool`, raw-leaf = each
leaf's own training rate.

| arm                    | leaves | no train data | thin   | pooled Brier | thin-leaf pooled | thin-leaf raw |
| ---------------------- | ------ | ------------- | ------ | ------------ | ---------------- | ------------- |
| all stops (production) | 2291   | 177           | 448    | 0.06656      | 0.14857          | 0.05506       |
| terminals excluded     | 2107   | 174           | 436    | 0.06034      | 0.11144          | 0.05769       |
| through stops only     | 1617   | **0**         | **69** | 0.06208      | **0.08082**      | 0.07788       |

Read the last three columns, not the fourth: overall Brier is dominated by the
largest leaves and each arm scores a different population, so cross-arm overall
comparison is not like-for-like (terminals-excluded "wins" it at 0.06034 while
being worse everywhere the estimator is actually load-bearing). What is
like-for-like:

- **Every held-out leaf has training data** under through-only — 177 leaves
  that previously needed a fabricated prior are gone, because they were
  off-skeleton stops that appear in one window and not the next.
- Thin leaves fall 448 -> 69 and their Brier 0.14857 -> 0.08082.
- Pooled and raw converge (0.06208 vs 0.06195, and 0.08082 vs 0.07788 on thin
  leaves). The exchangeability violation measured in the earlier grade —
  pooling _losing_ to raw by 1.8x on the leaves it exists to serve — was mostly
  terminals and off-skeleton stops sitting in one hierarchy with through stops.
  Most of what the n-aware-shrinkage work was for dissolves with the population
  fixed instead.

Reproduction check, since the harness is new: the all-stops arm scored on all
ticks gives pooled 0.06598 / raw 0.06472 over 2,378 leaves and 2.21M trials, 499
of them thin, against the 0.06519 / 0.06414 / 2,375 / 2.2M / 497 recorded
earlier. Same estimator, same finding.

### AUC against the alert label: ordering reproduces, magnitude does not

Same window, severity floor 2, 45,292 route-ticks, prevalence 12.4-14.7% in every
one of the 24 UTC hours (the flatness the label was already condemned for):

| arm                | AUC    | mean score on normal | on disrupted | overall advance rate |
| ------------------ | ------ | -------------------- | ------------ | -------------------- |
| all stops          | 0.3388 | 0.5619               | 0.7215       | 0.5807               |
| terminals excluded | 0.4361 | 0.8217               | 0.9112       | 0.8470               |
| through stops only | 0.4641 | 0.8673               | 0.9099       | 0.8832               |

Ordering and sign match the 0.2175 -> 0.3899 pair recorded earlier; the absolute
values do not (different route-tick population, no presence mask). All three are
still inverted, so this table ranks the arms and grades nothing — the label is
still the blocker.

### Negative result to carry forward: p0 now saturates

The flat per-(route, direction, tod) baseline's median p0 moves 0.5333 -> 1.0000
once layovers are out. That is arithmetically right — at 5-minute cadence a
through stop's modal tick has every matched trip advancing — but it breaks the
classifier's design intent. `DISRUPTED_RATIO * p0` was meant to judge each line
against its own normal; with every cell's normal at ~1.0 the trip-wire collapses
to a global "under 50% advancing", which is the single global cutoff the
baseline-relative design existed to replace. The significance gate still fires on
the same evidence, so this is not a correctness bug, but the ratio form is now
doing nothing and should be reconsidered against a saturated baseline.

### What shipped

`gtfs_static.terminals` / `through_stops` / `stops_to_json`; a `counts_from_stop`
filter on `load_r2.build_movement_series{,_by_direction}` that recomputes
advanced_n/stalled_n from the raw `transitions` map (the archived counters are
already summed and cannot be narrowed after the fact); the trainer fitting the
advance baseline on through stops and publishing that same set as
`movement_through_stops` in params.json, from one static-timetable fetch per run
so the fit and the set cannot disagree; the Worker, viz and the offline
reconstruction paths counting the same way. The archived `transitions` map stays
raw and unfiltered — it is the only place terminal behaviour remains observable,
and it is what lets one definition apply to the whole archive instead of the
counters changing meaning at deploy.

Measured with two throwaway scripts at the repo root (both deleted after) over
`archive/vehicles/` and `archive/alerts/` via `murk exec`.

## 2026-08-12 — the 1-minute trace reads back: 3,457 single-hop traversals in 25 minutes, tracking the timetable within 3%

origin: agent

Deployed the per-minute trace (ddefa71) and built the reader. Trace objects land
at exactly 60s spacing, ~650 rows each, while archive/vehicles/ stays on its
300s cadence — the 5-minute pipeline's gate holds.

First reconstruction, over 25 minutes of live data (572 trips, 16,196 rows,
4,337 arrivals):

| outcome                                          | n     | share |
| ------------------------------------------------ | ----- | ----- |
| exact single hop                                 | 3,457 | 91.9% |
| interval-censored (arrival missed between polls) | 215   | 5.7%  |
| right-censored (in transit when last seen)       | 89    | 2.4%  |
| dropped: stop_seq backwards                      | 11    |       |
| dropped: arrival with no stop_seq                | 82    |       |

**92% of hops are cleanly measured at 1-minute polling.** That is the number the
whole trace was built for: at 5-minute polling the mean observed move spanned
2.71 stations, so nothing below the multi-station jump was measurable at all.

### Two bounds, not one measurement

A poll brackets a hop rather than pinning it, and the two ends are biased in
opposite directions:

| cut                  | n     | ratio to schedule (median) | p10   | p90   |
| -------------------- | ----- | -------------------------- | ----- | ----- |
| arrival -> arrival   | 3,420 | **1.033**                  | 0.667 | 1.667 |
| departure -> arrival | 797   | 0.422                      | 0.189 | 0.833 |

Arrival-to-arrival includes the dwell at the origin, so it is an upper bound.
Departure-to-arrival starts its clock at the first sighting that had already
left, which is systematically late, so it is a lower bound — and a weak one:
only 1,014 of 3,761 traversals catch the train in transit at all, and those skew
toward the slow hops that are easiest to catch. The upper bound is the usable
measurement, and it lands 3% over the timetable.

**Negative result: `hop_seconds` does not need a dwell correction.** The obvious
objection to comparing arrival-to-arrival against a departure-to-arrival
timetable is that it double-counts station dwell. Measured instead of assumed:
NYCT sets arrival_time == departure_time on 95.7% of stop_times rows (200k
sampled), median 60s on the 4.3% that differ. The schedule allocates no dwell,
so the scheduled hop IS its arrival-to-arrival time and the comparison is
like-for-like. Real trains still spend ~30s standing at each stop that the
timetable never allocates, which is most of the gap between the 120s observed
median and the 90s scheduled one.

### Reconstruction rules that cost real effort to get right

Two of these were wrong in the first draft and produced plausible-looking
numbers anyway, which is the reason they are written down:

- **Right-censoring needs in-transit evidence, not just absence.** A trip last
  seen standing at a stop FURTHER ALONG than its last recorded arrival has
  already completed the hop — we simply could not time it (the feed omitted
  stop_seq). Recording "still running at T" there is false, not unverified,
  and stretches every fitted traversal. The rule that holds: censor only when
  the last sighting is in transit toward a stop other than the last arrival.
- **A departure observed at the same instant as the next arrival is not a
  departure.** It bounds travel time below by zero, which is no information.
  Emitting 0 there would drag any fit down hard; those records carry
  moving_seconds=None instead, and 2,443 of 3,457 exact hops are in that state.
- A feed gap after the last in-transit sighting does not weaken the censoring
  bound — the train demonstrably was still moving when last seen, and the
  endpoint is the feed's own vehicle_ts, not a poll time. A gap that swallows
  an arrival surfaces as a stop_seq jump instead, which is interval-censored.

Interval-censored spans are dropped from the survival samples rather than split
evenly across their hops: the fitters in survival.py carry a right-censored
likelihood only, and "each of these 2 hops took at most 200s" is an upper bound
they cannot express. Splitting would fabricate observations. At 5.7% of
traversals the discard is small, and TraceStats reports it.

Reproducible: `murk exec -- uv run python -m training.trace`.

## 2026-08-12 — the through-stops retrain landed, but three silent faults sat between the code and the artifact

origin: agent

The through-stop counting fix had been deployed and inert since 8e88644: the
Worker falls back to counting every stop until params.json carries
`movement_through_stops`, and the retrain is manual. Publishing it turned out
to require fixing three things first, none of which announced themselves.

### 1. `dwell_movement` had silently gone to zero cells

The first dry run reported `dwell_movement_cells: 0` against the 25 routes the
live params carried. One line on stderr: `movement dwell skipped (math domain
error)`, swallowed by the fail-soft wrapper that exists so an archive hiccup
can't block a publish.

The cause is numerical, not statistical. `pooled_dwell` fits a shared shape
across routes and then sweeps each route's log-scale over e^4..e^15.9 by golden
section. When a state's episodes are all one tick long — two disrupted samples
of 294s and 300s — the shared shape fits at **172.95**, and at the top of that
sweep `(294/86679)**172.95` underflows to exactly 0.0. `math.log(0)` raises.

Fixed by evaluating the likelihood through the logarithm: `log_z = shape *
(log t - log scale)`, with `log(1+z)` as a softplus so neither end can overflow.
`training/dwell._loglogistic_survival` had the mirror-image bug — at that shape
`(t/scale)**shape` raises OverflowError in Python while the TS twin in dwell.ts
quietly yields Infinity, so the two sides disagreed only on the inputs that
crash. After the fix: 32 cells, and the published block is back to 25 routes.

The general lesson is about the guard, not the math: `t = max(t, _MIN_DURATION)`
keeps `t` positive, which looks like it protects `log(t)` and does — but `z` is
a _power_ of a ratio and has its own range. Flooring the input does not bound
the intermediate.

### 2. Every local retrain since 2026-08-10 stamped a stale `code_sha`

The published params claimed `code_sha: fa513e5`, a commit from 2026-07-17,
with `dirty: null` — for a run that produced a feature that commit predates.
The previous live params said the same thing.

`.build-sha` is a gitignored file that `trainer deploy:ci` writes into the
checkout before the container build, because the image excludes `.git`.
`code_provenance()` consulted it _above_ `git rev-parse HEAD`, so once a local
deploy has run, the file shadows the live tree forever and `dirty` can never be
computed. Reordered: env var, then the live git tree, then the file, then
unknown. The file is the container's only source; in a checkout the tree that
actually ran the code is both more accurate and the only thing that can answer
whether it was dirty.

### 3. The Worker's committed movement-transition stream is not yet a source

`_movement_dwell` preferred `v1/movement_transitions` whenever it was non-empty,
falling back to replaying `published_condition` ticks. That stream started
filling at the deploy that introduced it, so the preference flipped this week —
silently, on presence alone. Over 2026-07-30..08-12 it holds **13 route
transitions against the replay's 124**; episode census 3 vs 28, and cells with
enough completed episodes to speak for themselves 0 vs 3.

The two sources do agree: 10 of the 13 match a replay record on route, target
state and onset within one tick. Exact-key agreement is 0/28, because the two
debounce the onset differently — which is why the loose matcher is the honest
one here. So this is a change of source, not of signal, and it is still a model
change that has to be graded rather than triggered by a deploy. Pinned to the
replay; the switch is its own piece of work.

### The publish

`v1786578716` over 2026-07-30..08-12: 28/28 routes, `movement_through_stops`
with **1,659 stops across 25 routes**, dwell_movement 25 routes (32 cells, 7
with an atom), baseline 210 cells. Live snapshot at the next boundary tick:
26/30 routes movement-sourced, the 4 `unknown` being GS, 7X, 6X and FX. No
`advance counters include every stop` line in the Worker log, which was the
whole point.

Two fits that the through-stops commit had left unscoped are now scoped by the
same set: `write_segment_params`' per-leaf baseline (1,658 cells, one per
published through stop) and `write_segment_dwell`'s vehicle reconstruction.
They were fitting over layovers while the Worker scored against a through-stop
normal.

**Still false in the live artifact:** `v1786578716` was published before the
provenance fix, so it carries the stale sha. The corrected stamp needs one more
publish from a recorded commit.

## 2026-08-12 — the degradation label was mostly measuring the clock: hourly bins cut degraded prevalence 11.0% -> 1.8%

origin: agent

The assigned_n degradation label is meant to be the truth a movement model gets
graded against. Over 2026-08-03..08-13 it called **11.0% of all judgeable
route-ticks degraded**, and the per-hour breakdown said most of that was not
disruption at all:

| UTC hour | ET    | prevalence, tod_bin | prevalence, hourly |
| -------- | ----- | ------------------- | ------------------ |
| 10       | 06:00 | **26.8%**           | 5.9%               |
| 03       | 23:00 | **18.9%**           | 2.6%               |
| others   |       | 4.6-14.7%           | 0.2-8.0%           |

Both spikes sit at a `tod_bin` edge. The bins are 4-6 hours wide (00-05, 06-09,
10-14, 15-19, 20-23), so the 06:00-09:59 median is set by the 08:00 rush peak
and a route running its genuine 06:00 service reads at a third of "normal".
Measured on 14 synthetic weekday mornings ramping 6 -> 30 trains, the wide
bucket's median is 21 and 06:00 service scores 6/21 = 0.29, under the 0.5
degrade floor, every single morning.

Switched this label — and the `recovery_independent` grading path that consumes
the same call — to `momentarily.hmm.schedule_bin`: ET (weekday|weekend, hour),
which already existed for the schedule-rate channel and is already mirrored in
the Worker. `tod_bin` is untouched; the HMM emission channel still scores the
live service ratio against the `(route, tod_bin)` baseline shipped in
params.json, and that pairing has to keep agreeing with the Worker.

Result over the same window: 1,122 baseline cells instead of 131, degraded
prevalence **11.00% -> 1.79%**, events 210 -> 142, distinct routes 23 -> 14 (the
nine that vanish were flagged only by the bin edge), and the share of onsets
landing in the top two ET clock hours 42% -> 30%.

**Negative result: finer than hourly is not worth it.** Same window, same
thresholds, four granularities:

| bin            | cells | judgeable ticks | events | degraded | top-2 ET hours |
| -------------- | ----- | --------------- | ------ | -------- | -------------- |
| tod_bin (4-6h) | 131   | 69,636          | 210    | 11.00%   | 42%            |
| hourly         | 1,122 | 69,546          | 142    | 1.79%    | 30%            |
| half-hour      | 1,197 | 55,963          | 66     | 1.22%    | 26%            |
| quarter-hour   | 2,364 | 55,621          | 54     | 0.90%    | 26%            |

Half-hourly buys 4 points of edge concentration and costs **20% of every
judgeable tick** (69,546 -> 55,963) plus two more routes, because cells fall
under `min_samples`. Note the cell counts: half-hour yields 1,197 where the
split predicts ~2,244, so nearly half the buckets never reach 20 samples in an
11-day window. Quarter-hour drops ~47%. Hourly is the last granularity that
costs no coverage at all (69,546 of 69,636, a 0.13% loss).

The residual concentration after the fix is ET 05:00 (27 onsets) and 06:00
(16) — the hours service ramps hardest within the hour. That is the part a
half-hour bin would actually fix, and it is not worth a fifth of the label.

## 2026-08-12 — per-segment traversal baselines: 1,338 of 1,924 hops fit their own curve off eight hours of trace

origin: agent

`training/traversal.py` turns the minute trace's per-(trip, hop) traversals
into a per-(route, direction, from_stop, to_stop) baseline, plus the deviation
of a live hop from it. First fit, over 2026-08-12 16:30Z..08-13 00:14Z (473
snapshots, 7h45m — the whole archive, which started today):

|                                       |           |
| ------------------------------------- | --------- |
| exact single hops fitted              | 74,706    |
| distinct hops seen                    | 1,924     |
| hops fitted on their own data         | **1,338** |
| hops anchored on their scheduled time | 257       |
| hops omitted, thin and unscheduled    | 329       |
| median own-fit hop                    | 109.1 s   |
| median p90/median within a hop        | 1.351     |
| median observed/scheduled             | 1.056     |

Eight hours is already enough for 70% of hops to clear 20 observations, which
was the open question — the median hop gets 47 samples.

### Arrival-to-arrival, not travel time

`Traversal` carries two clocks and `to_dwell_samples` returns the
departure-to-arrival one, which sounds like the right thing to compare against
a departure-to-arrival timetable. It is the wrong one to FIT, for two measured
reasons: only 24,320 of 74,706 exact hops (26.5%) ever caught the train in
transit and so have a departure at all, and the ones that do are selected for
being slow — a hop still in motion when the next minute's poll lands. Fitting
the tail-detector on a tail-biased quarter of the data is backwards.

Arrival-to-arrival is also the rider's quantity, and the timetable is
comparable to it anyway: NYCT allocates no dwell on 95.7% of stop_times rows,
which is why observed/scheduled lands at 1.056 rather than somewhere near 1.3.

Only EXACT single hops reach a cell. A RIGHT-censored traversal has `to_stop =
None` — the train was last seen heading somewhere it never reached — so there
is no segment to file it under, and filing it by from_stop would pool a branch
point's two successors. It is not a right-censored observation of a known hop;
it is an observation of an unknown one. 390 dropped, against 5,295 interval
spans.

### Thin hops take their level from the timetable, not from the population

Pooling raw seconds across segments would put a 60-second hop and a 400-second
hop in one distribution. The exchangeable quantity is the RATIO to the
scheduled time, so a thin hop gets the population's ratio curve rescaled by its
own scheduled hop — level from the timetable, shape and dwell allowance from
the population. A thin hop the timetable does not name gets no cell at all.

### The signal the advance rate could not reach

The slowest own-fit segments run **4.03x, 3.88x and 3.82x** their scheduled
time (n = 93, 64, 104). A binary "did it leave" cannot express that at all, and
at 5-minute polling those stations were mostly interior to a multi-station jump
and never observed.

**Bug found by a degenerate fixture:** `fit_loglogistic` overflowed on a sample
with zero variance (30 hops all timed at 180 s). The MLE shape is at infinity,
and Nelder-Mead's expansion step reached log-shape 776 — `math.exp` raises long
before the likelihood says stop. Bounded the simplex to log-shape +/-6 and
log-scale +/-20; shape 400 is already a point mass, so the box is inert on real
data. Same family as the log-space likelihood fix earlier today, different
mechanism: that one underflowed inside the likelihood, this one overflowed in
the parameterisation on the way to it.

Not graded. There is one evening of archive and no held-out window; the epic
says weeks, and nothing here is retroactive.

## 2026-08-12 — the advance trip-wire graded against the assigned_n label: 7 firings in 54,000 ticks, zero agreement. Retuning does not fix it and neither does the estimator

origin: agent

The advance trip-wire (`load_r2.classify_direction`) had never been graded
against a label that could see it. The assigned_n degradation label now can be
trusted enough to try (see the hourly-bins entry above), so: 10 days of vehicle
archive, through-stops counting, the live baseline, scored per (route, tick)
against the label's degraded/normal. Read every count below as agreement with
that proxy, not with ground truth — the last section is about the gap between
the two.

```
label base rate                                        1.795%
current (ratio 0.5, alpha 0.05, min_matched 3)
  graded 53,993   fired 7   agree 0   disagree 7   missed 276
```

Seven firings in fifty-four thousand judgeable route-ticks, and not one of them
lands on a labelled degradation. Every knob was swept — min_matched 3/8/15,
ratio 0.5/0.25, alpha 0.05/0.001, and their combinations — and every setting
either fires less or fires zero. There is nothing to retune.

### The baseline estimator IS wrong, and fixing it changes nothing

`compute_advance_baseline` takes p0 as the MEDIAN of per-tick advance
fractions. With layovers excluded, **71.3% of ticks have zero stalls**, so the
median sits on the ceiling and gets floored to 0.9990. That is not the cell's
advance rate:

|             | median of tick fractions | pooled advanced/matched |
| ----------- | ------------------------ | ----------------------- |
| median cell | 0.9990                   | **0.9443**              |

p0 − true rate is +0.033 at the median and +0.135 at p90, and **73 of 210 cells
publish p0 = 0.9990 for a real rate between 0.80 and 0.95**. That matters
because `_binom_lower_tail(advanced, matched, p0)` is a significance test and
takes p0 as a Bernoulli rate: at 0.999 with 10 matched trips a single stall
reads significant, in a cell where one stall in ten is ordinary.

So it was fixed offline — pooled advanced/matched, with a 0/10/20% one-sided
trim to keep outage ticks from dragging the rate down. The saturation goes away
(median p0 0.9990 -> 0.9443, cells at/above 0.99 drop from 74.3% to 8.6%). The
trip-wire does not improve:

| baseline         | ratio | fired | agree | disagree | agreement rate |
| ---------------- | ----- | ----- | ----- | -------- | -------------- |
| median (shipped) | 0.5   | 7     | 0     | 7        | 0.000          |
| pooled, trim 0   | 0.5   | 7     | 0     | 7        | 0.000          |
| pooled, trim 0   | 0.7   | 896   | 3     | 893      | **0.003**      |
| pooled, trim 0.1 | 0.7   | 1,129 | 3     | 1,126    | 0.003          |
| pooled, trim 0.2 | 0.7   | 1,421 | 3     | 1,418    | 0.002          |

At a 1.795% base rate, an agreement rate of 0.003 is five times WORSE than
firing at random. Against this label the wire is not weak, it is
anti-correlated — the same inversion three movement scores showed against the
alert truth, now reproduced against a proxy that shares no feed with the
alerts.

**Why, and it is structural.** assigned_n degradation is service being
WITHDRAWN: fewer trains dispatched. The advance rate measures whether the
trains that ARE out keep moving. Those are different failure modes, and their
coverage is anti-correlated — withdraw service and the movement channel has too
few matched trips to judge at all. Of 1,249 labelled degraded ticks only 276
overlap a judgeable movement call, and at min_matched=8 only 5 do.

### What this settles

The estimator fix is NOT being shipped. It is correct on its own merits and it
moves the live classifier's operating point, and by measurement it buys
nothing — publishing an ungraded operating-point change that improves no metric
is the churn that produced 18 params versions in 46 days. It waits for a truth
that can see freezing rather than withdrawal, which is what the traversal model
is for.

Reproducible from `training.degradation_label` + `load_r2.classify_direction`
over any 10-day window; the sweep is three nested loops over the knobs.

## 2026-08-13 — two arms were sharing one forecast field, and the mix scored worse than either: AUC 0.378 -> 0.856

origin: agent

`route_status[route].condition` has been movement-primary for a while.
`route_status[route].inference.p_normal_in_30min` was not: it came from
whichever arm `recovery_source` named, which was usually the alert-HMM, timing
the alert regime on the alert clock. Both numbers sat in the same object, so a
reader had every reason to think the forecast described the condition.

Graded properly for the first time — each prediction against the condition we
actually published 30 minutes later, six days of the stream:

| rows                   | n      | AUC       |
| ---------------------- | ------ | --------- |
| movement-sourced       | 4,741  | **0.856** |
| alert-sourced          | 20,497 | 0.261     |
| all of them, one field | 25,238 | **0.084** |

The mixture scoring below both components is the whole finding. The two arms
put their probabilities on different scales, so ranking across the union tracks
which arm answered rather than which route is at risk. Simpson's paradox, in
production, in a field consumers read.

### The rule

A forecast is published only when it is about the condition that was published.
Movement qualifies by construction — same arm, same clock, same regime. An
announced resume time qualifies too: it is a fact about the route rather than a
rival model of it. Everything else is withheld (null), the way the 60- and
120-minute horizons already are. Applied to the same six days: **AUC 0.378 ->
0.856** on the rows that still publish, and once the movement arm is sourcing
normally that is about 80% of them (on 08-12, 6,635 of 8,352 ticks).

With the alert arm no longer forecasting, `projectForward` fell out of
`snapshot.ts` entirely, along with the whole `else { /* Normal now */ }` branch
and the `toNormal` transition-matrix split. The alert arm still estimates
`recovery_minutes`; it no longer estimates probabilities.

### The same bug in minutes

`recovery_minutes` is the same claim on a different scale, so it was graded the
same way — time to the next tick published normal:

| arm / published condition | n     | MAE             |
| ------------------------- | ----- | --------------- |
| movement / disrupted      | 10    | **3.5 min**     |
| hmm / disrupted           | 23    | 13.5 min        |
| hmm / suspended           | 12    | 33.8 min        |
| hmm / not_scheduled       | 4,258 | **1,134.8 min** |

Nineteen hours, on the dominant population. Those are overnight routes: the
published condition is `not_scheduled` and the alert arm was timing its own
regime's return, which has nothing to do with when the trains come back.

Two fixes. The schedule countdown now gates on the PUBLISHED condition rather
than the alert shadow, so an announced resume is used wherever one exists. Where
none exists — the ordinary overnight gap — the estimate is withheld through the
mechanism the contract already has for "this is not a prediction":
`recovery_indeterminate` with the value at the ceiling, exactly as the
outlived-every-dwell case has always done. 4,293 rows a week stop publishing a
confidently wrong number.

**Not made nullable, deliberately.** `recovery_minutes` is an integer in the
published snapshot contract and the Home Assistant integration reads it; the
forecast fields could widen to null because their siblings already were.
Making the recovery block nullable is a consumer migration, not a bug fix.

### Worth remembering

`Number.isFinite(null)` is `false`, and `scrubCorruptInferences` used it as the
validity check for every probability. Widening one field to nullable would have
silently deleted the entire inference block for ~60% of routes. The type system
could not see it: `Number.isFinite` takes `unknown`.

## 2026-08-13 — the scheduled reference was wrong three ways, and the one named in the bug report was the smallest: service-day pooling misprices 26% of hops, the express/local lumping misprices none

origin: agent

Setting out to fix the skip-stop inversion in the traversal measure — the modal
chain crediting a bypassing train with the time of every stop it skipped — and
finding that per-pair keying had already killed it, that the express/local
lumping it was supposed to also fix does not exist in the data, and that the
reference was meanwhile pooling the weekend timetable into weekday hops.

Window: 2026-08-12 16:30Z..08-13 02:56Z, 650 trace snapshots, 109,257 arrivals,
95,294 exact single hops, 6,953 interval-censored spans. All percentages below
are of observed single hops unless stated.

### Decomposing the reference error

`hop_seconds` was one median per (route, direction, from, to) over every trip in
the feed, measured departure-to-arrival. Three things are wrong with that, and
they are separable:

| change to the reference                            | observed hops mispriced >10% | >25%  |
| -------------------------------------------------- | ---------------------------- | ----- |
| service-day slicing (weekday vs weekend)           | **26.0%**                    | 16.0% |
| clock (departure-to-arrival -> arrival-to-arrival) | 4.9%                         | 3.5%  |
| both                                               | 27.9%                        | 17.1% |

**Service-day pooling is the dominant error and nobody had named it.** A quarter
of weekday hops were being judged partly against Saturday and Sunday run times.
The clock is smaller than the 95.7% arrival==departure figure suggests, because
the 4.3% that differ carry real scheduled holds — up to 240s at a timepoint.

Fixing both recentres the population ratio: observed/scheduled median 1.050 ->
1.000, and `own_cells.median_ratio_to_schedule` 1.056 -> 1.011.

### The named bug: real, structural, and worth 0.16%

At hop granularity the modal chain is already gone — `traversal.py` keys per
(route, direction, from, to) and only accepts spans the realtime feed calls one
hop, so there is no chain to sum along. What survives is narrower: the feed
publishes its OWN stop sequence, so a train that bypasses a station reports two
consecutive stops the timetable puts a station between. Resolving each traversal
against its trip's own stopping pattern finds **148 of 95,294 such spans (0.16%)**
and drops them; before, they landed in a cell as if they were direct hops.

Real, and correctly fixed. Not the 42,596-transition problem the 5-minute
prototype had.

### Negative result: express and local do not disagree about a shared hop

The premise was that pooling express and local trips over the same pair
misprices both. 1,415 of 1,923 weekday hop keys (85.4% of observed hops) are
served by more than one path code, so if the effect existed it would be
everywhere. Ratio dispersion on exactly those keys, trip-exact scheduled times
against the pooled per-key median:

| reference on multi-pattern keys    | IQR        | MAD    | sd     |
| ---------------------------------- | ---------- | ------ | ------ |
| the trip's own scheduled times     | 0.4667     | 0.2000 | 0.5832 |
| median over trips serving the pair | **0.4653** | 0.2000 | 0.5855 |

The pooled median is marginally _tighter_. Physically obvious in hindsight: on a
consecutive pair both patterns serve, express and local cover the same track at
the same scheduled speed. Express differs in which pairs it has, and per-pair
keying already separates those. Over all keys the same comparison gives IQR
0.4750 (trip-exact) against 0.4833 (pooled) — a 1.7% difference against a 26%
one from the service day.

So `Timetable` stores per-pattern STOP LISTS, which is what detects a bypass and
what a span has to be summed along, and takes its TIMES from the service day's
per-pair medians. 433 patterns and 12,248 stops, against 565,093 per-trip
stop_time rows for no measured gain.

### Matching a realtime trip to the timetable

Only 81% of realtime trip ids match a static trip outright: NYCT dispatch
origins drift off the schedule, reroutes get an `X` path code, and SIR/SS are not
in this feed at all. Two things recover most of the rest.

**The path code determines the stop list.** 433 (service, path) pairs across
20,621 trips, not one carrying a second stop list. So the pattern can be keyed on
the path code rather than the trip.

**Unanimity beats picking.** Realtime truncates some codes (`W..N` for
`W..N30R`), and a service day can run two calendars. Taking the answer only when
every candidate pattern agrees, and falling back to the day's per-hop median
otherwise, lifts pattern coverage 78.4% -> 90.1% against a unique-prefix rule.
Layered with the median fallback:

| how a hop gets priced                 | share     |
| ------------------------------------- | --------- |
| the trip's own pattern                | 89.9%     |
| the service day's median for the pair | 8.8%      |
| unpriceable                           | **1.14%** |

Key-level coverage 83.0% -> 85.9% (1,715 of 2,066 keys, then 1,718 of 2,000).
The 14% that stay unpriced are thin keys the timetable never scheduled — 1.1% of
traffic.

### The trip id's origin field is hundredths of a minute

Needed because a service day is not a calendar date. `060250` is 602.50 minutes
past midnight, 10:02:30 — **not** 06:02:50. Read as hundredths it reproduces the
trip's own first scheduled arrival exactly for 20,311 of 20,621 trips (the other
310 off by a flat 90s); read as HH:MM:SS, 273. The `:50` endings are the tell:
they are half-minutes, and no subway leaves at fifty seconds past.

That gives a service-day rule with no wall-clock cutoff — rank the candidate
midnights by how near each puts the observation to the trip's own scheduled
origin, and let the calendar veto a day that never ran the pattern. Over 106,172
arrivals it put every one on the right day, none more than 50 minutes early
against its own origin. A fixed 4am cutoff was tried first and is wrong in the
other direction: it misfiles trains put into service before their scheduled
origin, which a naive subtraction sends back a full day (1,488 arrivals, 1.4%).

### The measurement floor is the poll, not the reference

Observed hop times cluster hard on 60/120/180s — the 1-minute poll brackets an
arrival rather than pinning it, and 31% of measured hops land on an exact
multiple of 60. The timetable is on a 30-second grid. Where the two grids line
up, the measure looks three times sharper than where they do not:

| scheduled hop          | n      | ratio IQR |
| ---------------------- | ------ | --------- |
| 60s (on the poll grid) | 9,071  | 0.517     |
| 90s (off it)           | 32,418 | 0.522     |
| **120s (on it)**       | 22,325 | **0.150** |
| 150s (off it)          | 8,812  | 0.387     |
| 300s                   | 1,383  | 0.200     |

A 120s hop is not physically three times more predictable than a 90s one. Ratio
IQR sits near 0.48 overall under every reference tried, and a large part of that
is the poll rather than the trains. **No further work on the scheduled side
moves the single-hop measure much.**

Which points at the 6,953 interval-censored spans currently discarded. Resolved
against the trip's own pattern, 91% of them price, and their ratio is far
tighter than single hops because the same +/-60s lands on a 2-3x longer span:

| span              | ratio IQR | sd        | median |
| ----------------- | --------- | --------- | ------ |
| exact single hop  | 0.478     | 0.576     | 1.000  |
| interval, 2+ hops | **0.221** | **0.306** | 0.983  |

They are deliberately kept out of the population ratio curve — that curve is
transplanted onto thin SINGLE hops, and a distribution tightened by averaging
would understate their spread. But as a line-speed measure they are 2.7x
cleaner than what the model currently uses, off data it currently throws away.

### The disrupted-vs-normal comparison is no longer NaN. It is worse than that

Re-run under the fixed reference, joining each hop to its route's alert state at
the enclosing 5-minute tick:

| cut                 | share of route-ticks | disrupted median | normal median |
| ------------------- | -------------------- | ---------------- | ------------- |
| Delays or Suspended | **94.8%**            | 1.000            | 1.000         |
| Delays              | 84.7%                | 1.000            | 1.000         |
| severity_sum >= 100 | 26.2%                | 1.000            | 1.000         |
| Suspended           | 34.1%                | **1.033**        | 1.000         |

Only the suspended cut separates, by 3.3% on the median (p99 3.625 vs 3.000) —
directionally right, where the 5-minute scores came out inverted, but small. The
other three say nothing because the label says nothing: 95% of route-minutes on
an ordinary weekday carry a Delays or Suspended alert. This is the same wall the
advance rate hit. The measure is not gradeable until the label is.

## 2026-08-13 — the timetable is now the instrument, not an input: thin cells abstain, and drift reads 1.0117 against feed 20260807

origin: self

Follow-on decision to the entry above, taken deliberately as an architectural
constraint rather than on the coverage numbers. Movement is the only sensor. A
segment with too little movement of its own now says nothing instead of
borrowing a level from the schedule.

### What changed

`traversal_baseline` no longer has a SCHEDULED cell source. Every level is
fitted from that segment's own traversals; under `MIN_HOP_SAMPLES` there is no
cell. `TraversalCell.source` is gone with it — one source needs no label.

|                          | before | after     |
| ------------------------ | ------ | --------- |
| cells                    | 1,723  | **1,357** |
| keys abstaining          | 284    | **663**   |
| share of traffic covered | 99.1%  | ~97%      |

The 366 cells that went away were carrying 2.1% of traffic on borrowed levels.
That is the price, and it shrinks: those segments cross the sample floor on
their own as the trace archive fills.

### Why, in one line

A reference that feeds the fit cannot then detect the fit drifting.

### The drift measure, which is the point

`schedule_drift` compares fitted cells against the timetable they were not
fitted from. First reading, 2026-08-12 16:30Z..08-13 03:00Z:

|                                     |                                                   |
| ----------------------------------- | ------------------------------------------------- |
| feed_version                        | `20260807-H-rockaways-extension-removed`          |
| cells compared                      | 1,347 (10 fitted cells the timetable never names) |
| median fitted / scheduled           | **1.0117**                                        |
| p10 / p90                           | 0.7946 / 1.3516                                   |
| share of segments >= 1.25x schedule | **16.0%**                                         |

The fleet runs its own timetable to within 1.2%. One segment in six has a
NORMAL that is a quarter over schedule — some of those are standing slow orders,
so the level is not the alarm. The change in it is, and this is the baseline
that change gets measured from.

### The ruler can move too, so it is on the record now

The static feed is a snapshot, not a standing truth. Facts as of today: the
object at `rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip` was Last-Modified
2026-08-07, six days old, self-versioned
`20260807-H-rockaways-extension-removed` — the name says it dropped a branch —
and it declares itself valid 2026-05-26..2026-10-31.

Three gaps that made the drift measure meaningless until closed:

- `fetch_gtfs_zip` re-downloads every run. No caching, no ETag check, no pin.
- **Nothing recorded which feed version a measurement used.** `provenance.py`
  still has no GTFS field; `ScheduleDrift` now carries `feed_version` so at
  least the measure is self-describing. The trainer artifact does not yet.
- **Nothing checked the validity window.** The vehicle archive reaches back to
  June; the trace archive is 30 days. Replaying a window from before this feed
  took effect would have compared trains against a schedule that did not exist
  when they ran — and Rockaways service was _removed_ in this version, so that
  is not hypothetical. `Timetable.covers` now guards it: an out-of-window
  observation still fits its cell and carries NO scheduled time, which keeps it
  out of the drift measure by construction rather than by remembering to filter.
  `n_outside_feed_window` reports it; the current window reads 0.

### Also

Dropped the pricing metaphor ("priced", "mispriced", "unpriceable") that had
spread through three modules. A timetable does not price anything; it says how
long a hop is scheduled to take.

## 2026-08-13 — correction: the bypass filter was the schedule choosing the training set

origin: self

Both entries above describe bypasses as detected and DROPPED. That was wrong and
is now changed: they are counted and kept.

A bypass is a span the realtime feed calls one hop and the trip's own timetable
puts a station inside. Excluding it looked like hygiene and was two mistakes.
The train covered that stretch without stopping — physically the same movement
an express makes on the same pair, so it is a real measurement of it. And
bypasses cluster on bad days, so removing them taught the baseline a cleaner
normal than the one riders got, using the schedule to decide what the model was
allowed to learn from. Under a movement-only constraint the timetable does not
get to choose the training set.

`hop_samples` now admits observations on realtime evidence alone — EXACT, has a
destination, one hop by the feed's own stop sequence — and reads the timetable
only to attach a comparison. `TraversalStats.n_dropped_bypass` is `n_bypass`.

Re-run over 2026-08-12 16:30Z..08-13 04:00Z (the archive is still filling, so
this window is larger than the one above, not just a rerun of it):

|                                     |            |
| ----------------------------------- | ---------- |
| traversals                          | 148,337    |
| bypasses, kept                      | 202        |
| cells                               | 1,533      |
| median fitted / scheduled           | **1.0058** |
| share of segments >= 1.25x schedule | 14.45%     |

## 2026-08-13 — the planned-work grade ran end to end and graded almost nothing: two days of trace cover disjoint clock bands, and the 7's daily work blacks out its own control

origin: self

`training/planned_work.py` had 436 lines of tests and no callers — the measure
had never touched real data. It has now. The harness works; the answer is that
the archive cannot yet supply the primary grade, and one whole class of window
may never supply it.

### What the first run saw

|                                        |                                                        |
| -------------------------------------- | ------------------------------------------------------ |
| trace snapshots                        | 1,329 (Wed 12:20 ET -> Thu 10:41 ET, 22.3h)            |
| traversals                             | 171,960 (159,374 exact)                                |
| announced windows in the alert archive | 394                                                    |
| **over the trace span**                | **2**, both `Express to Local` on the 7, naming 1 stop |
| unknown alert types                    | 0                                                      |

2 of 394 is not a shortfall, it is the rate: ~2.2 geo-scoped windows a day
against a 22-hour archive.

### The secondary grade ran and found nothing, which is the expected answer

Difference-in-differences on the boundary hops, 4 affected keys against 36
control keys:

| window          | inside | affected lift | control lift | effect     |
| --------------- | ------ | ------------- | ------------ | ---------- |
| Wed 15:00-22:00 | 321    | 1.0099        | 0.9982       | **1.0118** |
| Thu 06:15-10:00 | 248    | 0.9990        | 0.9942       | **1.0048** |

An express told to run local does not slow the hops beside it down; it stops
making its own long hops. Duration is the wrong instrument for this type and
the module says so — 1.007 is that prediction confirmed, not a miss.

### The primary grade returned zero rows, and zero was ambiguous

`pattern_shift` produced nothing for either window. The control arm wants the
same band of the local clock on a comparable service day, and the funnel dies
entirely at that filter:

| window          | on-route hops | inside | same service class | **+ same clock band** |
| --------------- | ------------- | ------ | ------------------ | --------------------- |
| Wed 15:00-22:00 | 8,726         | 3,815  | 4,911              | **0**                 |
| Thu 06:15-10:00 | 8,726         | 2,254  | 6,472              | **0**                 |

Not a property of these two windows. The trace's two calendar days cover
Wed 12:20-24:00 and Thu 00:00-10:41 — **disjoint clock bands**, so no window of
any type, on any route, could have drawn a control from this archive.

That had to be dug out with a throwaway script, because an empty
`pattern_shift` reads the same whether the pattern held or the comparison never
ran. Those are opposite conclusions. `control_supply` now counts the traversals
the matched arm had to work with, and the report carries
`coverage_no_control_period` beside `graded_coverage`: this run reads 0 graded,
2 with no control period, which is a measurement that did not happen rather
than a detection that failed.

### The sharpest finding: recurring work erases its own control

The 7's `Express to Local` runs 15:00-22:00 and 06:15-10:00 every weekday. Its
control period is the same clock band on the adjacent weekday, which is the same
work. The blackout that keeps other announced disruption out of the baseline
correctly removes all of it:

| tonight's window           | control traversals in the trace | after blackout |
| -------------------------- | ------------------------------- | -------------- |
| 7 Express to Local, 15:00  | 3,815                           | **0**          |
| 6 Part Suspended, 21:30    | 1,376                           | 1,376          |
| L Part Suspended, 22:45    | 593                             | 593            |
| SI Special Schedule, 23:00 | 299                             | 299            |
| A Reroute, 23:45           | 1,588                           | 1,588          |
| A Stops Skipped, 23:45     | 1,588                           | 1,588          |
| E Stops Skipped, 23:45     | 781                             | 781            |

`Express to Local` is 278 of the 394 announced windows. Where it recurs daily,
an adjacent-day control cannot exist, and the answer is not to relax the
blackout — a control drawn from the same work is not a control. That subset
needs a different comparison (a non-adjacent unaffected day, or the local as the
within-window control the way the 7/7X split already reports it). Until then the
gradeable supply is the other 116 windows, not the headline 394.

### What lands next, and it is hours away, not weeks

13 gradeable windows are announced in the next 4 days, and the six above already
have their Wednesday-night control sitting in the trace. Re-running over
2026-08-12..08-14 should produce the first coverage rows the measure has ever
returned from real movement, on multi-stop part suspensions and reroutes — the
shape of work this measure was actually built for.

## 2026-08-13 — correction: the control diagnostic counted a route where it had to count a service

origin: self

Two corrections to the entry above, both caught in review before the code left
the tree.

### The diagnostic could have certified a comparison that never happened

`control_supply` shipped summing control traversals over the whole ROUTE.
`pattern_shift` emits a row only when ONE service has both arms — that is the
entire reason it reports the 7X apart from the 7. So a route-level total can
report control on the strength of the local while the express the work actually
named had none, and the runner would score the empty result
"compared, found nothing" instead of "never ran". That is the precise confusion
the field was added to remove, reintroduced one level down.

It now returns `dict[service, int]`, and `_service_of` is shared with
`_shift_one` so the grade and the diagnosis cannot disagree about which service
a traversal belongs to. The empty result splits into three states, because there
are three:

|                              |                                              |
| ---------------------------- | -------------------------------------------- |
| `graded_coverage`            | a service had both arms                      |
| `coverage_no_paired_service` | some service had control, none had both arms |
| `coverage_no_control_period` | no service had a control arm at all          |

**No published number changes.** Re-run over the same window: 0 graded, 2 with
no control period, 0 with no paired service — and the six windows in the table
above resolve to a single service each (`{'6': 1376}`, `{'L': 593}`,
`{'SI': 299}`, `{'A': 1588}`, `{'E': 781}`), with the 7 an empty dict rather
than 3,815 -> 0. The defect was latent, not active. Recorded anyway: a
denominator that is right by accident is still the wrong denominator, and
`Express to Local` — 278 of 394 windows, the one class that routinely carries
two services — is exactly where it would have fired first.

### The origin line was wrong

The entry above reads `origin: self`. It should read `origin: artifact`: nothing
about disjoint clock bands was found by inspection, the grader's own output
surfaced it on first run. Left uncorrected in place, per the append-only rule.

## 2026-08-13 — the advance rate was never the weak arm: as a continuous score it reads 0.925 where movement is visible at all, and the progress ratio reads nothing

origin: artifact

`training/progress.py` grades both movement arms against the assigned_n
degradation label over the trace's own window (Wed 12:20 -> Thu 19:55 ET,
31.6h, 1,891 snapshots, 268,813 traversals, 246,473 priced hops). Two results,
and the arm I expected to win lost.

### Both arms, same window, same denominator

Scores are stated so HIGHER MEANS WORSE. Intervals are a stratified percentile
bootstrap over 2,000 resamples. Acute is graded against NORMAL ticks only.

|                          | progress ratio              | stalled share               |
| ------------------------ | --------------------------- | --------------------------- |
| graded against the label | 7,790                       | 7,820                       |
| degraded among them      | 54                          | 51                          |
| pooled AUC               | 0.535 [0.440, 0.628]        | 0.606 [0.507, 0.712]        |
| **within-route AUC**     | **0.476 [0.380, 0.579]**    | **0.684 [0.618, 0.759]**    |
| acute-onset AUC          | 0.605 [0.441, 0.762] (n=14) | 0.755 [0.581, 0.893] (n=17) |
| median, normal ticks     | 1.000                       | 0.043                       |
| median, degraded ticks   | 1.017                       | 0.250                       |

The progress ratio's within-route interval straddles 0.5 and is centred just
below it. Against this label, over this window, it has no signal. The medians
say why in one line: when assigned_n collapses, the trains still running cover
their hops in 1.017x their booked time instead of 1.000x. **Withdrawn service
does not slow down the trains that remain.** The same ticks carry six times the
stalling, 0.25 against 0.043.

### The reversal, and it reverses my own entry from yesterday

On 2026-08-12 the advance arm was written up here as anti-correlated: 7 firings
in 53,993 ticks, 0 agreement, 0.003 when loosened, "five times WORSE than
firing at random." That entry graded the TRIP-WIRE — `classify_direction`'s
thresholded verdict with its p0 baseline and binomial test. This one grades the
underlying continuous rate, and the rate separates the classes at 0.684
within-route, interval clear of 0.5.

Both are true and they do not conflict. At a 0.65% base rate a ranker at AUC
0.68 still has dismal precision at any threshold, which is exactly what the
sweep found. The correction is to the CAUSE, not the numbers: the wire's
failure was never evidence that trains-not-moving is uninformative about
service being withdrawn. It is evidence that a significance test against a
saturated p0 is the wrong decision rule on top of an informative quantity.

### The real constraint is that there is nothing to look at

Raising the matched-trip floor makes the score BETTER and the measurement
rarer — the signature of a coverage limit, not a skill limit:

| min_matched | degraded ticks judgeable | coverage | within-route AUC         |
| ----------- | ------------------------ | -------- | ------------------------ |
| 3           | 51                       | 25.5%    | 0.684 [0.612, 0.752]     |
| 8           | 19                       | 9.5%     | **0.925 [0.903, 0.945]** |
| 15          | 0                        | 0%       | no answer — empty class  |
| 25          | 0                        | 0%       | no answer — empty class  |

A small-denominator artifact would have gone the other way. This one sharpens
to 0.925 as the floor rises, then runs out of data entirely.

The number underneath it is the sharpest thing this run produced. Across the
193 degraded ticks carrying a movement row at all:

|                                  | matched trips at through stops |
| -------------------------------- | ------------------------------ |
| median normal tick               | 15                             |
| **median degraded tick**         | **0**                          |
| degraded ticks with exactly zero | **50.8%**                      |

Not "few". Zero, on half of them. Yesterday's entry put this at 276 of 1,249
degraded ticks judgeable and called the coverage anti-correlated; the median is
the honest statement of it. A movement arm cannot be graded on the ticks that
matter most, because withdrawing service withdraws the observations a movement
call is made from. Every AUC above describes the mild tail where trains are
still out.

### One latent defect, no published number moved

`grade` first computed the acute AUC with chronic ticks swept into the NEGATIVE
class, scoring each arm for failing to rank a fresh collapse above a route
already known to be down. `_split` now names both classes and drops anything in
neither. The acute numbers moved by 0.0002 and 0.0002, because the scored
chronic ticks are ~0.5% of a 7,700-strong negative pool. Latent, not active,
and recorded for the same reason the `control_supply` one was: the comparator
was right by accident.

### What this settles

The progress-ratio-vs-advance-rate comparison the epic has carried as
unanswerable now has an answer against this label: advance rate, decisively, on
the judgeable quarter. That does NOT retire the progress ratio. The two measure
different failure modes and this label contains only one of them — withdrawal,
not slowness — so the result is as much a statement about the answer key as
about the arm. The instrument that can see slowness is the planned-work grade,
which is announced, geo-scoped, and as of tonight finally has its control band
in the trace.

Two things NOT shipped on the strength of this. The advance-baseline estimator
fix stays parked: it moves the live operating point, and nothing here shows a
threshold that works at a 0.65% base rate. And 31.6 hours with 51 positives is
one window on one pair of weekdays, with intervals wide enough that the
pooled-versus-within-route gap is the only comparison I would defend.

## 2026-08-13 — correction: nine episodes, not fifty-one ticks. The advance arm's win is not established, and the 0.925 was one incident

origin: self

The entry above is wrong where it says "decisively", and its headline number is
the worst offender. Caught in review before the code left the tree; the numbers
it reports were real, the uncertainty around them was not.

### The bootstrap counted repetitions as evidence

`auc` resampled TICKS. A disruption is one episode observed every five minutes,
so fifty consecutive degraded ticks on one line are close to one observation,
not fifty. Resampling them independently reports an interval far too narrow —
which is precisely how a handful of incidents came to read as decisive.

`_clusters` now groups each maximal run of consecutive same-class ticks on one
route, and the bootstrap resamples those. The point estimates are unchanged.
The intervals are not, and neither is the conclusion:

| within-route AUC      | by tick (wrong)          | by episode (right)       |
| --------------------- | ------------------------ | ------------------------ |
| progress ratio        | 0.476 [0.380, 0.579]     | 0.476 [0.428, 0.556]     |
| **stalled share**     | **0.684 [0.618, 0.759]** | **0.684 [0.450, 0.850]** |
| acute, progress ratio | 0.605 [0.441, 0.762]     | 0.605 [0.365, 0.751]     |
| acute, stalled share  | 0.755 [0.581, 0.893]     | 0.755 [0.352, 0.989]     |

**The stalled share's interval now straddles 0.5.** 51 degraded ticks are
**9 episodes**; the acute cut's 17 ticks are **3**. The advance arm's point
estimate is still the better of the two, and that is now the whole claim —
"decisively" is withdrawn. Nothing here separates it from chance at 95%.

### The 0.925 was a single incident, and the tight band around it was an artifact

The floor sweep row I put in bold was selected after seeing the sweep, which
alone makes it exploratory. It is worse than that:

| min_matched | degraded ticks | **episodes** | coverage | within-route AUC        |
| ----------- | -------------- | ------------ | -------- | ----------------------- |
| 3           | 51             | 9            | 25.5%    | 0.684 [0.450, 0.850]    |
| 8           | 19             | **1**        | 9.5%     | 0.925 — no interval     |
| 15          | 0              | 0            | 0%       | no answer — empty class |

Nineteen ticks of ONE episode on one route. Its old band of [0.902, 0.948] was
a degenerate resample: with one cluster, every bootstrap draw returns that same
cluster, so the interval collapses to a point and reads as maximum confidence
built from a single incident. `auc` now withholds the interval entirely when
either side has under two clusters, which is the same can't-judge contract the
rest of this module uses, applied to the uncertainty rather than the value.

So the sweep does not show the score "sharpening" to 0.925. It shows the
measurement running out of episodes, with one left at min_matched=8 and none at 15. The direction is still inconsistent with a small-denominator artifact, and
that is all it is good for.

### What survives unchanged

The coverage census, which is a count and not an inference: median 0 matched
trips on a degraded tick against 15 on a normal one, and 50.8% of degraded
ticks with exactly zero. And the descriptive medians — 1.000 vs 1.017 for the
progress ratio, 0.043 vs 0.250 for the stalled share. The mechanism those point
at (withdrawn service does not slow the trains that remain) is unaffected by
how many episodes produced it, because it is a statement about direction and
magnitude, not about significance.

What the real finding is, then: **this window cannot settle the comparison.**
Thirty-one hours contains about nine independent degradations, and no interval
built on nine episodes will separate two arms this close. That is a supply
problem with the same shape as the planned-work grade's, and it wants more
days, not more analysis.

## 2026-08-13 — correction to the correction: five episodes, not nine. Cutting runs on the scored ticks split single disruptions at their coverage gaps

origin: self

The entry above fixed the bootstrap to resample episodes instead of ticks and
reported nine of them. Nine was still too many, for a reason that is the same
coverage problem wearing a different hat.

`_clusters` cut its runs over the keys an arm had SCORED. A movement arm judges
about a quarter of the degraded ticks, so one continuous disruption arrives as
scattered ticks with unscored gaps in between, and every gap started a new
"episode". The clustering was undone by exactly the sparsity it was introduced
to survive.

Episode identity now comes from the full label (`_episode_ids`), which has
every tick, and the arm's ticks are filed under those boundaries:

|                      | episodes by scored ticks | episodes by label |
| -------------------- | ------------------------ | ----------------- |
| degraded, either arm | 9                        | **5**             |
| acute                | 3                        | 3                 |

| within-route AUC      | 9 episodes (wrong)   | 5 episodes (right)       |
| --------------------- | -------------------- | ------------------------ |
| progress ratio        | 0.476 [0.428, 0.556] | 0.476 [0.409, 0.528]     |
| **stalled share**     | 0.684 [0.450, 0.850] | **0.684 [0.451, 0.919]** |
| pooled, stalled share | 0.606 [0.253, 0.867] | 0.606 [0.256, 0.963]     |

Five. Thirty-one hours of trace against a label carrying 17 disruptions over
the same span yields **five** that any movement arm could judge, and the
stalled share's interval now runs from worse-than-random to nearly perfect.
Every directional conclusion in the first entry stands withdrawn; what is left
is a point estimate ordering and a census.

The census is the part that keeps surviving these corrections, and it is worth
noticing that it survives them because it is a count rather than an inference:
17 disruptions in the window, 5 judgeable, median 0 matched trips on a degraded
tick, 50.8% at exactly zero. Three passes at the uncertainty have not moved any
of those, and they say the same thing each time — the arms are not being beaten
on skill, they are being starved of the observations that would let them play.

## 2026-08-13 — correction: the origin lines on both grading corrections were wrong

origin: agent

The two correction entries above are both headed `origin: self`. Neither was
found by a human. Both defects — the tick-level bootstrap, and episode
boundaries cut from scored keys instead of the label — were surfaced by
adversarial review of the change before it left the tree, so both should read
`origin: agent`. Left uncorrected in place, per the append-only rule.

Worth recording that the review caught all three methodology defects in this
stream and the grader's own output caught none of them. The run reported its
numbers cheerfully in every broken state: a confident [0.902, 0.948] built from
one incident is not a shape any assertion in the harness could have flagged,
because nothing was wrong with the arithmetic. Only the meaning was wrong.

## 2026-08-16 — the planned-work measure ranks announced work by severity without being told the ordering, and part suspensions slow the hops beside them by a third

origin: artifact

Four days of accumulation turned the measure from "never graded anything" into
its first per-type read. 6,379 trace snapshots, continuous Wed 12:30 -> Sun
22:46 ET with no gaps, 720,058 traversals, 527 announced windows.

|                     | 08-13 | 08-14 | 08-16  |
| ------------------- | ----- | ----- | ------ |
| over the trace span | 2     | 15    | 31     |
| gradeable           | 2     | 13    | 20     |
| **graded**          | **0** | **6** | **13** |
| no control period   | 2     | 7     | 7      |

### The instrument validates itself: the ordering is monotone in how much of the route the work removes

Share of the route's hop keys that stop appearing inside the window:

| type             | rows | median vanished |
| ---------------- | ---- | --------------- |
| Suspended        | 1    | 0.4375          |
| Reroute          | 4    | 0.2716          |
| Stops Skipped    | 4    | 0.1583          |
| Part Suspended   | 9    | 0.1250          |
| Special Schedule | 2    | 0.0263          |

Nothing tells this measure that a full suspension removes more service than a
reroute, or a reroute more than a skipped stop. It ranks them in that order
anyway, off movement alone. That ordering is the closest thing to a validation
the measure has had, and it is worth more than any single row in the table.

### Part suspensions are visible on BOTH instruments, and the duration one is the surprise

Difference-in-differences on the hops at the boundary of the closed stretch:

| type               | rows   | median effect |
| ------------------ | ------ | ------------- |
| **Part Suspended** | **10** | **1.3175**    |
| Reroute            | 3      | 1.0491        |
| Express to Local   | 5      | 1.0233        |
| Stops Skipped      | 2      | 0.9895        |

The hops beside a part suspension take **a third longer**, and it is not one
outlier: 8 of 10 rows sit above 1.0, spread 1.29 to 3.07, with the two below
at 0.77 and 0.81. Physically this is what single-tracking and turning trains
short does to the segments either side of a closed stretch, and it is the
answer to the question left open on 08-14 — the type with the most announced
supply (61 windows) is not invisible to this measure, it is the one it reads
most strongly.

Two smaller confirmations. `Stops Skipped` comes in just UNDER 1.0: skipping a
station makes the remaining hops quicker, which is the right sign. And
`Express to Local` reads 1.0233, the third time that type has measured as
approximately nothing (1.007, then 1.023) — duration is the wrong instrument
for it, exactly as the module has claimed from the start, now on five rows
instead of two.

### What the 08-14 entry got right by waiting

That night's read put Part Suspended at 0.0213 vanished and I flagged it as
either a broken measure, a stop-scope gap, or thin data. It was thin data: the
same type reads 0.125 over 9 rows. The two-and-a-half-hour first-day fragments
carried 46 to 112 traversals each and were measuring the part of an overnight
window before the work bites. Writing that up as a finding would have put a
wrong mechanism in the record for the sake of reporting something.

### The honest limits

`n_affected_keys` is **1 or 2** on every part-suspended row. A window names a
stretch, and only the hops at its boundary survive to be timed — everything
inside vanishes and is counted by the other instrument. So each row's effect
rests on one or two segments, albeit with 10 to 113 traversals apiece and a
control arm of 15 to 74 keys. What carries the 1.32 is agreement ACROSS ten
independent windows, not depth within any one of them.

`Suspended` is a single row; 0.4375 is an anecdote. Seven of twenty gradeable
windows still have no control period at all, all of them the recurring
`Express to Local` blackout. And the duration rows carry no service
attribution, so unlike the coverage rows they cannot separate a 7 from a 7X.

## 2026-08-16 — the shippable segment model graded for the first time: not inverted, strongest exactly where the work physically constrains, and short of significance on every cut

origin: artifact

`planned_work` grades the raw traversal data. `training/segment_grade.py`
grades the thing we would actually ship: `traversal.deviation`, a live hop
scored against its own segment's fitted normal. Every movement score graded in
this repo before now has come out at or below chance, so the question was
binary before it was quantitative.

**It is not inverted.** That is the result. Everything after this is about how
far short of a claim it stops.

|                                 |                                                               |
| ------------------------------- | ------------------------------------------------------------- |
| gradeable announced windows     | 20                                                            |
| graded                          | 15                                                            |
| abstained                       | 5 (2 no fitted cell on any boundary hop, 3 thin affected arm) |
| windows above 0.5               | **11 of 15**                                                  |
| sign test                       | **p = 0.118**                                                 |
| median AUC                      | 0.5705                                                        |
| median deviation, affected hops | 1.0116                                                        |
| median deviation, control hops  | 0.9727                                                        |

### Where the signal actually lives

| type               | windows | median AUC | median affected deviation |
| ------------------ | ------- | ---------- | ------------------------- |
| **Part Suspended** | 6       | **0.674**  | **1.244**                 |
| Express to Local   | 5       | 0.568      | 1.008                     |
| Reroute            | 2       | 0.537      | 1.007                     |
| Stops Skipped      | 2       | 0.491      | 1.120                     |

Part suspensions again, and the magnitudes on the four that work are not
subtle: the 1 at 1.32x its own normal (AUC 0.95), the 6 at 1.42x (0.86), the J
at 1.17x (0.66), and the L at **2.46x** (0.69). This corroborates the
difference-in-differences result from the same weekend, which put the same type
at 1.32x by a completely different construction — one measures boundary hops
against control hops, the other against the segment's own fitted history. Two
instruments, one answer.

### The two windows that go the other way are both the N, and both read FASTER

| route | AUC   | affected deviation |
| ----- | ----- | ------------------ |
| N     | 0.431 | 0.864              |
| N     | 0.448 | 0.893              |

Not noise scattered around 0.5 — two windows on the same route, both saying the
boundary hops ran about 12% QUICKER than their own normal during a part
suspension. The plausible mechanism is that the N's trains bypass the closed
stretch rather than terminating short, and a hop run nonstop is genuinely
faster. Bypasses are deliberately kept in the baseline fit (they really did run
that stretch), so this is a real measurement rather than an artifact, and it
means "part suspension" is not one intervention. Worth resolving before the
type's 1.24 median is quoted as though it described all six.

### The sample is smaller than fifteen, and the unit rule only got it halfway

Grading per ANNOUNCED window rather than per local-day row was deliberate:
`planned_work` splits a period per calendar day, and those fragments are the
same work, so treating them as independent repeats the tick-versus-episode
error that cost three corrections on the label grade. That much this module
gets right by construction — it iterates unsplit windows.

It does not go far enough. Of the 15 graded windows, **all five Express to
Local are route 7**, the same recurring daily work observed on five different
days, and two of the six part suspensions are the same N. So the effective
independent sample is nearer eight or nine distinct work patterns, and the
sign test's fifteen is optimistic. The honest reading of p = 0.118 is therefore
"weaker than that", not "close to significant".

### What is settled and what is not

Settled: the score is directionally correct, it abstains rather than guessing
when a segment has no fitted cell or an arm is thin (5 of 20 windows), and its
magnitude tracks the physical severity of the work on the one type with enough
windows to look at.

Not settled: anything with a p-value. The baseline withheld 112,139 traversals
as inside announced work to avoid grading the model against itself, which is
correct and costly — it is why cells exist for only some boundary hops. More
distinct work patterns is the only thing that moves this, and the collector
supplies them at a few a night without any further work.

## 2026-08-16 — correction: the entry above called it "shippable" while fitting from the future. Refitting each window from its own past only, the answer holds

origin: agent

The entry above grades `traversal.deviation` and calls it the model we would
ship. It fitted every cell from clean traversals across the WHOLE archive,
including days after each window opened. A model running live at 21:30 on a
Thursday has only the past, and a segment whose 20 samples arrive on Saturday
has no cell at all yet. So that result measures whether the signal is IN the
data — retrospective — and is not by itself a deployable claim.

`baseline_before` (`--causal`) now refits per window from traversals that
FINISHED before its start. One fit per window, 3.5 minutes instead of 60
seconds. It is strictly harsher, and the honest surprise is how little it costs:

|                    | retrospective        | causal (past only)   |
| ------------------ | -------------------- | -------------------- |
| graded             | 15                   | 16                   |
| abstained thin     | 3                    | 4                    |
| above 0.5          | 11 of 15             | 11 of 16             |
| sign test          | p = 0.118            | p = 0.210            |
| median AUC         | 0.5705               | 0.5794               |
| **Part Suspended** | **0.674, dev 1.244** | **0.670, dev 1.245** |

Part suspensions land in the same place to three decimal places on the
deviation and within 0.004 on the AUC. Whatever the retrospective fit was
borrowing from the future, it was not what produced the result. The
deployable claim and the data claim agree.

Both modes are kept and neither is allowed to stand in for the other, because
they answer different questions: `--causal` asks what a live model would have
seen, the default asks whether the signal exists at all. A future run reporting
one of them as the other is the defect this entry exists to prevent.

Unchanged: still not significant, and the effective sample is still nearer
eight distinct work patterns than sixteen windows.

## 2026-08-16 — correction: the causal comparison was run against a bigger archive than the retrospective one, and on equal footing the AGGREGATE result is a coin flip

origin: agent

The table in the entry above compares 15 retrospective windows against 16
causal ones. Those are two different fetches — the alert archive grew between
the runs — so it was not a controlled comparison, and "the answer holds" was
asserted from mismatched inputs. Re-run properly: one fetch, both modes graded
off exactly the same bytes (7,008 snapshots, 775,193 traversals, 593 announced
windows, 28 gradeable).

|                    | retrospective                 | causal (past only)            |
| ------------------ | ----------------------------- | ----------------------------- |
| graded             | 22                            | 21                            |
| above 0.5          | 12 of 22                      | 13 of 21                      |
| **sign test**      | **p = 0.832**                 | **p = 0.383**                 |
| median AUC         | 0.5439                        | 0.5542                        |
| **Part Suspended** | **n=6, AUC 0.674, dev 1.242** | **n=6, AUC 0.670, dev 1.245** |

Two findings, pulling opposite ways.

### The causal question is settled, and favourably

Now that it is a fair test, past-only fitting costs almost nothing: one window
moves from graded to abstaining, and part suspensions match to 0.004 on the AUC
and 0.003 on the deviation. A model with only its own history reads what the
retrospective fit read. That claim is now earned rather than asserted.

### The aggregate result died, and it should never have been the headline

"11 of 15, p = 0.118" became **12 of 22, p = 0.832** on eight more windows.
That is not a near-miss that needs more data; it is regression to chance, and I
reported the small-sample version as the finding two entries ago.

The deeper mistake is that the pooled sign test was the wrong statistic from
the start, and this module's own sibling says so: `planned_work._median_by_type`
groups by type precisely because **the types are different experiments**. An
`Express to Local` window is one this measure is documented as unable to see by
duration — it has now read approximately 1.00 four separate times. Pooling it
with part suspensions and asking "how many windows beat 0.5" dilutes a real
effect with windows where no effect is expected, and the answer drifts toward
0.5 as more of them arrive. Which is exactly what happened.

### What actually survives, and it is narrow

Part suspensions, on six windows: median AUC ~0.67, affected hops at ~1.24x
their own normal. That number has now been stable across a retrospective fit, a
causal fit, and an archive that grew from 20 to 28 gradeable windows, and it
agrees with the independent difference-in-differences estimate of 1.32x from
`planned_work`. Everything else here is unproven, and the aggregate is not
evidence in either direction.

Next run must report per type and drop the pooled sign test, or keep it only as
a diagnostic clearly marked as pooling incommensurable experiments.

## 2026-08-16 — the baseline blackout tested only a traversal's start, so hops running INTO the work were fitted as normal. No published number moves

origin: agent

`baseline_outside` withheld a traversal when the window contained its START
time. A hop has duration: one that began at 21:25 and arrived at 21:45 spent
most of itself inside a window that opened at 21:30. Those are also the
traversals nearest the work, so admitting them dragged the fitted normal toward
the disrupted level and made the deviation read closer to 1.0. **The bias ran
against detection** — the direction that quietly passes a broken model rather
than failing a good one. `baseline_before` already accounted for duration, so
the two blackouts disagreed with each other.

Both now share `_overlaps_work`, which tests the traversal's whole span against
every announced window on its route. It withholds materially more: 164,694
traversals against 112,139, and 1,592 fitted cells against 1,687.

And it changes nothing. Same fetch, both modes:

|                    | retrospective              | causal                    |
| ------------------ | -------------------------- | ------------------------- |
| graded / above 0.5 | 22 / 12                    | 21 / 13                   |
| sign test          | p = 0.832                  | p = 0.383                 |
| median AUC         | 0.5443                     | 0.5542                    |
| **Part Suspended** | **AUC 0.6745, dev 1.2429** | **AUC 0.6703, dev 1.245** |

Part suspensions have now held at AUC ~0.67 and ~1.24x through a retrospective
fit, a causal fit, an archive that grew 20 -> 28 gradeable windows, and a
blackout fix that removed 52,555 more traversals from the baseline. Four
different ways of being wrong, same answer.

The wider view got worse, as it should have with more windows. `Reroute` now
has 8 windows and reads 0.492 — nothing. `Stops Skipped` 0.481. `Express to
Local` ranks marginally above chance (0.565) with a magnitude of 1.007, which
is a ranking signal with no practical size and matches its four prior readings
of approximately nothing. One type carries this measure, and it is the one whose
work physically constrains the track the trains are still using.

## 2026-08-17 — the N does not run faster during its part suspension, it runs UNCHANGED on two hops. Part suspensions are one mechanism, not two

origin: self

Two N windows read AUC 0.43/0.45 with affected deviations of 0.864/0.893 while
the 1, 6, J and L read 1.17 to 2.46, and I filed that as the type splitting into
two mechanisms — trains bypassing a closed stretch rather than terminating
short. That was a story built on an aggregate. Looking at the hops themselves,
there is no second mechanism.

### The whole arm is two hops, and neither of them moved

Both windows are 23:45-05:00, naming R09, R11, R13, R14, R15, R16. Twelve hops
qualify as boundary hops; ten of them have almost nothing inside the window. Two
carry the arm:

| hop              | in-window median | out-of-window median | n in / out |
| ---------------- | ---------------- | -------------------- | ---------- |
| R17N->R16N (Wed) | 118 s            | 121 s                | 14 / 527   |
| R17N->R16N (Thu) | 120 s            | 121 s                | 15 / 526   |
| R08S->R09S (Wed) | 159 s            | 170 s                | 7 / 465    |
| R08S->R09S (Thu) | 166 s            | 170 s                | 14 / 458   |

One to eleven seconds on two-minute hops, off 33 and 23 observations across a
five-hour overnight window. That is not a route running faster under
disruption; it is a route whose boundary hops were not measurably touched, on an
arm too thin to say much either way. An AUC of 0.43 on 33 observations is a coin
flip, and I read a mechanism into it.

So the type does not split into "slows down" and "speeds up". It splits into
DETECTED (1, 6, J, L) and NO MEASURABLE CHANGE ON A THIN ARM (N, twice). The
bypass-versus-terminate-short hypothesis has no support here and is withdrawn;
what the N actually shows is that a part suspension does not necessarily
congest its own boundary.

### The absolute deviation was not comparable across windows

Chasing this surfaced a real reporting defect. `median_affected_deviation`
scores hops against a cell level fitted over ALL hours, while a window covers
particular ones — so an overnight window carries whatever the overnight hour
was doing, mixed into the effect. `WindowGrade.effect` now divides the affected
arm's deviation by the control arm's over the same minutes, which absorbs the
hour the same way `planned_work`'s difference-in-differences does, in deviation
space. It moves the headline:

| type               | windows | above 0.5 | median AUC | median effect | raw deviation | sign p |
| ------------------ | ------- | --------- | ---------- | ------------- | ------------- | ------ |
| **Part Suspended** | 6       | 4         | 0.670      | **1.201**     | 1.245         | 0.688  |
| Express to Local   | 6       | 5         | 0.586      | 1.042         | 1.004         | 0.219  |
| Stops Skipped      | 2       | 1         | 0.494      | 1.023         | 1.080         | 1.000  |
| Reroute            | 7       | 3         | 0.498      | 0.926         | 0.981         | 1.000  |

Part suspensions come down from 1.245 to 1.201 once the hour is taken out. In
the same measurement, the control arms sat at 0.96-1.08, so the correction is
small — but "small once measured" is a different statement from "assumed
negligible", and only one of them is defensible.

### The pooled sign test is now labelled as a diagnostic in the code

`Report` renames it `pooled_sign_test_p` and carries the reason in a comment
beside the field: it pools types that are different experiments, `Express to
Local` has read approximately 1.00 on every window ever graded, and the 11-of-15
to 12-of-22 collapse was composition rather than evidence. `by_type` now carries
its own per-type sign test, and nothing in that table is significant — 4 of 6
at p=0.688 is the best part suspensions can claim.

What is still true after all of this: part suspensions are the only type with a
magnitude worth the name, at ~1.2x control on six windows, agreeing with an
independent difference-in-differences estimate. That has survived every attempt
to break it. It is still not significant.

## 2026-08-17 — recurring work does not need a different control, it needs coverage: the answer key is six days deep and every comparable day carries the work

origin: self

The bug said `Express to Local` recurs daily, so its adjacent-day control is the
same work, so that subset "needs a different comparison — a non-adjacent
unaffected day, or the local as a within-window control." I built two of those
comparisons and measured a third. All three fail, and the premise is wrong: the
control definition was never broken.

### The estimator already admits a free day

`_is_control` restricts nothing to adjacent days. Given the nearest free
comparable weekday for the R's 12:45-18:00 reroute, a traversal at Fri 08-07
12:55 passes every filter — service class 0 == 0, clock band true, blackout
clear, `_is_control` true — with no code change. The 15 ungraded windows are not
waiting on a statistic. They are waiting on archive.

### What the 15 actually are, and it is broader than the bug says

| work               | windows | cadence                                    |
| ------------------ | ------- | ------------------------------------------ |
| 7 Express to Local | 6       | every weekday, 06:15-10:00 and 15:00-22:00 |
| R Reroute          | 6       | every weekday, 07:15-09:00 and 12:45-18:00 |
| N Stops Skipped    | 2       | every Sat and Sun, 06:00-23:00             |
| 4 Stops Skipped    | 1       | continuous, 2026-04-27 to 2026-08-18       |

Not one class but four cadences, including a 113-day continuous window that no
adjacent-day control could ever serve.

### Negative: the out-of-band control does not recover the primary

Same service class, same day, the hours the work does NOT run, normalised the way
07953e6 normalised the duration effect — `vanished` on hops touching a named stop
minus `vanished` on hops touching none, so pattern diversity in the reference
cancels. Graded against the matched-day `vanished` on the 12 service-rows where
both exist:

|                | matched-day | out-of-band      |
| -------------- | ----------- | ---------------- |
| mean           | 0.1976      | **-0.0616**      |
| median         | 0.1250      | **-0.0684**      |
| pearson r      | —           | **0.564** (n=12) |
| sign agreement | —           | **6 of 12**      |

Sign agreement at chance, and biased 0.26 the wrong way: it reports "nothing
detected" where the primary sees a quarter of pairs vanish. The D row is the
mechanism — matched `vanished` 0.0, out-of-band effect **-0.5078**, because the
unnamed arm lost 51% of its pairs to out-of-band pattern diversity while the
named arm lost none. Subtracting two unequal inflations does not cancel them.
Rejected.

### Negative: the static schedule cannot be the reference for this work

The obvious escape from needing an unaffected PERIOD is an unaffected SCHEDULE —
the booked stop pattern, pre-registered and immune to recurrence. It is
contaminated at exactly the window that motivated the bug. Feed
`20260807-H-rockaways-extension-removed`, covering 2026-05-26 to 2026-10-31,
carries these path codes for the 7 on the window's service day:

`7..N27R, 7..N35R, 7..N36R, 7..N38R, 7..S95R, 7..S96R, 7..S97R, 7..S99R`

Zero 7X patterns. The supplemental feed has already absorbed the Express to
Local, so the schedule agrees with the disrupted movement by construction —
graded against itself, passing no matter how bad the model is. Meanwhile the
realtime feed still publishes 787 trips inside the window declaring `7X..N`, of
which 670 hops are consecutive local stops and 117 skip 3-4 stations. The
movement shows the work; the schedule cannot certify it.

### Negative, caught before it shipped: a countdown to a control we cannot certify

Having concluded the fix was a diagnostic — report how far the archive must reach
— the first version searched +/-60 days and named 2026-08-04 as the 7's nearest
free weekday, 8 days back. It is unbacked. `archive/windows/` holds **6
publication days** (2026-08-12 to 08-17), and an alert that ran and expired
before the first snapshot is absent from the record, not inactive. Absence of an
announcement is evidence of a free day only where we hold snapshots. Bounded to
certified days, every one of the 15 reads:

|                              |                            |
| ---------------------------- | -------------------------- |
| `coverage_no_control_period` | 15                         |
| `coverage_reach_unknown`     | **15**                     |
| `answer_key_days`            | 6                          |
| weekday windows              | certified 4, covered 4     |
| N weekend windows            | certified **1**, covered 1 |

`certified == covered` everywhere: every comparable day the answer key can speak
for carries work on the route. The N's weekend windows have exactly one candidate
day in six days of coverage, which is the whole finding in one number.

### Three defects in the diagnostic itself, all found in review

All three would have certified a control that never existed.

- Coverage read as `first..last` rather than as a set. `load_windows` supports
  non-contiguous coverage, so a hole between the endpoints would have counted as
  evidence of free service.
- The window was excluded from its own blackout and only its START date skipped,
  so for the 4's 2026-04-27 to 2026-08-18 closure, 113 days of the closure would
  have certified as a control for itself.
- `_band_on` projected the clock band as `midnight + seconds`. On 2026-03-08,
  which springs forward, local 05:00 is not 18,000 seconds after midnight — the
  band landed at **06:00**, an hour off, twice a year. Both endpoints are now
  built with `datetime.combine`, the way `split_by_local_day` already builds day
  boundaries.

### Where this leaves the measure

Nothing about the grade changed and no published number moves: 13 graded, 22
duration-graded, medians unchanged. What changed is that `NO_CONTROL_PERIOD` no
longer reads the same whether a control is one week out of reach or cannot exist.
Both archives retain 3,650 days and began 2026-08-12, so the countdown starts
there and the windows grade themselves as coverage accumulates — the 7's own
announced record shows free weekdays from 2026-09-07, which the record will be in
a position to certify by then. The gradeable supply today is 13 of 28 over the
span, not 28, and not the headline 593.

## 2026-08-17 — the reach diagnostic read one band off a four-piece closure, and the tests passed because my multi-day case never crossed a service class

origin: artifact

The pre-commit gate blocked the entry above with a fourth defect in the same
diagnostic, after 839 tests and both linters had passed. It is the same mistake as
the other three, which is the part worth recording: the diagnostic answering a
slightly different question than the grade asks. I wrote that exact warning into
`_band_on`'s docstring and then made the error three more times.

### What it got wrong

`control_reach` read one clock band and one service class off the UNSPLIT window.
The grade does not: `pattern_shift` and `control_supply` both iterate
`split_by_local_day`, because a piece is the only thing carrying one band and one
class. A Friday-23:45-to-Monday-05:00 closure is four pieces on three timetables —
Friday late night, all Saturday, all Sunday, Monday morning — and its Saturday
piece needs a SATURDAY control.

Read unsplit, only Friday's weekday class is ever considered, so a free Saturday
in the record is not even a candidate. On that fixture:

|                     | day        | lag  | certified | covered |
| ------------------- | ---------- | ---- | --------- | ------- |
| unsplit (rejected)  | **None**   | None | 1         | 1       |
| per-piece (shipped) | 2026-08-08 | -7   | 3         | 2       |

It reports no reach where a control exists — the opposite direction from the
earlier 2026-08-04 defect, and just as wrong.

### Why the tests missed it

`test_a_multi_day_closure_blocks_its_own_later_dates` used Wed 23:45 to Fri 05:00.
Three pieces, all weekdays, all one service class — so it exercised splitting
without ever crossing the boundary that splitting exists for. A multi-day fixture
that stays inside one timetable is not a multi-day fixture. The replacement runs
Friday to Monday and fails against the unsplit logic.

### What moved in the published numbers

Nothing that matters, and one thing worth noting. Still 15 blocked, all
`certified == covered`, `coverage_reach_unknown` 15. The 4's 2026-04-27 to
2026-08-18 closure now reports **360** candidate (piece, day) pairs rather than 4,
because its 113 daily pieces are each searched against the six covered days — the
earlier 4 was the unsplit window's single band, which is precisely the bug. The
`Express to Local` duration median read 1.0118 then 1.0124 across the two runs at
identical n_rows=6; that is the live day accumulating twelve more minutes of
trace between them, not the change.

## 2026-08-17 — the reach blackout filtered routes when building the set and then ignored them when testing it, blocking the 2 for work on the 5

origin: artifact

A second gate round on the same diagnostic, and the fifth instance of the same
mistake. `control_reach` built its blackout from windows sharing a route with the
target, then tested only temporal overlap — so for a window naming two routes, work
on either one marked the day covered for both.

That is a false negative, and the grade disagrees: `coverage_state` calls a window
graded when ANY service pairs, so a window on the 2 and 5 against work running on
the 5 alone still has a free control arm on the 2. On that fixture:

|                                       | day        | lag  | certified | covered |
| ------------------------------------- | ---------- | ---- | --------- | ------- |
| routes ignored at the test (rejected) | **None**   | None | 2         | 2       |
| per-route (shipped)                   | 2026-08-13 | +1   | 2         | 1       |

Not hypothetical: **13 of the 593** announced windows in the record name more than
one route, the 2/5 reroute among them. A day now counts as covered only when EVERY
route the window names is blacked out over the band.

No published number moves — 13 graded, 22 duration-graded, all 15 blocked windows
still `certified == covered`, coverage medians byte-identical to the run before the
diagnostic existed. The fix only ever changes multi-route windows, and none of the
15 currently blocked are.

## 2026-08-17 — the reach diagnostic rejected a whole band on any overlap while the grade only rejects an instant: five of six defects were the same disagreement

origin: self

Sixth defect in `control_reach`, found after the commit was already pushed. It
judged a candidate band covered if ANY blackout overlapped it. `_is_control` does
not: it tests one traversal's own timestamp against `Window.contains`, so work
occupying part of a band leaves the rest admissible.

Against a 12:00-16:00 band with a 12:00-13:00 closure on the candidate day:

|                         | day        | lag  | certified | covered |
| ----------------------- | ---------- | ---- | --------- | ------- |
| band-level (rejected)   | **None**   | None | 2         | 2       |
| instant-level (shipped) | 2026-08-13 | -1   | 2         | 1       |

And the grade agrees with the second one, which is the part that settles it:
`control_supply` on the same blackout returns `{'J': 1}` for a traversal at 14:00
and `{}` for one at 12:30. The control arm is real; the diagnostic was denying it.

Fixed by sweeping the band and asking whether any instant escapes every blackout,
with spans sorted because blackouts can overlap each other. Both ends closed,
matching `Window.contains`, so a blocked span resumes at `hi + 1`.

### The pattern, which is the actual lesson

Six defects in one small diagnostic, and five were the same mistake: the
diagnostic answering a question slightly different from the one the grade asks.
One band instead of per-piece bands. Routes filtered when building the blackout
but ignored when testing it. A whole band instead of an instant. Each was a false
reading in production output, and none was caught by the tests I wrote for it —
the multi-day fixture stayed inside one service class, the route fixture named one
route, the overlap fixtures used exactly-aligned bands.

The guard that works is not another unit test of the diagnostic in isolation. It
is asserting the diagnostic against the MEASURE: the new test drives
`control_supply` over the same blackout and requires it to admit a traversal
exactly where reach claims a control exists. Any future divergence between the two
now fails a test rather than publishing a number.

### What moved

Nothing in the published numbers, again: 13 graded, 22 duration-graded, 15 blocked
all `certified == covered`, coverage medians byte-identical to the run before the
diagnostic existed. Today's recurring work occupies bands that align exactly with
the windows' own, so no candidate band was ever partially covered. The fix changes
readings only where a blackout overlaps part of a band, which the archive will
produce as soon as the announced work stops lining up this neatly.

Two existing expectations moved with it, both correctly. The Wed-23:45-to-Fri-05:00
fixture went from covered 7 to **5**: its Wednesday whole-day band has admissible
hours before 23:45, and the grade would take them. And the Friday-to-Monday
fixture needed its partial blocker widened to the full 00:00-05:00 band to still
isolate the Saturday, because a 02:00-03:00 closure no longer blocks 00:00-02:00.

## 2026-08-20 — recalibrating the movement disrupted arm to match assigned_n was rejected: the two feeds measure different things, and the trains that run are moving fine

origin: self

With the independent assigned_n truth wired into review (0pb), the movement-primary
published arm scored only 0.144 disrupted-recall against it, so the obvious next
move was to loosen the movement disrupted threshold (`DISRUPTED_RATIO`, currently
0.5·p0) and catch more. Ran the cross-tab first. It killed the idea.

Over 2026-08-15..08-20, against a causal advance baseline (2026-08-01..08-14), of
the 785 assigned_n-disrupted route-ticks:

| what the movement feed sees                         | ticks | share |
| --------------------------------------------------- | ----- | ----- |
| unjudgeable — too few matched trips / no baseline   | 550   | 70%   |
| no movement row at all                              | 23    | 3%    |
| judgeable, and reads **normal** (advancing >0.5·p0) | 212   | 27%   |
| judgeable, reads **disrupted**                      | **0** | 0%    |

For the 212 judgeable cells the worst-direction posterior advance ratio is median
**0.75·p0**, min 0.53, and a quarter sit at or above **1.0·p0**. Not one dropped
below the disrupted floor. Lowering `DISRUPTED_RATIO` to 0.8 to catch them would
fire "disrupted" on routes advancing at three-quarters of a noisy baseline — well
inside normal variation — corrupting what the movement arm means.

The reading is that assigned_n collapse (the dispatcher pulled trips — a SUPPLY
cut) and movement-disrupted (the trains present are FROZEN — a FLOW problem) are
different axes of "degraded." A route routinely has one without the other, and the
70% unjudgeable is the fingerprint of the supply cut: pull the trips and movement
has too few left to judge. So the low cross-recall is correct, not a calibration
miss. 07h's premise — recalibrate to raise recall — is wrong; the follow-up is to
publish assigned_n as its own service-level axis rather than fold it into the
movement disrupted state.

## 2026-08-22 — the drawer's alert-HMM posterior is one-hot on 24 of 29 routes, and its regime clock resets on 14 of them in a single tick

origin: self

Reading the route drawer against the live feed to explain "P(suspended) 100.00%,
Regime age —" on the 2, the J and the 4. The section is honest about _which_ arm it
shows (the alert-aware read, not the published movement condition), but two of its
four numbers are not usable as displayed.

In one snapshot (generated_at 1787432444), of the 29 routes carrying an inference:

| what the alert filter shows                                  | routes  |
| ------------------------------------------------------------ | ------- |
| posterior numerically one-hot (max p > 0.999999)             | 24 / 29 |
| `regime_age_seconds == 0` — argmax flipped on this very tick | 14 / 29 |

The posteriors are not near-certain, they are saturated: the 2 reads
`p_normal = 2.0e-44`, `p_disrupted = 4.1e-33`, `p_suspended = 1`. Six independent
likelihood channels multiply per tick (`logEmission`: Poisson count, four
Bernoullis, binomial movement, Gaussian service), so the log-odds run to hundreds
and the argmax is decided by whichever channel moved last. That is why half the
system changes regime in the same tick — and since `regime_entered_at` only
advances on an argmax change (`hmm.ts:192`), the drawer's "Regime age" is the age
of that flapping alert argmax, NOT the age of the movement regime the badge is
published from. Age "—" is `fmtMinutes(0)`: it flipped this tick.

Two display consequences worth separating from the model problem: `fmtMinutes`
rounded the hour remainder on its own, so 1379.7 minutes printed as "22h 60m"
(fixed); and for a movement regime in state `normal`, `movementRecovery` returns
`recovery_minutes = 0` by construction and `p_normal_in_30` is P(_stays_ normal),
so the "Recovery forecast" block renders "Median —" against "P(normal in 30m) 93%"
— the label is wrong for normal routes, not the number.

Also confirmed the supply axis reads far above baseline routinely, not just below
it: same snapshot has the 2 at 217%, the J at 175%, the 1 at 169%, while the axis
only degrades below 50%. A bare "Normal / 217%" pair is unreadable without the
thresholds, which is what the new drawer meter marks.

## 2026-08-22 — a fixed ratio-vs-median threshold cannot mark "more trains than usual": p90/median spans 1.09x to 12.0x across cells, and MAD is degenerate in 14% of them

origin: agent

Marking the supply axis's over-baseline readings on the route cards needed a
threshold, and ">100% of the hourly median" was the obvious one. It fires on the
majority of the grid. Measured over the same 28-day window the live sidecar is
built from (2026-07-26..08-22, 8,065 archived trip-updates ticks, 193,017
(route, tick) samples, 1,129 cells at >=20 samples), for the MEDIAN cell:

| threshold vs own cell median | share of that cell's own history flagged |
| ---------------------------- | ---------------------------------------- |
| > 1.00x                      | 35.8%                                    |
| > 1.10x                      | 16.7%                                    |
| > 1.25x                      | 1.7%                                     |
| > 1.50x                      | 0.0%                                     |

Note it is 35.8% and not ~50%: `assigned_n` is a small integer count, so the
median sits ON a heavily-tied value — the median cell has 29.2% of its readings
exactly AT its median (max 100%). A median partitions the cell's historical
values with ties, and says nothing about what share of a cross-sectional
snapshot will exceed it. Claiming "half by construction" is wrong twice over.

**The reason no single multiple works is that the ratio is not comparable across
cells.** Measured p90/median: median 1.14x, p25 1.09x, p75 1.29x, max 12.00x
(FS we22 has median 1 train and p90 of 12). So 1.14x is an ordinary afternoon
for one cell and a once-a-month event for another. At the small end the ratio is
pure quantisation: GS/FS at median 1 train means one extra train is +100%, while
the F and 7 at rush (median 37-38) price the same extra train at +3% — a 33x
spread in the number for an identical physical event.

Robust z-score via MAD is not the fix either: MAD is exactly 0 in 161 of 1,129
cells (14%), IQR in 86 (8%), because a line that always runs the same count at
3am has no spread to normalise by. A modified z-score is undefined or infinite
for one cell in seven.

What this points at is per-cell empirical quantiles in the baseline sidecar
(p10/p50/p90 beside the median it already ships), scored as "above this cell's
own p90" rather than any global multiple — the same shape `compute_dwell_quantiles`
already uses for dwell. Not built yet; the card threshold moved to a global 1.25x
as a stopgap that at least does not fire on most of the grid (13 of 22 judgeable
routes at 1.00x, 4 of 30 at 1.25x).

Loose thread worth pulling: > 1.50x flagged 0.0% of the median cell's history in
this window, yet the live snapshot minutes later had the 1 at 170% and the 2 at
188% at we23. Either late-Saturday service is genuinely off the historical
distribution or the denominator for those cells is wrong.

## 2026-08-22 — the supply denominator is not soft: the 1 and 2 really did run weekday-level service on a Saturday. But the ratio cannot tell a rare day from a recurring second mode, and a per-cell percentile can

origin: agent

Chased the suspicion that the we23 baseline for the 1/2 was wrong, because they
read 170%/188% of their usual late-Saturday service. The denominator is fine and
the readings are real. What is wrong is the measure.

Per-ET-date median `assigned_n` across all hours, weekends only, over
2026-07-20..08-23 (9,935 archived ticks):

| route | 07-25 | 08-01 | 08-08 | 08-15 | 08-22  | weekday level |
| ----- | ----- | ----- | ----- | ----- | ------ | ------------- |
| 1     | 13    | 13    | 12    | 13    | **20** | 21-23         |
| 2     | 11    | 11    | 11    | 11    | **21** | 25-26         |
| C     | 11    | 11    | 9     | 10    | 11     | 14-15         |

On Saturday 08-22 the 1 and 2 ran at nearly weekday levels all day while the C
ran an ordinary Saturday. No `Extra Service` alert was published for either line
— `assigned_n` is the only witness. So the axis worked: it caught a real service
anomaly the alert feed never announced.

**The finding that matters for the measure.** Those two readings look identical
as ratios and are nothing alike as percentiles:

| cell     | n   | median | p90 | p95 | observed | ratio | percentile | >1.25x median? | >p90?  |
| -------- | --- | ------ | --- | --- | -------- | ----- | ---------- | -------------- | ------ |
| 1 @ we23 | 120 | 11     | 13  | 17  | 18       | 164%  | 96th       | yes            | yes    |
| 2 @ we23 | 120 | 9      | 16  | 16  | 16       | 178%  | 88th       | yes            | **no** |

The 2's cell is BIMODAL — value counts `{6:1, 7:5, 8:42, 9:42, 10:6, 14:8, 15:1,
16:15}`, the 14-16 mode occurring on 2 of 5 weekends. Sixteen trains is that
line's ordinary high mode, sitting exactly at its own p90, and the ratio calls it
a bigger deviation (178%) than the 1's genuinely once-a-month 18 (164%). A
median-plus-multiple cannot represent a two-mode cell at all; the percentile
separates them correctly with no tuning.

This is the concrete case for moving the axis off ratio-vs-median and onto
per-cell quantiles, beyond the cross-cell comparability argument recorded above.

## 2026-08-23 — the Models page reliability panel graded the movement forecast against the alert arm: same forecast, AUC 0.406 published vs 0.961 correct

origin: self

`p_normal_in_30min` is published only when `recovery_source` is `movement` or
`schedule` (`worker/src/snapshot.ts:878`) — it is a movement dwell-curve survival
probability. `calibrate()` took its outcome from `future.condition`
(`training/eval.py:670`), the alert-shadow regime. `published_arm()` had existed
for exactly this since the two arms diverged, and `_grade_recovery` already used
it; `calibrate()` was never migrated.

`build_eval` computed both blocks all along. `build_calibration` published only
the shadow one, dropping `calibration_movement` and `calibration_arm`, so the
page could never see the correct pairing and the feed did not even say which arm
it graded.

Live feed 2026-08-22 (`code_sha 1982fd6`), 30-minute horizon, one forecast:

| graded against                             | n      | Brier  | BSS vs clim | AUC       |
| ------------------------------------------ | ------ | ------ | ----------- | --------- |
| `condition` (alert-shadow) — published     | 35,078 | 0.0175 | -0.031      | **0.406** |
| `published_condition` (movement) — dropped | 34,740 | 0.0009 | -0.108      | **0.961** |

The tell was in the strata. Shadow `not_normal_now`: `mean_pred` 0.9939 against
`mean_outcome` 0.5049, and 35,017 of 35,078 samples in the single [0.9,1.0)
reliability bin. That reads as a degenerate constant forecast, but the forecast is
not constant — it is conditioned on the _movement_ state while the stratum splits
on the _alert_ condition, and the two are nearly independent. Sharpness of 0.99
against a realized 0.50 was cross-arm disagreement being reported as forecast
error.

**The correction is not "publish the better number".** The movement arm has
`not_normal_now` n=**4** and `unknown_share` 0.247, so its AUC 0.961 rests on 28
non-normal outcomes out of 34,740, and its BSS vs climatology (-0.108) is _worse_
than the shadow arm's (-0.031). Swapping 0.406 for 0.961 would have moved the
dishonesty rather than removed it. Both arms now ship with their label, sample
count, and coverage, and the panel draws both series.

Two things this did not fix, recorded so they are not mistaken for it:

- Horizons 60 and 120 grade n=0 permanently. Those fields are emitted only for
  `recovery_source === 'schedule'` (`worker/src/snapshot.ts:881-884`) and
  `calibrate()` skips every schedule row (`training/eval.py:653-655`). The gating
  was deliberate (journal 2026-08-12, BSS -0.00 to -1.30) but the page still drew
  two blank calibration plots, which read as "no skill" rather than "withheld".
  The panel now says which it is; whether the horizons should exist at all is open.
- Recovery interval coverage is 1.8% against a nominal 50%, and both arms lose to
  per-route climatology. Neither is a reporting artifact. The 99.1% normal base
  rate makes "always say normal" a genuinely hard bar, and with 608 not-normal
  samples on the shadow arm and 4 on the movement arm, this window cannot
  distinguish a good model from a bad one on the cases that matter. That is a
  collection problem, not a charting one.

## 2026-08-23 — one "Recovery forecast" heading was answering four different questions, and three of them wrongly

origin: self

Correction to the entry above, which called the regime-probabilities filter an
"alert-HMM" and its one-hot posterior "saturated, not confident". Both are wrong
framings. `logEmission` (hmm.ts:134-157) folds in the binomial movement and
Gaussian service channels whenever they are available, one tick lagged
(index.ts:361-378), so the filter is alert-PRIMARY, not alert-only. And one-hot
IS its confidence — the problem is that the confidence is miscalibrated and
cannot hedge, not that it is missing.

The drawer fix: the published numbers under one "Recovery forecast" heading mean
four different things, and the reader had no way to tell which. Mapping the live
snapshot's 29 inference-carrying routes onto the branches:

| what the numbers actually mean                                                   | routes |
| -------------------------------------------------------------------------------- | ------ |
| published normal — median 0/0/0 by construction, horizon is P(_stays_ normal)    | 19     |
| published unknown — we declined to judge, so every number is withheld            | 7      |
| `not_scheduled`, `recovery_source: 'hmm'` — no announced resume to count down to | 3      |
| real time-to-normal estimate off the dwell curve                                 | 0      |

The 19 normal routes were rendering "Median —" (that's `fmtMinutes(0)`) directly
above "P(normal in 30m) 93%", which reads as a broken forecast rather than "this
line is fine and 93% likely to stay that way". The 3 not_scheduled routes took the
withholding path, so a naive "the status came from train movement" message would
have been a lie about them — their condition comes from the schedule.

Also: `recovery_indeterminate` is two different silences. The movement arm sets it
when the regime outlived every measured dwell OR when the fitted curve's median
hits the 24h clamp (snapshot.ts:974), so copy asserting "beyond every one we've
measured" is wrong for the clamp case; the message now names the ceiling that
`recovery_minutes` already carries. The hmm arm sets it via the withholding gate
(snapshot.ts:855), which is not a bounding failure at all.

## 2026-08-23 — "collection problem" was wrong: 91 days are archived, eval grades 7, and a 7-day window's real support swings 17.8x

origin: self

An earlier entry today closed by calling the thin disrupted sample "a collection
problem, not a charting one." That was wrong on both halves, and the numbers say
so.

**There is no shortage of archive.** `v1/predictions/` holds 24,972 files across
91 days (2026-05-25..2026-08-23, 705,344 rows). `training/eval.py`'s `--days`
defaults to 7 with no upper bound, and `training/prune.py:61` retains 90. So the
published grading reads 7 of 90 available days, and widening it is an argument,
not a data-collection campaign.

**The per-tick n was never the sample size.** Six consecutive 7-day windows on
the published arm — tick counts indistinguishable, support wildly not:

| window start | tick-rows | episodes |
| ------------ | --------- | -------- |
| 2026-08-16   | 58,464    | 26       |
| 2026-08-09   | 58,551    | **89**   |
| 2026-08-02   | 58,464    | 16       |
| 2026-07-26   | 58,464    | **5**    |
| 2026-07-19   | 57,868    | 8        |
| 2026-07-12   | 56,280    | 8        |

A flat advertised n of ~58k hides a **17.8x** swing in independent evidence. Any
week-over-week metric move on this page was dominated by which week got sampled.
The current window's 26 incidents sit behind a displayed 52,287 — a ~2,000x
overstatement of independent samples.

**Negative result, with the number that killed it.** The first cut of
`episode_support` fed the raw window to `scorecard.model_episodes`, which resolves
the arm via `published_arm` — and that falls back to the alert-shadow `condition`
for rows predating 2026-07-11. Over the full archive it scores **1,781** episodes
against **153** on the published arm alone: a 12x inflation, entirely the shadow
arm's flapping, reported under a movement-arm label. Fixed by filtering to rows
carrying `published_condition`, counting the excluded legacy rows, and publishing
the span actually covered — so `--days 91` now reports 153 episodes over 43d
covered with 348,516 rows excluded, rather than a number that looks 12x better.

Also: the machinery already existed. `training/episodes.py` and
`training/scorecard.py` have segmented and graded episodes on the right arm for a
while; the counts just went to `docs/review/<date>-shadow-hmm/summary.json` and
never to `v1/eval.json` or `v1/calibration.json`. The page was tick-only because
nothing carried the episode count to it, not because it was hard to compute.

Standing constraint the fix cannot lift: `published_condition` is 43 days old, so
the movement arm's support grows with the calendar and caps at the 90-day
retention. 28d is the conservative widening (fully inside the published era);
beyond ~43d there is nothing yet to read.

## 2026-08-23 — the clamp fights the ceiling on every route because "disrupted" is 94% ordinary Delays, not because the corpus is thin

origin: self

Chasing why `_cap_self_loops` fires on 23/28 routes (normal), 28/28 (disrupted)
and 27/28 (suspended). Two hypotheses died on the way to the answer; both are
recorded with the number that killed them.

**Dead hypothesis 1 — "the 14-day training corpus is too thin."** The code comment
at `train_em.py:113` says EM on a thin or mostly-quiet corpus drives self-loops
toward 1.0, so widening the corpus should let real per-line dynamics emerge.
Patched `_cap_self_loops` to record the pre-clamp diagonal and refit both widths:

| corpus  | normal over cap | disrupted | suspended | mean excess     |
| ------- | --------------- | --------- | --------- | --------------- |
| 14 days | 21/28           | 22/28     | 14/28     | +0.014 … +0.046 |
| 28 days | 23/28           | 18/28     | 19/28     | +0.016 … +0.048 |

Doubling the data changed nothing. EM wants stickier regimes at any corpus size,
so the clamp is not a thin-data artifact and widening the corpus buys nothing.

**Dead hypothesis 2 — "the ceiling is calibrated circularly."** `MAX_SELF_LOOP`
was set from median regime dwell in `v1/regime_transitions`, and that stream is
the filter's own argmax flips — `worker/src/grading.ts:11-13` literally calls it
"ground-truth dwell times." A loop on paper. But the numbers refuse to close it:
the cap implies a **48 min** median dwell and the filter's measured argmax dwell
is **15 min** (median, n=231 over 28d). The cap is ~3x _looser_ than the
behaviour it supposedly came from, so it is not enforcing its own justification.

**What is actually true.** Of the 1,086 route-ticks where the filter says
not-normal, split by the severity of the alert behind them:

| tier | meaning           | ticks     | share              |
| ---- | ----------------- | --------- | ------------------ |
| 3    | suspension        | 2         | 0.2%               |
| 2    | severe delays     | 60        | 5.5%               |
| 1    | ordinary `Delays` | **1,024** | **94.3%**          |
| 0    | planned work      | 0         | correctly excluded |

The latent "disrupted" regime is routine `Delays`, which the MTA posts constantly.
Planned work is properly excluded, so that earlier suspicion was wrong too. And
the duration distributions explain the EM fight exactly:

| population                   | n     | median | mean    | mean/median |
| ---------------------------- | ----- | ------ | ------- | ----------- |
| raw alert presence, any tier | 1,692 | 55 min | 284 min | 5.2x        |
| raw alert presence, tier>=2  | 45    | 50 min | 56 min  | **1.1x**    |
| filter argmax not-normal     | 231   | 15 min | 24 min  | 1.6x        |
| movement arm                 | 136   | 5 min  | 11 min  | 2.2x        |

Severe events are nearly symmetric and perfectly representable by a geometric
self-loop. The heavy tail is _entirely_ tier<2. A single self-loop fitted across
both populations splits the difference — EM lands on 0.965 (~97 min) and the cap
clips to 0.930 (~48 min), which happens to sit near the tier>=2 median of 50 min.
The cap is accidentally about right for real disruptions, which is probably why
this went unnoticed.

The missing input is severity, and it is missing by construction:
`training/load.py:36-38` — "Counted (HMM-included) alert_types active on this
route-tick. Only the R2 truth builder populates it — used by the review to grade
ground truth by severity; **the HMM training path leaves it empty**." Grading is
severity-aware (`CANONICAL_SEVERITY_FLOOR`, truth_version 2 = severe-only); the
model is not. It predicts when the `Delays` banner clears and is scored on when
the severe event ends.

That single mismatch is the best available explanation for the two worst numbers
on the page — recovery MAE 74 min and IQR coverage 1.8% against a nominal 50% —
since both are "how long will this last" inheriting a dwell learned from a
population 94% composed of the wrong events.

## 2026-08-23 — the movement binomial is the only emission channel that scales with fleet size: 0.192 nats per matched trip under bootstrap params, while 99.83% of 35,492 graded forecasts sit in a single bin

origin: agent

The 2026-08-22 entry above established that the posterior is one-hot on 24 of 29
routes and named the cause as "six independent likelihood channels multiply per
tick, driving log-odds to hundreds". It did not say which channel. Two separable
things below: a structural fact about the channels, and a measurement of the
output. The bridge between them is NOT established — see the limits at the end.

**Structural.** In `_log_emission` (`src/momentarily/hmm.py:366-400`, mirrored in
`worker/src/hmm.ts:134-158`), six of the seven channels evaluate one scalar or
one flag per tick, so their log-likelihood ratio between states is a constant
that does not depend on how much data the tick contained. The binomial movement
channel is the exception: `_log_binomial(advanced_n, matched_n, rate)` treats the
cross-matched trips as `matched_n` independent Bernoulli trials of the route's
hidden state, so its LLR is `matched_n * KL(rate_normal || rate_disrupted)` —
linear in the trip count. This holds for any parameter values; only the
coefficient changes. `matched_n` is `advanced_n + stalled_n` over both directions
(`worker/src/movement_state.ts:380-402`, `worker/src/vehicles.ts:116-190`), with
`MIN_MATCHED_TRIPS = 3` as a floor and **no cap**.

**Sized with the checked-in bootstrap emissions only**
(`training/run_filter.py:34-52` plus the `EmissionParams` defaults at
`src/momentarily/hmm.py:131-136`), on an observation with no alerts,
`service_ratio` 1.0, and `advanced_n` at exactly `0.6 * matched_n` — the normal
state's own advance rate, so there is no anomaly anywhere in the input. LLR
normal vs disrupted, nats:

| channel                         | n=3      | n=20     | n=40     | n=80      | n=160     |
| ------------------------------- | -------- | -------- | -------- | --------- | --------- |
| Poisson `alert_count`           | 3.70     | 3.70     | 3.70     | 3.70      | 3.70      |
| Bernoulli `has_delays`          | 0.90     | 0.90     | 0.90     | 0.90      | 0.90      |
| Bernoulli `has_service_change`  | 0.90     | 0.90     | 0.90     | 0.90      | 0.90      |
| Bernoulli `has_planned`         | 0.86     | 0.86     | 0.86     | 0.86      | 0.86      |
| Bernoulli `has_suspended_alert` | 0.05     | 0.05     | 0.05     | 0.05      | 0.05      |
| Gaussian `service_ratio`        | 0.89     | 0.89     | 0.89     | 0.89      | 0.89      |
| **Binomial movement**           | **0.83** | **3.84** | **7.68** | **15.36** | **30.73** |
| total                           | 8.12     | 11.14    | 14.98    | 22.66     | 38.02     |
| movement share of positive LLR  | 10%      | 35%      | 51%      | 68%       | 81%       |

Per-trip discrimination here is `KL(0.6 || 0.3) = 0.192` nats, contributed
whether or not anything is wrong, and the crossover where movement carries the
majority of the evidence is around `matched_n = 40`. For contrast the same
observation with trips advancing at the disrupted rate (`0.3 * matched_n`,
n=160) gives **-22.11** nats: under these params the channel is a hard
classifier, not a nudge. A settled prior mixed through the bootstrap transition
matrix predicts `(0.95, 0.04, 0.01)`, worth `ln(0.95/0.04) = 3.17` nats of pull
toward staying put, which 38 nats of emission would beat 12:1 — the shape that
would produce one-hot posteriors and single-tick argmax flips together rather
than as two symptoms.

**Measured, independent of the above.** Over the 35,492 graded 30-minute
forecasts in the public `v1/eval.json` (window ending 2026-08-23):

|                              | alert-shadow arm             | movement arm    |
| ---------------------------- | ---------------------------- | --------------- |
| forecasts in the 0.9–1.0 bin | 35,432 / 35,492 (**99.83%**) | 35,097 / 35,157 |
| forecasts in 0.8–0.9         | 60                           | 60              |
| forecasts below 0.8          | **0**                        | **0**           |
| AUC                          | 0.375                        | 0.361           |
| BSS vs climatology           | -0.033                       | -0.534          |

And the one subset where the answer should not be "fine", the 552 ticks whose
line is not normal _now_: mean forecast **0.9963**, mean outcome **0.5272**. The
model says 99.6% where the truth is 53%.

Both arms saturate identically, on the same 60 mid-bin predictions, so this is
not the grading mismatch recorded on 2026-08-23 above. Checked deliberately,
because that entry's lesson was that a panel can grade the wrong arm and
manufacture an AUC.

**Limits — what this does not establish.** The live fitted emissions were never
read: `PARAMS_KEY = 'state/params.json'` (`worker/src/params.ts:15`) sits outside
the public `v1/` prefix (`worker/src/index.ts:96`), and no fitted params are
checked into the repo. Raw per-tick `Observation` values are not persisted either
— `PredictionRecord` keeps `p_normal` and friends but drops `alert_count`,
`matched_n`, `advanced_n` and `service_ratio` (`worker/src/grading.ts:28-90`). So
the 0.192 nats/trip, the 81% share, and the `matched_n = 40` crossover are
properties of the bootstrap parameters, not measurements of production, and EM
refits `advance_rate` per route and tod_bin (`src/momentarily/hmm.py:658-832`).
The live runner-up masses are in the right family — route 1 publishes
`p_disrupted = 2.6864e-17`, and `-ln` of that is 38.16 nats — but attributing
those nats across channels needs the fitted params and a real `matched_n`, and
inverting one number into the other assumes exactly the equality that is
untested. Two things would settle it: log the per-channel log-likelihood terms
for one tick from inside the Worker, or persist `matched_n` and `advanced_n` on
`PredictionRecord`.

Same error class as the `severity_sum` Gamma channel removed earlier
(`hmm.py:128-133`, "double-counted the count evidence and saturated the
posterior") — but where that double-counted a bounded quantity, this multiplies
by an unbounded trip count. No fix applied. The direction suggested by the
structure is to score the advance _rate_ once per tick rather than once per
trip, which under bootstrap params would put the channel at 0.192 nats instead
of 30.73 at n=160; untested, and it would need the EM fit redone.

## 2026-08-24 — the recovery grade was measuring an arm mismatch, not the model: MAE 102 → 29 min, IQR coverage 8.7% → 46.9%

origin: agent

Built the per-retrain segmentation the widened eval window needed
(`recovery_by_params_version` in `training/eval.py`, published in `v1/eval.json`
and trimmed into `v1/calibration.json`), and it immediately overturned the
standing causal story for the two worst numbers on the Models page.

**What the segmentation shows.** 28 days, 2026-07-28..08-24, 226,432 predictions,
728 transitions. MAE/IQR are per-tick; `incid` is the incident count on the arm
being graded, which is the number that says how much independent evidence a
segment carries:

| params trained | ticks | incid   | MAE        | IQR coverage | span         |
| -------------- | ----- | ------- | ---------- | ------------ | ------------ |
| 2026-07-26     | 60    | 13      | 30.1m      | 38.3%        | 07-28..08-02 |
| 2026-08-02     | 14    | 2       | 17.9m      | 78.6%        | 08-02..08-03 |
| 2026-08-03     | 0     | 0       | —          | —            | 08-03..08-03 |
| 2026-08-03     | 89    | **33**  | **32.5m**  | **39.3%**    | 08-03..08-11 |
| 2026-08-11     | 16    | 2       | 99.0m      | 6.2%         | 08-11..08-12 |
| 2026-08-12     | 12    | 12      | 11.7m      | 0.0%         | 08-12..08-13 |
| 2026-08-13     | 884   | **197** | **116.8m** | **2.6%**     | 08-13..08-24 |
| pooled         | 1,075 | 259     | 102.3m     | 8.7%         | whole window |

Two segments clear a 20-incident floor and they disagree by 3.6x on MAE and 15x
on coverage. The pooled number sits between them and flatters the model now
running.

**Dead hypothesis — "the 2026-08-13 retrain made the filter flap, and a
dwell-based estimate cannot hit short regimes."** Regime durations moved the
wrong way: median non-normal dwell was 5.0 min before 08-13 (n=78) and 10.0 min
after (n=286), mean 18.3 → 36.9 min. Regimes got _longer_, so flapping is not
what broke the estimate.

**What is actually true.** The predicted quantity collapsed, not the truth. Over
the gradeable population (non-normal on the shadow arm, determinate), split by
the arm that produced `recovery_minutes`:

| recovery_source | n   | median predicted | median IQR width | zero-width |
| --------------- | --- | ---------------- | ---------------- | ---------- |
| `hmm`           | 192 | 50.0m            | 80.0m            | 1%         |
| `movement`      | 883 | **0.0m**         | **0.0m**         | **99%**    |

`movement` became the dominant source at the 08-11 retrain and carried 857 of
884 rows by 08-13. Those rows read zero for a good reason: **870 of the 872
movement-sourced zeros (100%) sit on routes whose `published_condition` is
`normal`**, while the shadow condition that selected them says `disrupted` (515)
or `suspended` (357).

So the movement arm is right. It says "this route is running normally, there is
nothing to recover from" and emits 0 — `worker/src/snapshot.ts:963-967`,
`1029-1033`. The grader picks the row because a _different_ arm calls the route
disrupted, then reads that 0 as a forecast of instant recovery against the
shadow's regime clock. `_grade_recovery` gates ticks on `arm(p)` but grades
`recovery_minutes`, which comes from whichever arm `recovery_source` names, and
the two are not the same arm. The guard for exactly this failure is already in
the code and already describes it (`training/eval.py:861-867`: "A route that is
already normal isn't recovering — it predicts recovery_minutes=0, and grading
that against time-until-the-next-disruption swamps MAE and pins IQR coverage
near zero"); it just tests the shadow stream, which is not the stream the zero
came from.

Excluding movement-sourced rows the way `schedule` rows are already excluded
(`training/eval.py:876`):

|                        | ticks | incidents | MAE       | IQR coverage |
| ---------------------- | ----- | --------- | --------- | ------------ |
| as published           | 1,075 | 259       | 102.3m    | 8.7%         |
| movement rows excluded | 192   | 52        | **29.0m** | **46.9%**    |

Against a nominal 50%, the recovery forecast is close to calibrated. **Not
applied.** It changes published grading semantics and makes the headline numbers
look 3.5x better, which is precisely the kind of change that needs sign-off
rather than an agent's initiative.

**What this does and does not overturn.** The standing diagnosis held that MAE
and IQR coverage were severity conflation — a dwell inherited from a population
94% composed of ordinary `Delays`. That conflation is real and separately
verified: the training path carries no severity input at all. What is now
measured is that it does not _explain these two metrics_. Across the 08-13
boundary the severity mix is flat while the recovery source inverts:

|                    | tier>=2 share | tier 1 share | `hmm` | `movement` |
| ------------------ | ------------- | ------------ | ----- | ---------- |
| pre 08-13 (n=182)  | 3.3%          | 96.7%        | 89.6% | 10.4%      |
| post 08-13 (n=893) | 6.3%          | 93.7%        | 3.2%  | 96.8%      |

Same conflation on both sides — marginally _more_ severe events afterward, which
under the conflation hypothesis should have improved the numbers rather than
degrading coverage from 39.3% to 2.6%. The 2026-08-03 params scored MAE 32.5 min
and 39.3% coverage over 33 incidents under exactly this mix. So the conflation
stays on the books as a modeling problem worth fixing on its own merits, and is
struck as the cause of these two numbers. A three-state severity split was
queued against an acceptance criterion requiring MAE and IQR coverage to move
materially; on this evidence it could not have moved them, and would have been
recorded as a failed diagnosis for the wrong reason.

**Instrument note, learned twice on the way.** A segment's `low_sample` flag
counts incidents, not ticks: the regression test's v200 has more graded ticks
than v100 (50 vs 44) and one incident against 22, so a tick threshold waves
through exactly the thin segment the flag exists to catch. And it counts them
via `per_regime.n` — regimes on the arm being graded — not via `episode_support`,
whose incident count is computed on the published movement arm and therefore
sizes a different metric. Reaching for the adjacent number would have rebuilt
the same arm-mismatch error class one field over, inside the very instrument
built to find it.

## 2026-08-24 — the crowding estimate's unit was wrong by 10x, and the version people actually want doesn't need the ridership feed at all

origin: self

Building the hourly-ridership ingest and the platform-crowding estimate off
it. The design as written was "entry rate x time since last train, published
as '~3 trains' worth waiting'". Both halves of that survived
contact with the data; the copy did not.

Replayed the real trace for 2026-08-20 07:00-11:00 ET (240 minutes, 982
directional platforms) against per-complex hourly entry rates aggregated from
data.ny.gov 5wq4-mkjj over a 4-week window, splitting each complex's rate
evenly across the platforms observed in service. 28,670 observed platform gaps:
p50 6.0 min, p90 13, p99 31, max 232. Implied riders waiting, with gaps over 30
minutes abstained: **p50 28, p90 86, p99 270, max 1302**. A ten-car train holds
roughly 1,000 riders, so p99 is 0.27 of a train and 0.018% of published
estimates reach one train load at all. "~3 trains' worth" is not a rounding
difference from "~28 people" — it is the wrong unit by an order of magnitude.
The surface publishes riders.

The cap is load-bearing, and the reason is the same degenerate-baseline
failure class the movement classifier's terminal over-flagging already names. Uncapped, the top implied-crowding platforms in the window are
G21N Queens Plaza (median gap 232 min, 3,142 riders), R31N Atlantic Av (231
min, 2,049) and A24N 59 St-Columbus Circle (208 min, 2,037) — terminal and
relay platforms whose "gap" is an artifact of not being a through-service
platform in that direction, not a crowd. A 30-minute cap abstains 296 of 28,670
gaps (1.04%) and takes the maximum estimate from 3,142 to 1,302.

The finding that changes what to build next: **expressing crowding relative to
the same platform's own usual crowd cancels the ridership baseline exactly.**
crowd_now / crowd_usual = (rate x minutes_since) / (rate x usual_headway) =
minutes_since / usual_headway. The rate divides out. So the "how bad is it right
now vs usual" gauge needs a scheduled-headway baseline (not implemented) and
needs nothing at all from the ridership feed. The ridership ingest
earns its keep on exactly two things the ratio cannot do: the absolute
headcount, and ranking platforms against each other at one instant. Worth
knowing before anyone wires the ridership artifact into a percentile gauge.

Third thing, a collection footgun. Deriving "a train cleared this platform"
from `stopped === true` sightings alone misses **7.0% of station departures**
(1,078 of 15,382 measured over a 3-hour window): NYCT dwells are frequently
shorter than the 1-minute poll, so a train can arrive and leave between two
snapshots and never be seen stopped. A stopped-only rule therefore reports
last_train_at one full headway stale on 7% of platforms, which roughly doubles
the estimate there. `crowding.ts` carries its own trip->stop map and counts a
departure when a trip's stop_id changes, which by construction catches every
observed departure. That carry is deliberately NOT vehicle_stops.json — see the
step-0 hazard comment in index.ts; it lives in state/station_wait.json, is
updated with `max`, and is therefore idempotent under a retried cron minute.

Two things the estimate structurally cannot see, both published in
`platform_crowding.method.excludes` rather than left as folklore: free
in-system transfers are never fare-swiped, so a transfer complex undercounts;
and the hourly feed has no exits column at all (the legacy 4-hour turnstile
data that had one was retired end-2022), so a platform crushed by an arriving
train letting out reads as empty.

Postscript, because the departure rule got challenged in review and the
challenge was reasonable-sounding. The objection was that stamping a platform
on every `stopped === true` sighting measures time since ARRIVAL, and that a
departure should instead be counted only when the immediately preceding
observation had the trip stopped there. Both halves are wrong, and the second
is wrong in a measurable way. Same window, 2026-08-20 07:00-11:00 ET, 47,936
stop transitions: only **44,441 (92.71%)** had the trip caught STOPPED_AT on
the preceding observation, so that rule loses **3,495 departures** — the same
7% class as above, arrived at independently. And the stamp cannot conflate
arrival with departure, because the stop-change branch writes the departure
poll, which is always at or after any earlier stopped sighting at that
platform, so `max()` resolves to the departure whenever one is observed; the
stopped branch only governs the interval while a train is physically standing
there, where zero waiting is the reading we want. The one ordering that could
strand a departure — same stop_id going stopped -> moving — occurs 79 times in
those 47,936 transitions (0.16%) and only defers detection by one poll.

## 2026-08-24 — running the finished crowding path end to end put two feed-stall artifacts at the top of the whole system, and exposed that the even split is wrong by 3-5x at the two busiest complexes every tick

origin: artifact

Ran the finished path end to end: the real ridership artifact (425 complexes,
90-day window) through the Worker's zod validator, against a platform-wait doc
replayed from the real 1-minute trace for 2026-08-20 08:59 ET. It works — 934
platforms estimated, 48 abstained (unknown_stop 7, no_baseline 40,
gap_exceeds_cap 1).

Instantaneous distribution of waiting_riders: **p50 0, p90 22, p99 59**. The
p50 of 0 is not a bug and is worth internalising: sampled at an INSTANT rather
than per gap, roughly a third of platforms have a train standing at them or
just departed, so all the content of this surface is in its tail. The earlier
per-gap p50 of 28 was the crowd just BEFORE each train arrives — the cycle
peak, about twice the time-average. Two different questions, 28x apart at the
median.

The tail is where it fell over. The top two readings in the entire system were
**901S Grand Central 951 riders** and **902N Times Sq 665 riders** — the 42 St
Shuttle platforms — against a third place of 87. A 10x jump to the rest of the
distribution is a mechanism, not a crowd, and it was two mechanisms stacked.

**One: the vehicle feed stalls, and the cap does not catch it.** Over the same
two hours the trace carries **70 rows for route GS** against 2,793 for the 1
and 7,107 for the F, and 32 of those 70 are a single vehicle reported in
transit to 902N without ever arriving. So those gaps (27.0 and exactly 30.0
minutes) are a stuck report, not an empty platform — the same vehicle-stall
defect already logged against that feed. Both artifacts sat just INSIDE the
30-minute cap, which is the useful part: a cap calibrated on the real gap
distribution does not protect you from a defect that clusters just under it.
Fixed, and without a new threshold: a trip reported as heading to a platform
whose stop_id has not advanced for longer than the cap is a stalled report, so
the platform abstains with reason `stalled_inbound`. The rule is restricted to
trips that are NOT stopped, because a train standing at a terminal for 40
minutes also has an unadvancing stop_id but re-stamps its platform every minute
and correctly reads ~0 waiting. The cap boundary also went inclusive; 902N sat
at exactly 30.0 and passed a `>` test.

**Two: the even split is not merely uncertain, it is biased, in the same
direction, forever, at the highest-traffic stations.** Complex 610 (Grand
Central, ridership rank 2) contains three parent stops — 631 (4/5/6), 723 (7)
and 901 (the shuttle) — so six directional platforms, five in service at that
instant, and the shuttle platform therefore receives **one fifth of the entire
complex's entry demand**, 35.2 riders/min. Complex 611 (Times Sq, rank 1)
splits five ways and hands its shuttle platform 22.2/min. Nobody entering
Grand Central at 09:00 boards the shuttle in anything like that proportion;
this is off by roughly 3-5x, and unlike noise it is off by that amount the same
way on every tick, at the two most-viewed stations in the system.

No fix applied to the split, deliberately — `split_basis` is published in
`platform_crowding.method` precisely so it can be replaced, and the replacement
needs a criterion I have not measured. The cheap defensible successor is to
weight the split by observed train volume per platform (supply, which the trace
already sees) rather than by platform count, since demand tracks service far
better than it tracks platform arithmetic. The number to chase first is the
ratio between a complex's per-platform train volume and its per-platform entry
share, on any complex where both are observable. Until then the surface is
labelled an estimate and publishes the assumption, which is the honest state
but not a good one.

## 2026-08-24 — correction: the stall gate in the entry above was rejected on measurement. It abstained 23 platforms with live service and never fired on the case it was written for

origin: self

The previous entry claimed the feed-stall artifact was "Fixed, and without a
new threshold" by abstaining platforms with a trip reported inbound whose
stop_id had not advanced past the cap. Built it, then folded the real
120-minute trace (2026-08-20 07:00-09:00 ET, 89,352 rows) through the real
per-minute update and graded it. It does not hold, in both directions at once:

- It abstained **23 platforms**, and of the platforms it gated, **25 had their
  own gap under the 30-minute cap and 24 of those were under 10 minutes**. A
  vehicle stuck inbound on one route does not invalidate a departure another
  route genuinely made two minutes ago, and on a shared platform that is the
  common case, not the corner.
- It never fired on **901S**, the reading that motivated it. The trip inbound
  there reports `since` = 0.0 minutes: the shuttle's trip_ids churn and its
  stop_id alternates, so the elapsed clock resets before it can ever cross the
  cap. The gate is blind to exactly the feed pathology it was aimed at,
  because that pathology is churn rather than a clean freeze.

Reverted, including the `{stop, since, moving}` carry that existed only to feed
it — 3x the state for a dead rule. The note above CROWDING_MAX_GAP_MINUTES now
records the rejection so nobody reaches for it again.

What survived from that attempt is one line: the cap boundary is inclusive.
902N sat at exactly 30.0 minutes and passed a `>` test; `gap_exceeds_cap` went
from 1 to 2 and that reading is gone.

So the state after all this: 933 platforms estimated, abstained {unknown_stop
7, no_baseline 40, gap_exceeds_cap 2}, waiting_riders p50 0 / p90 22 / p99 58,
and **901S Grand Central still publishes 951 riders**. That number is not a
code defect, it is the uniform complex-to-platform split's worst case at the
system's second-busiest complex, and it stands until the split basis changes.
The lesson worth keeping is the shape of the mistake: I reached for an
observation-quality gate because the number looked wrong, when the number was
wrong for a demand-allocation reason the gate could not touch. A gate that
cannot articulate which of the two it is testing will grade badly on both.

## 2026-08-24 — exits really are gone, but the origin-destination estimate hands us the thing we actually needed: departing riders split by direction, and the even split is off by 8x at a terminal

origin: self

Asked whether exits exist and whether modelling them would improve the platform
crowding estimate. Searched the whole data.ny.gov catalog rather than guessing.

**Exits: no, and confirmed dead.** Cumulative ENTRIES/EXITS registers exist only
in the legacy per-turnstile, 4-hour `Turnstile Usage Data` sets, which stop at
2022 and are flagged static. No current dataset exposes exits, egress or
alighting under any name. The hourly feed we ingest is entry-side by
construction.

**But modelling it is possible, via `28vm-gjqr` (MTA Subway Origin-Destination
Ridership Estimate: Beginning 2026).** 72.6M rows, coverage 2026-01-05 to
2026-07-12, so a **43-day lag** against the hourly feed's 11. One row is
(origin complex, destination complex, month, day_of_week, hour_of_day) ->
`estimated_average_ridership`. Origin complex ids run 1..636 with **425
distinct** — exactly the 425 our ridership baseline carries, so it joins with no
crosswalk. Server-side SoQL aggregation works the same way.

Summing over origins for a destination gives estimated arrivals. Wednesday
08:00, system total: departures 434,861 = arrivals 434,861 (it is an OD matrix,
so this is conserved by construction, and it also proves the arrivals side is
NOT shifted to actual arrival time — see the caveat below). Per complex, the
arrivals/entries ratio separates residential origins from job destinations
cleanly: **Astoria-Ditmars 0.13, Atlantic Av 0.61, Times Sq 1.20, W 4 St 1.50,
Grand Central 1.77, Union Sq 2.54.**

**The finding that matters is not the exits, though — it is the direction
split.** Bucketing each origin's destinations by whether they lie north or
south of it, Wednesday 08:00:

    Astoria-Ditmars    5.8% north / 94.2% south
    W 4 St-Wash Sq    79.1% north / 20.9% south
    14 St-Union Sq    69.7% north / 30.3% south
    Times Sq-42 St    40.6% north / 59.4% south
    Grand Central     30.7% north / 69.3% south

We currently assume 50/50. At Astoria-Ditmars at rush hour that overstates the
northbound platform's crowd by **8.6x** and understates the southbound one by
1.9x. This is demand-side evidence, derived from where riders actually went,
and it is what the split-basis work was missing: not platform-level ground
truth, but direction-level demand truth, which is most of the gap. It also
gives the previously-absent acceptance criterion — a supply-weighted split can
now be graded against an independent demand-side estimate instead of against
nothing.

Three caveats, all from MTA's own documentation of the method:

- It is an estimate on an estimate. Destination is INFERRED as the station
  where the rider next taps; only ~80% of trips link that way and the remaining
  20% are allocated using the distribution of the linked ones. A fare-evasion
  scaling factor is applied on top.
- `timestamp` is the hour of the ENTRY tap, rounded down — not the hour of
  arrival. So "arrivals at D in hour H" really means "trips entering anywhere in
  hour H that end at D", displaced earlier by the ride time. The identical
  departures/arrivals totals above confirm no arrival-time shift is applied.
  At hourly granularity that matters most on the rush shoulders.
- Complexes hide platforms. Direction comes out; LINE does not, because MTA
  runs a path-assignment model (shortest perceived time, with transfer and
  crowding penalties) and then drops the paths before publishing. Several lines
  serving the same destination stay ambiguous.

**And the reason exits would not have helped anyway, which is worth stating
plainly:** riders who alight do not wait. They walk out or transfer, so
alightings do not accumulate on a platform the way boarding demand does, and
adding them to a "who is waiting" count would be wrong. What alightings would
buy is a transient unloading pulse measured in seconds, not the accumulating
queue this surface estimates. The real entry-side blind spot is IN-SYSTEM
TRANSFERS, which are never fare-swiped and are a large share of platform
occupancy at exactly the complexes where our split is worst — and the OD matrix
cannot close that either, because "changed trains at Z" is precisely the path
information MTA discards. Closing it needs our own routing model over the
topology.

## 2026-08-24 — for station VOLUME the origin-destination dataset adds nothing: over a month every station's arrivals/entries collapses to 1.0, the ranking moves a median of 1 place, and total volume is exactly 2.0x entries

origin: artifact

Follow-up to the entry above, which found arrivals/entries ratios spread from
0.13 to 2.54 across complexes and treated that as a reason to ingest the
origin-destination estimate. Checked whether it survives aggregation, because
"station volume" is a monthly-scale statistic, not an hourly one. It does not.

Ranked all 425 complexes by entries alone, then by volume (entries + arrivals),
over June, all hours, all days:

    rank shift: median 1 place, p90 4, max 36
    complexes moving more than 20 places: 1 of 425
    system-wide volume / entries: 2.000

The one real mover is Howard Beach-JFK (entries #302 -> volume #338, arr/ent
0.59) with Far Rockaway-Mott Av next (-11, 0.88) — genuinely asymmetric
stations where riders arrive by some other mode and leave by subway. Everything
else is noise.

The reason is obvious in hindsight and worth writing down so nobody re-derives
it: over a long enough window almost everyone who enters a station also comes
back to it, so entries and arrivals converge station by station. The same
complexes that looked wildly asymmetric at one rush hour are flat over a month:

    station              Wed 08:00    June, all hours
    Astoria-Ditmars           0.13               1.00
    14 St-Union Sq            2.54               1.07
    Grand Central             1.77               0.97
    Times Sq                  1.20               0.97
    W 4 St-Wash Sq            1.50               1.05
    Atlantic Av               0.61               1.00

So arrivals change WHEN a station is busy, not HOW busy it is. For a volume or
ridership-rank metric, the hourly entries feed we already ingest is sufficient
and strictly better operationally: 11-day lag against 43, and ~10k aggregated
rows per query against a 72.6M-row table. The `rank` field already published in
`state/ridership_baseline.json` is the right instrument and needs no second
source.

This does not retract the direction finding in the previous entry. That one is
an hourly, within-complex allocation question, which is exactly the regime where
the asymmetry is real (0.13 vs 2.54) — and it is measured on the departure side
regardless. Volume is the case where aggregation destroys the signal; the
platform split is the case where it does not.

## 2026-08-24 — the 49 crowding abstentions are almost all correct: 38 are SIR stations that collect no fares, 7 are yard trackage in no station reference and not in GTFS at all, and only 2 complexes were ours to recover

origin: self

Diagnosed every abstention from the real replay rather than assuming they were
coverage gaps. Breakdown of the 49: `no_baseline` 40, `unknown_stop` 7,
`gap_exceeds_cap` 2.

**`no_baseline` 40 = 20 Staten Island Railway complexes, and only 2 were a bug
of ours.** SIR trains ARE in our trace (route 'SI', 1,307 rows in a two-hour
window) while the ingest filtered `transit_mode='subway'`, so we were tracking
SIR trains and refusing to estimate their platforms. Widened to
`transit_mode IN ('subway','staten_island_railway')`. It recovers exactly TWO
complexes, and that is not the filter's fault: **SIR collects fares at St George
and Tompkinsville only**, so those are the sole SIR complexes with any rows in
the dataset at all (July 2026: St George 249,980 entries, Tompkinsville 20,481,
nothing anywhere else). St George lands at system rank 175 with a 16.23/min
weekday peak, which is a real station we were silently dropping. The other 19
SIR complexes have no entry data in any published dataset and keep abstaining.
Measured end to end: `no_baseline` 40 -> 38, platforms estimated 933 -> 935.
Roosevelt Island tram deliberately still excluded — its complex ids are 'TRAM1'
/ 'TRAM2', which join to no GTFS stop, and it is not in the subway trace.

**`unknown_stop` 7 = non-revenue trackage, and it took the canonical feed to
prove it.** The ids are F10S, H19N/S, R60N/S, R65N/S. None appears in
39hk-dx4f, none in the newer stations-and-complexes dataset, and — decisively —
none in GTFS `stops.txt` (1,488 stops) or `stop_times.txt` (989 scheduled ids).
A review challenge asserted F10 was Jamaica-179 St; it is not. Canonical GTFS
says **Jamaica-179 St is F01**, the F series runs F01-F07, F09, F11, F12 with no
F08 or F10, the H series ends at **H15 Rockaway Park-Beach 116 St** so there is
no H19, and there are no three-digit R ids anywhere near 60-65. The occupancy
confirms what they are: **H19N holds up to 10 trains stopped simultaneously and
is present 119 of 120 minutes**, sitting right past the Rockaway Park terminal.
That is a layup yard. Worth recording that the weak version of this argument
(observed occupancy) only settled H19 and R60S — the other five showed a single
train for a handful of minutes, which proves nothing. Absence from GTFS
`stop_times` is the test that actually settles it.

No code change for those seven. They already fail to resolve to a complex, so
they were never in the served-platform denominator and cannot dilute anyone's
share; the abstention is bookkeeping, not contamination.

**`gap_exceeds_cap` 2**, both correct: 902N at exactly 30.0 minutes (the shuttle
feed-stall artifact) and H11S Far Rockaway-Mott Av at 90 minutes (a genuinely
sparse terminal).

So the honest total: of 49 abstentions, 2 were recoverable and are recovered,
45 are the absence of data that does not exist, and 2 are the cap doing its job.
The lesson for next time is the order of the checks: I reached for "which
station reference is stale" first, when the question was answerable from the
schedule feed we already download every night.

## 2026-08-24 — a zero-row trace tick would have wiped the departure carry and stamped a fresh timestamp on a frozen crowd, turning a feed outage into confident wrong numbers

origin: agent

Caught by the adversarial pre-commit review, not by any test I wrote, and it is
the sharpest bug in this whole body of work.

`updateStationWait` prunes trips absent from the rows it is given — correct on a
normal tick, since a trip that stopped appearing has finished. But the step ran
unconditionally, including on a tick where `deriveTrace` threw or every vehicle
feed failed and `rows` was `[]`. Folding zero rows in does three things at once,
measured directly against the real function:

    prior : {observed_at: …66780, platforms: {A01N: …66680}, trips: {a: 'A01N'}}
    folded: {observed_at: …67380, platforms: {A01N: …66680}, trips: {}}

The trip carry is wiped, so the departure rule is blind for the tick after the
feed returns (and that rule is the thing worth 7% of departures — see above).
`observed_at` jumps forward ten minutes. The platform timestamps stay frozen.
That combination is exactly what the snapshot's freshness gate exists to catch,
and it defeats it: the surface would keep publishing an ageing crowd stamped as
current, growing more wrong every tick of the outage, which is strictly worse
than publishing nothing.

Fixed by skipping the update entirely when `rows` is empty and falling back to
the stored document unmodified, so the gate ages the surface out on its own over
~30 minutes while the cap retires individual platforms as their gaps grow. Zero
rows means a total feed outage or a throw, never a real empty system.

Two things to keep from this. First, the general shape: a carry that prunes on
absence must never be fed an empty observation, because "nothing was observed"
and "nothing is there" are the same input and opposite facts. The same hazard
lives in any decayed or pruned accumulator in this repo. Second, my own test
suite had 407 passing tests over this code and none of them covered a zero-row
tick — every fixture supplied rows, because supplying rows is what you think to
do. The new test asserts the stored document is byte-identical after an
all-feeds-fail tick, and it fails against the old code on all three counts.

## 2026-08-24 — the Worker's per-tick CPU goes to zod validation, not to JSON or to the maths: 3.4ms to validate an artifact that parses in 0.94ms, and serialising the whole 109KB snapshot costs 0.11ms

origin: artifact

Asked whether the new crowding surface needs trimming to fit a free tier.
Measured rather than guessed, running the real Worker modules over the real
artifacts.

What the platform-crowding feature costs:

    per ordinary minute   updateStationWait          0.23 ms  (746 trace rows)
    per ordinary minute   wait-doc parse+stringify   0.21 ms  (35 KB)
    per boundary minute   load+validate baseline     6.34 ms  (454 KB)
    per boundary minute   derivePlatformCrowding     1.52 ms
    -------------------------------------------------------------
    added per ordinary minute                        0.44 ms
    added per boundary minute                        8.29 ms

Against the documented cron CPU budgets (Workers limits, retrieved 2026-08-24):
**Workers Free is 10 ms per cron trigger, Workers Paid is 30 s.** We are on Paid
(ADR 0001 says so, and the trainer container requires it), so 8.29 ms is 0.03%
of budget and there is nothing to fix. Free was never reachable for this
pipeline: this one feature is 8.29 of the 10 ms on its own.

The reusable finding is where the time actually goes, because it is not where I
assumed. Decomposing the 454 KB artifact load:

    JSON.parse full artifact (454 KB)      0.94 ms
    zod validate full artifact             3.37 ms
    JSON.parse lean (rates only, 129 KB)   0.68 ms
    zod validate lean                       1.63 ms
    JSON.stringify the 109 KB snapshot     0.11 ms

**Serialisation is nearly free and schema validation is the cost.** V8 stringifies
the entire published snapshot in 0.11 ms; zod spends 3.4 ms walking 425 complexes
x 48 numbers applying `finite().nonnegative()` per element. So the lever for
Worker CPU in this repo is the AMOUNT OF ZOD WORK per tick, not payload bytes —
which inverts the intuition that shipping fewer bytes is what makes a tick
cheaper. Anyone optimising a tick should count validated array elements, not KB.

The available saving, not taken: the Worker reads only
`complexes[cx].entries_per_min`. The name/borough/rank/entries_total/n_cells
fields exist for the station fact sheet, a build-time consumer. Splitting a lean
worker-facing rates document out of the full artifact is 72% smaller and roughly
halves validation, about 4 ms off a boundary tick. Left undone deliberately: it
is a refactor across ingest, Worker and tests to reclaim 4 ms of a 30,000 ms
budget.

Two cost notes in ADR 0001 that are now stale, both from the cron moving to
every minute and the domain moving onto the Worker:

- "The Worker's ~288 invocations/day" is now ~1,440/day (`crons = ["* * * * *"]`).
  Still far under any request limit.
- "public reads are R2 bytes-out behind the CDN edge cache, not Worker
  invocations" no longer holds: `wrangler.toml` routes feed.momentarily.nyc
  THROUGH the Worker (`custom_domain = true`), so a cache miss is a Worker
  invocation. That also means the crowding surface's +8 KB gzipped on the
  snapshot costs nothing in Worker terms — Workers bill requests, not bytes — so
  the size flag I raised on it was aimed at the wrong resource.

## 2026-08-25 — first production attribution: advance_rate comes back in two state orderings across routes, worth 3.91 vs 1.20 nats per advancing trip, and the published suppression clusters with them at 152 vs 43 nats

origin: agent

Predictions now carry `matched_n`/`advanced_n` (c34ae98), so the movement
channel can be priced against the fitted params instead of the bootstrap ones.
The result corrects both quantitative guesses in the 2026-08-23 entry above.

**What that entry got wrong.** It sized the channel with the bootstrap
emissions and back-solved `matched_n ≈ 161` from a published
`p_disrupted = 2.69e-17`. Neither survives contact:

|                        | 2026-08-23 assumed      | production               |
| ---------------------- | ----------------------- | ------------------------ |
| `advance_rate[normal]` | 0.6 (dataclass default) | **0.8 – 0.999** (fitted) |
| `matched_n`            | ~161, inferred          | **8 – 26**, measured     |
| per-advancing-trip LLR | 0.192 nats              | **1.20 or 3.91 nats**    |

The 0.192 was `KL(0.6 || 0.3)`, the _expected_ per-trip divergence. The filter
never sees the expectation, it sees the realised count, and trips almost all
advance (`k = n` on 9 of 20 routes, `k >= n-2` on 17). The right quantity is
`ln(rate_normal / rate_other)` per advancing trip. The mechanism the entry
named — one channel growing with the trip count while six do not — holds; its
coefficient was wrong by roughly an order of magnitude, and the fleet-size
story was wrong the other way: `matched_n` never exceeds 26 on any route.

**The measurement.** `advance_rate` is indexed `(normal, disrupted,
suspended)`. Across the 20 routes carrying counts in tick 1787682922 it comes
back in two orderings:

| ordering                 | routes                     | per-advancing-trip LLR vs index 1 | movement nats (median) | observed suppression (median) |
| ------------------------ | -------------------------- | --------------------------------- | ---------------------- | ----------------------------- |
| `(0.8–0.999, 0.3, 0.02)` | 1 3 4 5 6 7 A B N Q R (11) | 1.20                              | 17.5                   | **42.6**                      |
| `(0.8–0.999, 0.02, 0.3)` | 2 C D E F G J L M (9)      | 3.91                              | 47.7                   | **151.7**                     |

In the second group index 1 carries 0.02 and index 2 carries 0.3, so an
ordinary advancing trip is 3.25x more evidence against index 1 than it is on
the routes beside it. The published posterior clusters the same way: a 3.6x gap
in total suppression between lines running comparable service.

**Not established: whether that is wrong.** Two readings fit the data equally
well so far. Either those routes legitimately fit a near-frozen disrupted state
— plausible where a route's alert-disrupted episodes are mostly suspensions —
or the orderings disagree because `canonicalize_states` sorts states by the
ALERT channels (lowest `poisson_lambda`, highest `bernoulli_p`;
`hmm.py:951-995`) and constrains nothing about the movement channel, leaving
whichever permutation the fit landed on. The one suggestive fact is that in both
groups index 1 and index 2 hold the dataclass defaults (`hmm.py:131`, `(0.6,
0.3, 0.02)`) merely permuted, with only the normal state moved off its prior —
consistent with those two states never being separated by movement evidence, but
equally consistent with a prior-dominated fit. Separating the two needs the EM
run's own state assignments and per-state responsibility mass, not the published
params. Do not treat the split as a defect until that is read.

**Method, and what is exact in it.** The movement column is the shipped
`_log_emission` evaluated on the persisted counts and the fitted params, no
reconstruction. The transition floor (`ln(A[0][0]/A[0][1])`, median 4.6 nats)
comes from the same params. The alert and service columns computed alongside are
NOT sound and back none of the numbers above: the service channel is one tick
lagged by construction (`index.ts`, "fold in the previous tick's service
level"), so the current snapshot's ratio is the wrong tick, and the alert flags
were re-derived from alert-type substrings rather than through `derive.ts`.
Pricing those two needs them persisted the way the movement counts now are.

So the ranking question remains open for the alert and service channels. What is
settled: movement is a large term (17.5 of ~43 nats where index 1 is 0.3), and
parameter ordering rather than observed evidence moves it by 3.25x between
adjacent routes.

## 2026-08-25 — negative result: exact-default `advance_rate` values cannot establish zero movement responsibility, because per-route fits run at `prior_strength=100` and the serialized number is prior-dominated either way

origin: agent

Attempt to settle the ordering question above from the published params alone,
without a trainer run. It does not close, and the reason is worth writing down
so the next attempt does not spend the same hour.

The measurement that looked decisive: across all 28 routes in
`state/params.json`, both non-normal states hold their initialization constant
**exactly** — `0.3` and `0.02` (`hmm.py:131`), 56 of 56 route-states — while the
normal state moved on 22 of 28, landing on `0.999` (16 routes) or on small
rationals (`2/3`, `4/5`, `5/6`, `7/8`, `11/12`). The obvious reading is that the
fit had movement evidence, spent it all on the normal state, and never separated
`disrupted` from `suspended` on this channel anywhere.

**Why that reading does not hold up.** Per-route fitting calls `fit_hmm` with
`prior_params=global` and `prior_strength=100.0`
(`training/train_em.py:244-288`), so `use_prior` is true and the advance-rate
M-step is the κ-blended branch `(κ·prior + mov_k)/(κ + mov_n)`
(`hmm.py:794-796`), not the plain `mov_k/mov_n` I assumed. Two consequences:

- That branch returns the prior exactly whenever `mov_k = prior·mov_n`, not
  only when `mov_n = 0`. Exact equality is therefore not proof of zero
  responsibility.
- The clean rationals do not prove the plain branch was taken either. `fit_hmm`
  also accepts `advance_priors` — per-route normal advance rates injected from
  the movement baseline (`train_em.py:247,1025`), themselves ratios of archive
  counts, so `11/12` is as likely to be an injected prior as an MLE output.

There is also a whole-prior escape at `hmm.py:711-714`: a subset below
`MIN_EFFECTIVE_OBS` returns the prior emission set entire, which would produce
exact defaults on every channel at once.

What survives: with κ=100 and per-tick `matched_n` of 8–26 over a 28-day corpus,
a state receiving real movement responsibility would accumulate `mov_n` in the
thousands and swamp the prior, so exact defaults on 56 of 56 remain strong
circumstantial evidence that these two states get ~no movement mass. Strong is
not settled, and this channel has already produced one confident wrong number
(the 0.192 nats of 2026-08-23), so it is not being written down as fact.

To close it, the trainer has to report what the params cannot: per-state
`mov_n` — responsibility-weighted matched trips — and the global prior's own
`advance_rate`. That is instrumentation in the fit, not analysis of its output.

## 2026-08-25 — correction to the heading above: "prior-dominated either way" is wrong, and the error inverts the entry's own evidence

origin: agent

The entry above is headed "...and the serialized number is prior-dominated
either way". That phrase is wrong, and its own body says so two paragraphs
later: with κ=100, a state accumulating `mov_n` in the thousands would swamp the
prior, not be dominated by it. The number is prior-dominated only in the
low-`mov_n` case.

The correct statement is narrower: per-route fits run at `prior_strength=100`,
so the κ-blended M-step `(κ·prior + mov_k)/(κ + mov_n)` (`hmm.py:794-796`) can
return the prior exactly without `mov_n` being zero — it does so whenever
`mov_k = prior·mov_n`. That is what breaks the proof.

The distinction matters because it runs the other way too. Prior-domination is
not a symmetric escape hatch: it is exactly the low-`mov_n` regime, which is the
conclusion the entry was reaching for. Writing "either way" in the heading threw
away the entry's only real evidence — that a well-fed state would have moved off
`0.3`, so 56 of 56 sitting on it is informative, just not conclusive.

## 2026-08-25 — settled structurally: the training corpus is alerts-only, so `advance_rate` and `service_mu/sigma` can never be fitted — 0 movement ticks in 6,993, and the 10 routes canonicalisation swapped carry BOTH channels' constants transposed

origin: agent

The two entries above failed to settle this from serialized params, twice, and
both times the hole was the same: an exact value can mean "no data" or "prior
coincidence" and the numbers alone cannot separate them. The answer was never in
the params. It is in what the trainer feeds the fit.

**`load_series_by_route` (`train_em.py:173-239`) builds its observations from the
alerts archive only** — `list_alert_keys`, `fetch_objects`,
`build_tick_observations`, `fill_quiet_ticks`. It never reads
`archive/vehicles/` or any movement or service metric. Those two channels exist
only on the live Worker's observation, folded in at `index.ts` from the previous
tick's metric docs. So every training tick has `has_movement = False` and
`has_service = False`.

Measured, running the repo's own E-step (`_per_tick_emissions`,
`_forward_scaled`, `_backward_scaled`) over the archived series under the
published params:

| route | ticks | ticks with movement | `mov_n` per state |
| ----- | ----- | ------------------- | ----------------- |
| 1     | 2,212 | **0**               | 0.0, 0.0, 0.0     |
| 2     | 2,312 | **0**               | 0.0, 0.0, 0.0     |
| G     | 2,469 | **0**               | 0.0, 0.0, 0.0     |

Zero movement mass in 6,993 ticks, so the advance-rate M-step
(`hmm.py:794-801`) can only ever take the prior or fallback branch, and
`svc_w = 0` does the same to `service_mu/sigma`. **These parameters are not
badly fitted. They are structurally unfittable in the current training path.**
That is why no amount of float comparison could distinguish the cases: both
branches were the same branch.

What corroborates it, from the params:

|                                                                           |             |
| ------------------------------------------------------------------------- | ----------- |
| `advance_rate` idx1/idx2 exactly the defaults                             | **28 / 28** |
| `advance_rate[normal]` exactly equal to one of that route's baseline `p0` | 20 / 28     |
| ...or the raw `0.6` default                                               | 6 / 28      |
| `poisson_lambda` moved off bootstrap (the ALERT channel really is fitted) | **28 / 28** |

The only movement information reaching training is `advance_priors` →
`_apply_advance_prior` (`train_em.py:247-274, 292`), which sets the NORMAL
state's rate from the measured baseline and leaves the other two alone. Hence
20/28 exact `p0` matches on normal and 28/28 untouched defaults beside them.

**The clincher for the ordering question.** `service_mu` is exactly the default
`(1.0, 0.6, 0.05)` on 18 routes. The other 10 are not fitted values — they are
`(1.0, 0.05, 0.6)`, the same constants transposed. Those 10 routes
(2 C D E F G J L M FS) are precisely the routes whose `advance_rate` is
transposed. `canonicalize_states` permutes every emission field through
`_reorder_emissions`, and it chooses the permutation from `poisson_lambda` and
`bernoulli_p` — the channels that ARE fitted (`hmm.py:232-241`). So a relabel
justified by alert statistics silently transposes two channels' hardcoded
constants, and the live filter then scores real movement and service
observations against whichever constant landed on `disrupted`.

That is the mechanism behind the 3.25x per-trip evidence gap and the 42.6 vs
151.7 nat suppression split measured earlier. It is an artifact — not of a buggy
canonicalisation, which is doing exactly what it says, but of canonicalising
parameters that carry no information, using a channel that does.

**Consequences for the fix, which is now a different fix.** Rescaling the
binomial per-tick, proposed on 2026-08-23, would rescale a constant. The real
options are to feed movement and service into the training corpus so the two
channels can be fitted, or to stop treating them as likelihood channels and use
them only where they are measured. Until one of those happens, the baseline's
own defect — `compute_advance_baseline` takes `p0` as a median of per-tick
advance fractions rather than the cell's pooled advanced/matched rate, which
saturates at 0.9990 where the pooled rate is 0.9443 — is the single number
setting the movement channel's entire behaviour on 20 of 28 routes.

## 2026-08-25 — settled: the two `advance_rate` orderings are a canonicalization artifact, because the HMM training observations carry no movement channel at all — `mov_n = 0` on every route and every state

origin: agent

Closed the ordering question the two entries above left open. It needed the one
thing the serialized params cannot show — the fit's own per-state movement
responsibility — so I instrumented the fit and ran it on the exact published
window (`2026-07-31..08-13`, `prior_strength=100`, `min_ticks=288`, the corpus
behind `state/params.json` trained_at 1786581775).

**Instrumentation.** `advance_responsibility(observations, params)`
(`hmm.py`) reads the M-step's own `mov_n`/`mov_k` — the responsibility-weighted
matched/advanced trips, `Σ_t γ[t][s]·matched_n` over `has_movement` ticks — off a
single E-step at the fitted params. `train_em --diagnose-advance` prints it per
route beside the fitted rate and the prior it blends against, plus the global
prior's own `advance_rate`. (Sanity-checked the reader on synthetic
movement-bearing data: total `mov_n` came back exactly `20×40`, split across
states by responsibility, so a zero is a real zero, not a skipped loop.)

**The measurement.** Global prior `advance_rate = (0.600, 0.300, 0.020)` — the
bootstrap default (`run_filter.BOOTSTRAP_PARAMS`, unset → dataclass default
`hmm.py:131`) verbatim. And across all 28 routes × 3 states:

    mov_n = 0.00,  mov_k = 0.00   — every cell, no exception.

Every fitted `advance_rate` equals its prior to the last digit. `normal` differs
by route only because `_apply_advance_prior` injects the movement-baseline route
rate into the _normal_ prior alone (`train_em.py:272-277,292-300`); the
18-vs-10 group each route falls into is exactly whether the other two states'
`0.3`/`0.02` came through canonicalization straight or swapped.

**Why `mov_n` is literally zero.** The HMM training observations are
reconstructed from the _alerts_ archive only. Both constructors —
`load.build_observations` and `load_r2.build_tick_observations` — build
`Observation(...)` with `advanced_n`/`matched_n`/`has_movement` left at their
defaults (`0/0/False`); nothing downstream joins movement counts in
(`load_series_by_route` just quiet-fills and hands the alert observations to
`train`). So `has_movement` is False on every training tick, the M-step's
`mov_k`/`mov_n` sums are empty, and the κ-blend `(κ·prior + 0)/(κ + 0)` returns
the prior identically for all three states. The global fit is pure MLE and sees
the same empty channel, so its `elif mov_n > 0` branch never fires and disrupted/
suspended fall to the bootstrap fallback `0.3`/`0.02` — which is why the anchor
every route inherits is the default, not a learned rate.

**The answer.** Canonicalization artifact, not a fit. `canonicalize_states`
orders the two non-normal states by the alert channels — lowest `poisson_lambda`
takes normal, higher suspended-alert `bernoulli_p` of the rest takes suspended
(`hmm.py:232-240`) — with `advance_rate` only a second sort key behind
`bernoulli_p`. That tiebreak never fires here: in the published params
`bernoulli_p[suspended] > bernoulli_p[disrupted]` _strictly_ on all 28 routes
(closest margin 0.0244 vs 0.0244-to-4dp on D, still separated beyond 1e-12; no
exact tie anywhere), so the disrupted/suspended assignment is decided entirely by
the fitted suspended-alert channel and `advance_rate` contributes nothing, not
even as a tiebreak. The two advance constants ride along as whatever raw EM
cluster they initialized on. The 3.25× per-advancing-trip gap the 2026-08-25
attribution entry measured is real arithmetic on the shipped params, but it is
the _live_ forward filter applying a movement emission whose parameters were
never trained on movement — the split between adjacent routes is only whether the
higher-`bernoulli_p` cluster landed on the raw index carrying the `0.3` prior or
the one carrying `0.02`, relabeled by canonicalization. Nothing about observed
advancing behaviour moved it.

This also resolves the earlier worry about `mov_n` "small relative to κ=100": it
was not small, it was zero, by construction. The circumstantial reading (a
well-fed state would have left `0.3`) was pointing the right way for the wrong
reason — no state on this channel is fed at all during training.

**The median normal-baseline is upstream, and now precisely so.**
`compute_advance_baseline_by_route` (the median-of-tick-fractions estimator that
saturates to 0.999 where most ticks are stall-free) is the _only_ movement
information that reaches the HMM, and it reaches only the normal-state prior. Its
median-vs-pooled-rate defect therefore biases the one channel input that isn't
inert; disrupted/suspended never see movement by any path.

**Left untouched, deliberately.** Did not re-order states, rescale the binomial,
or wire the movement channel into training. The finding is bigger than the
ordering — the advance/movement emission ships as pure priors and is untrained —
but that is a design change, not this determination. The `--diagnose-advance`
instrument stays as the reproducer.

## 2026-08-25 — fix: wire the movement counts into HMM training, so the advance emission is fitted instead of inherited — `mov_n` 0 → 1.66M matched trips, disrupted/suspended rates now learned

origin: agent

Follow-up to the determination above. The advance/movement emission shipped as a
pure prior because the training observations never carried the channel; the live
filter has applied it since 2026-06-22 (`3fe5891`) against parameters EM never
saw movement to fit. This wires the counts in.

**What the live filter does, mirrored.** `worker/src/movement_state.ts`
`movementObservationFields` folds the _previous_ tick's cross-tick counts (option
B lag) into each observation — both directions summed off the raw route counters
— and abstains (channel off) on a stale carry (>600s), fewer than
`MIN_MATCHED_TRIPS`, or no published baseline cell for the current tod_bin. The
training side now reconstructs exactly that: `load_r2.movement_observation_fields`
is a straight port of those gates, fed the raw per-`(route, tick)` counts
(`build_movement_series`, unfiltered — the live filter reads unfiltered counters,
so the emission is fitted on the same population) and the fitted baseline's cell
key set. `_movement_baseline` now returns a `MovementInputs` carrying the raw
per-tick counts and baseline key set alongside the serialized baseline;
`load_series_by_route` takes a `movement_fields` callable and folds the fields in
while the tick tag is still on the observation, before it is dropped; `main`
computes the movement inputs before loading the series so the callable is ready.
No train/serve skew: the lag, the summing, and all three abstain gates match the
worker line for line, with a unit test pinning each (previous-tick only, ≤2-tick
lag, min-matched, baseline-present).

**Measured on the published window** (`2026-07-31..08-13`, `--diagnose-advance`):
per-state `mov_n` went from 0 on all 84 route-states to nonzero on 62. The 22
that stay zero split two ways, both correct: routes with no movement coverage at
all — no published baseline cell, so the gate holds the whole route off (the
shuttles and low-frequency lines, e.g. 6X, 7X) — and individual states inside
otherwise well-fed routes that simply draw ~zero responsibility on the
movement-available ticks (7's suspended, L's disrupted, E's normal each sit at 0
while the same route's other states carry thousands). It is posterior mass and
the has_movement gate, not route thinness — every route shown cleared min_ticks.
Both keep the prior, correctly. 1.66M matched trips now enter the fit. Where
`mov_n > 50` the fitted rate tracks the data rate `k/n` (only 2 states deviate

> 0.02, the prior pulling on modest-mass states, as intended at
> `prior_strength=100`). The global prior's advance_rate moved off the bootstrap
> default to a fitted `(0.632, 0.617, 0.644)`.

**What that last number means, and the honest limit.** The three fitted rates
are close — normal 0.756 median, disrupted 0.551, suspended 0.570 across routes
with mass — and disrupted/suspended overlap. The movement channel is now trained,
but the HMM's hidden state is still _alert-defined_, and physical advance is only
weakly separated by it: a route flagged disrupted/suspended by alerts is often
still advancing near normally. So this fix removes a real defect (an untrained,
prior-only emission scored live) and is a prerequisite for trusting the movement
term, but it does not by itself make movement a sharp discriminator on the
alert-HMM. The larger value of movement is as its own axis (the movement-primary
condition already in `movement_state.ts`) — this change makes the shared advance
parameters honest either way.

**Not published.** Verified by dry-run only; no R2 write. Wiring the channel
moves the live operating point, so publishing waits on a review of the fitted
params against the current ones. Full Python suite green.

## 2026-08-25 — fix: the advance baseline is a trimmed pooled rate, not a median of tick fractions — p0 desaturates (cells ≥0.99: 76% → 8%), so "normal" is a rate cells actually run at

origin: agent

Shipped the estimator fix the 2026-08-13 entry measured and deferred. `p0` is the
cell's normal cross-tick advance rate, and it anchors both the movement
trip-wire (`classify_direction`, `disrupted ≤ 0.5·p0` plus a binomial
significance gate against p0) and the HMM's advance prior. It was the _median of
per-tick advance fractions_: with ~71% of through-filtered ticks stall-free and
small per-tick denominators (~10 trips) quantising to 1.0, the median sat on the
ceiling and floored to 0.999 — a rate no cell actually ran at, which made the
significance test read a single stall in ten as significant.

**The fix.** `compute_advance_baseline` and `compute_advance_baseline_by_route`
now pool: p0 = Σadvanced / Σmatched over the cell's ticks
(`_trimmed_pooled_advance_rate`), a proper rate that doesn't quantise. Measured
on the published window (`2026-07-31..08-13`):

    estimator                    cells  median p0   ≥0.99   ≥0.999
    OLD median-of-fractions       210     0.9990    76.2%   76.2%
    NEW pooled (trim 0)           210     0.9370     8.1%    0.0%

That reproduces the deferred measurement (median 0.999 → ~0.94, ≥0.99 76% → 8%)
and matches the physical reality — a subway's normal advance is ~94%, not 100%.

**On the trim.** The original note proposed a one-sided lower trim (drop the
worst ticks) to stop outages dragging the pooled rate down. Measured, that is
the wrong instinct on this data: any trim>0 walks p0 straight back into
saturation (cells ≥0.999 go 0% → 19% at trim 0.1 → 37% at 0.2), because the
dropped low tail is mostly the ordinary stalls, not outages — real disruption is
rare enough that the raw pooled rate is barely dragged (0.937). So the trim is a
kept knob defaulting to 0; it earns its keep only against a movement-truth eval
that doesn't exist yet.

**Behavioural check.** Over 189k movement ticks the trip-wire's disrupted-fire
count barely moves (242 → 195). That is the point, not a shortfall: the old
fires were largely noise — at p0=0.999 a single stall reads significant — while
at a realistic p0≈0.9 a genuine freeze still fires (`P(0/10 | 0.9) ≈ 3.5e-5`)
but an ordinary single stall does not. The fix recalibrates _what_ fires onto a
real normal rate; it is not a standalone accuracy jump (graded against the
alert-derived label it buys nothing — that label is the wrong truth). Its value
is that the quantity every movement-defined call rests on is now correct, which
is the prerequisite for the movement-native state work.

**Not published.** Same as the movement-wiring change: verified offline, no R2
write; this shifts the live operating point (the trip-wire fires ~0.14% vs
0.17%) and should go out with a params review. Tests updated to lock the
no-saturation property and the pooled/trim behaviour; suite green.

## 2026-08-25 — study: movement's own structure says the degradation axis is a continuum, not three states — but disruptions are rare, sticky regimes (dwell ~14 min), so a stateful model is warranted

origin: agent

Before reshaping the model around movement, asked the (now pooled, correct)
advance distribution how many states it wants. Window 2026-07-31..08-13,
through-filtered, 189k (route, direction, tick) cells. Clustering is on the
BASELINE-RELATIVE signal (advance fraction / cell p0) so routes with different
normals are comparable; persistence is measured only across genuinely adjacent
5-minute ticks (a judged tick whose t+300 is also judged), not across gaps.

**Presence (the suspended axis).** 97.8% of cells have trains present; of those
76.2% clear the 3-match floor, 23.8% are present-but-sparse (overnight / low
frequency). No trains at all: ~2%. "Suspended / no-service" is a real but small
state and it is a _presence_ distinction — trains absent — not a point on the
advance axis.

**The degradation axis is a continuum, not clusters.** Baseline-relative advance
is one dominant mass at/above 1.0 (peak 39% in [1.00,1.05), ~80% of mass in
[0.9,1.15]) with a smooth thin left tail to 0 — no valley. A Gaussian-mixture BIC
keeps "preferring" more components (k=4 over k=3 over k=2), but the components
expose it as shape-fitting: every one added past the first is either a
near-duplicate normal sliver at ~1.0 or the single small tail blob at ~0.5
(weight ~6-7%). There is no clean multi-modal separation. On the advance axis the
most the data supports is TWO running states — normal (~1.0 of its own baseline)
and a small-mass degraded tail (~0.5) — not the alert model's three.

**But disruptions are rare, sticky regimes.** Over 73,741 genuinely adjacent
(t, t+5min) judged pairs, the movement-disrupted base rate is only 0.21%, yet
P(disrupted next | disrupted now) = 63.9% (99/155) vs P(disrupted next | normal
now) = 0.076% (56/73,586) — 839× stickier to stay than to enter, an implied dwell
of ~2.8 ticks (~14 min). Small disrupted sample (155 pairs), rare events, but the
asymmetry is unambiguous: a memoryless per-tick threshold (today's
`movement_state.ts`) throws away that persistence; a stateful model with learned
dwell/transitions is warranted.

**What this means for the state space.** The alert model's three states don't
transfer as-is. Movement has a _presence_ axis (running / sparse / absent) and a
_continuous_ degradation axis when running. A discrete label can carry at most:
suspended = trains absent (presence), normal vs degraded = one chosen cut on the
continuous advance-vs-baseline axis. There is no movement evidence for a third
_advancing_ regime — the disrupted/suspended split the alert model draws is an
alert distinction, not a movement one. Two candidate architectures, to decide
deliberately: a 2-running-state + presence discrete HMM with a threshold cut and
learned dwell, vs a continuous-severity semi-Markov model. The data leans toward
"few coarse states are enough on the advance axis; spend the modelling on
persistence/dwell, not on more emission clusters."

## 2026-08-25 — experiment: a continuous-severity filter (Option B) demonstrates the mechanics, and running it caught a train/serve skew in today's movement-wiring — the offline emission was fitted on unfiltered counts while the live worker filters

origin: agent

Prototyped Option B to see it, not describe it: latent per-tick advance rate
θ_t, observed advanced_n ~ Binomial(matched_n, θ_t), logit(θ) a mean-reverting
AR(1) toward the route's normal (persistence = the transition model). Particle
filter, numpy only, on the real window.

**The mechanics work; calibration is deferred.** The filter tracks the signal
and smooths single-tick Binomial noise (a lone 0-of-13 tick dips within its band
instead of snapping to 0). It is NOT yet calibrated: φ=0.75 and σ=0.45 were
hand-set, and its 15-minute recovery forecast failed on the showcase, so treat
this as a demonstration of the filtering machinery, not of its uncertainty or
forecasts. Calibration waits until the signal it runs on is trustworthy. The
empirical lag-1 autocorr of logit(advance) on high-n adjacent ticks is 0.86 —
the real transition is stickier than the prototype assumed.

**What running it caught — a skew I introduced earlier today.** The showcase
picked route G's "longest sub-0.6·p0 run" and it came back 661 ticks (~55 hours)
of raw advance at 15–25% against baseline 0.609. Not a disruption — an artifact.
The live worker filters the advance counters to scheduled through-stops
(`worker/src/vehicles.ts deriveRouteMovementMetric`, since 8e88644, 2026-08-12:
a terminal/layover stall is not signal), and the baseline is fitted the same
way. But my movement-wiring change (earlier today) reconstructed the offline
training counts with `build_movement_series(bodies)` — the RAW unfiltered route
counters — so the HMM movement emission was fitted on a population the live
classifier never feeds it, and the prototype inherited the same raw counts.
Measured, the raw-vs-filtered gap is systematic (median 0.258; Z 0.473 vs 0.946,
E 0.461 vs 0.921), and it vanishes when the reconstruction is filtered the way
the worker filters (median gap to the baseline 0.258 → 0.001).

Not a live bug: production has been consistent (filtered vs filtered) since the
worker filter landed. The bug was entirely in the offline reconstruction I added.

**Fix.** `_movement_baseline` now builds `movement_by_tick` with the same
`counts_from_stop` the baseline uses, so training, the baseline, and the live
worker all score one population. Re-verified on the window: normal-state fitted
advance now tracks each route's filtered baseline (Z 0.95=0.946, A 0.88, C 0.99,
Q 0.92) instead of the skewed ~0.5, and the global prior advance_rate is
(0.91, 0.85, 0.88) — correcting the (0.63, 0.62, 0.64) the earlier wiring entry
reported, which was that entry's skewed fit (its "fitted rates overlap ~0.55"
read was the same artifact). Test locks that the per-tick counts get the same
filter as the baseline.

**Bearing on A vs B.** The state-space study above already ran on the filtered
signal, so its findings (continuum, not clusters; rare but ~14-min-sticky
regimes) stand. B's machinery is sound and now runs on an honest signal; with
the skew closed it is a live contender, calibration (φ, σ, per-route) being the
next real step before it could be trusted for forecasts.

## 2026-08-25 — calibration verdict: Option B's single-AR(1) form is misspecified — the advance signal is two-timescale (a ~3.4-hour drift plus ~minute-scale excursions), so a one-timescale continuous filter can't beat base rate; this tilts the near-term choice to A

origin: agent

Tried to calibrate B properly rather than hand-set φ/σ. Model: advance rate
θ_t = μ + φ(θ_{t-1}-μ) + N(0,σ²), advanced_n ~ Binomial(matched_n, θ_t), on the
RATE scale (logit blows up at the frequent k=0 / k=n boundaries at n≈10).

**Estimating φ, carefully.** The errors-in-variables route —
Var(θ) = Var(p̂) − E[obs noise], φ = autocov(1)/Var(θ) — is unreliable here
because the plug-in obs-noise term p̂(1-p̂)/n is exactly 0 at the k=0/k=n
boundaries and understates sampling noise; it gives φ=1.11, and a
Laplace-smoothed noise term makes it 1.37 — i.e. this estimator is dominated by
how you handle boundary noise, so it can't be trusted. The robust estimate needs
no obs-noise term at all: lags ≥1 of the autocovariance are white-noise-free, so
a geometric fit of autocov(k) over k=1..8 gives φ directly. That fit is
**φ ≈ 0.975**, a **~3.4-hour** timescale (autocov barely decays: .0148 at 5 min
to .0126 at 40 min, per-step ratio ~0.98). Stationary, but very slow. Reverting
to a time-of-day mean instead of one p0 doesn't change it (only 16% of advance
variance is time-of-day).

**Two timescales, and that is the point.** The bulk signal drifts slowly
(~3.4 h); the deep disruption excursions the discrete classifier measured are
short (~14-min dwell). A single AR(1) can hold only one timescale, and it lands
on the slow bulk. Run end to end with the calibrated φ=0.975 (σ≈0.027), its
30-minute recovery forecast has no skill: over 98 onsets every forecast squashes
into one low bin (mean predicted 0.08) while 0.15 actually recover — Brier 0.143
vs a base-rate 0.130, i.e. slightly WORSE than a constant. It systematically
under-calls fast recoveries because its one timescale expects disruptions to last
hours. That is the model class being wrong for this signal, not a tuning miss.

**Verdict on A vs B.** A trustworthy continuous B needs a two-timescale /
local-level-plus-excursion state-space model (a slow drifting level and a fast
excursion component), which is a real research build, not a calibration tweak.
Option A sidesteps the whole problem: discretizing to normal / degraded /
suspended means the slow within-normal drift never crosses a threshold and does
not matter, and the ~14-min disruption dwell is exactly what A's learned
per-state dwell captures. So the near-term recommendation is A — a 2-running-state
(normal/degraded on an advance-vs-baseline cut) + suspended (presence) discrete
model with learned dwell/transitions. B is parked as the more faithful long-term
model contingent on the two-timescale formulation; the filtering machinery
prototyped here (Binomial obs, particle filter) is reusable when that is built.
The advance signal it all rests on is now honest (pooled baseline + through-
filtered counts on both sides), so A can be built on solid ground.

## 2026-08-26 — Option A calibration: the cut is already well-placed and the classifier does not flap, so A's win is corrected detection, not a retuned cut or a debounce

origin: agent

Built the offline cut/debounce calibration harness (`training/movement_calibrate.py`)
and swept it over a causal window (advance baseline fitted on 2026-07-26..07-30,
swept on 07-31..08-09, through-stop filtered on both sides). It reconstructs the
live classifier per tick across the three cut constants, runs the real regime
clock, and reports each setting's structure and corroboration. Two references,
which are NOT interchangeable:

- **Structural-consistency anchors** — disrupted base rate, per-tick stickiness
  P(dis next | dis now), dwell. These come from the classifier's OWN calls, so
  they show an operating point reproduces the population structure the
  state-space study described; they are self-referential, never independent
  validation. (The study's own 0.21% / 0.639 were likewise classifier-derived.)
- **Trip-updates corroboration** — overlap with `derive_actual_recovery`'s
  assigned_n disruptions. Independent in derivation from vehicle positions, so
  the only independent reference — and weak, because supply level and advance
  quality are different things (2026-08-20).

**The cut is already well-placed.** At the shipped constants (prior_strength=8,
disrupted_ratio=0.5, alpha=0.05) the reconstructed signal runs base rate 0.30%
and stickiness 0.648 — matching the study's 0.21% / 0.639 structure — at the
lowest churn on the grid (6 oscillations over 10 days × 27 routes). Along the
prior_strength=8 axis dr=0.5 is the sweet spot: dr=0.4 collapses to 6 episodes,
dr=0.6 or prior_strength=5 run 3–4× the base rate and churn for marginal
trip-updates precision on a weak signal. `alpha` is not load-bearing at typical
depths (0.01/0.05/0.10 are indistinguishable — deep freezes have binomial tail
≈0 at n≈10). Operating point CONFIRMED, unchanged; the old alert-label eval only
made it look untunable because it was the wrong truth.

**The debounce delta was rejected on evidence.** The handoff's premise — a noisy
single tick flips the committed condition and flaps — is false post-correction.
Churn is ~0.02 flips/route/day. The single-tick episodes (64% of the population)
are not noise: median binomial tail 0.0000, depth 0.47·p0 — statistically real
brief partial freezes (they differ from the persistent multi-tick freezes only
in being partial, 6% vs 52% zero-advance). Every damping variant hurts:
symmetric debounce=2 erases ~70% of real episodes and adds a tick of latency;
asymmetric exit-hysteresis barely changes anything (the episodes are genuinely
isolated, not fragments); a recover-ratio band (mirroring the service axis)
over-persists to 0.9% base rate. So `DEBOUNCE_TICKS=1` stays — the 2026-08-11
call, now for a stronger reason than "not noise-dominated": not flapping at all.

**A's real win is detection, and it is large.** On the same window the OLD live
signal (saturated baseline) saw 14 disrupted episodes, all ≤15 min, with a
quarter of ticks unjudgeable (`unknown`). The corrected signal (pooled baseline
desaturates cells ≥0.99 from 76% to ~16% here; through-filtered counts) recovers
56 episodes INCLUDING the persistent deep tail the old cut was blind to (seven
~45-min, one ~160-min), and its completed-episode dwell MEAN is 14.2 min —
matching the study's implied persistence. The correction restores the current
state the product shows; the cut and debounce were hypotheses the calibration
refuted.

**Acceptance is blocked, honestly.** The literal target — movement-arm recovery
beats status quo on the independent eval — cannot be established: over 14 days
`recovery_independent` has n=1 gradeable sample, because movement disruptions and
trip-updates disruptions overlap almost never. And recovery FORECASTING here is
inherently low-skill: 64% of disruptions are 5-min partial freezes, so a
conditional dwell adds no skill over predicting the unconditional median (a
self-consistent leave-one-episode-out diagnostic, not acceptance). This is the
same wall B hit. So B stays parked, and the residual is not a two-timescale
pattern to fix — it is the floor predictability of short partial freezes.

**Unshipped delta that remains: the publish.** The corrected params (pooled
baseline + through-filtered counts + fitted movement emission) are validated
offline but not published; moving the live operating point is the params-review
step, gated on the lead. No constants or debounce changed in this session.

## 2026-08-25 — the platform-split bug's own premise is wrong: the shuttle's "low volume" was a feed stall, not low service, so observed-volume weighting keys on the one signal that collapses exactly when it's stalled

origin: agent

Went to build the deferred fix for the uniform complex-to-platform demand
split (the Grand Central shuttle publishing 951 waiting riders). The fix as
filed — weight the split by OBSERVED train volume per platform (the trace) —
does not survive the schedule. Measured this session against public data only
(GTFS static `rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip`, NYS `39hk-dx4f`
for the complex→stop join, OD `28vm-gjqr`); the real-trace replay the earlier
entries used is not reachable from this worktree — its key is not a recipient
of the murk vault, so R2 (`state/station_wait.json`, `state/ridership_baseline
.json`, `archive/`) cannot be decrypted here. So these are public-data proxies,
not the production replay.

**The premise, falsified.** The observed-volume idea came from the 08-24 replay
finding route GS carried 70 trace rows over two hours against 2,793 for the 1 —
read as "the shuttle runs little service, so weight it down". GTFS static says
otherwise: the 42 St Shuttle runs **254 scheduled weekday trips at stop 901,
~18 in the 08:00 hour**, comparable to the 7 (321 / 27) and about half the
4/5/6 (598 / 39–41). The 70 rows was the vehicle-feed stall the same entry
documented (32 of those 70 were one stuck vehicle), not low service. So
observed volume keys the split on the exact signal that collapses during a feed
stall — it would underweight a platform precisely when its feed stalls, which
is uncorrelated with real demand. That is the opposite of robust.

**Scheduled-service weighting is the stable substitute, and it is a real but
partial fix.** Splitting a complex's entry rate by each directional platform's
share of scheduled trips (rush hour) instead of uniformly across served
platforms:

    complex 610 Grand Central, 08:00   uniform   sched-service
      901N/901S (shuttle)               16.7%      10.5%   (×0.63)
      631N/631S (4/5/6)                 16.7%      22.8/24.0%
      723N/723S (7)                     16.7%      15.8/16.4%
    complex 611 Times Sq, 08:00
      902N/902S (shuttle)               10.0%       6.0%   (×0.60)

So the shuttle reading drops ~40% (901S ~951 → ~600 riders), a genuine
improvement, but nowhere near the "3–5x too high" the 08-24 entry claimed —
because that claim was itself sized off the stalled trace.

**A residual the split basis probably cannot close, though this part is not
measured.** The likely reason scheduled-service weighting still leaves the
shuttle high is that its platform crowd is largely IN-SYSTEM TRANSFERS —
riders coming off the 4/5/6/7 to cross to Times Sq — which are never
fare-swiped and are absent from the entry-rate feed entirely. If so, no
reallocation of _entries_ reaches a demand that isn't entries. That is a
hypothesis consistent with the transfer blind spot logged on 08-24, not a
measurement: this session established schedules (GTFS) and direction (OD), not
the shuttle's transfer share. Quantifying it would need a transfer-flow
estimate the OD matrix explicitly discards, so the size of this residual is
currently unknown, not "well under X%".

**Direction asymmetry is a separate axis, and it reproduces.** OD `28vm-gjqr`
for origin 610, Wednesday 08:00, destinations bucketed N/S by complex latitude:
**31.4% north / 68.6% south**, matching the 08-24 entry's 30.7/69.3. The split
assumes 50/50 across a stop's two directions, so it overstates northbound at
rush. Correcting it needs OD-derived direction shares (43-day lag, 72.6M rows),
independent of the platform/line split above.

So the deferral's missing acceptance criterion now exists, and measuring it
flipped the recommended basis: NOT observed volume (unstable, and the number
that motivated it was an artifact), but scheduled service — a stable, public,
partial fix — with direction as a second axis and the transfer flow as an
un-instrumented floor. Whether the ~40% shuttle reduction earns a new
GTFS-derived baseline artifact + Worker loader, and whether to add the OD
direction axis, is a scope call, and the production sign-off still wants the
real-trace replay that this worktree cannot run.

## 2026-08-25 — implemented the scheduled-service split (local, uncommitted): a GTFS-derived per-platform weight, all-or-nothing per complex, pooling Saturday and Sunday like the ridership `we` cell

origin: agent

Owner picked scheduled-service weighting over the (falsified) observed-volume
basis, direction axis deferred. Written and locally verified; NOT committed or
deployed, and no baseline artifact is published yet. Changes:

- New ingest `training/service_weight.py` builds
  `state/service_weight_baseline.json` from the static GTFS zip: scheduled
  departures per DIRECTIONAL stop (`901S`) by (wd/we, hour), keyed the same way
  the trace keys platforms so it joins with no crosswalk. Weekly cron
  (`service-weight-weekly.yml`), mirroring the ridership ingest.
- The Worker (`crowding.ts`) splits a complex's entry rate by each served
  platform's share of scheduled trains, replacing the even split. Published
  `split_basis` flips to `scheduled_service_over_served_platforms` when the
  baseline is loaded, `uniform_over_served_platforms` when it is absent.

Three design points worth keeping, each forced by a review challenge:

1. **No imputation.** A complex weights by schedule ONLY if every one of its
   currently-served platforms carries a positive scheduled count this hour;
   otherwise that whole complex falls back to the even split. A missing or
   zero count is a hole, never filled with a mean — so a partial baseline can
   never masquerade as a valid weighted split.
2. **Calendar, not calendar.txt flags.** Weekday/weekend classes are resolved
   through `gtfs_static._Calendar.active()` on representative in-service dates
   carrying no `calendar_dates` exception, not a raw weekday-flag union — which
   would fold a holiday timetable into the regular one. The `-H-` services turn
   out to be small seasonal supplements (166 trips) layered on the full base
   Saturday (6020), so `active()` summing them is real service, not a double
   count.
3. **`we` pools Saturday and Sunday**, because `schedule_bin` maps both onto
   `we` and the ridership `we` rate the split multiplies is itself pooled;
   a Saturday-only weight would misread every Sunday.

Verified locally on real public data (GTFS + the artifact through the Worker's
own zod loader) and deterministic unit tests: the GC shuttle platform's share
drops 16.7% -> 10.5% and Times Sq's 10.0% -> 6.0%, demand is conserved across
the reweight, and partial/zero/out-of-service platforms fall back cleanly.
Worker suite 413 pass, viz 117, python green, all typechecks clean. What is
still NOT closed: the end-to-end published `waiting_riders` replay against the
real `station_wait` + ridership baseline in R2, which this worktree cannot
decrypt — and actually publishing the baseline + deploying. The share
reduction is the mechanism of the fix and it is measured; the absolute
per-tick number wants that replay before a production sign-off.

## 2026-08-25 — parked the direction axis as a diagnostic, not a split input; the demand asymmetry is real and large, but nothing here maps it onto a physical platform

origin: agent

Decision was to park the OD direction axis as a validation instrument rather
than wire it into the live split. Built `training/od_direction.py` for it: a
manual diagnostic (no cron, no R2, no Worker, no published artifact) that, per
origin complex, measures the share of demand heading to a destination north vs
south of it by (weekday/weekend, hour) from OD `28vm-gjqr`, reported at each
complex's own busiest weekday hour so the headline is a real peak, not a thin
overnight cell. Pure reduction unit-tested; the per-origin query is origin-
filtered (the only shape the public endpoint serves without timing out — a
global group-by-origin scan times out, and even filtered runs 20-40s, so a full
425-complex sweep is a slow manual job, which is the honest reason it stays a
diagnostic).

Measured, weekday busiest hour, share north / south:

    Astoria-Ditmars (N W)        h08   6% / 94%
    Atlantic-Barclays (B Q ...)  h08  78% / 22%
    Flushing-Main St (7)         h07  25% / 75%
    Grand Central (4/5/6, 7, S)  h17  38% / 62%
    W 4 St (A C E ...)           h17  50% / 50%

So the asymmetry the live 50/50 split assumes away is real and sometimes huge:
Astoria at the AM peak is 6/94, a 0.44 departure from even. That is the size of
the bias we are choosing not to correct, now on the record.

**What this does NOT establish, and the metric I built and then deleted.** I
first added a `lat_dominance` number — the share of a complex's demand spread
that is latitudinal vs longitudinal — and used it as a "is the N/S geometry
trustworthy" gate. That was wrong and I removed it. It measures the DESTINATION
TRAJECTORY, not the LINE's orientation: an Astoria rider bound for Manhattan
travels south and west, so the trajectory reads half-longitudinal (lat_dominance
~0.36) even though the N W line itself runs north-south and the 6/94 asymmetry
is genuine. The metric cannot tell "crosstown line, N/S label is nominal" (the 7
at Flushing) apart from "N/S line, diagonal destinations" (Astoria) — it lands
low in both — so it can neither certify nor discard a reading. Do not re-add it
as a guard. Certifying that a measured N/S demand share maps onto a physical
platform needs each line's own direction geometry, which none of this reads;
until then these are demand asymmetries, not per-platform corrections.

That is the whole case for parking, and it is not "the bias is small" — it is
large. It is that (1) there is no per-platform ground truth to grade a direction
split against, (2) the source is a 43-day-lagged estimate-on-an-estimate, and
(3) turning a complex-level N/S demand share into a platform correction needs
line geometry we have not built. The instrument stays as the thing a future
direction split would be graded against.

## 2026-08-24 — silence made to talk: every baselined cell judged every tick, and the discrimination survives it at 3.3x

origin: agent

`segment_flow` judged 19.8 of 1,658 baselined cells per tick because
`MIN_EFF_MATCHED=5` over a ~25-minute decayed window needs sub-5-minute headways
at a single stop, and a cell with no trains simply left the accumulator. Absence
was treated as abstention. It is now evidence.

**The mechanism.** Each cell carries a third decayed sum beside advanced/matched:
`e`, the expected traversals over the same window, accumulated from a new
per-(cell, `schedule_bin`) rate the trainer publishes as `lam` in
`segment_params.json`. Running the expectation as its own decaying sum rather
than scaling the current bin's rate by `1/(1-DECAY)` is what keeps an hourly or
weekday/weekend bin edge honest — for the five ticks after 06:00 the window still
holds overnight traffic, and comparing that against a rush rate would read as a
collapse on every cell at every bin edge. With a constant rate the two forms
agree exactly.

The call is a Poisson lower tail on the effective count. A decayed sum of
per-tick Poisson counts is not Poisson — mean `Σd^i λ`, variance only
`Σd^2i λ` — so both sides scale by `(1+DECAY)`, the standard weighted-Poisson
effective count, which puts mean and variance back on each other.

Two decisions in it are derivations rather than knobs:

- **The quiet floor is where the test loses all power.** Below
  `-ln(THROUGHPUT_ALPHA) = 2.996` expected traversals, even a completely empty
  window sits above the tail threshold, so no observation could ever fire. A cell
  under it publishes a new `quiet` state — normal for now, by timetable — instead
  of abstaining. 88.6% of cell-ticks land there, which is what an overnight
  subway looks like.
- **The effective count floors, it does not round.** It is "how many complete
  traversals the window can account for", and rounding a fractional decayed
  remnant up to a whole traversal credits evidence never observed. Flooring also
  makes the call monotone in the observation; with rounding, a cell whose
  expectation was fading toward quiet flipped normal/disrupted purely on which
  side of .5 the remnant landed, once per cell per bin edge.

**Graded causally, 14 days fitted (2026-08-04..08-17) and 7 held out
(08-18..08-24), 1,765 scored ticks.** Truth is `degradation_label`'s assigned_n
label — unmodified, and with its `(route, schedule_bin)` baseline also fitted on
the leading window only, so an episode cannot lower the median it is measured
against. 96 episodes, 182 normal stretches, 15 routes. Both arms are the same
classifier over the same archive; the `before` arm is the same params doc with
the fit stripped.

|        | judged/tick | cells ever judged        | gradeable episodes |
| ------ | ----------- | ------------------------ | ------------------ |
| before | 19.8        | 494 / 1,658 (29.8%)      | 30 / 96            |
| after  | **1,657.8** | **1,658 / 1,658 (100%)** | **52 / 96**        |

Coverage moved as intended, and it bought 22 more episodes that could be scored
at all rather than only more cell-ticks.

| arm    | episode share            | normal share             | ratio    | CIs disjoint |
| ------ | ------------------------ | ------------------------ | -------- | ------------ |
| before | 0.401 [0.245, 0.611]     | 0.082 [0.051, 0.123]     | 4.88     | yes          |
| after  | **0.474 [0.398, 0.552]** | **0.142 [0.111, 0.180]** | **3.33** | yes          |

Share = disrupted over TESTABLE cells on the route (normal + disrupted), per
tick, bootstrapped over episodes. The cost is visible and real: the base rate on
genuinely-normal routes roughly doubles (8.2% → 14.2%) and the discrimination
ratio drops 32% relative, because the newly-covered cells are thinner and noisier
than the busy ones the advance branch already handled. The episode CI halves in
width (0.37 → 0.15) on 52 episodes instead of 30.

### Two statistics that had to be discarded to get there

**Dead metric 1 — "any disrupted cell on the route".** At ~1% of cell-ticks
disrupted and ~60 cells on a route, an any-cell alarm fires on nothing at all
~40% of the time. Measured: 78.1% on episodes against 44.5% on normal stretches
in the after arm. Saturated, and nearly powerless.

**Dead metric 2 — the same share over JUDGED rather than testable cells.** It
read 0.0246 [0.0043, 0.0663] on episodes against 0.0106 on normal, ratio 2.32
with overlapping CIs — an apparent collapse of the signal that was entirely a
denominator artifact. `quiet` is the classifier saying it has no power on that
cell; counting those in the denominator dilutes the share by however much of the
network happens to be asleep, so the comparison between arms became a comparison
between denominators.

And one asymmetry that inverted the answer before it was fixed: episodes have a
median of 6 ticks, normal stretches 76. Any per-unit "did it ever fire" rate
hands the longer population twelve times the chances, and the first cut of this
grade reported detection 45.3% against false alarms 60.1% on exactly that
artifact.

**Parity, because the grade is only worth its fidelity.**
`training/segment_replay.py` runs the Worker's actual accumulator — same EWMA,
same expectation sum, same two branches in the same order, over the same
baselined cell set — and `tests/fixtures/parity_segment_flow.json` pins the two
tick for tick, bit for bit. The two pre-existing offline replicas both
approximate the EWMA as a 6-tick trailing sum, deliberately and for other
purposes; neither can grade a change to this classifier. Python's `round()` is
banker's rounding and the Worker's `Math.round` is not, which desyncs on every
exact .5 — and decayed sums of integers times powers of 0.8 hit those often.

Standing caveat the numbers cannot lift: assigned_n and vehicle positions come
from the same upstream feed family. Independent in derivation, per
`degradation_label`'s own argument, not independent in source.

### Bakeoff against the other axis: they compose, and the pair beats either alone

The `feat/segment-status-map` branch bought the same coverage on the TIME axis
instead — `SEGMENT_DECAY` 0.8 -> 0.94 (25 -> 83 min window) and
`MIN_EFF_MATCHED` 5 -> 3 — selected against a route-level severity recall proxy
whose absolute values are all sub-1% and whose weakness that branch documented
itself. `segment_replay.Policy` makes the accumulator's two
levers replayable, so both axes and their cross can be scored on one statistic.
Same causal split, 110 episodes, 196 normal stretches, 1,958 scored ticks:

| arm        | decay | window | judged/tick | coverage | gradeable episodes | episode share        | normal share         | ratio    |
| ---------- | ----- | ------ | ----------- | -------- | ------------------ | -------------------- | -------------------- | -------- |
| status quo | 0.80  | 25 min | 20.0        | 1.2%     | 33/110             | 0.367 [0.224, 0.552] | 0.082 [0.052, 0.121] | 4.49     |
| window     | 0.94  | 83 min | 688.6       | 41.5%    | 82/110             | 0.020 [0.009, 0.047] | 0.010 [0.007, 0.014] | 2.00     |
| throughput | 0.80  | 25 min | 1,657.7     | 100%     | 58/110             | 0.486 [0.421, 0.519] | 0.133 [0.106, 0.166] | 3.65     |
| **both**   | 0.94  | 83 min | 1,652.8     | 99.7%    | **90/110**         | 0.334 [0.196, 0.419] | 0.064 [0.056, 0.073] | **5.21** |

Every arm but `window` has disjoint episode/normal intervals. The 41.5% here
independently reproduces that branch's own 42.28%, which is the cross-check that
makes the rest of the column trustworthy.

**The window axis alone does not survive this metric.** It judges 34x more cells
than the status quo and says `normal` for essentially all of them — 4,875
disrupted against 1,343,490 normal, a 0.36% disrupted rate — so its episode share
collapses to 0.020 and the separation stops being significant. Coverage without a
verdict that moves is not coverage of anything.

**The two axes are not substitutes.** Throughput supplies the discrimination the
wider window loses; the wider window supplies matched-count depth that makes 90
of 110 episodes scoreable against throughput-alone's 58. Crossed, the ratio beats
even the status quo (5.21 vs 4.49) while judging 83x more cells.

Which leaves exactly one open question, the one the decay knob has always
turned on: `both` spends 83
minutes of window, so its `entered_at` and its recovery forecast can lag a real
onset by up to that. Nothing here measures latency. `throughput` alone is the arm
that buys the coverage for free on that axis, and it is significant on its own.

### Onset latency, measured from the onset and not from a lead window

The open question is detection latency, so `_onset_latency` measures it. The first cut
used scorecard.onset_latency's 30-minute lead tolerance and returned medians of
-25, -30, -28, -30 minutes across the four arms — against a floor of exactly -30.
That is a censored distribution reporting its own boundary: this surface's alarm
rate is high enough that on most episodes it is already firing when the lead
window opens, so the search starts inside the alarm and every latency pins to
`-lead_sec`. The number said nothing about any classifier.

Measured from the onset tick itself, with episodes already alarming AT the onset
counted separately and excluded rather than scored as a zero:

| arm        | window | already alarming | measurable | detected | median | p90     | clean-before | detected |
| ---------- | ------ | ---------------- | ---------- | -------- | ------ | ------- | ------------ | -------- |
| status quo | 25 min | 21               | 89         | 4        | 12 min | 30 min  | 12           | 2        |
| window     | 83 min | 22               | 88         | 3        | 10 min | 30 min  | 53           | 1        |
| throughput | 25 min | 37               | 73         | **11**   | 15 min | 430 min | 14           | **6**    |
| both       | 83 min | 50               | 60         | 7        | 10 min | 35 min  | 27           | 3        |

**This is much less flattering than the lead-window run, and correctly so.** That
run credited `both` with 61 detections of 110; 50 of those were the alarm already
being on. Onset detection is weak for every policy — 3.4% to 15.1% of measurable
episodes — which is unsurprising when the median episode is 6 ticks long, but it
is the real number and the earlier one was base rate wearing a detection label.

**Throughput leads onset detection: 11 of 73, against the status quo's 4 of 89 and
the wider window's 3 of 88.** Roughly a 3x rate improvement, consistent with its
share separation. On 11 events, so quoted as a direction, not a measurement.

**No arm shows a latency penalty from the wider window.** `both` and `window`
median 10 minutes, `throughput` 15, status quo 12 — two to three ticks either
way, well inside the noise of 3-11 detections apiece. The fear that priced 0.94
against 0.98 does not appear here, but neither is it refuted: these counts cannot
rank policies on latency.

So the window stays unpriced, and the instrument is the reason. Two fixes it needs: measure
first crossing on the DEBOUNCED regime rather than the raw per-tick call, since
the debounced regime is what the published surface flips on; and find a truth
whose episodes outlast the window being priced, because a 30-minute median episode
cannot price an 83-minute accumulator either way.

### The bakeoff on the surface riders actually see: the wide window is safe only with throughput, and the pair still wins

The first bakeoff graded raw classifier calls. `published_states` now runs every
arm through the regime clock and grades the debounced state the snapshot actually
shows. This is the staleness test the window needed: an abstaining cell keeps its
last regime for up to an hour, so a wide accumulator can look precise internally
while publishing old evidence.

Same causal split; this archive pass held 94 episodes, 176 normal stretches and
1,735 scored ticks:

| arm        | window | call ratio | published ratio | published episode share | published normal share | onset detections | median     |
| ---------- | ------ | ---------- | --------------- | ----------------------- | ---------------------- | ---------------- | ---------- |
| status quo | 25 min | 3.72       | 3.20            | 0.188                   | 0.059                  | 1/76             | 20 min     |
| window     | 83 min | 1.86       | 1.55            | 0.009                   | 0.006                  | 2/76             | 30 min     |
| throughput | 25 min | 3.64       | 3.65            | 0.489                   | 0.134                  | **7/59**         | 35 min     |
| **both**   | 83 min | **5.32**   | **5.30**        | 0.341                   | **0.064**              | 5/49             | **10 min** |

Episode/normal intervals are disjoint for every arm except `window` alone.

**Full coverage removes the staleness mechanism.** Staleness enters through the
regime clock holding a cell's last state while the classifier abstains. The two
arms without throughput lose 14-16% of their separation between calls and the
published surface (3.72 -> 3.20, 1.86 -> 1.55). The throughput arms judge every
cell every tick, never enter that hold, and their scores are unchanged to the
second decimal (3.64 -> 3.65, 5.32 -> 5.30). The objection to an 83-minute window
was real for the old classifier and structurally void for the new one.

**On absence, the wider window is mechanically faster rather than slower.** The
expected-count accumulator is ~3.3x larger, so an empty route crosses the Poisson
threshold sooner; `both` median onset latency is 10 minutes against 35 for
narrow-window throughput. On 5 and 7 detections, a direction rather than a
measurement, but it agrees with the mechanism and the separation.

That is enough to retune the shipped pair to decay 0.94 / matched floor 3. The
window-only arm remains rejected: it publishes the weakest, non-significant
separation and the slowest onset despite judging 41% of the cells. The win comes
from the cross, not from accepting that branch unchanged.

## 2026-08-26 — reconciled onto main: throughput and the 0.94/83-min window ship, span-crediting is dropped by main's own grade, and the pair re-grades at 5.6x on the rebased tree

origin: agent

`feat/segment-throughput-branch` was cut before `feat/segment-status-map` (PR #6,
da5383a) landed on main, and the two overlapped hard — a straight merge produced
55 conflict blocks across 12 files. Rebased instead, semantically, one mechanism
at a time. Main advanced twice more during the work (segment-status-map, then
journey-enumeration + status-first pages + the HMM movement-emission fit); the
final branch sits on `2d7de2f` and fast-forwards.

**Two of the three commits carried over unchanged in intent.** Main had
independently adopted decay 0.94 / floor 3 (the same retune this branch reached
from the other direction), so the window question was already settled the same
way on both sides. The throughput branch layered on top with no contest: main's
segment core (`segment_flow.ts`, `state.ts`, `vehicles.ts`) was untouched by
every main commit after the fork, so the only real integration work was threading
the new `quiet` state through main's rewritten viz — the `@/lib/overlays` +
octilinear-diagram map architecture that replaced the old `projector` map this
branch had modified. `quiet` is now a first-class `SegmentState` (muted, ranked
above `normal` so a sparse overnight map never reads as a green all-clear, below
`unmeasured` since no verdict is more conservative than a benign one) across
`segments.ts`, `overlays.ts`, the trip page, and the station roll-up.

**The third commit — span crediting (`hops`) — was dropped, by main's grade and
not by merge order.** Main's `training/segment_coverage.py` had already measured
pattern-confirmed multi-stop hop crediting as its `expand` lever and rejected
shipping it: "EXPAND costs recall at every decay tested (0.80: 0.05% -> 0.03%;
0.94: 0.17% -> 0.10%; 0.98: 0.22% -> 0.13%) in exchange for coverage -- the
opposite of what the retune was for. Not shipped for that reason, not for the
~685KB trainer-side hop map it would additionally need." This branch had shipped
exactly that mechanism to the Worker (a live `PathPatternIndex` in `vehicles.ts`
feeding a `hops` field `segment_flow.ts` read in place of `transitions`), with no
countervailing grade. The throughput grade that justifies the whole branch was
itself measured on the TRANSITIONS stream — the graded windows all predate the
`hops` deploy and the production Worker never wrote a v2 archive — and the fit
(`lam`) and the observation (`matched`) must share one credit rule or the
expectation stops matching what gets counted. So the graded, self-consistent
configuration is throughput-on-transitions; keeping `hops` would have been
keeping it against the only grade that touches it. Reverted to `transitions`.

The repo's adversarial pre-commit reviewer flagged the removal on exactly the
right instinct: without `hops`, a 5-minute-cadence jump A>C credits only the
A-keyed cell, so an intervening cell B reads `matched = 0` against a real
expectation and could false-disrupt — and at this cadence ~90% of moves span
several stops. It is neutralized by the credit-rule coupling it could not see:
B's `lam` is fit from the same transitions stream that rarely credits B, so B's
expectation is correspondingly low and it reads `quiet`/`normal`, not
`disrupted`. The fresh grade below carries the proof — the shipped pair's
normal-run false-alarm share is 0.063, not inflated.

**Re-graded on the rebased tree** (transitions, published/debounced surface,
fit causally on the vehicle archive 2026-08-06..08-19 and scored on
08-20..08-26; 2,013 scored ticks, 114 assigned_n episodes, 202 normal runs, 16
routes):

| arm                | decay | window | judged/tick | coverage | gradeable eps | pub episode share    | pub normal share         | ratio    | CIs disjoint |
| ------------------ | ----- | ------ | ----------- | -------- | ------------- | -------------------- | ------------------------ | -------- | ------------ |
| status quo         | 0.80  | 25 min | 20.3        | 28.9%    | 34            | 0.279 [0.173, 0.443] | 0.056 [0.035, 0.082]     | 5.00     | yes          |
| window             | 0.94  | 83 min | 690         | 93.5%    | 84            | 0.012 [0.005, 0.029] | 0.006 [0.004, 0.008]     | 2.04     | **no**       |
| throughput         | 0.80  | 25 min | 1,658       | 100%     | 59            | 0.475 [0.380, 0.517] | 0.129 [0.100, 0.166]     | 3.67     | yes          |
| **both (shipped)** | 0.94  | 83 min | 1,652       | 100%     | **91**        | 0.350 [0.190, 0.454] | **0.063 [0.055, 0.071]** | **5.59** | yes          |

The pairing that ships — decay 0.94, floor 3, throughput on transitions —
publishes a 5.6x episode/normal separation with disjoint CIs and full coverage on
91 gradeable episodes, against 34 for the status quo. The throughput arm
reproduces the pre-rebase 3.65x at 3.67x, so the mechanism came through the
rebase and the `hops` removal intact. The window-only arm remains exactly what
the earlier bakeoff called it: the weakest and only non-significant separation
(2.04x, overlapping CIs) despite judging 93% of cells. The win is still the
cross, and it survives dropping the axis main had already graded away.

## 2026-08-26 — GTFS after-midnight rows wrap into the wrong service class: 18,909 of 565,093 stop_times rows (3.35%) run on the next wall-clock day

origin: agent

The pre-commit adversarial gate blocked the scheduled-service-split merge on a
bug the whole test suite passed: `hour_of()` mapped `24:xx`/`25:xx` departures
back to a wall-clock hour with `% 24` but left the count in the SERVICE day's
`wd`/`we` bucket, while the Worker's `serviceWeightFor` keys the lookup by
actual wall clock. A Friday 25:10 trip counted into `wd[1]`, but a rider on
that platform at Saturday 01:10 is read against `we[1]`; Sunday 25:10
contaminates `we` while Monday 01:10 reads `wd`. 3.35% of all scheduled
departures sit past 24:00, so every overnight cell near a weekday/weekend
boundary carried some wrong-class weight.

**The rejected fix, and why it is structurally dead, not just unshipped.** The
obvious repair — count a >=24:00 row into the class of the reference day's
successor via previous-day active service sets — looks exact and is not: the
artifact publishes ONE `wd` vector applied to all five weekdays, so Monday's
early hours (fed by Sunday-night, weekend-class service) and Tuesday-Friday's
(fed by weekday-night service) cannot both be right in a two-bin shape. Any
per-class predecessor rule just moves which morning is wrong. Exactness needs
day-of-week resolution in both the artifact and the Worker lookup.

Shipped instead: exclusion. `hour_of()` returns None for hour >= 24, an
excluded row can zero a platform's overnight cell, and a zero forces that
complex back to the even split for that hour — the pre-feature behavior,
never a wrong-class weight. The split therefore abstains overnight rather
than lying about which platform the demand goes to. Worth remembering the
shape of the catch: 456 worker tests and the full python suite were green on
both sides of this bug; only the semantic review gate saw it.

## 2026-08-27 — correction: not "every overnight cell near a boundary" — only cells that themselves carry >=24:00 rows, and only where the spillover crosses a class

origin: agent

The entry above overclaimed its blast radius. The wrap put weight in the WRONG
class only where a cell's own >=24:00 rows spill across a weekday/weekend
boundary (Friday-night weekday service landing Saturday morning, Sunday-night
weekend service landing Monday morning). A mid-week 25:10 row wraps into
`wd[1]`, which is the class Tuesday-Friday 01:10 actually reads — same-class
spillover, not corruption. So the corrupted set is: overnight cells containing

> =24:00 departures whose service day's class differs from the next wall-clock
> day's, a subset of the 3.35% of rows, not "every overnight cell near a
> boundary". The shipped exclusion drops all >=24:00 rows regardless, which is
> broader than the corruption but the only shape a two-bin artifact can carry
> honestly.

## 2026-08-27 — supply-ratio 1.7-2.1x on weekend late nights: denominator is correct, the thin 2-weekend median is the amplifier

origin: agent

Re-verification of the 170-208% supply readings (routes 1/2/J at
169/208/175% of baseline while the median route sits at 100%). Confirmed the
denominator is correct and the readings are real off-distribution surplus, not
a wrong-cell-join / daypart / thin-cell defect — but the amplifier is a real
sampling weakness in how the baseline window is sized and gated.

**The sidecar reproduces bit-for-bit from the archive.** `state/service_baseline.json`
is v1787792319 (generated_at == params_trained_at, i.e. baked by the retrain over
its training window `2026-08-14..2026-08-27`, NOT by the 28-day backfill tool).
Re-fetching the raw `archive/trip_updates/` for that window and re-running
`build_service_series` -> `compute_baseline(bin_fn=schedule_bin)` /
`compute_service_quantiles` reproduces every cell exactly: route 1 `we23`
median=13.0 (p10=8/p90=18), route 2 `we23` median=11.5 (p10=7/p90=16), route J
`we23` median=5.0 (p10=2/p90=8), route C `we23` median=2. So no wrong join, no
arithmetic error. The Worker's `serviceRatioFor` (movement_state.ts) divides live
`assigned_n` by `baseline[route][schedule_bin(observedAt)]` with the identical
`schedule_bin` the builder used — same cell, both sides.

**The reported ratios reproduce from those medians.** 22/13 = 169%, 24/11.5 =
209%, 9/5 = 180%. The live assigned_n behind them (~22/24/9 trains on a weekend
late night) is near a NORMAL WEEKDAY-MIDDAY level: the current Thu-12:40 snapshot
reads route 1 assigned=24 at ratio 1.045 (wd12 median ~23), route 2 assigned=27
at 1.038 — all ~1.0. The machinery is calibrated correctly everywhere except
where the baseline itself is thin.

**Why 2x is expected here, not a bug.** The training window is 13 days = exactly
2 weekends, so every `we<hour>` cell has only 4 independent nights. `compute_baseline`
gates on `min_samples=20` counting 5-minute TICKS (~12/hour, near-constant within
a night), so a weekend-hour cell clears the gate on ~2 nights though it is really
2-4 independent draws. Late-night weekend service is bimodal (reduced/trackwork
nights vs normal nights) and the median lands in the empty gap between the modes:
route 2 `we23` per-night medians are 8, 9, 16, 15 -> cell median 11.5 (no night
ran ~11.5); route J `we23` is 2, 2, 7, 7 -> cell median 5 (no night ran 5). A live
night at 24 (route 2) or 9 (route J) exceeds EVERY training night (max 16 and 8 =
p90), so it correctly reads above p90 — a genuine surplus measured against a
correct denominator that happens to sit between two modes.

**Actionable weakness (filed, not fixed — params/training-window side, eval on
v1787792319 protected):** the `min_samples` gate counts autocorrelated ticks, not
independent service-nights, so a 2-weekend window certifies weekend-hourly cells
that are really a 2-4-point summary of a bimodal distribution. The fix is a longer
dedicated baseline window (the backfill tool's 28-day default gives ~8 weekend
nights) and/or gating on independent-night count rather than tick count. Both
change what the artifact publishes and feed the emission `service_mu/sigma`, so
they are params-side and out of scope for a builder-only fix.

**Recompute gotcha for the next investigator:** `archive/trip_updates/{d}/` is
keyed by UTC date, but `schedule_bin` buckets by ET. ET weekend hours 20-23 are
UTC 00-03 of the next day, so a weekend-only fetch of the ET Sat/Sun UTC prefixes
silently drops every Sunday-ET late night (they live under the Monday UTC prefix)
— a partial fetch gave a `we23` median of 14.5 vs the true 13.0. Fetch the full
contiguous window (as train_em does) or the recompute will not reconcile.

---

## 2026-08-31 — recovery-grader-arm-mismatch

origin: agent

`_grade_recovery` in `training/eval.py` gated ticks on one condition arm but
graded `recovery_minutes`, which is produced by whichever arm `recovery_source`
names — a different arm. The shadow grade (`recovery_metrics`) gates on the
alert-shadow `condition` but was reading movement-sourced estimates too. On a
route the shadow calls disrupted while movement calls it normal, movement
correctly emits `recovery_minutes=0` (nothing to recover from); the shadow grade
then scored that 0 as a forecast of instant recovery against its own disrupted
clock, so every such row logged the full remaining regime as error and a coverage
miss.

Measured over 2026-07-28..08-24 (233,914 predictions, 752 transitions), shadow
recovery as published: n=1097, 269 incidents, MAE=100.4 min, IQR coverage=0.085.
906 of those gradeable rows were movement-sourced. Grading each arm only against
its own source — shadow reads `{hmm}`, the trip-updates grade reads `{movement}`,
cross-arm rows excluded and counted — leaves n=192, 52 incidents, MAE=29.0 min,
IQR coverage=0.469. Same model, same window; the 3.5x improvement is entirely the
removal of cross-arm rows, not a modeling change, and the shrunk support (192
ticks / 52 incidents) must ship beside the number so it does not read as one.

The guard for exactly this failure already existed and already described it, but
it tested the condition stream rather than the stream the estimate came from.
`recovery_minutes` still cannot express "no estimate" without overloading
`recovery_indeterminate`; the source gate sidesteps that but does not fix the
underlying contract.

## 2026-08-31 — publish-prep: the movement-trained retrain moves ONLY the advance emission live — service_mu/sigma are still prior-only (0 of 28 routes fitted), because training never carries the service channel

origin: agent

Retrained on the current window (`2026-08-18..08-31`, `prior_strength=100`,
`min_ticks=288`, `--dry-run`) to diff the fitted emissions against the live
published set (`trained_at=1787792319`, `2026-08-14..08-27`) before a real
publish. `baseline_cells=211`, `service_cells=131`, movement mass healthy — the
advance-baseline publish gate is nowhere near empty.

**The advance emission is genuinely fitted, and it is the only channel that
moves.** Per-state `advance_rate` shifts on every route with movement mass and
tracks the responsibility-weighted `k/n` from `--diagnose-advance` (e.g. N
disrupted 0.540→0.508 at `mov_n≈27k`; F normal 0.973→0.927; G disrupted
0.234→0.838 as EM re-seats G's alert-thin states). The global prior's
advance_rate is the fitted `(0.919, 0.903, 0.838)`. This is the intended
operating-point move.

**Negative result — service_mu/service_sigma are NOT fitted: 0 of 28 routes
deviate from the prior default.** After the retrain every route still ships
`service_mu=(1.0, 0.6, 0.05)` and `service_sigma=(0.3, 0.3, 0.15)` exactly (the
`DEFAULT_EMISSIONS` constants), identical to the currently published set. The
cause: `training/load_r2.py` builds each `Observation` with the alert flags,
tod_bin, and (since 2026-08-25) the folded movement fields, but never sets
`has_service`/`service_ratio`. So `svc_w=0` for every state and the M-step
(`hmm.py` `_m_step`) returns `prior.service_mu[s]` unchanged. The service
Gaussian is a prior-only channel in training — exactly the state the movement
channel was in before it was wired — while the live worker still scores
`service_ratio` against these constants (`worker/src/hmm.ts` gates on
`service_mu !== undefined`, which the published params satisfy). So a
"movement-trained params" publish moves the advance emission and leaves the
service emission's operating point untouched; the earlier assumption that this
retrain also fits the service channel does not hold.

**Canonicalization holds under the now-fitted advance tiebreak.** The concern
was that non-constant `advance_rate` feeding `canonicalize_states`' tiebreak
could re-label states. It does not: across all 28 routes the suspended slot is
still decided by the suspended-flag Bernoulli (`bernoulli_p`), which is strictly
ordered `suspended ≥ disrupted` everywhere — the smallest gap is R at 0.005, and
there is no exact tie for the advance term to break. Zero state assignments flip
on the fitted advance. One pre-existing interaction does shift, though: because
the prior `service_mu` constant is permuted along with a route's canonicalization
and never re-applied in canonical order, any route whose EM converges to a
non-identity permutation ships a severity-inverted `service_mu`
(`(1.0, 0.05, 0.6)` — disrupted 0.05, suspended 0.6). The retrain moves that
affected set from `{6X, W}` (published) to `{3, G, N, W}`. Harmless while the
channel is prior-only, but it is a latent mis-order that a real service fit would
need to fix first; out of scope for this publish.

## 2026-08-31 — negative result: the published movement condition and the independent assigned_n label are statistically independent, so assigned_n cannot adjudicate movement detection (0/173 severe, base≈false-alarm≈0.1%)

origin: agent

New reusable harness `training/movement_validation.py` (+ `tests/test_movement_validation.py`)
grades the published movement-primary condition (worker/src/movement_state.ts
deriveMovementState, mirrored by load_r2.build_movement_truth, through the regime
clock at the live DEBOUNCE_TICKS=1) against the only reference in the archive that
is independent IN DERIVATION of the vehicle-position signal it reads: the
trip-updates `assigned_n` degradation label (training.degradation_label). Vehicle
positions themselves cannot adjudicate the condition — the condition is derived
from them. Episode-bootstrap detection / false-alarm / onset-latency, CIs,
counted exclusions, following segment_coverage.py.

**Window.** Advance + assigned_n baselines fitted causally on 2026-07-28..08-10
(14d); scored held-out 2026-08-11..08-31 (21d), the current operating point
(pooled through-filtered advance baseline, prior_strength=8 / disrupted_ratio=0.5
/ alpha=0.05). 6,039 scored movement ticks, 211 advance cells; truth = 252
assigned_n episodes (173 severe ≤0.15×baseline, 79 partial, 0 unrateable), 381
confirmed-normal runs, 18 routes, episode median 9 ticks (45 min), 1,126
(route, schedule_bin) baseline cells.

**The reconstruction fires, but flat across the label.** Call mix over 111,578
judged route-ticks: 110 disrupted, 111,468 normal, **0 suspended** — disrupted
base rate **0.099%** (same order as the 0.30% the 08-26 calibration measured on a
different fortnight; the classifier is alive, not dead). Then, disrupted arm,
raw calls:

- detection.all: 1 of 84 gradeable episodes ever fired — tick_rate **0.000
  [0.000,0.002]**, unit_rate **0.012 [0.000,0.036]**.
- detection.severe (near-suspension, supply ≤15% of baseline, the subset where
  the two feeds MUST physically coincide): **0 of 40 gradeable episodes**, tick
  and unit rate **0.000**. Published surface: **0 of 70**.
- detection.partial: unit_rate 0.023.
- false_alarm on confirmed-normal runs: 217/381 gradeable, tick_rate **0.00101
  [0.00058,0.00163]**, unit_rate 0.194 [0.143,0.249].

**The finding.** false-alarm tick-rate (0.101%) ≈ disrupted base rate (0.099%) ≈
detection tick-rate (~0.0–0.1%). The movement condition fires at ~0.1% of ticks
whether supply is normal, partially cut, or fully collapsed — its firing carries
essentially **zero mutual information** with assigned_n, in either direction. At
base rate 0.099% over episodes of median 9 ticks, chance predicts ≈0.75 firings
across the 84 gradeable episodes; we observed 1. Detection is at chance. This
replicates and strengthens the 2026-08-20 cross-tab (0/785 assigned_n-disrupted
ticks read movement-disrupted) at n=252 episodes / 111k route-ticks.

**Why, and what it means.** assigned_n measures SUPPLY (trains dispatched);
movement measures FLOW (whether running trains advance). A route with 85% of its
trains pulled but the remaining 15% moving fine is supply-collapsed and correctly
movement-normal — so even on the severe tail the flow signal does not (and
arguably should not) fire. The independence is a property of the two signals, not
a defect in the classifier. Consequence: **assigned_n cannot serve as a detection
truth for the movement flow axis — it can neither confirm nor refute a
movement-disrupted call.** No independent detection truth for the flow axis exists
in the archive: vehicle positions ARE the signal, assigned_n is orthogonal, and
the alert feed lags with no true-negative class (escalation.py). Onset latency is
therefore undefined here: over episodes the vehicle feed could actually judge
(≥1 scored movement tick on the route), 1 of 84 all / **0 of 40 severe** detected.
The exclusions are themselves telling — **168 of 252 episodes (133 of 173
severe) had NO judged movement tick anywhere in their span**: a supply collapse
leaves too few cross-tick matches to form a movement call at all, so the flow
signal is not merely silent on these, it is structurally blind to them (counted
as n_ungradeable, never charged to detection).

**Reconstruction limit, stated not hidden.** Offline suspended is vehicle-only
(build_movement_truth keys it on vehicles_n==0) and fired 0 times, so `not_normal`
≡ `disrupted` throughout; the live worker additionally reads suspended when
assigned_n==0, which this harness cannot reproduce. But that live suspended arm is
gated on assigned_n itself, so validating it against an assigned_n truth is
circular — it would only confirm the two feeds agree a dead route is dead, not
that the FLOW signal detects anything. The genuinely independent question is the
disrupted arm's, and its answer is above.

**Verdict for the promotion gate.** This validation does NOT
supply independent detection evidence for promoting the movement condition, and
by construction it cannot — the axis is orthogonal. What it DOES establish: the
published condition's absolute alarm rate on independently-confirmed-normal supply
is very low (0.10% of ticks [0.06,0.16]), i.e. it does not flood normal service
with disrupted calls — but since that equals the base rate, it is a low-base-rate
property, not proof of supply-concordance. Recommendation: do not treat
cross-feed detection agreement as a cleared gate; there is no archived truth that
can clear it. The movement condition's trustworthiness rests on its calibration
and low false-alarm behaviour (settled 2026-08-26) plus the alert-corroboration
lower bound (escalation.py), not on independent detection agreement. Harness run
offline only; nothing published.

---

## 2026-08-31 — supply axis abstains on thin cells: an independent-night gate on the published (schedule_bin) baseline, tod_bin emission untouched

origin: agent

Closed the actionable weakness the 2026-08-27 entry filed (weekend-hourly
`we<hour>` cells reading 1.7-2.1x off a bimodal 2-4-night median) on the
publication side, so the supply axis no longer presents an un-trustworthy median
as a confident reading. The supply axis itself (per-route `service_condition` +
`service_ratio`/`service_low_ratio`/`service_high_ratio`/`service_percentile`,
distinct from the movement `condition` flow axis) was already published and
consumed by the viz (route-card glyph + "Trains running" drawer + percentile
Gauge, all gating a null reading out honestly); the only missing piece was the
thin-cell treatment, so this is a targeted addition, not a new surface.

**The fix is abstention, not a clamp.** `compute_baseline` and
`compute_service_quantiles` (training/load_r2.py) gain an opt-in `min_nights`
gate that counts DISTINCT ET calendar dates per cell on top of the existing
`min_samples` tick floor — because assigned_n is ~constant within an hour, a
dozen 5-min ticks from one night are one autocorrelated draw, not twelve, which
is exactly why the old `min_samples=20` gate certified a 2-night cell. A cell
below the night floor is OMITTED (no baseline -> `serviceRatioFor` returns null
-> `service_condition` 'unknown' -> the viz already renders "No reading… nothing
to compare it against for this hour"). No ratio is fabricated or clamped; the
reading is withheld until enough independent nights exist.

**Applied to the published axis only.** `_service_baseline` (train_em.py) passes
`min_nights=SERVICE_MIN_NIGHTS=8` to the schedule_bin `hourly` baseline + its
quantiles (both, so `set(baseline) == set(quantiles)` still holds and the Worker
can always pair a low/high ratio with its median), and leaves the tod_bin
emission denominator at the default `min_nights=1`. So the frozen HMM operating
point / `service_mu/sigma` emission channel is byte-for-byte unchanged — this is
a sidecar-generation change, out of the protected v1787792319 eval's path. The
offline degradation label keeps its own ungated `compute_baseline` (a truth
signal, not a mirror of the published axis), so its grade is unperturbed too.

**8 nights, and the window that feeds it.** 8 = a full month of weekend nights
(4 weekends x Sat/Sun); a normally-thin `we<hour>` cell abstains until it has
that many independent nights instead of publishing a between-modes median. The
backfill tool's default window widened 28 -> 35 days (5 weekends = 10 Sat/Sun
nights per cell) so a regenerated sidecar clears the floor with margin rather
than landing exactly on it.

**Evidence.** New unit tests in tests/test_load_r2.py: a 2-night/24-tick cell
clears `min_samples=20` but is withheld at `min_nights=8`; an 8-night cell
publishes; `min_nights=1` is a proven no-op (identical to the current behaviour);
baseline and quantiles gate identically under the same floor. tests/test_train_em.py
asserts `_service_baseline` threads `SERVICE_MIN_NIGHTS` into BOTH the schedule_bin
baseline and its quantiles while the tod_bin call stays ungated. Full suites green:
Python 1052 passed, worker (vitest) 456 passed, viz (node --test) 231 passed.

**Not fixed here (still filed):** the Sat/Sun pooling into one `we` bin (a real
split candidate if weekend service differs materially) and the actual sidecar
regeneration — the live v1787792319 sidecar was baked over a 13-day window, so
its weekend cells will abstain under the new floor until a 35-day backfill
republishes them. Regeneration needs R2 and is a deploy step, staged not run.
## 2026-08-31 — fit-or-drop verdict on the service emission: DROP. Wired assigned_n into training, fit it, and the Gaussian does not separate the HMM states — median per-state mu spread 0.15 on a ~1.0 scale, severity-inverted on 15 of 28 routes

origin: agent

Follow-up to the same-day publish-prep entry, which established that the service
Gaussian is prior-only in training (0 of 28 routes fitted) while the live worker
scores `service_ratio` against the `DEFAULT_EMISSIONS` constants. Two things were
done here: fix the latent canonicalization mis-order the earlier entry flagged
(prerequisite), then wire `service_ratio`/`has_service` into training exactly as
the movement channel is wired, fit it on the archive, and decide fit-or-drop on
the numbers.

**Canonicalization fix (prerequisite, ships regardless of the verdict).**
`canonicalize_states` permuted _every_ per-state emission tuple by the fitted
state order, including channels the fit never touched. A prior-only channel's
values are the default constants sitting at their canonical indices, so on any
route whose EM converges to a non-identity permutation the reorder shipped a
severity-INVERTED `service_mu` — e.g. suspended scored against `mu=1.0`. `fit_em`
now derives which channels the corpus never enables (no movement / no service
tick anywhere) and tells `canonicalize_states` to hold those in canonical order
rather than reindex them (`prior_only_channels`, `hmm.py`). This also protects
the movement channel on the ~5 zero-movement routes (`6X`, `7X`, …). Pinned by
`tests/test_hmm.py`: a witness test shows the old reorder inverting
`(1.0,0.6,0.05)→(0.6,0.05,1.0)` and the fixed path holding it canonical, plus an
end-to-end `fit_em` case with no service tick.

**Training wiring + parity.** `service_observation_fields` (`load_r2.py`) is a
line-for-line port of `worker/src/movement_state.ts serviceObservationFields`:
previous tick's carried `assigned_n` (option-B one-tick lag, walked back to
`MAX_SERVICE_METRIC_LAG_SECONDS=600`), divided by the `(route, tod_bin)` baseline
median, gated off when the route is absent from the carried snapshot or the cell
has no baseline; `assigned_n==0` is admitted as a real zero-supply reading (no
`min_matched` floor, unlike movement). Pinned by six parity tests in
`tests/test_load_r2.py`. A `--diagnose-service` flag + `service_responsibility`
(the service analog of `advance_responsibility`) report per-state responsibility
mass and the data-implied mean.

**The fit (window `2026-08-18..08-31`, `prior_strength=100`, `--diagnose-service`).**
Service is now genuinely fitted — `svc_w` is in the hundreds-to-thousands of
ticks on most routes. But the fitted per-state `service_mu` does not separate the
three HMM states:

- Median spread `max(mu)-min(mu)` across states = **0.153** (scale ≈ 1.0), with
  per-state `sigma ≈ 0.25` — the state densities overlap almost completely.
- Only **12 of 28** routes come out monotone (`normal>disrupted>suspended`);
  **16 of 28** are non-monotone / severity-INVERTED in the fit itself (e.g. `2`:
  1.02/0.85/0.99; `N`: 1.07/0.83/1.03; `L`: 1.26/0.88/0.98 — suspended ≥
  disrupted, or disrupted ≥ normal on `A`,`B`,`C`,`D`,`R`).
- **11 of 28** routes have a zero-mass disrupted or suspended state, so the
  Gaussian ships the prior there anyway — the exact prior-only hazard the
  canonicalization fix addresses.
- At a genuine half-service reading (`ratio=0.5`, itself far below the fitted
  suspended means of ~0.9), the channel adds a **median +0.70 nats** toward
  suspended over normal, and pushes the **wrong way on 7 of 28** routes.

**Why (already on record).** `assigned_n` (dispatched-train supply) is
statistically independent of the alert-defined disruption axis the HMM states are
anchored on — the 2026-08-31 negative-result entry measured 0/785
assigned_n-disrupted overlap and 0/173 on severe. EM cannot learn a separating
Gaussian from a signal orthogonal to the states; it fits ≈1.0±0.25 in every
state. And the alert posterior is one-hot on 24/29 routes with per-tick log-odds
in the hundreds (2026-08-22/23), so a sub-nat channel that points the wrong way
half the time cannot move it. The supply information is not lost by dropping it:
the published `service_condition` axis reads `assigned_n` on its own hysteresis
band, and the live suspended arm already gates on it independently.

**Verdict: DROP.** The trainer no longer ships the service Gaussian in params
(`_params_to_json` strips `service_mu`/`service_sigma`, both top-level and
per-tod-bin). The worker's `service_mu`/`service_sigma` are already OPTIONAL
(`worker/src/params.ts`; `worker/src/hmm.ts` only scores the term when
`em.service_mu !== undefined`), so omitting them turns the channel off live via
the same back-compat gate pre-service params used — no worker deploy required.
The production fit stays service-free (the fold runs only under
`--diagnose-service`) so the alert/movement params are not refit under a
service-scored E-step the worker never runs. Dry-run only; nothing published or
deployed.

## 2026-09-01 — retrain-publish-stomped-the-service-sidecar

origin: self

`train_em`'s publish path called `write_service_baseline` over its own HMM
training window — default `--days 14` — so every retrain overwrote
`state/service_baseline.json` with a 14-day sidecar. The published schedule_bin
axis gates each cell on `SERVICE_MIN_NIGHTS = 8` distinct nights; a 14-day window
holds only ~4 weekend (Sat/Sun) nights per late-night cell, so every weekend cell
abstained. Observed live: the service-drop params publish replaced the 35-day
backfilled baseline (**1122 cells**, `v1788228846`) with a **562-cell** one whose
weekend cells all abstained, and it had to be restored by re-running
`backfill_service_baseline` (`v1788230125`). The tod_bin emission baseline
(params-embedded, training-window) was not the problem — only the published
schedule_bin sidecar.

Fix: decouple the sidecar window from the training window. The publish now
computes the schedule_bin sidecar over a dedicated `SERVICE_SIDECAR_WINDOW_DAYS =
35` window (35d = 5 weekends = 10 Sat/Sun nights per cell, margin over the floor
of 8), reusing the training-window result only when the training window is
already ≥ 35d (so a wide retrain does no second fetch, and a `--days 60` run gets
a 60d sidecar). The constant is shared with `backfill_service_baseline`, so a
retrain publish and a standalone backfill write the same sidecar. No flag to
remember: the DEFAULT invocation now refreshes the sidecar correctly instead of
regressing it. The tod_bin emission baseline still ships from the training window
so the frozen operating point is untouched.
## 2026-09-01 — wired the movement false-alarm bound into review.py as a recurring scorecard, and the supply-baseline seam that keeps it from reproducing the standalone 0.00101

origin: agent

`training/movement_validation.py` produced the movement condition's promotion
numbers as a one-off CLI run: false-alarm upper bound 0.00101/tick
[0.00058, 0.00163] on assigned_n-confirmed-normal runs, detection 1/84 all /
0/40 severe (gradeable), 168/252 episodes with no judged movement tick at all.
Extracted the report-building/grading into a pure `build_validation_report(
movement_truth, service_series, service_baseline, ...)` so `main()` and
`training.review` compute the bound one identical way, and wired it into
`review.py` (embedded under `summary["movement_validation"]`, printed each run).
The bound now recomputes from the archive window every review, carrying its
episode-bootstrap CI, gradeable/offered support, and counted exclusions
(`n_ungradeable`, `n_unrateable`) — no longer a number someone has to remember
to re-run.

Non-obvious seam worth pinning: the review's monitored false-alarm number is
NOT the standalone tool's number under identical methodology, and shouldn't be
expected to reproduce 0.00101 to the digit. The standalone tool fits the
assigned_n supply baseline on a held-out leading sub-window (`--fit-days 14` of
21) and scores the remainder; `review.py` reuses its already-loaded `tu_baseline`
= `compute_baseline(tu_series)` fit on the review window ITSELF — deliberately,
because that is `degradation_label`'s own design (a per-(route, schedule_bin)
median resists an outage lowering its own reference), and it lets the whole
window be scored instead of a 7-day tail. The movement side stays causal either
way (advance baseline fit on the pre-window). So the two runs sit on different
supply-baseline windows by construction; the review number is the monitored one,
the tool number is the pinned reference. Treat a drift between them as expected
window/methodology difference, not a regression, unless the review number leaves
the tool's CI by a wide margin.

Separately audited gate 4 (does the published surface label flow distinct from
supply): met at the copy level. The two axes are separated at the data layer
(`viz/lib/feed.ts`) and every render site pulls the same helpers, so wording
can't drift between pages. Flow is worded around movement everywhere
("moving normally / slowly or stalling", badge Normal/Disrupted/Suspended,
map "advancing / not advancing"); supply around counts-vs-baseline ("of usual
trains", "supply low/normal", gauge "fewer/more trains than usual"). The
sharpest disambiguation is the drawer's "Trains running" note
(`viz/app/page.tsx:774`): "How many trains are out, not how well they move. A
line can run few trains that all move fine, or a full set that crawls," and the
"Predicted status" note ("can differ from the status above, which follows train
movement alone"). No copy fix needed. One structural gap, reported not fixed:
the models page (`viz/app/models/page.tsx`) grades/explains the flow/condition
model only — the word "supply" never appears there; a panel explaining the
supply axis's derivation (assigned_n / trip-updates -> service_ratio ->
degrade/recover thresholds) does not exist on that page, only in per-surface
tooltips and `viz/README.md`.
## 2026-09-01 — publish the movement regime's own entered_at so the drawer can time the badge, not the HMM argmax

origin: agent

The route drawer's "regime age" row read `inference.regime_entered_at`, the
HMM argmax clock (advances only when `roll.filter` flips its top state), while
the badge beside it is movement-primary (`condition`/`condition_source ===
'movement'`, driven by the movement regime in `state/movement_state.json`). The
two clocks are unrelated: in a live snapshot 14/29 routes flipped the HMM
argmax in the same tick, so the row commonly read `—` (fmtMinutes(0)) or a few
minutes next to a movement badge that had actually held for hours. The badge's
own clock — the movement regime's `entered_at` — was read in `buildSnapshot`
(`resolvePublishedCondition`, `movementRecovery`) but never published on
RouteStatus, so viz could not show it and could not honestly derive it (segment
`entered_at`s are per-segment, not the route movement regime).

Fix, one field: `route_status.condition_entered_at: int | null`, the epoch the
PUBLISHED condition began, filled only by the arm that can honestly time the
badge. `resolvePublishedCondition` now returns `entered_at` alongside
`{condition, source}` so the precedence lives in one place: `movement` →
the movement regime's `entered_at`; `schedule` (a planned not_scheduled) → null
(the Worker tracks the announced END via `scheduled_resume_at`, never the start
of the non-run); `unknown`/`hmm` → null. Invariant: non-null **iff**
`condition_source === 'movement'`. The drawer shows "<state> for X" under the
badge only when non-null; `heldFor()` floors sub-minute to "under a minute" so
a just-changed badge doesn't render fmtMinutes' `—`. The model section keeps
`inference.regime_entered_at` under its honest label "Model regime age".

No new zod input surface: `entered_at` is already validated on
MovementRegimeSchema — this only threads it to the output. Schema regenerated
(`scripts/export_schema.py`), one additive nullable int. Worker 456, viz 233,
python parity 2, both typechecks clean. Not deployed: the Worker must ship for
`condition_entered_at` to appear on the live feed; viz reads it back-compat
(field absent → no row).

## 2026-09-01 — observed-headway wait signal: schedule reference false-alarms 44.5% of confirmed-normal service, own-cell 9.3%; stop-level history is only 21 days
origin: agent

Built the offline observed-headway reconstruction at a per-route/direction
reference stop and its own-cell typical-actual AWT/CV baseline, and graded the
wait signal against the movement and supply axes and the alert feed over
2026-08-12..09-02. Numbers a reader would not have guessed:

Substrate. Stop-level timing is reconstructable ONLY from archive/trace/ (the
per-minute vehicle census), which begins 2026-08-12 — 21 days. archive/trip_updates/
reaches back to 2026-06-15 but carries only the compact assigned_n service metric,
no stop_time_updates, so it cannot yield a headway however far back it goes. So the
headway corpus is 21 days, not the 90 the assigned_n archive spans. Volume:
3.69M reconstructed arrivals, 129,531 successive-train headways at 50 reference
stops, 234,664 five-minute tick-aligned AWT readings. Reference-stop rule (the
max-scheduled-trips through-stop) lands on a served-by-all-patterns stop
(coverage 1.000) for all 25 routes x 2 directions, so no train is missed.

Feed stalls are a real concern but did not occur: the trace's max inter-poll gap
over 21 days is 120s (one missed minute); zero gaps exceed the 240s stall
threshold, so 0 of 129,531 headways are feed-gap-excluded. The handling exists
and is unit-tested; the archive just never exercised it.

The load-bearing result (why the baseline must be own-cell, not schedule). On
confirmed-normal ticks (movement AND supply both call normal), a schedule-SWT
reference (AWT > 1.25xSWT) fires on 0.445 [0.424, 0.466] of ticks, night-
bootstrapped over 334 route-nights; the own-cell p90 reference fires on 0.093
[0.087, 0.099]. A timetable baseline flags ~45% of confirmed-NORMAL service as
excess wait, 4.8x the own-cell rate. Mechanism: SWT = sched_headway/2 assumes
CV=0, but delivered service always has CV>0, so AWT > SWT by construction of real
bunching, not degradation. The 2026-08-22 Saturday (1/2 ran weekday-level
service) is the vivid instance: the schedule reference flags 52 of 92 route-cells
that day; an illustrative ungated own-cell baseline places the delivered AWT
inside p10..p90 on 90 of 92. (The disciplined 8-night own-cell baseline ABSTAINS
on every weekend cell — 0 of 1023 fitted — because only 3 weekends of stop-level
history exist; 939 of 1109 weekday cells fit, median 14 nights.)

False alarms vs the movement arm's 0.00101/tick certified bound (gate a,
confirmed-normal, 145,619 ticks / 334 route-nights). A bare above-p90 flag fires
0.0929 [0.0869, 0.0990] — the ~0.10 percentile identity, not an alarm; a
sustained (>=6-tick / 30-min) above-p90 flag fires 0.0527 [0.0478, 0.0579]; a
twice-typical (AWT > 2xp50) flag fires 0.0191 [0.0164, 0.0218]. All land 19x-92x
above the movement bound. Read: the wait signal is a continuous reading, not a
drop-in rare-event alarm at the movement discipline. Caveat that bounds all of
gate (a): the reconstructed headways ride the same ATS/vehicle-position stream
the movement detector reads, so movement-confirmed-normal is a consistency
reference, not an independent one.

Severity tiers (own-cell quantile exceedance, first-pass cutpoints) disagree with
the movement condition almost completely and track the alert feed instead. Of
2,523 tier-2 (severe) headway ticks, the movement condition independently called
not-normal on 1; 2,418 (95.8%) coincide with a Delays alert. 105 severe ticks are
headway-only (no movement, no Delays) and every one of the top disagreement
windows is feed-clean (coverage and group-freshness >= 0.99), so they are genuine
functional disagreements, not common-mode blindness. Tier prevalence over fitted
(weekday) cells: 83.5% normal / 13.3% degraded / 3.2% severe; tier-2 episode
dwell median 20 min, p90 55, max 135.
## 2026-09-01 — the MTA Major Incidents log is a monthly aggregate with zero trace overlap: it cannot adjudicate movement-arm misses

origin: agent

Joined the MTA's official Major Incidents log to our archive as the first
EXTERNAL truth source (`training/major_incidents.py`). The mission premise —
that this log is the only identified path to measuring the movement detector's
misses, because every internal signal rides the same ATS-sourced feeds — is
sound about the SOURCE but defeated by its GRANULARITY and its TIMING. Two
structural facts, both measured, bound what the join can ever do:

1. NOT INCIDENT-LEVEL. The published dataset (NYS Open Data `ereg-mcvp`) is a
   MONTHLY AGGREGATE: 5791 rows, one per (month, division, line, day_type,
   category) carrying a COUNT, 2015-01..2026-07, updated 2026-08-28. No
   timestamps, no per-incident rows. The only sibling (`g937-7k7c`,
   Delay-Causing Incidents, a sub-major superset) has the identical
   monthly-aggregate shape (24640 rows). There is no incident-level public
   companion — metrics.mta.info renders these same aggregates. A month x line
   count cannot be aligned to an archive window by timestamp, so the achievable
   join is PREVALENCE ANCHORING, never per-episode miss adjudication.

2. ZERO TRACE OVERLAP. Movement condition and headway severity are reconstructed
   only from archive/trace. R2 substrate bounds, probed 2026-09-01:
     alerts        2026-06-03 .. 2026-09-02
     trip_updates  2026-06-15 .. 2026-09-02   (supply axis)
     vehicles      2026-06-21 .. 2026-09-02
     trace         2026-08-12 .. 2026-09-02   (movement + headway)
   The dataset's last published month is 2026-07; trace begins 2026-08-12, 15
   days after that month ends. Incident-months overlapping each signal: alerts
   {2026-06, 2026-07}, supply {2026-06, 2026-07}, movement {}, headway {}. The
   movement arm's misses therefore remain STRUCTURALLY UNMEASURABLE against this
   source — not at the wrong granularity, but with no overlapping day at all.
   Movement-arm miss rate: WITHHELD; n episode-alignable incidents = 0.

The only signals that overlap the incident-months are the common-mode alert feed
and supply axis — the very signals this external truth was meant to check. Even
their prevalence anchor is confounded, and instructively so. MTA majors: June 56
(our coverage partial: alerts from 06-03, supply from 06-15), July 86 (full).
Against our canonical severe-only truth (tier>=2, planned excluded):

  - EPISODE count collapses to 0 (June) / 1 (July) network-wide, because without
    the predictions presence-mask every open-ended severe alert runs to the
    window end and is dropped as a >24h standing advisory. That measures the
    missing mask, not the feed — rejected.
  - Mask-free, closure-free ROUTE-DAYS (days a route carried any severe alert)
    instead INFLATE: June 274, July 447 severe route-days of ~930 (30 routes x 31
    days), with routes like W/N/A/M/J at 30/31 days — near-permanent Severe
    Delays advisories, not 30 distinct incidents.

So the one joinable signal misses the MTA scale in BOTH directions depending on
how you bound it, because an MTA "major incident" (a discrete 50+-train
operational event) and our alert "severe" state (an advisory the MTA leaves up
for hours to weeks) are different objects with different temporal semantics and
are NOT subtractable. This is not a failure of the join; it is the positive case
for wanting an operationally-defined external source — and the demonstration that
THIS external source, at monthly aggregate with no timestamps and no trace
overlap, still cannot supply the movement-miss truth 2bc.22 asks for. What it
CAN anchor: a top-severity PREVALENCE (≈52-86 major incidents/month network-wide,
Signals the largest category — 41 of 86 in July), useful to kt3's severity-tier
framing, useless for per-episode adjudication.

Line mapping blind spots, documented as data: JZ fans to both J and Z (the MTA
counts the Nassau St skip-stop pair as one line, so a per-route join
double-counts it); the three named shuttles map S 42nd->GS, S Rock->H,
S Fkln->FS; no SI and no express variants in the source. Ingest/mapping/coverage
logic is hermetic and tested (18 cases); the R2 prevalence read lives behind
main(). 2023->2024 methodology break labeled by `era_for`; the joined window is
entirely post-2024, so the break does not bite here.

## 2026-09-01 — negative result: an online-FDR layer (LORD++/ADDIS) over the movement p-values cannot clear a fleet FDP<=0.05 gate — the Bayesian pre-screen already controls the operating point, and the only false-labelling truth is orthogonal to the flow signal

origin: agent

New offline harness `training/online_fdr.py` (+ `tests/test_online_fdr.py`, 21
cases) replays three alert rules over the fleet's archived movement-detector
p-values in causal, tick-major order: the current fixed binomial gate, LORD++
(Ramdas et al. 2017), and ADDIS (Tian & Ramdas 2019). The p-value is the same
`_binom_lower_tail(advanced_n, matched, p0)` the detector's significance gate
already computes; a route takes its worst candidate direction, as
derive_movement_state does. Fisher (2024) is why LORD++/ADDIS are the
dependence-robust choices: they control FDR under a local-PRDS form of positive
dependence, which the tick/route-correlated stream has, with no modified
recursion. Everything pure over its inputs; R2/alert reads live behind main().

**Harness validated to the digit.** On the journal's certified window
(fit 07-28..08-10, score 2026-08-11..08-31), the operating-point fixed gate
reproduces the published movement false-alarm bound **0.00101/tick
[0.00056, 0.00164]** (2026-08-31 entry: 0.00101 [0.00058, 0.00163]) — the
p-value extraction and episode/run bootstrap are faithful.

**Two surfaces, because the "116 FA/day" premise is about a surface the deployed
detector does not use.** The binomial tail is computed ONLY after the Bayesian
posterior screen (post <= 0.5*p0) inside classify_direction. So there are two
streams: the *operating point* (screened candidates — the p-values that actually
exist) and the *binomial surface* (every judgeable route-tick tested at p<=0.05 —
the uncontrolled-multiplicity scenario the lit review costed at ~116/day).

Certified window (21 scored days, 111,727 judged route-ticks, 88 movement
episodes / 49 escalation-corroborated, 381 confirmed-normal supply runs):

| surface / stream | alerts | FA/day | FA tick-rate | fleet FDP | corrob kept |
|---|---|---|---|---|---|
| op / fixed | 110 | 5.2 | 0.00101 [.00056,.00164] | 0.532 [.36,.67] | 49/49 |
| op / LORD++ | 108 | 5.1 | 0.00100 | 0.537 | 48/49 |
| op / ADDIS | 112 | 5.3 | 0.00102 | 0.541 | 49/49 |
| surface / fixed | 9056 | 399 | 0.07734 [.063,.093] | 0.994 [.991,.996] | 49/49 |
| surface / LORD++ | 952 | 42 | 0.00822 [.006,.011] | 0.956 [.93,.97] | 37/49 |
| surface / ADDIS | 2278 | 94 | 0.01827 | 0.978 | 41/49 |

**Finding 1 — at the operating point the layer is inert.** The posterior screen
already collapses the fleet to ~110 near-decisive candidates over 21 days
(p_median ~ 1e-19); LORD++/ADDIS reject the same 108–112 and move FA (0.00101 ->
0.00100–0.00102) and FDP (0.532 -> 0.537/0.541) within noise. The fleet
multiplicity the premise worried about does not exist where the detector runs —
the *posterior screen*, not the binomial gate, is what bounds the fleet count.
(Note ADDIS can reject 112 > the fixed gate's 110: its wealth-grown threshold
exceeds 0.05 up to its 0.25 cap, so the online streams are NOT a subset of the
fixed gate on the candidate stream.)

**Finding 2 — on the raw binomial surface the layer works as multiplicity
control but still fails the gate.** The fixed p<=0.05 surface is wildly
uncontrolled (9,056 alerts, 399/day, FA 0.077). LORD++ cuts that 9.5x to 952
(42/day, FA 0.0082) and ADDIS 4x to 2,278 — a real, large bound on fleet alarm
VOLUME. But realized FDP stays **0.956 / 0.978**, nowhere near 0.05, and LORD++
buys its volume cut by shedding 12 of 49 corroborated episodes (75.5% retained;
ADDIS 41/49). A higher-signal 36-day window (07-29..09-02, 187k ticks) shows the
same shape with retention holding 154/154 when the procedure spends more wealth,
and the operating-point FA there is 0.00160 — so the certified 0.00101 is
window-specific, not invariant.

**Why the gate cannot be cleared, and it is not the rule's fault.** The realized
FDP's "false" class is a rejection on an assigned_n-confirmed-normal SUPPLY run.
Supply and movement-flow are orthogonal-in-derivation (2026-08-31: 0 mutual
information), so a supply-normal tick is NOT a movement-null — it may be a genuine
flow freeze supply cannot see. LORD++/ADDIS control FDR against the movement
null; the only archived truth that can label a false discovery measures a
different, orthogonal axis. So the 0.956–0.994 FDP is an upper bound that
conflates supply-invisible true freezes with false alarms and cannot be driven to
0.05 by any threshold rule. Corroboration is a soft "true" label here too: the
alert feed reads disrupted on 45% (cert) to 81% (36-day) of judged ticks, so
49/49 corroborates largely because an alert is usually up. Onset back-dating is
0.0 across every stream by construction — the reference episodes are cut from the
fixed-gate movement truth, so any stream reproducing those ticks fires at onset;
the latency metric discriminates nothing in this design.

**Gate verdict (per variant, honest failure).** No variant meets the joint gate
(per-route FA <= 0.00101 AND fleet FDP <= 0.05) on either surface. Operating
point: per-route bound held (~0.00101) but FDP ~0.53 >> 0.05 and the layer is
inert. Binomial surface: the layer bounds volume 4–9.5x but FA stays 0.008–0.018
(> 0.00101) and FDP 0.96–0.98 (>> 0.05), with corroborated-episode loss.
Extends 2026-08-31: the wall is not the thresholding rule (LORD++/ADDIS provably
bound the fleet alarm COUNT the premise cited) but the absence of a flow-axis
truth to control FDR against. Offline only; nothing published or wired live.

---

## 2026-09-01 — night-gating the supply baseline: the weekend-late false-alarm amplifier is a fit-WINDOW-length effect, not a within-window cell partition

origin: agent

Offline eval of the supply baseline's independent-night gate (`SERVICE_MIN_NIGHTS=8`
on `compute_baseline`/`compute_service_quantiles`), against the frozen
v1787792319 artifact, via `training/service_night_gate_eval.py`. Fit the
baseline+quantiles on a trailing window, score above-p90 false alarms on a
DISJOINT confirmed-normal window (out-of-sample — in-sample above-p90 is ~0.10 by
construction because every night feeds its own p90; the amplifier only bites
nights the baseline never saw).

**The gate lowers confirmed-normal weekend-late FA ~7x, same-cell.** Paired
counterfactual, fit_end 2026-08-11, score 08-12..09-01: on the SAME 169
weekend-late (route, schedule_bin) cells, the SAME 29 confirmed-normal
(route,night) clusters / 348 ticks, the only thing varied is fit-window breadth.
A 13-day fit (what a tick-gate would ship) reads above-p90 FA 0.0402
[0.0057,0.0920], 14/348; a 35-day fit (the window the night gate forces) reads
0.0057 [0.0000,0.0172], 2/348. Point ratio 7.1x, but the 95% night-bootstrap CIs
TOUCH at ~0.017 — directional, not decisively separated at n=29 clusters. Likely
conservative: the whole score window sits in the low-supply mode (a regime shift
around 2026-07-20 took weekend late service from ~16 to ~9 trains and it stayed
there), so score nights rarely exceed even a thin p90; the effect is larger when
service returns to the high mode.

**The benefit is temporal (window adequacy), NOT the within-window partition
gk0z hypothesized.** On a single 35-day window, 169 of 184 weekend-late cells
already clear the 8-night gate at 0.57% FA; the 15 that don't are merely
sparsely-covered, and show 0% FA on their few nights — the thin cells inside one
window are not systematically worse. The amplifier lives across window LENGTH:
on 13 days every weekend-late cell has ~4 nights (0 clear the gate, all 172 are
silenced); on 35 days ~10 nights. So the gate's real function is to REFUSE a
thin-window publish and defer until ~a month accrues, at which point the same
cells publish at 6-7x lower FA — not to discriminate good cells from bad within
a window.

**Abstention cost is small and redundant.** Steady-state (35-day window) the gate
silences 15/184 weekend-late cells; over the score window those carry 3
confirmed-normal and 34 disrupted nights, silencing 4 flagged disrupted-nights
and 0 spurious — and all 4 are nights the alert feed already marks disrupted, so
the supply axis silencing loses redundant signal, not unique signal.

**The named bimodal targets (1/2/J we-cells) reproduce structurally but their
forward FA is UNMEASURABLE here.** Over 35 days their per-night medians are
plainly bimodal — route 1 we22/we23 span 2.1x (9..19, 8..16), route 2 we23 2.12x
(8..17, median 9, p90 17), route J we22/we23 6.6-7.3x — matching the journal's
1.7-2.1x, and the 35-day p90 spans both modes (route 2 we23 p90=17 covers the
high nights) where a thin 13-day low-mode fit (p90~9) would flag every one. BUT
routes 1/2/J carried an acute alert (delays/suspension/unplanned service-change)
on EVERY weekend late night in the score window: 0 confirmed-normal weekend-late
nights, so their individual out-of-sample FA cannot be measured — the supply
axis on those exact cells is also where the alert feed is least silent.

**Confirmed-normal is alert-consistency, and must keep planned advisories.**
First cut excluded `has_planned`; routine "Planned -" trackwork advisories
blanket nearly every weekend night, which zeroed every named-cell score night.
Excluding only ACUTE alerts (planned kept; `has_service_change` already drops
planned via its prefix guard) is correct: a genuinely reduced night reads LOW and
never fires the above-p90 surplus flag, so keeping planned nights eligible cannot
inflate the FA. This is a consistency reference (shared ATS-sourced service
state), not an independent one.

Verdict: night-gating measurably (directionally, CIs touching) lowers
confirmed-normal weekend-late FA at negligible, redundant abstention cost — but
the mechanism is window-adequacy deferral, not the cell partition the bead
proposed, and the named 1/2/J cells can't be individually graded in a window
where the alert feed never calls them normal.

## 2026-09-01 — night-gate FA benefit does NOT survive a window shift: the 7x was one transient; revises the entry above

origin: agent

Follow-up to the entry directly above (same-day night-gating eval). Reran the
SAME paired counterfactual (same night-bootstrap, same acute-only
confirmed-normal, same 35d-vs-13d fit design, same tool) over two more score
windows to try to separate the touching CIs. The effect did not strengthen — it
COLLAPSED, and the requested high-mode window turned out infeasible in-archive.
Three runs side by side, above-p90 confirmed-normal FA on the SAME shared cells,
only fit-window breadth varied:

- **August, fit_end 08-11, score 08-12..09-01** (original): 169 cells, 29 night
  clusters. short-13d 0.0402 [0.0057,0.0920] 14/348 vs long-35d 0.0057
  [0,0.0172] 2/348. 7.1x, CIs touch.
- **August, fit_end 08-04, score 08-05..09-01** (widened to 4 weekends): 167
  cells, 37 clusters. short-13d 0.0090 [0,0.0203] 4/444 vs long-35d 0.0090
  [0,0.0203] 4/444. **1.0x — identical, the amplifier is gone.**
- **mid-July high-supply score, fit_end 07-07, score 07-08..07-19** (requested):
  **0 shared cells.** trip_updates coverage starts 2026-06-15 and the high-supply
  mode ends ~07-19, so a 35d trailing fit reaches back only ~23d (~6 weekend
  nights); no weekend-late cell clears the 8-night gate, night_pass is empty, and
  the paired comparison is unmeasurable. A stable-high-mode test with the fixed
  35/8 design cannot be run in this archive.

**What this means.** The 7x in the first entry rested on ~12 excess alarmed ticks
from ONE transient — the late-August (~08-22..27) return-to-high-service nights,
scored against a 13d fit ending 08-11 that sat entirely in the low-supply mode.
Move fit_end one week earlier and those exceedances leave the confirmed-normal
set, and the thin- and wide-fit p90s classify every score tick identically
(4/444 both). So the false-alarm reduction is NOT a robust property of the gate;
it is a fragile, window-placement-specific artifact carried by a single regime
transient, and it does not reproduce.

**What DOES survive** is the mechanism, not the payoff: the gate's only lever is
window-adequacy deferral (13d = ~4 weekend nights, all cells silenced; 35d = ~10,
they publish). It never triages good cells from bad WITHIN a window — the 15
sparse abstainers on a 35d window showed 0% FA, not elevated. So the earlier
"night-gating measurably lowers FA" reads too strong; the honest claim is that
the gate defers a thin-window publish, and whether that deferral prevents real
false alarms depends entirely on whether a supply regime shift happens to land in
the deferred window — which in the one measurable case it did (7x) and in the
shifted case it did not (1x). Verdict stays open and moves toward negative on the
FA payoff; the deferral rationale stands on its own.

Boundary-bug note (pre-commit review): the short paired-fit was sliced at UTC
midnight while every night concept here is ET, so it folded 20:00-23:59 ET of the
eve of short_start — the weekend-late band itself — into the short window's p90.
Fixed to cut at ET midnight (_et_midnight) with a pinning test; rerunning both
windows (fit_end 08-11 and 08-04, same seeds/design) reproduced the figures above
BIT-FOR-BIT (short 0.0402 14/348 / long 0.0057 2/348; and 0.0090 4/444 both) — the
leaked eve band was too few ticks to move any nearest-rank p90 across an above-p90
decision, so no reported number changes.

## 2026-09-01 — CORRECTION: the night-gate FA benefit was a measurement artifact; two review-caught bugs reverse it to "no measurable effect". Revises both 2026-09-01 night-gate entries above

origin: agent

The pre-commit adversarial review caught two real bugs in
`training/service_night_gate_eval.py`; fixing them erases the false-alarm
benefit reported in the two entries above. Both prior entries' FA figures are
SUPERSEDED by this one.

**Bug 1 — the quietest, most-normal nights were dropped from the denominator.**
`build_tick_observations` emits a row only for a (route, tick) some alert's
informed_entity named, so an entirely alert-free route-night (full weekend
service, no trackwork advisory) has NO observation and vanished from
`night_labels` — making "confirmed-normal" contingent on planned-advisory
coverage and systematically excluding the high-supply full-service nights, which
are exactly the ones that read above a trackwork-suppressed p90. Quantified on
the two score windows: 11 (08-12..09-01) and 9 (08-05..09-01) weekend-late
service route-nights were alert-free-but-witnessed (other routes observed at
those ticks -> genuinely alert-free, not an archive gap; 0 pure-gap nights).
Fixed: `night_labels(obs, service_ticks=...)` labels a service night normal when
alert-free AND the archive was live over it (coverage witness), excluding true
gaps.

**Bug 2 — hourly cells of one night counted as independent bootstrap draws.**
`false_alarm_rate` made each (route, schedule_bin, night) its own resampling unit,
so a route's we20..we03 hours on one night counted as up to 8 independent draws —
the same pseudo-replication the tick-autocorrelation memory warns against.
Fixed to cluster by (route, NIGHT), aggregating all of a night's hourly cells
into one unit, matching headway_eval. This does not move the point estimate, only
the cluster count and CI width.

**Corrected numbers (paired same-cell counterfactual, above-p90 confirmed-normal
FA, short-13d vs long-35d fit):**

- score 08-12..09-01 (169 cells): was short 0.0402 / long 0.0057 (7.1x, n=29
  "nights"). NOW short 0.0870 [0.0225,0.1785] 94/1081 vs long 0.0759
  [0.0063,0.1719] 82/1081, **ratio 1.15x, n=13 route-nights, CIs heavily
  overlap.** The 7x gap is gone; both fits over-fire ~8% on the quiet
  full-service nights, and even the 35-day p90 does not cover them.
- score 08-05..09-01 (167 cells): was short 0.0090 / long 0.0090 (1.0x, n=37).
  NOW short 0.0044 [0,0.0098] 4/913 vs long 0.0044 [0,0.0098] 4/913, **1.0x,
  n=12, identical.**

**Revised verdict (supersedes both entries above): night-gating does NOT
measurably lower confirmed-normal weekend-late false alarms.** Short and long fit
are statistically indistinguishable (1.0-1.15x, CIs overlapping), and the FA
LEVEL is set by window placement (0.44% at one week's shift, ~8% at another), not
by the gate. The apparent 7x rested entirely on the two bugs: excluding the quiet
high-supply nights and treating correlated hours as independent draws. What still
holds is only the mechanism, not a payoff — the gate defers a thin-window publish
(window adequacy); it does not demonstrably prevent false alarms. The bimodality
of the named 1/2/J cells is real (per-night spans 2.1-7.3x, unchanged), but a
wider fit window does not cover the high mode any better than a thin one here, so
the gate is not the lever that fixes it.

Witness-hardening note (final review iteration): the alert-free-normal coverage
witness was changed from reconstructed alert observation ticks (which a
long-running alert can extend across a collection gap) to trip-updates
snapshot-presence (the snapped observed_at of the fetched bodies, one per cron
tick regardless of alert quietness), with a pinning test that a synthetic alert
spanning a gap does NOT witness it. Reran both windows: bit-identical to the
figures above (0 archive-gap nights, ~120s max inter-poll cadence), so no number
changes — the hardening removes a latent fabrication path, not a measured error.

Two further hardenings (final review iteration) — figures updated, verdict
unchanged. (a) NIGHT KEY: the resampling/label unit is now the SERVICE night
(ET hours 0-3 roll to the prior date), so Sat 23:00 + Sun 01:00 are one cluster
and Mon 00-03 stays in Sunday's weekend-late night (Sat 00-03, being Friday's
night, drops out); membership follows the service night, not schedule_bin.
(b) ALERT WITNESS: the archive carries no per-tick alert-fetch liveness (the
trip-updates fresh_feeds is vehicle feeds only), and alerts is a separate fetch
from the cron, so an alert-free night is called normal only when BOTH a
trip-updates snapshot exists over it AND some alert-version was archived
system-wide that service night; a witnessed-cron night with no alert archived is
an alerts outage and is excluded. Reran the paired same-cell counterfactual:

- score 08-12..09-01: shared cells 169 -> 88 (merged service-nights + Mon-edge +
  outage exclusions raise the 8-night bar), short-13d 0.1234 [0.0192,0.2933]
  77/624 vs long-35d 0.1042 [0.0032,0.2804] 65/624, ratio 1.18x, n=13
  route-nights, CIs heavily overlap.
- score 08-05..09-01: short 0.0052 [0,0.0139] 3/576 vs long 0.0052 [0,0.0139]
  3/576, identical, n=12.

The point estimates shifted (was 0.087/0.076 and 0.0044/0.0044) but the verdict
is IDENTICAL: night-gating does not measurably lower confirmed-normal
weekend-late FA — short and long fit stay statistically indistinguishable
(1.0-1.18x, overlapping CIs) and the FA level is set by window placement
(0.5%-12%), not the gate. All prior-entry FA figures are superseded by these.

Label-semantics addendum (final review round): "confirmed-normal" here is
night-witnessed ALERT-QUIET, not tick-certified normal. The archive carries no
per-tick alert-fetch liveness record (fresh_feeds covers vehicle feeds only;
alertsFreshness exists only in the live snapshot), so the witness proves the
alerts fetch succeeded at least once that service night — a mid-night alerts
outage could still mislabel later quiet ticks. The PAIRED short-vs-long verdict
scores identical ticks under identical labels in both arms, so any label error
is shared by construction (its direction is indeterminate without per-tick
liveness); the
ABSOLUTE FA levels (0.5%-12%) should be read as alert-quiet rates under a
night-granular witness, not certified-normal rates. A tick-certified label
needs a per-tick alert-fetch success record the Worker does not archive today.

## 2026-09-02 — the nearest-rank off-by-one that hopped the published p90 across the bimodal gap: FS/we20's upper service bound was 18 trains where its normal mode tops out at 6

origin: agent

Consolidation pass on the eval seams, prompted by four review-caught bugs in one
day across three independently-written tools. Extracted training/eval_common.py
and migrated the duplicated seams onto it. One of the "small" fixes turned out
not to be small.

THE BUG. Nearest-rank was implemented as `int(q*n)` in two places
(headway.py `_nearest_rank`, load_r2.py `compute_service_quantiles`). The correct
0-indexed rank is `ceil(q*n)-1`, which equals `floor(q*n)` for every NON-integer
product and is one rank lower exactly when `q*n` lands on an integer. A test
asserted the buggy values (p10=2.0, p90=10.0 for the ten values 1..10, where
nearest-rank gives 1.0 and 9.0), which is why it survived review twice.

WHY IT WAS NOT COSMETIC. n=120 is the common weekend-late cell size, and
0.9*120=108 is an integer, so every such cell took rank 108 instead of 107. On
the bimodal cells that motivated the whole weekend-baseline investigation, ranks
107 and 108 sit on OPPOSITE SIDES of the bimodal gap. Measured on the live
35-day window (2026-07-29..09-01, 1122 published cells):

  FS/we20  sorted[100:] = [4,6,6,6,6,6,6,6,18,18,18,18,18,18,18,19,19,19,19,19]
           rank 108 -> p90 = 18      rank 107 -> p90 = 6
  J/we23   rank 108 -> 16            rank 107 -> 9
  1/we21   rank 108 -> 22            rank 107 -> 17

So the published upper service bound for the Franklin Ave shuttle's weekend
evening was 18 trains, three times the top of its normal mode. p90 moved on
116/1122 cells (mean -1.586), p10 on 44/1122 (mean -1.045). The bound is what
the Worker divides by the cell median to get service_high_ratio, so the meter
could effectively never read high on these cells, and every above-p90 false-alarm
rate measured against them was understated. Rebuilt and published the sidecar
(state/service_baseline.json v1788375377, 27 routes, 1122 cells);
params_trained_at 1788229972 unchanged, so the frozen model is untouched.

FLOATING POINT, CHECKED NOT ASSUMED. `ceil(q*n)-1` would re-introduce the same
off-by-one if `q*n` came out a hair above an integer in binary. Scanned q in
{0.10, 0.50, 0.90} for every n in 1..200000 against exact rational arithmetic:
zero mismatches. The implementation still snaps the product (`round(q*n, 9)`) so
that stays true of a q added later.

GATE FIGURES AFTER THE FIX (the weekend night-gate counterfactual, same
--paired design, score 2026-08-12..09-01, 88 shared cells, n=13 route-nights):
long-35d fit 0.1138 [0.0064, 0.2901] 71/624 vs short-13d 0.1234 [0.0192, 0.2933]
77/624 — ratio 1.08x, down from 1.18x, CIs still heavily overlapping. The
short-fit figure is byte-identical to the pre-fix run because 13-day cells do not
hit an integer product; the long fit rose (65 -> 71 alarmed) because the corrected
p90 is a tighter threshold. The verdict does not move: night-gating still shows
NO measurable false-alarm benefit, only window-adequacy deferral. Decision taken:
keep the gate on deferral grounds, leave the tod_bin emission ungated.

NEGATIVE RESULT — the liveness witness cost nothing, so it buys certainty rather
than trading it. Gated the wait signal's confirmed-normal label on the same
two-witness rule the supply night-gate eval uses (a collected vehicle body with
this route's own line-group feed fresh, AND an alert version archived somewhere
on the system that service night). Expected a 10-15% sample loss. Measured over
2026-08-12..09-02: both-axes-only 112,384 route-ticks, witnessed 112,384 —
ZERO dropped. Breakdown: 0 ticks with no collected body, 0 with the route's own
feed group stale, 0 nights with no alert version. The archive was simply healthy
across this window, the same way the trace's stall handling has never once fired
(max inter-poll gap 120s over 21 days). So the label is now PROVEN rather than
assumed at no cost — but the rule is untested against a real outage, exactly like
the stall path, and its value will only show up in a window that has one.

Refreshed wait figures on the corrected quantiles (2026-08-12..09-02, 944
baseline cells): gate-a above_p90 0.0954 [0.0894, 0.1015] (was 0.0929), sustained
0.0551 (was 0.0527), twice_typical 0.0191 (unchanged — it is p50-referenced and
p50's rank did not move); gate-b schedule reference 0.4430 [0.4239, 0.4626] (was
0.445) vs own-cell 0.0956 (was 0.093). The above-p90 rates rose because the
corrected p90 is tighter, which is the expected direction. Exposure differs from
the earlier entry (402 route-nights / 153,374 tick-readings against 334 /
145,619); the witness rule drops nothing and the service-night key can only
merge nights, so the growth is not from either change. Most likely the supply
axis abstains on fewer cells as the trip-updates archive lengthens past the
8-night gate — NOT verified, and flagged here rather than asserted.

SEAMS NOW SHARED, AND WHAT DELIBERATELY IS NOT. training/eval_common.py owns
et_date, service_night, et_midnight, snap_tick, nearest_rank, and the two
witness builders; headway.py, load_r2.py, headway_eval.py and
service_night_gate_eval.py all import them, retiring three copies of the ET-date
helper and one each of the service-night, ET-midnight and witness builders. Each
tool's CLUSTERING UNIT was left alone on purpose: the wait and supply evals
cluster by night, the movement and online-FDR tools score episodes, and episodes
are genuinely their unit. Forcing those onto nights would have moved the
published detection-latency figure as a side effect of a refactor, which is the
category of change that caused this pass in the first place.

## 2026-09-02 — CORRECTION to the entry above: the witness result is an audit trail, not proof of normality; and the raw-retention decision was left half-made

origin: self

Review caught two overclaims in the entry above. Both are mine and both are the
same kind of error — treating an absence of contrary evidence as positive
evidence.

1. "The label is now PROVEN rather than assumed" is wrong, and the sentence
following it (that the rule is untested) contradicts it in the same breath. What
the zero-exclusion measurement establishes is narrow: over 2026-08-12..09-02 the
two-witness rule removed 0 of 112,384 route-ticks, so adopting it cost no sample.
That is all. It does NOT establish that those ticks were normal:

  - The alert-side witness is NIGHT-granular. A mid-night alerts outage following
    one successful fetch is still indistinguishable from genuine quiet in this
    window. The per-tick liveness record the Worker now archives closes that gap,
    but only for windows after its deploy — not retroactively for this one.
  - A filter that has excluded nothing has not been shown to discriminate. It is
    unexercised, exactly like the trace stall path it resembles, and its value is
    only demonstrable in a window containing a real outage.

Correct reading: each label now CARRIES positive evidence that the archive was
recording, where before it carried none. That is an improvement in auditability,
not a certification of normality. "Normal" here remains the absence of a detected
problem under a night-granular witness.

2. Raw retention was encoded half-way. archive/vehicles/ and archive/trip_updates/
moved from unpoliced to an explicit 3650-day window, which is defensible on its
own (silence replaced by policy, 3 MB/day combined). But framing that as "keep the
historicals" while archive/trace/ is still deleted at 30 days is misleading,
because trace is the ONLY prefix whose retention materially costs anything and by
far the most valuable one to keep: it is the per-minute per-train census from
which stop-level timing, headways and traversals are all reconstructed. Measured
2026-09-02, at R2 Standard list ($0.015/GB-month):

  prefix                window   steady GB   $/mo
  archive/trace/            30d        2.7   0.04   <- 89.9 MB/day
  archive/vehicles/       3650d        9.1   0.14   <- 2.5 MB/day
  archive/trip_updates/   3650d        1.8   0.03   <- 0.5 MB/day
  archive/traversals/     3650d        6.6   0.10   <- 1.8 MB/day, derived

  trace at 365d = 32.8 GB (~$0.49/mo); at 3650d = 328 GB (~$4.92/mo), which is
  16x every other prefix combined and still under five dollars a month.

So the 30-day trace cap is not a cost decision — at these volumes nothing here is
a cost decision. It is a decision about whether stop-level history older than a
month is worth keeping, and today the answer encoded in the code is no: no model
can ever be evaluated on more than 30 days of stop-level data, and the headway
corpus is permanently capped at ~a month however long the project runs. That
tradeoff was made when trace was the largest prefix by two orders of magnitude
and is worth revisiting on its own terms rather than inheriting it by silence.
Deliberately NOT changed here: it is a standing decision with a documented
rationale, and changing it belongs in its own change with its own reasoning.

## 2026-09-02 — arithmetic correction: the 10-year trace ratio is 18.7x, not 16x

origin: self

Review caught a bad denominator in the retention table above. The claim "328 GB
... which is 16x every other prefix combined" divided by 20.2 GB, which is every
prefix INCLUDING trace's own 2.7 GB at its current 30-day window — so the
comparison quietly counted part of its own subject. Every other prefix combined
is 17.52 GB (vehicles 9.12 + trip_updates 1.82 + traversals 6.57, all at 3650d),
and 328.1 / 17.52 = 18.73. Corrected ratio: ~18.7x, on ~$0.26/mo of other-prefix
storage.

Every other figure in that table is unchanged and re-verified: trace 2.70 GB at
30d, vehicles 9.12, trip_updates 1.82, traversals 6.57; trace 32.8 GB (~$0.49/mo)
at 365d and 328.1 GB (~$4.92/mo) at 3650d. The conclusion the table was drawn to
support does not move — a decade of raw trace is still under five dollars a
month, and the 30-day cap is still an analytical decision rather than a cost one.

Noting the shape of the mistake because it is the same one the entry two above
corrects in a different guise: both came from letting a convenient aggregate
stand in for the quantity actually named. There the aggregate was "no witness
excluded anything" standing in for "the labels are proven"; here it was "all
prefixes" standing in for "every OTHER prefix". A number is only defensible
against the sentence it is written into.

## 2026-09-03 — the obvious fix for the self-contradictory inference block is the wrong one

origin: agent

J published `inference.is_disrupted=true` with `p_disrupted=0.999992` beside
`recovery_minutes=0`, `[0,0]`, `recovery_indeterminate=false`. The tempting fix
is to stop deriving `is_disrupted` from the alert shadow and derive it from the
published, movement-primary condition — the badge said `normal`, so the flag
should have said false and the whole contradiction disappears. That fix is
wrong, and what makes it wrong is not in worker/ at all.

Three viz surfaces gate the recovery string on this exact field, not on the
published condition: viz/app/page.tsx:485, viz/app/lines/page.tsx:94,
viz/app/lines/[route]/page.tsx:187. So `is_disrupted` is not really "is this
line disrupted" — operationally it is "is the recovery number in this block
worth showing". For a route with no movement read the published condition is
`unknown`, `publishedNotNormal` is false, and the alert-HMM arm supplies a
genuine estimate: live H published 80 [55,100] that way. Deriving the flag from
the published condition turns H's flag false and silences an honest estimate,
trading a self-contradictory row on J for a silently emptied row on H. The
consumer that reads the flag decides what the flag is allowed to mean.

So the arms stay as they are and the composition is guarded instead: withhold
via the existing ceiling convention (1440 + `recovery_indeterminate=true`)
whenever `is_disrupted` is true and the arm that produced the recovery block was
timing a `normal` regime. The single-arm rows are untouched by construction —
the schedule arm only runs on `publishedNotNormal` and the alert-HMM arm only on
`condition !== 'normal'`, so only the movement arm can report "nothing to
recover from" under a flag that says otherwise. Pre-fix, the four
behaviour-preservation cases in worker/test/inference_arm_agreement.test.ts all
pass and only the J case fails, which is the evidence that the guard is
load-bearing on exactly one row shape.

`p_normal_in_30min` is deliberately left published on the withheld row. It
forecasts the PUBLISHED condition, which is `normal`, so 0.998 "still normal in
30 min" is correct and is the best-measured number in the block (AUC 0.856 on
movement-sourced rows vs 0.261 on alert-sourced). Nulling it would discard real
signal to tidy up a field-naming problem.

The residual smell is untouched and belongs to the contract, not the model: a
rider-facing boolean whose real meaning is "which arm answered" cannot be made
coherent while `recovery_minutes` is a non-nullable integer, because "no
estimate" has to be spelled as a real number that means something else.

## 2026-09-03 — correction: the arm enumeration in the entry above was wrong, and wrong in the way that entry warned about

origin: agent

The entry above closes with a claim that does not survive: "the schedule arm
only runs on publishedNotNormal and the alert-HMM arm only on
condition !== 'normal', so only the movement arm can report 'nothing to recover
from' under a flag that says otherwise." The first guard I shipped keyed off
exactly that enumeration —

    recovery_source !== 'movement' || movementRegime?.state !== 'normal'

— and it has a reachable hole on the schedule arm.

derive.ts's two alert predicates are not complements, and the code says so in
plain sight (derive.ts:256): `alert_count` counts "real-time alerts and any
other id", i.e. everything not `lmm:planned_work:*`, while `has_realtime_alert`
matches only `lmm:alert:*`. An alert in a third namespace is therefore COUNTED —
so `effectiveCondition` does not force `normal` and `is_disrupted` goes true —
while leaving `has_realtime_alert` false, which is precisely the
`!schedule.hasRealtimeAlert` precondition `scheduleRecovery` needs. Add an
active planned window and the schedule arm is selected on a route the flag calls
disrupted. Then `remaining = max(0, round((resume - now) / 60))` rounds to 0 for
any tick within 30 seconds of the announced resume, and `scheduledResumeAt`'s
containment test is `now <= end` inclusive, so `overdue = now >= resume` fires at
the boundary too. Published object: `is_disrupted: true`, `recovery_minutes: 0`,
`[0, 0]`, `recovery_indeterminate: false`, `overdue: true` — the J defect exactly,
on the arm the enumeration had assumed safe. Reproduced in
worker/test/inference_arm_agreement.test.ts; pre-fix the two defect rows fail and
the five behaviour-preservation rows pass.

The irony is that the pre-existing gate's own comment (snapshot.ts:1094-1096)
already argues against what I did: it is assembled at the single return point
rather than per-arm because "there are five ways these values get set and a
per-arm gate would silently miss the next one added." I wrote a per-arm gate
anyway, one screen below that sentence, and it missed one of the five.

The guard now keys off the answer instead of the arm: withhold when
`is_disrupted` is true and the recovery block is a determinate, ZERO-WIDTH zero
(`!indeterminate && minutes === 0 && low === 0 && high === 0`). Zero width is
load-bearing, not decoration — a fitted curve with a sub-30-second median
publishes `0` with a non-zero upper bound, which says "probably imminent, could
be a while" and is a legitimate forecast. Only the zero-width determinate zero
asserts the disruption is provably already over, which is the thing that cannot
be true beside a live disruption claim. That predicate is also arm-count
independent, so the next arm added inherits it.

Generalisable, and the reason this is worth an entry: when a guard's safety
argument is "the other N branches can't reach this state", the argument is only
as good as an exhaustive reading of all N, and two predicates that sound like
complements ("is it real-time" / "does it count as disruptive") frequently are
not. Prefer a guard keyed on the invariant the object must satisfy over one
keyed on the paths believed to violate it.

## 2026-09-03 — the guard was masking an overloaded encoding, and the mask was worse than the bug

origin: agent

Third and last correction to the two entries above. Both call the schedule
sub-30-second rounding a third "producer" of the contradictory object, and the
entry above calls it "durable coverage of the schedule-arm zero". Neither
survives. The commit hook rejected the branch on it, correctly.

The case I pinned was a planned resume 20 seconds in the FUTURE
(`overdue = false`). That is a truthful imminent recovery. `Math.round(20/60)`
published it as a determinate `[0,0]`, and my composition guard then converted
that to `1440 / recovery_indeterminate=true` — so a route about to come back in
under a minute published "we cannot say when this recovers". A false unknown is
user-visible and strictly worse than the zero it replaced, and I had written a
test asserting that behaviour as correct.

The actual defect was never composition, it was ENCODING. `round` overloaded 0
to mean two different things on the same arm:

  - "the announced time has already passed" (the `overdue` clamp), and
  - "the announced time is less than 30 seconds away".

The first contradicts a live disruption claim. The second is a perfectly good
forecast. Guarding the collision papers over the overload and, in the branch I
nearly landed, resolves it in favour of the wrong one. Fixing the encoding costs
one operator: the countdown ceils, so a strictly-positive wait is structurally
>= 1 (no clamp needed), and 0 belongs to `overdue` alone. Ceil also never
under-promises the wait, which is the right direction for an ETA, and it is
identical to round on whole-minute resumes — every pre-existing schedule
assertion (1800s->30, 5400s->90, 30min->30) is untouched, which is why the
change is invisible to the other 33 test files.

Only after that does the invariant predicate become exactly right rather than
approximately right: a determinate zero-width zero now means one thing, so
`is_disrupted && determinate && [0,0]` is precisely the contradiction and
nothing else. The guard did not change. Its PRECONDITION got established.

Generalisable, and sharper than the lesson in the entry above: a guard that
fires on a value is only sound if that value is unambiguous, and mine was not.
Before adding a sentinel check, ask what else can produce the sentinel — if the
answer includes something truthful, the encoding is the bug and the guard will
convert a correct answer into a withheld one. "Withhold when unsure" is not
free; it has a cost paid by the consumer, and it is the wrong trade against a
number that was simply rounded badly.
## 2026-09-03 — the movement_transitions key had no scope, so segment writes ate route writes for 8 days

origin: agent

`writeMovementTransitions` keyed on `observedAt` alone —
`v1/movement_transitions/<date>/<ts>.jsonl` — and `index.ts` calls it twice per
tick: route clock at index.ts:897, segment clock at index.ts:937, same
`observedAt`. R2 `put` replaces. The second write therefore deleted the first
on every tick that had a segment change.

Numbers, from an archive scan over 2026-08-12..09-03: route-scope
records per day ran 13, 25, 5, 0, 0, 35 through 08-26 while segment ran 90-238,
then segment jumped ~60x on 08-27 (8733, 8170, 10068, 10928, 8525, 6571) and
route went to exactly 0 on that same date and stayed there for eight days. The
mechanism is confirmed structurally, not just by correlation: no archived object
has ever contained both scopes. 2026-08-20 — 106 files, 101 segment-only, 5
route-only. 2026-08-26 — 84 files, 79 segment-only, 5 route-only. 2026-08-29 —
287 files, all segment-only. A same-key overwrite produces exactly that
partition; an append or a scope-partitioned key never could.

Fixed by putting scope in the key (`<ts>-<scope>.jsonl`) rather than merging both
scopes into one put. The deciding factor is the reader:
`training.eval.load_movement_transitions` lists by whole-day prefix and filters
on the record's `scope` FIELD, never parsing the key — so scope-in-key needs
zero reader change and the entire pre-fix archive keeps loading unchanged beside
the new shape. Merging into one put would instead have coupled the two writers,
which sit in different try blocks with independent fail-soft behaviour: a
segment-step failure would have taken the route records with it.

`writeMovementTransitions` now groups by scope internally and puts one object
per scope present, instead of reading the scope off `records[0]`. That removes
the caller invariant entirely — no input can mislabel a key, and no two puts
from one tick can collide — rather than documenting an invariant that a third
callsite would silently break.

Audited the rest of the write path while here: `writeMovementTransitions` is the
only writer under this prefix (two callsites, both in index.ts), and both are
awaited sequentially inside one `scheduled` handler, so there was never a
concurrency race — the loss was purely the shared key.

Not recoverable: the eight days of route-scope records are gone, overwritten in
place. `training.movement_backfill` can reconstruct that window offline from
`archive/vehicles`, which is what the movement dwell fit already reads, so the
hole blocks the *committed-stream* switch rather than the fit
itself. Its module docstring claimed segment-scope regimes "have never been
wired up online"; they are, and they were the thing doing the overwriting.

## 2026-09-03 — correction: the route-scope loss is ongoing, not eight closed days

origin: agent

The entry above says "the eight days of route-scope records are gone" and "the
key is scope-partitioned now". Both read as a completed cutover. They are not.
The fix lives in worker/src/grading.ts in an uncommitted worktree and has NOT
been deployed — deploying is the user's call — so the deployed Worker is still
keying on `<ts>.jsonl` and is still overwriting route scope on every tick with a
segment change, right now. The correct statement is: route scope has been empty
since 2026-08-27 and stays empty until that file ships. Eight days is the count
at the time of writing, not a closed interval.

The same overclaim had leaked into three docstrings and is corrected there:
`training.eval.load_movement_transitions` had dated the key-shape boundary to
2026-09-03, which would have told a future reader that anything after that date
is scoped; it now describes the two shapes and explicitly says not to infer a
cutover date from it. `training/movement_backfill.py` had "the key is
scope-partitioned now" and a closed 08-27..09-03 window; both now say the loss
continues until deployment. grading.ts's own module comment said "until this
suffix was deployed".

Worth naming the shape of the error, because it is a specific hazard of writing
a fix and its documentation in the same sitting: the tense of the code under
your cursor is not the tense of production. A docstring that describes an
archive is a claim about deployed behaviour and dates it, and every one of these
three would have been read later as evidence about what the bucket contains. A
key-shape boundary in particular is exactly the kind of assertion a future
reader would trust instead of listing the bucket.
## 2026-09-03 — the severity signal the HMM never had, and the tier-1 channel that would have undone it

origin: agent

The latent `disrupted` regime was measured at 94.3% ordinary tier-1 Delays
(1024/1086 not-normal route-ticks) while the grading truth counts severity >= 2
only. Closing that on the training path turned up three things worth keeping.

**The severity was being computed and then dropped, not never computed.** The
standing note pointed at `training/load.py` saying the HMM path leaves
`disruptive_types` empty — but that is the LOCAL collector loader, which the
trainer never calls. `load_r2.build_tick_observations`, the one it does call, was
already populating it. The break is one step later: `load_series_by_route`
discards the `TickObservation` tag and hands EM a bare `Observation`, and
`Observation` had no severity field at all. Severity reached the boundary and
died there. Same conclusion, different line — and the difference matters because
the fix belongs on `Observation`, not on the loader.

**The two loaders were independent copies.** `load.py` and `load_r2.py` each
carried their own observation build, including their own `_match`, and there were
three copies of the quiet-tick fill. That duplication is exactly why a channel
could reach one path and not the other. Both now call one shared builder with a
parity test asserting identical output at both floors.

**The obvious cheap fix would have re-created the confound.** Splitting
`has_delays` into ordinary and severe per-state Bernoullis looks like the minimal
change, and it is wrong. EM is unsupervised: it maximises likelihood, and the
tier-1 cluster carries roughly 17x the mass of the severe one, so nothing stops
it re-discovering ordinary Delays, taking the `disrupted` index, and taking the
disrupted dwell back with it. Capacity to separate two populations is not an
incentive to separate them. So tier-1 is removed from the state definition
outright — floor 2 scores only tier >= 2, exactly `truth_version 2`'s population
— and survives only as non-scored provenance (`max_severity_tier`,
`has_minor_alert`) so diagnostics can segment the same severe episodes under both
arms. The floor-independence of `max_severity_tier` is what makes the pre/post
comparison meaningful: both arms are scored against the identical episode set.

Two details that would have produced wrong numbers if taken from the textbook:

*The discrete KS.* Scoring a fitted self-loop against episode durations needs a
KS statistic, and the continuous convention — reading the ECDF on both sides of
each jump — floors it at the largest atom's probability. At a = 0.8 that is 0.20
on a *perfectly matching* sample, which would have made the statistic unable to
separate the arms at all. It now compares right-continuous step functions at
every integer up to the longest episode, and a matching sample scores under 0.05.

*Censoring direction.* Episodes touching either window end are excluded rather
than truncated. Truncating biases every dwell statistic downward — the direction
that flatters the fit — so the cheap choice is the one that would have manufactured
a positive result.

Consequence accepted and pinned by a test rather than left to surprise a refit:
every Service Change type maps to tier 1 under `mapping.severity_tier`, so floor
2 silences `has_service_change` entirely. That is `truth_version 2`'s own stance,
not an oversight; disagreeing with it means moving `TRUTH_VERSION`.

The floored arm cannot be published: `worker/src/derive.ts` counts every
non-planned alert, so a floored fit's emission distribution is one the Worker
never produces, and shipping it would be a train/serve mismatch.
`mapping.LEGACY_SEVERITY_FLOOR` exists to make that contract checkable, and
`train_em.main` refuses the write before it reads a credential.

**Not yet measured.** The refit that decides whether any of this helps — pre-clamp
diagonals and tier>=2 dwell fit, floor 1 vs floor 2 on one window — is blocked on
R2 credentials, not on code. A synthetic fixture confirms only that the
instrumentation fires and is legible: disrupted over-cap 1/1 -> 0/1, mean excess
+0.0281 -> 0.0000, pre-clamp a11 0.9581 -> 0.9000, with `normal` moving the other
way (0/1 -> 1/1) as predicted once tier-1 ticks read quiet. That is a fixture, not
evidence, and its severe episodes are a degenerate point mass at 10 ticks that no
geometric can fit, so its dwell-fit numbers say nothing. The instrumentation is
now permanent behind `--diagnose-severity` instead of being hand-patched into
`_cap_self_loops` a third time.

## 2026-09-03 — the severity floor measures negative, because the clamp pins 28 of 28 routes

origin: agent

Refit both arms on 2026-08-06..09-03, 28 routes, all fitted on their own data,
identical corpus and prior — only `--severity-floor` differs.

```
                over_cap        mean_excess       max_excess
normal      22/28 -> 26/28   0.0118 -> 0.0173  0.0247 -> 0.0246
disrupted   24/28 -> 20/28   0.0370 -> 0.0305  0.0691 -> 0.0677
suspended   18/28 ->  8/28   0.0278 -> 0.0068  0.0689 -> 0.0552
```

Both gates fail. Disrupted over-cap 24/28 -> 20/28 against a <=12/28 gate, and
mean pre-clamp excess 0.0370 -> 0.0305 against a <=0 gate. The pooled tier>=2
dwell fit is *bit-identical* between arms — mean log-likelihood -5.6172 and KS
0.1565 in both, on n=600 uncensored episodes. Of the 14 routes carrying any
severe episode, the fit improves on 1, worsens on 2, and is unchanged on 11.

**Why nothing could move.** Under the serving build the shipped disrupted
self-loop is pinned at exactly the 0.93 cap on **28 of 28 routes**. Every route
ships the same disrupted dwell and it is the cap's value, not any route's fit:
`_cap_self_loops` is doing the modelling and EM's disrupted row is discarded
wholesale. The training population therefore cannot influence the shipped
number, and "the dwell is learned from the wrong population" is not operative for
the quantity that ships — it is not learned from any population. Under floor 2
that falls to 21/28, and only two of the seven newly-unpinned routes carry a
severe episode, which is exactly why C3 barely registers.

The severity signal *does* work on its own terms: global pre-clamp `a11` moves
0.9834 -> 0.9744, so EM genuinely wants a shorter disrupted dwell once ordinary
Delays stop defining the state. It is just an order of magnitude short of the
ceiling, and the ceiling is what ships.

**The accident is confirmed to 2.2 minutes.** The cap implies a 47.8-minute
median; the pooled empirical tier>=2 median measures 50.0. The suspicion that the
cap "happens to sit near the tier>=2 median of 50 min, which is probably why
nobody noticed" is now measured on n=600 episodes instead of 45. The shipped
number is already nearly right for the severe population *by coincidence*, which
is why a severity floor has nothing to improve — and why retuning the cap is a
worse idea than when that warning was written, not a better one. Moving it would
break the one quantity that is accidentally correct.

So the 94%-ordinary-Delays conflation is real, is now fixed, and has been
measured as **not** the cause of either symptom it was proposed to explain:
recovery MAE and IQR coverage (refuted 2026-08-24, grader arm mismatch) and now
the dwell misfit. Two independent refutations of one causal story. What the
refit actually exposes is that a single geometric self-loop under a hard ceiling
cannot represent the disrupted dwell for *any* route — 28/28 pinned is the
evidence — which is a hazard-shape question, not a severity question.

One lead recorded and deliberately not chased: the **suspended** row responds
~4x more strongly than the disrupted row this work targeted, over-cap 18/28 ->
8/28, mean excess 0.0278 -> 0.0068. The plausible mechanism is that suspension
episodes stop being padded by adjacent ordinary-Delays ticks once tier-1 no
longer competes for the non-normal states. That is a hypothesis from one table,
not a result, and it is the largest single movement in the whole comparison.

The negative result cost nothing to keep: the training path now carries severity,
the two loaders share one observation builder instead of being independent copies
(which is how severity reached one path and not the other), and the pre-clamp
diagonals are permanently instrumented rather than hand-patched a third time.
None of that depended on the gates passing — which is the point of having written
the gates down first.

## 2026-09-03 — correction: the severity conflation is measurable, not fixed

origin: agent

The entry above says "the 94%-ordinary-Delays conflation is real, is now fixed,
and has been measured as not the cause of either symptom." The middle clause is
wrong and is withdrawn.

Floor 2 is diagnostic-only by construction — the trainer refuses to publish a
floored fit while `worker/src/derive.ts` counts every non-planned alert, and
that refusal is the point of the change, not a limitation of it. The default and
only publishable build is still floor 1. So the production training path counts
alerts exactly as it did before, and the latent disrupted regime is still 94.3%
ordinary tier-1 Delays in every params set that actually ships. Nothing about
shipped behaviour changed.

Stated without the overclaim, what landed is: severity now *reaches* the training
observation, the two loaders share one builder rather than being independent
copies, the pre-clamp diagonals and tier>=2 dwell fit are permanently
instrumented, a severity floor is parameterised and measurable end to end, and a
guard prevents publishing the floored build while serving disagrees. That is an
instrumented negative experiment plus the plumbing to run it — a capability, not
a behavioural fix.

The distinction is worth the correction because it changes what a reader would
do next. "Fixed" invites someone to assume the disrupted regime is now
severity-clean and reason downstream from that; it is not, and on this
measurement there is no reason to make it so, since the shipped disrupted dwell
is the 0.93 cap on 28 of 28 routes irrespective of the training population.

Numbers, gates and verdict in the previous entry are unaffected. Only the claim
about the state the codebase is left in needed narrowing.
## 2026-09-03 — the committed movement_transitions stream cannot supply censoring at all, and losing it biases atom_p up by 0.105

origin: agent

Groundwork for switching `dwell_movement`'s source from the `published_condition`
tick replay to the Worker's own committed `v1/movement_transitions`. The switch
was gated on the stream spanning a training window AND the two fits grading at
least as well as each other. The measured span/coverage/agreement/grade numbers
need R2 and are pending a credential grant, but two findings fell out of the
code alone, and neither is fixed by letting the stream accumulate longer — which
is the assumption the gate was written under.

**1. The stream carries transitions only, so it has no censoring information.**
`v1/movement_transitions` is a transition log; there is no committed per-tick
census beside it. So `dwell.dwell_samples_by_cell` falls to
`dwell._open_regimes`, which can only read a route's still-open regime off that
route's LAST transition record, and is blind to any route that never
transitioned inside the window. Measured on a controlled case (three
transitions on route A, plus a QUIET route holding `normal` for the whole
window):

| censoring source                | routes with cells | QUIET's normal cell        |
|---------------------------------|-------------------|----------------------------|
| stream-only inference           | `['A']`           | absent entirely            |
| prediction-derived census       | `['A', 'QUIET']`  | n=0, n_censored=1, 1396059s |

QUIET does not get a censored cell under the stream — it gets no cell. Per
`pooled_dwell`'s own docstring the steadiest routes are "the ones this
estimator exists to serve", so a clean cutover to the committed stream drops
exactly them. Reaching back for the prediction-derived open-regime map would
re-import the replay's tick census and so would not be a cutover at all.

**2. Losing the census biases the one-tick atom rate UP.** Mechanism:
`pooled_dwell._atom_counts` counts a right-censored observation as informative
only once it has outlived the atom (`d > atom_sec`), and never as an atom. A
census-only open *disrupted* regime is therefore pure non-atom evidence, and
dropping it raises `n_atom / n_informative`. Isolated with A/B/C each holding
three one-tick and one 2400s disrupted exit, final regime closed so both
configurations agree all three sit in `normal` at window end — leaving D and E's
open disrupted regimes, which no transition record exists for, as the only
difference:

| configuration                 | cells | atom_p                        |
|-------------------------------|-------|-------------------------------|
| stream-only                   | 6     | 0.75 (A/B/C)                  |
| census, D/E withheld          | 6     | 0.75 (A/B/C)  ← isolation check |
| census + D/E open disrupted   | 8     | 0.645 (A/B/C), 0.6397 (D/E)   |

Dropping the census moves atom_p 0.645 → 0.75: +0.105 absolute, +16% relative,
and two cells lost. Since `atom_p` is P(episode lasts exactly one tick), an
inflated atom makes the recovery forecast systematically too optimistic in the
first tick — the region the atom was introduced to fix in the first place.

NEGATIVE RESULT, recorded because the sign is the whole point: the first attempt
at (2) measured atom_p moving the *other* way, 0.6 → 0.645, and would have
supported the opposite conclusion. The construction left A/B/C's final disrupted
regime open, so stream-only inference handed them a spurious censored disrupted
observation and the two configurations differed in two places at once; the D/E
effect was swamped by the artifact. The isolation check row above (census with
D/E withheld reproducing stream-only exactly) is what makes the third row
attributable to D/E and nothing else. A two-configuration comparison that
differs in two places measures neither.

Methodology note for the pending grade, same failure mode in a different guise:
a sparse source publishes fewer cells and so abstains on every held-out episode
whose (route, state) it never reached, taking its mean CRPS over an easier
subset. On synthetic data the committed-side fit scored 108 of 218 held-out
episodes against the replay's 218/218, and the two means were not comparable
quantities. The grade has to be PAIRED on the episodes every fit can score
(there: replay 4.933 vs committed 5.029) with the abstention count reported
beside it, or the sparser source wins by declining to answer.

## 2026-09-03 — correction: "no censoring at all" overstates it; the stream is blind to no-transition routes specifically

origin: agent

The entry above is headlined "cannot supply censoring at all", which is wrong as
written and wrong in the direction that flatters its own conclusion. Correcting
it in place is not an option here, so: the precise claim is narrower.

`dwell._open_regimes` DOES supply a right-censored observation for any route
that transitioned inside the window — it reads that route's still-open regime
off the `new_state` of its last transition record. The stream-only path is
blind only to routes with NO transition in the window, because there is no
record to read a regime off. That is the QUIET row in the table above, and it
is the real deficit: it covers the steadiest routes, which is why it bites.

The evidence for the narrower claim was already sitting in the same entry and I
read past it twice. Route A gets a cell under stream-only censoring in the
first table. And the discarded first attempt at the atom_p measurement failed
precisely BECAUSE stream-only inference supplied a censored observation —
A/B/C's final disrupted regime was left open, so the stream handed them a
spurious censored disrupted sample and moved atom_p the other way. An
experiment whose failure mode is "the stream censored something" cannot support
"the stream censors nothing".

Neither of the two numbers moves: QUIET still gets no cell rather than a
censored one, and withholding the census still moves atom_p 0.645 -> 0.75
(+0.105, +16% relative) with two cells lost. What changes is the scope of the
mechanism, and therefore what a fix would have to do — the gap is confined to
no-transition routes, not to censoring generally.

Same shape of error as the two retention-table corrections above: a convenient
aggregate ("no censoring") standing in for the quantity actually measured
("censoring for no-transition routes"). Third instance in this journal, so
worth naming as a habit rather than an incident — the headline of a finding
drifts broader than its experiment unless the experiment's scope is written
into the sentence.

## 2026-09-03 — withdrawn: "do not switch the dwell_movement source" was never a measured verdict

origin: agent

Withdrawing the framing, not the numbers, of the two entries above.

The dwell_movement source switch was gated on measured quantities: the
committed stream's span over a real training window, its per-route coverage
against the tick replay, onset agreement, and graded fits on held-out ticks.
None of those were measured — the R2 credential grant never arrived, and there
is no local archive. Every number in the two entries above came from synthetic
fixtures constructed by hand to isolate one mechanism at a time.

Those fixtures cannot satisfy the gate, and they cannot overturn it either.
They support a mechanism and a prediction: that a switch would degrade the fit
through lost censoring rather than through any difference in signal, and that
the cell-count criterion will fail when someone reads the window. That is a
hypothesis with a mechanism behind it. It is not a finding about the archive,
and reporting it as "do not switch" borrowed the authority of measurement for
evidence that had none.

The defensible state is: blocked, no verdict in either direction. The census
gap is filed separately so the mechanism survives whichever way the real
numbers land.

What each claim can actually carry:

- supportable: "on a controlled fixture, withholding the open-regime census
  moves atom_p 0.645 -> 0.75 and drops 2 of 8 cells, because _atom_counts
  credits a censored observation as non-atom evidence only once it outlives
  the atom"
- supportable: "the stream is a transition log, so stream-only inference is
  blind to routes with no transition in the window"
- NOT supportable without reading the archive: "the committed stream grades
  worse than the replay", "do not switch"

This is the third correction of the same shape in one session — the retention
table's denominator, the "no censoring at all" headline, and now the verdict
itself. The first two were single claims drifting broader than their
experiments; this one is the whole task's conclusion doing it, which is the
same mistake with more at stake. The rule that would have caught all three: a
sentence may only claim what its experiment's scope allows, so write the scope
into the sentence and the overreach becomes unsayable.

## 2026-09-03 — the committed route-scope transition stream is not accumulating, it is being overwritten: segment and route share one R2 key

origin: agent

Measured the dwell_movement source switch on the real archive (window train
2026-08-14..2026-08-27, holdout 2026-08-28..2026-09-03, route scope,
debounce_ticks=1). Verdict: do not switch. The reason is not the one the gate
anticipated, and not the one the two entries above predicted either.

Headline numbers, against the 13-vs-124 baseline from 2026-08-12:

| source          | records | routes | episodes | span                    |
|-----------------|---------|--------|----------|-------------------------|
| committed route | 93      | 21     | 37       | 08-11T23:50 → 08-26T13:35 |
| tick replay     | 324     | 22     | 102      | 08-14T00:00 → 09-03T13:05 |

Onset agreement rose from 10/13 (0.77) to **86/93 (0.9247)**, so where the
stream speaks it agrees with the replay: this is genuinely a change of source
and not of signal. Route coverage 0.9545 (only `M` missing). Fitted medians
agree exactly (median ratio 1.0). Cell count holds: replay 44, committed 41.

And yet the stream's last route record is 2026-08-26, with **zero route-scope
records for the eight consecutive days 08-27..09-03**, plus zeroes on
08-22/23/24. Per-day route vs segment counts:

| date  | route | segment |
|-------|-------|---------|
| 08-20 | 5     | 138     |
| 08-26 | 35    | 90      |
| 08-27 | 0     | 8733    |
| 08-29 | 0     | 10068   |
| 09-03 | 0     | 6571    |

**Root cause.** `worker/src/grading.ts` `writeMovementTransitions` builds its
key from `observedAt` alone — no scope component — and R2 `bucket.put`
overwrites. `worker/src/index.ts` calls it twice per tick: route scope at :897,
then segment scope at :937. Every tick with a segment change destroys that
tick's route records. Route records survive only on ticks where the segment
clock happened to be quiet. Segment volume rose ~60x on 08-27 (~150/day to
8–11k/day) and route scope hit zero on exactly that date.

Confirmed by reading every file's scope set on sampled whole days: **never a
single mixed-scope file.** 08-20: 106 files, 101 segment-only, 5 route-only.
08-26: 84 files, 79 segment-only, 5 route-only. 08-29: 287 files, 287
segment-only, 0 route-only. A same-key overwrite produces exactly that; an
append or a scope-partitioned key never could. Filed as its own P0 bug.

Also stale as of this measurement: `movement_backfill.py`'s module docstring
says segment-scope regimes "have never been wired up online". They are, at
8–11k records/day, and they are the reason route scope is gone.

**Two retractions, both mine.**

The entries above predicted the census gap would be the binding constraint —
that the cell count would collapse without the prediction-derived open-regime
map. It did not: 41 cells vs 44, comfortably inside a 10% bar. The census
mechanism is real (independently reproduced on two fixtures) but it is not what
is stopping this switch, and the census work it motivates is a nice-to-have rather
than a prerequisite. I extrapolated a hand-built fixture to production and got
the wrong deficit.

Second, subtler, and the more useful lesson: the criterion I wrote gated on
cell COUNT, and count survived while quality did not. `n_own` collapses 16 → 7
and `n_atom` 42 → 18 — the committed fit publishes nearly as many cells, but
fewer than half carry a one-tick atom and most are pooled rather than
self-supported. A threshold on `n_cells` is blind to that. A criterion is only
as good as the quantity it names, and I named the easy one.

The held-out grade could not be computed at all: the holdout offers 4 replay
episodes, 2 scorable by every fit, and 0 from the committed stream. n=2 is not
a grade — `segment_grade.MIN_ARM_HOPS` is 10 for exactly this reason ("an AUC
over three hops is not a weak result, it is not a result"), so it abstains
rather than reporting a number that would look like evidence.

Net: the gate assumed the stream needed time. It does not need time; it needs
the write fixed. Once that lands and a window accumulates, criteria 1/2/4 are
worth re-running — agreement and cell count already pass, and on this evidence
the switch looks more likely to be justified than not.

## 2026-09-03 — the dwell source grader dropped boundary-straddling regimes; latent, and provably inert on the graded window

origin: agent

The train/holdout split in the source grader selected training transitions with
`exited_at < split_ts`. A regime that began inside the training half but exited
during the holdout was therefore discarded entirely, when what the boundary
actually tells us is `dwell > split_ts - regime_entered_at` — a right-censored
observation, and typically a long one, since long regimes are the ones most
likely to straddle a boundary. Dropping them biases every fit short.

Worse, it biases them unequally. The replay fit recovers most of that evidence
from the per-tick census (`movement_open_regimes`), which observes every live
route regardless of whether it transitioned. The committed-source fit has no
census, so for it the loss is total — and that configuration is precisely the
one the cell-count criterion grades. The defect handicapped the source it was
being used to judge.

Fixed by censoring straddling regimes at `split_ts`, derived from each source's
OWN records (`regime_entered_at < split_ts <= exited_at`). For the committed
source that keeps the cutover clean: its own record proves the regime was open
at the boundary, so nothing is borrowed from the replay's census.

**Effect on the graded window: none.** Re-ran and diffed the full report —
byte-identical. Two independent reasons, both checked directly rather than
assumed:

1. The committed stream's straddle map is empty. Nothing spans 2026-08-28
   because the stream is already dead from 08-27, so there was nothing on that
   side to drop.
2. The replay's straddle map holds 6 routes (B, C, D, 4, 7, R, all `normal`),
   and all 6 were already in the 22-route census map, which supersedes.
   Straddle routes absent from the census: none.

So every reported number stands on a now-correct split: agreement 86/93,
cells 41 vs 44, `n_own` 16 → 7, `n_atom` 42 → 18, criterion 4 abstained.

Recording it anyway, because "no effect on this window" and "harmless" are
different claims. The two reasons it cancelled are both artefacts of this
particular window — one of them is the very outage the measurement exists to
report. On any window where the committed stream is alive the straddle map is
non-empty and the old code would quietly understate that source. When the
write bug is fixed and the criteria are re-run, they have to be re-run with the
corrected split or the committed source is graded with a handicap it does not
deserve.

A latent bug that cancels for reasons unrelated to its logic is still a bug,
and a measurement that agrees with itself for the wrong reason is not
corroborated. The diff is the evidence here, not the agreement.
## 2026-09-03 — the movement dwell arm's whole history of skill numbers is quoted against a hindsight baseline, and the mixture it is judged on was already shipped

origin: agent

Two findings, neither of which needed the archive, plus the committed harness
that was missing.

**The "next iteration" this arm is still tracked against has been shipped since
2026-08-12.** The one-tick point mass with a per-route shrunk rate and a
left-truncated conditional tail is `pooled_dwell.atom_fits` / `mixture_cell`,
wired through `train_em._movement_dwell(atom_sec=TICK_SECONDS)` (commit
46d138f). The figures still being quoted for this arm — CRPS skill -0.10, and
an offline -0.088 -> -0.023 — predate it; the causal grade after it landed was
continuous -0.1391, single global atom -0.0582, per-route shrunk -0.0606. So
the open question was never "build the mixture", it was "re-measure", and that
could not be done: the five scripts that produced those numbers lived in /tmp
and were deleted. `training/movement_dwell_grade.py` is that grading, committed
and tested, runnable as one command.

**Every skill number this arm has ever been judged by is measured against
perfect hindsight, and nobody said so.** `recovery_dist_report` builds its CRPS
baseline from the empirical CDF of the graded population's OWN durations
(recovery_dist.py:195). The model is a forecast; the baseline is not. That is
defensible as a fixed yardstick, but it means "-0.06 vs climatology" does not
mean "worse than a forecaster using climatology" — it means "worse than a
forecaster who already knew this window's duration distribution". The harness
now reports both: `oracle_skill` (that yardstick, kept so the numbers stay
comparable to the record) and `causal_skill`, the ratio against the SAME
climatology fitted on the train window only. On synthetic one-tick-dominated
ticks (23 matched episodes, 61% one tick) the causal climatology's own
oracle_skill is -0.0028, so there the hindsight advantage is ~0.3% of CRPS and
the gap really is the model's; whether that holds on real data is exactly the
unmeasured question, and it is the number the verdict on this arm should turn
on. Flagging the distinction rather than the conclusion: the archive run is
blocked on an R2 grant.

A second methodological repair in the same harness: variants disagree on
coverage (`n_no_curve` differs, because which routes have a train-window cell
depends on the form), so grading each on its own scorable subset compares CRPS
means over different populations AND against different baselines. All variants
are now scored on the episode intersection, which also makes the oracle
baseline literally identical across rows — asserted in a test rather than
assumed.

**The atom estimator leans on a per-tick census, and the deficit is narrower
than "the stream has no censoring".** `dwell._open_regimes` does censor any
route that transitioned in the window, reading its open regime off the
new_state of its last transition. The blind spot is routes with NO transition —
which, since a route completes a `normal` regime only by leaving normal, is
exactly the steadiest ones. That bites the mixture specifically:
`pooled_dwell._atom_counts` credits a right-censored observation as evidence
ABOUT the point mass only once it has outlived the atom, and never as an atom
itself, so a never-transitioned route sitting in a disrupted regime is pure
NON-atom evidence. Drop it and the fitted one-tick rate can only rise.
Measured on two independent fixtures: on the source side atom_p 0.645 -> 0.750
(+0.105 absolute, +16% relative, 2 cells lost); on the harness's own fixture
0.601 -> 0.625 with the never-transitioned route's cell disappearing.

The trap that makes such a delta easy to misattribute, recorded because the
first attempt on the source side hit it: transition-derived inference can hand
a mover a SPURIOUS censored observation, so a fixture whose movers still sit in
open not-normal regimes at the boundary moves atom_p for two reasons at once
and can get the right sign from the wrong cause. The isolation check that
separates them is asserting that a censused fit with the never-transitioned
routes withheld reproduces the no-census fit EXACTLY (verified: identical
atom_p 0.625 on all three routes); only the remaining delta is attributable.
Consequence for the arm: model form and source selection are not separable
here, because a transitions-only source biases the published p0 up and shrinks
coverage before any question of signal quality arises.

NOT MEASURED, and the reason: the refreshed CRPS numbers on the current archive
(three more weeks of data, and a movement truth that changed underneath the
2026-08-12 grade — the baseline desaturation plus through-stop filtering took
one window from 14 episodes to 56, adding the 45-160min multi-tick tail the old
cut was blind to). `murk_get` reports no active grant covering this thread for
the R2 keys, and the key file was deliberately not slot-linked around it. The
first run when access lands should be a REPRODUCTION check on the historical
window (train 2026-06-21..07-25, eval 07-26..08-11), where oracle_skill ought
to land near continuous -0.139 / shipped -0.061 before any refreshed number is
trusted.

## 2026-09-03 — correction: the dwell grading harness is not committed, it is uncommitted in a worktree

origin: agent

The entry above calls `training/movement_dwell_grade.py` "committed" twice —
"the committed harness that was missing" and "that grading, committed and
tested". Both are wrong. Nothing was committed: the harness and its tests exist
only as working-tree changes on a worktree branch, awaiting review. Verified
against the record rather than from memory — `git log` shows no commit for
either file.

The word was doing real work in that entry's argument, so the correction is not
cosmetic. The contrast being drawn was against the five /tmp scripts that
produced the 2026-08-12 numbers and were deleted, and the property that
mattered there was DURABILITY. An uncommitted file in one worktree does not yet
have it. What is true today: the grading is written down as a module with tests
instead of as a throwaway script, and it will be durable once the diff is
reviewed and lands. Until then the 2026-08-12 numbers remain the last ones
anybody can reproduce, and they remain unreproducible.

Same shape of error as the two corrections from 2026-09-02: a word that asserts
more than was done, in a sentence whose point depended on exactly that word.

## 2026-09-03 — the movement dwell arm was never worse than climatology: the baseline it lost to was a hindsight oracle, and per-route p0 is the only part the floor actually blocks

origin: agent

Refreshed the arm with a grading harness rather than throwaway scripts, and
three of the four things this task was opened to establish came out different
from the brief.

**The harness reproduces the record before anything else here is believed.**
Historical window, historical truth (vehicles-derived, train 2026-06-21..07-25,
eval 07-26..08-11, 64 episodes, one-tick share 0.609), all five forms on one
matched population:

| variant     | CRPS  | oracle_skill | causal_skill | mean PIT |
| ----------- | ----- | ------------ | ------------ | -------- |
| shipped     | 7.824 | -0.0807      | **+0.0314**  | 0.537    |
| global_atom | 7.796 | -0.0767      | +0.0350      | 0.538    |
| continuous  | 8.375 | -0.1568      | **-0.0368**  | 0.588    |
| pooled      | 7.740 | -0.0691      | **+0.0418**  | 0.535    |
| km_pooled   | 8.078 | **-0.1157**  | 0.000        | 0.769    |

oracle_skill lands on the record: continuous -0.157 against the journal's
-0.1391, shipped -0.081 against -0.0606, on 64 episodes against 74 (mine fits
the advance baseline in-window; the 2026-08-12 run fit it on a clean earlier
window and applied it forward, so the populations are close, not identical).

**The decisive number is km_pooled's own oracle_skill: -0.1157.** That is the
same climatology, fitted causally on the train window, scored by the metric
this arm has been judged by since it opened — and it loses to that metric by
about as much as the model did. `recovery_dist_report` builds its CRPS baseline
from the empirical CDF of the graded population's OWN durations
(recovery_dist.py:195), so "worse than climatology" was never a comparison
against a climatology forecast; it was a comparison against a forecaster who
already knew the eval window's duration distribution. Against the honest
baseline the shipped mixture is POSITIVE, +0.0314.

**On the current production source the sign flips outright.** The replay
(published_condition) source, train 2026-08-12..08-24, eval 08-25..09-03, 34
episodes / 22 matched / 12 no-curve, one-tick share 0.559:

| variant     | CRPS  | oracle_skill | causal_skill | mean PIT |
| ----------- | ----- | ------------ | ------------ | -------- |
| shipped     | 1.351 | **+0.5862**  | -0.0288      | 0.504    |
| global_atom | 1.349 | +0.5867      | -0.0277      | 0.504    |
| continuous  | 1.535 | +0.5298      | **-0.1691**  | 0.510    |
| pooled      | 1.316 | **+0.5969**  | **-0.0022**  | 0.505    |
| km_pooled   | 1.313 | +0.5978      | 0.000        | 0.741    |

The window is forced, not chosen: published_condition carries its first
NOT_NORMAL route-ticks on 07-11, runs 0-12/day through 08-11, jumps to 21-95/day
from 08-12, then collapses to 0-2/day after 08-27. 08-12..08-29 is the only
dense stretch this source has.

Honest reading of the two tables together: **the arm is at parity with a
climatology forecast** (+0.031 and -0.029, both within noise at n=64 and n=22),
and clearly better than the continuous curve it replaced. Neither "the mixture
fixed it" nor "the floor wins" was the right frame; the acceptance gate was
comparing a forecast to an oracle.

**PIT lobes are gone.** This investigation opened on [0,0,38,0,14,0,0,2,8,14] — a
38-episode bin that was 100% one-tick and a 22-episode high lobe that was 100%
multi-tick. Shipped now reads [3,0,1,5,2,3,2,4,0,2] at mean PIT 0.504.

**Per-route p0 is the one thing the floor really does block.** `pooled` — a
single unconditional cell per state, served to every route — equals or beats
the per-route form on BOTH populations (+0.0418 vs +0.0314; -0.0022 vs -0.0288).
The estimator says so in its own output: parent_p 0.7400 with every one of 19
routes shrunk into 0.7308-0.7452, a 0.014 spread across routes whose raw rates
span 0.000 to 1.000. So 2026-08-26's "conditional dwell adds no skill over the
unconditional median" is confirmed under CRPS and located precisely: the
CONDITIONING is floor-limited, the point mass is not. The point mass is worth
+0.068 and +0.140 of causal skill over the continuous form on the two windows.

NEGATIVE RESULT, and it corrects an emphasis of my own from earlier today: the
per-tick census deficit is NOT binding in production. Same window with
`--no-census`: cells 43 -> 41, n_own 20 -> 20, n_atom 42 -> 40, censored samples
22 -> 21, and every grade identical to three decimals (shipped CRPS 1.351, same
PIT histogram). The mechanism is real — a never-transitioned route in a
disrupted regime is pure non-atom evidence, reproduced at atom_p 0.645 -> 0.750
on a hand-built fixture — but a fixture built to isolate a mechanism says
nothing about its production magnitude, and here the magnitude is two cells and
zero skill. Same shape of error the source thread made and corrected in the
opposite direction.

Two harness bugs worth recording because both fail SILENTLY as "no episodes"
or "wrong population", and both were invisible against the vehicles source:
`extract_episodes` reads truth at exact 5-minute grid epochs while the
prediction stream stamps `ts` with real publish time (…210, …132), so an
unsnapped truth map graded 0 episodes on a window full of disruptions; and
admitting every non-normal state includes `not_scheduled`, which is absent
service rather than disruption and took the one-tick share from 0.61 to 0.034
and distinct durations from 4 to 19. An episode is a NOT_NORMAL run.

Recommendation left for the lead, not implemented: keep the mixture, drop the
per-route dimension. It grades equal-or-better on both populations, removes the
shrinkage machinery from the published path, and closes the 12-of-34 no-curve
coverage gap that per-route cells open for routes missing from the train window.

## 2026-09-03 — correction: the unconditional-mixture recommendation does not close the no-curve coverage gap, and the entry above had the evidence to know that

origin: agent

The entry above recommends collapsing the movement dwell block to an
unconditional pooled mixture and claims it "closes the 12-of-34 no-curve
coverage gap that per-route cells open for routes missing from the train
window". That claim is false, and its own table refutes it: every variant in
the acceptance run, `pooled` included, reports n_no_curve = 12. Identical.

The mechanism, stated correctly: the pooled variant fits one cell and then
serves it to the routes it saw in TRAINING. A route that appears only in the
eval window is still absent from the lookup, so it still grades as no-curve.
Pooling changes where a cell's parameters come from; it does not by itself
change which keys resolve.

Closing that gap needs a different change — a lookup-level fallback from an
unknown (route, state) to the state's pooled cell — which is not what was
recommended, is not implemented, and is not measured anywhere in this work. It
is also not measurable through the harness as built: the matched-population
design scores every variant on the episodes ALL of them can score, so a variant
with better coverage cannot show its advantage there by construction. Any
coverage claim would need a separate full-coverage measurement.

What survives of the recommendation is the part that was actually measured:
the unconditional mixture equals or beats the per-route form on skill on both
populations (+0.0418 vs +0.0314 historical, -0.0022 vs -0.0288 current) and
removes the per-route shrinkage machinery from the published path. The coverage
sentence should be struck.

Third correction today of the same kind, and worth naming as a pattern: a
claimed benefit that no measurement in front of me supported, sitting one line
away from the table that contradicted it.

## 2026-09-03 — correction: "never worse than climatology" overstates the evidence; what is established is that the old comparison used the wrong baseline

origin: agent

The headline two entries above reads "the movement dwell arm was never worse
than climatology". That is stronger than the numbers in its own tables support,
and the direction of the overstatement is the flattering one.

What the measurements do establish, and this part is solid: the negative skill
this arm has been judged by was computed against `recovery_dist_report`'s CRPS
baseline, which is the empirical CDF of the graded population's own durations
(recovery_dist.py:195) — hindsight, not a forecast. On the historical population
that baseline beats a CAUSAL fit of the same climatology by 0.116 skill, so the
old "-0.10 vs climatology" reading conflated "worse than climatology" with
"worse than an oracle". That conflation is real and is the finding.

What is NOT established is the headline. Against the causal climatology the
shipped mixture reads +0.0314 on the historical window (n=64) and -0.0288 on
the current replay window (n=22). The second number is NEGATIVE. Two windows,
opposite signs, both small, on 64 and 22 episodes, with no interval computed on
either — I did not bootstrap them, so "within noise" is an assertion I have not
earned either way. The defensible sentence is: current causal evidence is
NEAR-PARITY and underpowered, and the sign is not resolved.

Three claims from those entries, re-stated at the strength the evidence carries:

- "The headline claim is false as stated" — holds only for the original
  metric on the current source, where oracle_skill is +0.586. It does NOT hold
  as a statement about a climatology forecast, where the current point estimate
  is negative.
- "At parity with a climatology forecast" — better read as "not distinguishable
  from parity at these sample sizes", which is a statement about power, not
  about the model.
- The point-mass result (+0.068 and +0.140 causal skill over the continuous
  form) is the one comparison that keeps its sign and its magnitude across both
  windows and both truths. It is the finding here that does not need hedging.

Fourth correction today, and the same shape as the other three: a summary
sentence asserting more than the table beneath it. Recording the count because
the pattern is the useful part — the tables were right every time; the prose
over-reached every time.

## 2026-09-03 — every recovery-skill consumer in the repo was oracle-relative, and only one of the five could be fixed rather than labelled

origin: agent

Audited every reader of the recovery CRPS baseline after the movement dwell
re-grade. Five surfaces publish a skill number, and all five were computing it
against the graded window's own empirical duration CDF:
`recovery_dist.recovery_dist_report`, `scorecard.episode_recovery` (both arms,
via `episode_scorecard` into the review summary), `backtest.grade_recovery_timing`'s
printed line, `movement_dwell_grade.grade_variants` (already reporting the
distinction), and the models page's `RecoveryScoreCard` through
`viz/lib/recovery_dist.ts`. Two more surfaces mention the baseline in prose and
carried no number: the go/no-go template's graduation criterion and the models
page intro.

The report now fits a CAUSAL climatology — the same empirical-CDF forecast, but
over durations from a window the caller supplies that closes before the graded
one — and emits `causal_skill` beside a renamed `oracle_skill`. The rename is
the load-bearing part: `skill` and `baseline_crps` are gone, so no consumer can
keep quoting the hindsight figure without typing the word oracle.

How large the difference is on a hand-checkable case: a graded window that
recovers in 1-2 minutes against a training window that said 10 gives the SAME
model `oracle_skill -1.0` and `causal_skill +0.8`. That is a whole verdict's
worth of spread with nothing about the model changing — the mechanism behind
the -0.1157 a causally-fitted climatology scored against itself on the real
movement population.

What could not be fixed, only labelled, and this is the part worth remembering:
`episode_scorecard` is handed exactly ONE window of truth episodes, and
`backtest.grade_recovery_timing` loads truth only from `eval_start`. Neither has
a pre-window episode population in memory, so neither can fit a causal
climatology without new data loading. Both therefore publish `causal_skill:
null` — visible, never a silent fallback to the oracle number. The published
review summary still has no honest skill column; it now has an honestly named
dishonest one. Rejected the cheap substitute of fitting the baseline from
`train_trans` dwell spells in backtest: those are HMM regime spells, not
severe-only truth episodes, so the ratio would be taken against a different
population and would read as a real number while comparing two different things.
