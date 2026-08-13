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
still-open regime from a route's *last transition record*, so the 17 routes
with zero transitions in the window supply neither an event nor a censored
observation and never reach the pooling step — the steadiest routes, which are
the ones the pooling exists to serve, are still the ones excluded. Those routes
carry 62% of normal route-ticks.

Counterfactual over 50,866 predictions (08-05..08-11), pooled curves vs the
live geometric projection, graded against the model's own condition at t+H:

| horizon | covered subset (n=19,217) | fleet (n=50,663) |
| --- | --- | --- |
| 30 min | Brier 0.01734 -> 0.00326 (-81%) | 0.01714 -> 0.01180 (-31%) |
| 60 min | 0.03746 -> 0.00392 (-90%) | 0.03895 -> 0.02623 (-33%) |
| 120 min | 0.06052 -> 0.00416 (-93%) | 0.06380 -> 0.04242 (-34%) |

On the covered subset mean predicted `p_normal_in_120` moves 0.7582 -> 0.9870
against an actual base rate of 0.9968, so the sharpness bias really is closed
where a cell exists. The number to chase is the fleet column: it only moves if
open regimes are reconstructed from prediction rows rather than from transition
records.


## 2026-08-11 — open regimes from prediction rows: 11 -> 29 normal cells

origin: agent

Follows the entry above, which named the number to chase. `_open_regimes` read
each route's still-open regime off its *last transition record*, so a route with
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
open *now* — `open_regimes_from_predictions` takes `window_end` and ignores
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

| arm | n | Brier | BSS vs persistence | AUC |
| --- | --- | --- | --- | --- |
| shadow (`condition`), h30 | 55,390 | 0.01709 | -7.93 | **0.780** |
| movement (`published_condition`), h30 | 41,132 | 0.01605 | -16.37 | **0.145** |
| movement, h60 | 40,807 | 0.03640 | -40.27 | 0.144 |
| movement, h120 | 40,081 | 0.06059 | -55.48 | 0.142 |

AUC 0.142-0.145 is not weak, it is inverted: random is 0.5, and inverting the
forecast would score ~0.855. The mechanism is not subtle. Of the 29
movement-disrupted ticks in the window, **29/29 have shadow `condition ==
"normal"`** — `effectiveCondition` (worker/src/snapshot.ts) hard-returns `normal`
whenever `disruptiveAlertCount === 0`, so movement-disrupted-while-alerts-silent
is by construction the case the filter is most confident about. Mean
`p_normal_in_30min` is **0.9166 on movement-disrupted ticks vs 0.8780 on
movement-normal ticks**: the forecast is *more* sure of health exactly where
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

| period | unknown share | gradeable share |
| --- | --- | --- |
| 2026-07-11 | 3.7% | 92.2% |
| 2026-07-12 .. 2026-08-02 | ~100% | ~0% |
| 2026-08-03 | 64.1% | 34.9% |
| 2026-08-04 .. 2026-08-11 | 15-17% | 68-78% |

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

| debounce_ticks | n episodes | median (min) | min | max | routes ≥3 eps | survival vs 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 17 | 5.0 | 4.95 | 34.92 | 3 of 6 | 100% |
| 2 | 4 | 10.125 | 10.0 | 34.92 | 0 of 2 | 23.5% |
| 3 | 1 | 49.95 | 49.95 | 49.95 | 0 of 1 | 5.9% |

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

| debounce_ticks | n episodes | median (min) | min | max | cells ≥3 eps | survival vs 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 828 | 10.0 | 5.0 | 435 | 35 of 79 | 100% |
| 2 | 411 | 30.0 | 10.0 | 765 | 26 of 59 | 49.6% |
| 3 | 216 | 55.0 | 15.0 | 1090 | 25 of 49 | 26.1% |

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
*changed* any cell's state writes nothing, even while it publishes
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

| day | disrupted calls | ticks w/ >=2 disrupted | adjacent (gap0) | share |
| --- | --- | --- | --- | --- |
| 08-04 | 3391 | 286 | 11 | 3.8% |
| 08-05 | 4236 | 286 | 15 | 5.2% |
| 08-06 | 3949 | 287 | 26 | 9.1% |
| 08-07 | 3601 | 285 | 0 | 0.0% |
| 08-08 | 4559 | 287 | 0 | 0.0% |
| 08-09 | 5089 | 287 | 0 | 0.0% |
| 08-10 | 4379 | 287 | 0 | 0.0% |
| 08-11 | 5210 | 251 | 37 | 14.7% |

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

| division | n | mean log-scale | within-division variance |
| --- | --- | --- | --- |
| BMT | 9 | 10.899 | 0.027 |
| IND | 9 | 10.841 | 0.023 |
| IRT | 10 | 10.529 | 0.401 |

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
telling): eta² swings to 0.272 and F *drops* to 0.561 (p=0.529), the
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

| from_state | n exits | to_normal | dest breakdown |
| --- | --- | --- | --- |
| disrupted | 213 | 1.000 | normal=213 |
| suspended | 0 | n/a | (no completed episode in window) |

19 routes contribute to the 213, every one individually 100% to normal (7:45,
G:29, M:23, R:16, W:14, 6:14, E:11, J:10, D:10, N:9, A:8, C:5, 4:5, F:3, Q:3,
B:3, H:2, 1:1, 2:1, 5:1).

`source=predictions` (2026-08-04..2026-08-12, 8 days, the faithful published
vocabulary — trusted for the `not_scheduled` question the vehicle source
structurally cannot answer):

| from_state | n exits | to_normal | dest breakdown |
| --- | --- | --- | --- |
| disrupted | 17 | 1.000 | normal=17 |
| suspended | 0 | n/a | (no completed episode in window) |

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

| baseline | candidate ticks (>=2 disrupted) | adjacent ticks gap0 | adjacent ticks gap1 | share ticks gap0 | mean cluster size gap0 | share disrupted-segments-in-multi gap0 | gap1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| self (53-day self-trained) | 5257 | 21 | 21 | 0.40% | 1.0016 | 0.274% | 0.285% |
| published (live segment_params.json) | 5489 | 21 | 21 | 0.38% | 1.0016 | 0.271% | 0.283% |

gap=0 and gap=1 give the IDENTICAL 21 adjacent ticks on both baselines —
loosening `DEFAULT_MAX_GAP` still has no support. The two baselines land
within ~4% of each other on the denominator (5257 vs 5489 candidate ticks —
the published baseline calls slightly more ticks judgeable) and produce the
exact same 21 numerator. (8963 total reconstructed ticks either way; 5257
of them, 58.7%, have >=2 disrupted segments somewhere on the network — same
order of magnitude as the 8-day run's 55.6%, not a new finding on its own.)

Per-week breakdown (self-trained baseline; published tracks it within 1-2
ticks/week everywhere it differs):

| week | candidate ticks | adjacent ticks gap0 | share |
| --- | --- | --- | --- |
| 06-21..06-27 | 0 | 0 | n/a (no disruption observed yet) |
| 06-28..07-04 | 0 | 0 | n/a |
| 07-05..07-11 | 20 | 0 | 0.0% |
| 07-12..07-18 | 948 | 11 | 1.16% |
| 07-19..07-25 | 1185 | 9 | 0.76% |
| 07-26..08-01 | 1304 | 1 | 0.08% |
| 08-02..08-08 | 1255 | 0 | 0.0% |
| 08-09..08-12 | 545 | 0 | 0.0% |

5 of 8 weeks are exactly zero; all 21 adjacent ticks fall in a 3-week span
(07-12..08-01). Traced identity through the whole window with
`advance_incidents` (not just counting candidate ticks) to see how many
DISTINCT real incidents that span represents: exactly **3**, agreed on by
both baselines down to the segment keys and timestamps —

| incident | segments | first seen (UTC) | last seen (UTC) | span |
| --- | --- | --- | --- | --- |
| 1 | 7\|south\|701S, 7\|south\|702S | 2026-07-16 04:20 | 2026-07-16 07:55 | 215 min |
| 2 | A\|north\|A48N, A51N, A55N | 2026-07-24 18:15 | 2026-07-24 19:45 | 90 min |
| 3 | 7\|south\|701S, 7\|south\|702S | 2026-07-27 12:30 | 2026-07-27 12:45 | 15 min |

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

| horizon | n | AUC old→new | Brier old→new | BSS vs persistence old→new |
| --- | --- | --- | --- | --- |
| 30min | 50,061 | 0.150→0.762 | 0.01587→0.00066 | -16.27→+0.29 |
| 60min | 49,831 | 0.150→0.493 | 0.03609→0.00102 | -39.87→-0.16 |
| 120min | 49,201 | 0.150→0.397 | 0.06004→0.00178 | -58.08→-0.75 |

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
arm error — but the sign says the movement arm calls "recovered" *before* the
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

| date | unknown share | gradeable share |
| --- | --- | --- |
| 07-30 | 100.0% | 0.02% |
| 07-31 | 99.9% | 0.08% |
| 08-01 | 100.0% | 0% |
| 08-02 | 100.0% | 0% |
| 08-03 | 64.1% | 34.9% |
| 08-04 | 16.9% | 78.4% |
| 08-05 | 17.3% | 77.8% |
| 08-06 | 17.1% | 78.2% |
| 08-07 | 16.8% | 78.3% |
| 08-08 | 16.3% | 70.8% |
| 08-09 | 16.5% | 68.9% |
| 08-10 | 15.3% | 77.6% |
| 08-11 | 16.3% | 78.1% |
| 08-12 (today, n=522, partial day) | 13.8% | 82.8% |

Coverage is healthy now: today's 13.8% unknown / 82.8% gradeable sits inside
(slightly better than) the stable ~15-17%/~78% band every day has held since
the 07-12..08-02 empty-baseline gap closed on 08-03. Nothing resembling the
22-day 100%-unknown blackout has recurred.

Adoption, today (n=522, post-deploy) vs before deploy (n=8,352, 08-11, almost
entirely pre-deploy — the deploy landed ~23:48 UTC 08-11):

| field | value | before deploy | today |
| --- | --- | --- | --- |
| recovery_source | movement | 0.29% (24) | **82.76% (432)** |
| recovery_source | hmm | 99.71% (8,328) | 17.24% (90) |
| condition_source | movement | 83.66% (6,987) | 86.21% (450) |
| condition_source | unknown | 16.34% (1,365) | 13.79% (72) |
| published_condition | disrupted | 0.14% (12) | 0% (0) |

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

| subset | 30min AUC / BSS (n) | 60min AUC / BSS (n) | 120min AUC / BSS (n) |
| --- | --- | --- | --- |
| all (sanity) | 0.762 / +0.29 (50,061) | 0.493 / -0.16 (49,831) | 0.398 / -0.75 (49,297) |
| trustworthy | 0.728 / **+0.40** (14,278) | 0.395 / **-0.00** (14,053) | 0.352 / **-0.53** (13,606) |
| censored | 0.624 / -0.32 (35,783) | 0.449 / -0.63 (35,778) | 0.317 / -1.30 (35,690) |

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
*worse* from day 21 to day 28). Four experiments, detailed below, all point
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

| ticks | minutes | count | share |
| --- | --- | --- | --- |
| 1 | 5 | 150 | 70.4% |
| 2 | 10 | 27 | 12.7% |
| 3 | 15 | 11 | 5.2% |
| 4 | 20 | 4 | 1.9% |
| 5-119 (10 distinct values) | 25-595 | 21 | 9.8% |

Route-volume buckets (full-archive, in-sample fit — every route's own
completed-episode count over all 53 days vs. CRPS skill on its own
episodes):

| bucket (own episodes) | n routes | n episodes | mean CRPS | baseline CRPS | skill |
| --- | --- | --- | --- | --- | --- |
| 1-2 | 4 | 5 | 0.89 | 0.00 | undefined (zero-variance baseline, n too small) |
| 3-5 | 5 | 19 | 17.31 | 15.87 | -0.090 |
| 6-10 | 4 | 37 | 8.27 | 7.67 | -0.078 |
| 11+ | 7 | 152 | 7.34 | 7.05 | -0.040 |

Training-window sweep — fit on the first N days (2026-06-21 onward), grade
on the SAME fixed held-out window every time (2026-07-30..08-12, n=63
episodes, held fixed so N is the only thing that changes):

| N days | train window ends | n train episodes | routes w/ cell (own/pooled) | n scored | skill |
| --- | --- | --- | --- | --- | --- |
| 7 | 06-28 | 29 | 10 (3/7) | 51 | -0.243 |
| 14 | 07-05 | 77 | 13 (8/5) | 55 | -0.150 |
| 21 | 07-12 | 90 | 16 (9/7) | 62 | -0.142 |
| 28 | 07-19 | 118 | 20 (10/10) | 63 | -0.170 |
| 35 | 07-26 | 138 | 20 (14/6) | 63 | -0.164 |

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

| model | n scored | n no-curve | mean CRPS | baseline CRPS | skill |
| --- | --- | --- | --- | --- | --- |
| a. shipped continuous log-logistic AFT (`partially_pooled_dwell`) | 97 | 33 | 6.92 | 6.36 | -0.088 |
| b. per-route empirical/KM (`compute_dwell_quantiles`, min_samples=5) | 25 | 105 | 5.45 | 3.82 | -0.426 |
| c. pooled climatology (one KM curve over all routes' pooled samples) | 97 | 33 | 7.15 | 6.36 | -0.126 |

Restricted to the n=25 intersection every model can score (apples-to-apples,
same episodes, same baseline):

| model | mean CRPS | baseline CRPS | skill |
| --- | --- | --- | --- |
| a. shipped continuous | 4.42 | 3.82 | -0.157 |
| b. per-route KM | 5.45 | 3.82 | -0.426 |
| c. pooled climatology | 4.68 | 3.82 | -0.224 |

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

| PIT lobe | n | share | composition |
| --- | --- | --- | --- |
| [0.2,0.3) | 38 | 50.0% | **100% one-tick**, 100% own-fit routes with n=5 own events (6, G, R) |
| [0.4,0.5) | 14 | 18.4% | **100% one-tick**, 100% pooled-parent routes with n=1-2 own events (J, M, W) |
| [0.8,1.0) | 22 | 28.9% | **100% multi-tick** (median 12.5min, max 95min), mix of own (15) and pooled (7) routes, all 6 routes represented |
| other (<8min gap) | 2 | 2.6% | route 6, multi-tick, own-fit |

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

| split | n routes | n scored | mean CRPS | baseline CRPS | skill |
| --- | --- | --- | --- | --- | --- |
| own-fit | 3 | 55 | 4.98 | 4.53 | -0.099 |
| pooled-parent | 3 | 21 | 5.68 | 5.01 | -0.133 |

Cross-checked on the independent local fit (8 own-fit routes incl. n=32/15
event routes 7/M, 5 pooled-parent routes at n=1 each):

| split | n routes | n scored | skill |
| --- | --- | --- | --- |
| own-fit | 8 | 60 | -0.094 |
| pooled-parent | 5 | 37 | -0.079 |

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

| model | mean CRPS | baseline CRPS | skill |
| --- | --- | --- | --- |
| a. shipped continuous (closed-form check, same math as the curve version) | 7.04 | 6.36 | -0.108 |
| d. one-tick spike + conditional continuous tail mixture | 6.50 | 6.36 | **-0.023** |

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

| encoding | 20px monochrome result |
| --- | --- |
| gap 0 / 4 / 8 units on a 24 grid (spacing only) | fails — disrupted vs suspended indistinguishable |
| gap 0 / 5 / 11 (spacing widened to the frame) | marginal, needs a side-by-side reference |
| split topology + symmetric height drop | passes, read correctly cold |

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

| model | CRPS | baseline | skill | mean PIT |
| --- | --- | --- | --- | --- |
| continuous (what shipped before) | 7.451 | 6.541 | -0.1391 | 0.579 |
| single global atom | 6.922 | 6.541 | **-0.0582** | 0.545 |
| per-route shrunk atom (shipped) | 6.938 | 6.541 | **-0.0606** | 0.544 |

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

| model | PIT histogram | mean PIT |
| --- | --- | --- |
| continuous | [0,1,7,15,25,0,1,1,8,16] | 0.579 |
| mixture, raw PIT | [0,0,0,0,0,0,0,48,13,13] | 0.790 |
| mixture, randomized PIT | [8,8,4,8,6,6,6,2,13,13] | 0.544 |

The middle row is not miscalibration. PIT is only uniform for a *continuous*
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
is *deterministically* correct. Gated at the single point where the Inference is
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

| predictor | Brier | logloss | vs pooled |
| --- | --- | --- | --- |
| pooled (what ships) | 0.06519 | 0.23645 | |
| raw leaf rate | **0.06414** | 0.23453 | -0.00105 |
| route level | 0.18876 | 0.54609 | +0.12357 |
| system level | 0.24298 | 0.67903 | +0.17779 |
| pooled, fitted on normal ticks only | 0.06505 | 0.23640 | -0.00014 |

Restricted to the 497 THIN training leaves (n_A < MIN_LEAF_N=20) — the leaves
partial pooling exists to serve:

| predictor | Brier | logloss | vs pooled |
| --- | --- | --- | --- |
| pooled | 0.12451 | 0.43691 | |
| raw leaf rate | **0.06908** | 0.36334 | -0.05543 |
| system level | 0.19412 | 0.58022 | +0.06960 |

So: pooling is a small net loss overall and a **1.8x Brier loss exactly where it
is supposed to help**. It does beat route- and system-level pooling by a wide
margin (0.065 vs 0.189 and 0.243), so leaf specificity is doing real work — the
hierarchy *above* the leaf is nearly worthless here. It is the direction of the
shrinkage that is wrong, not the idea of a leaf-level estimate.

### Why: exchangeability is violated, and not in the direction I guessed

| | held-out advance rate |
| --- | --- |
| thin leaves (n_A < 20) | 0.8532 |
| voting leaves (n_A >= 20) | 0.5830 |
| centre thin leaves are shrunk toward | **0.9888** |

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

| stops covered in one 5-min look | count | share of moves |
| --- | --- | --- |
| 1 (an actual segment) | 90,159 | **18.2%** |
| 2 | 158,975 | 32.2% |
| 3 | 192,192 | 38.9% |
| 4 | 42,354 | 8.6% |
| 5+ | 8,612 | 1.2% |

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

| position in the line | stalls | moves | stall rate | share of all stalls |
| --- | --- | --- | --- | --- |
| first stop (origin terminal) | 244,009 | 28,261 | **89.6%** | **63.5%** |
| destination terminal | 12,419 | 2,434 | 83.6% | 3.2% |
| one in from a terminal | 6,324 | 19,611 | 24.4% | 1.6% |
| mid-line | 96,273 | 474,532 | **16.9%** | 25.1% |
| not in a scheduled chain | 24,953 | | | 6.5% |

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

| score | AUC |
| --- | --- |
| advance rate, as production computes it (all stops) | 0.2175 |
| advance rate, terminals excluded | 0.3899 |
| progress ratio, terminals excluded | 0.4229 |

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

  * **4 acute onsets in the entire week, across 4 routes**, giving 24 positive
    route-ticks out of 36,297.
  * AUC 0.6025 progress ratio vs 0.5744 advance rate. Both now the right side of
    0.5, and the progress ratio's mean is 0.0391 LOWER before an onset while the
    advance rate moves +0.0019 (i.e. not at all).
  * With 24 positives the CI on AUC is about +/-0.12. This is a direction, not a
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

| position | stalls | moves | stall rate | share of all stalls |
| --- | --- | --- | --- | --- |
| chain endpoint | 294,928 | 36,316 | **89.0%** | **76.8%** |
| mid-line (through) | 64,103 | 488,522 | 11.6% | 16.7% |
| not in the skeleton at all | 24,947 | 7,090 | **77.9%** | 6.5% |

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

| arm | leaves | no train data | thin | pooled Brier | thin-leaf pooled | thin-leaf raw |
| --- | --- | --- | --- | --- | --- | --- |
| all stops (production) | 2291 | 177 | 448 | 0.06656 | 0.14857 | 0.05506 |
| terminals excluded | 2107 | 174 | 436 | 0.06034 | 0.11144 | 0.05769 |
| through stops only | 1617 | **0** | **69** | 0.06208 | **0.08082** | 0.07788 |

Read the last three columns, not the fourth: overall Brier is dominated by the
largest leaves and each arm scores a different population, so cross-arm overall
comparison is not like-for-like (terminals-excluded "wins" it at 0.06034 while
being worse everywhere the estimator is actually load-bearing). What is
like-for-like:

  * **Every held-out leaf has training data** under through-only — 177 leaves
    that previously needed a fabricated prior are gone, because they were
    off-skeleton stops that appear in one window and not the next.
  * Thin leaves fall 448 -> 69 and their Brier 0.14857 -> 0.08082.
  * Pooled and raw converge (0.06208 vs 0.06195, and 0.08082 vs 0.07788 on thin
    leaves). The exchangeability violation measured in the earlier grade —
    pooling *losing* to raw by 1.8x on the leaves it exists to serve — was mostly
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

| arm | AUC | mean score on normal | on disrupted | overall advance rate |
| --- | --- | --- | --- | --- |
| all stops | 0.3388 | 0.5619 | 0.7215 | 0.5807 |
| terminals excluded | 0.4361 | 0.8217 | 0.9112 | 0.8470 |
| through stops only | 0.4641 | 0.8673 | 0.9099 | 0.8832 |

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

| outcome | n | share |
| --- | --- | --- |
| exact single hop | 3,457 | 91.9% |
| interval-censored (arrival missed between polls) | 215 | 5.7% |
| right-censored (in transit when last seen) | 89 | 2.4% |
| dropped: stop_seq backwards | 11 | |
| dropped: arrival with no stop_seq | 82 | |

**92% of hops are cleanly measured at 1-minute polling.** That is the number the
whole trace was built for: at 5-minute polling the mean observed move spanned
2.71 stations, so nothing below the multi-station jump was measurable at all.

### Two bounds, not one measurement

A poll brackets a hop rather than pinning it, and the two ends are biased in
opposite directions:

| cut | n | ratio to schedule (median) | p10 | p90 |
| --- | --- | --- | --- | --- |
| arrival -> arrival | 3,420 | **1.033** | 0.667 | 1.667 |
| departure -> arrival | 797 | 0.422 | 0.189 | 0.833 |

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

  * **Right-censoring needs in-transit evidence, not just absence.** A trip last
    seen standing at a stop FURTHER ALONG than its last recorded arrival has
    already completed the hop — we simply could not time it (the feed omitted
    stop_seq). Recording "still running at T" there is false, not unverified,
    and stretches every fitted traversal. The rule that holds: censor only when
    the last sighting is in transit toward a stop other than the last arrival.
  * **A departure observed at the same instant as the next arrival is not a
    departure.** It bounds travel time below by zero, which is no information.
    Emitting 0 there would drag any fit down hard; those records carry
    moving_seconds=None instead, and 2,443 of 3,457 exact hops are in that state.
  * A feed gap after the last in-transit sighting does not weaken the censoring
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
a *power* of a ratio and has its own range. Flooring the input does not bound
the intermediate.

### 2. Every local retrain since 2026-08-10 stamped a stale `code_sha`

The published params claimed `code_sha: fa513e5`, a commit from 2026-07-17,
with `dirty: null` — for a run that produced a feature that commit predates.
The previous live params said the same thing.

`.build-sha` is a gitignored file that `trainer deploy:ci` writes into the
checkout before the container build, because the image excludes `.git`.
`code_provenance()` consulted it *above* `git rev-parse HEAD`, so once a local
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

| UTC hour | ET | prevalence, tod_bin | prevalence, hourly |
| --- | --- | --- | --- |
| 10 | 06:00 | **26.8%** | 5.9% |
| 03 | 23:00 | **18.9%** | 2.6% |
| others | | 4.6-14.7% | 0.2-8.0% |

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

| bin | cells | judgeable ticks | events | degraded | top-2 ET hours |
| --- | --- | --- | --- | --- | --- |
| tod_bin (4-6h) | 131 | 69,636 | 210 | 11.00% | 42% |
| hourly | 1,122 | 69,546 | 142 | 1.79% | 30% |
| half-hour | 1,197 | 55,963 | 66 | 1.22% | 26% |
| quarter-hour | 2,364 | 55,621 | 54 | 0.90% | 26% |

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

| | |
| --- | --- |
| exact single hops fitted | 74,706 |
| distinct hops seen | 1,924 |
| hops fitted on their own data | **1,338** |
| hops anchored on their scheduled time | 257 |
| hops omitted, thin and unscheduled | 329 |
| median own-fit hop | 109.1 s |
| median p90/median within a hop | 1.351 |
| median observed/scheduled | 1.056 |

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

| | median of tick fractions | pooled advanced/matched |
| --- | --- | --- |
| median cell | 0.9990 | **0.9443** |

p0 − true rate is +0.033 at the median and +0.135 at p90, and **73 of 210 cells
publish p0 = 0.9990 for a real rate between 0.80 and 0.95**. That matters
because `_binom_lower_tail(advanced, matched, p0)` is a significance test and
takes p0 as a Bernoulli rate: at 0.999 with 10 matched trips a single stall
reads significant, in a cell where one stall in ten is ordinary.

So it was fixed offline — pooled advanced/matched, with a 0/10/20% one-sided
trim to keep outage ticks from dragging the rate down. The saturation goes away
(median p0 0.9990 -> 0.9443, cells at/above 0.99 drop from 74.3% to 8.6%). The
trip-wire does not improve:

| baseline | ratio | fired | agree | disagree | agreement rate |
| --- | --- | --- | --- | --- | --- |
| median (shipped) | 0.5 | 7 | 0 | 7 | 0.000 |
| pooled, trim 0 | 0.5 | 7 | 0 | 7 | 0.000 |
| pooled, trim 0 | 0.7 | 896 | 3 | 893 | **0.003** |
| pooled, trim 0.1 | 0.7 | 1,129 | 3 | 1,126 | 0.003 |
| pooled, trim 0.2 | 0.7 | 1,421 | 3 | 1,418 | 0.002 |

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