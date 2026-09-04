<!--
Go/no-go memo template for the HMM shadow-validation review (the
graduation decision). After running `python -m training.review`, copy this into
that run's docs/review/<date>-shadow-hmm/ dir as memo.md and fill every <…> from
its summary.json + the 4 PNGs. NOTE docs/review/ is gitignored (local artifacts),
so record the FINAL decision in the graduation-tracking issue, as the earlier
review did. Delete this comment in the copy.

Decision rule of thumb: the headline is the event-based scorecard (§1) — graded per
incident episode. Graduate a field only if it (a) detects real incidents at an
acceptable rate, (b) scores positive recovery CRPS skill vs a CAUSAL duration-
climatology baseline — one fitted on a window BEFORE the graded episodes — on enough
episodes, and (c) is not dominated by false alarms the independent movement truth also
disputes. On (b), read the causal column only. The hindsight baseline (the graded
window's own duration CDF, reported as `oracle_skill`) is not clearable by an honest
forecast: a causally-fitted climatology loses to it by -0.1157, so any gate stated
against it fails forecasts for being causal rather than for being wrong. Oracle skill
is retained for comparability with pre-2026-09 numbers and never decides. NOTE the
review feed publishes `causal_skill` null today (it is handed one window of truth
episodes, with no pre-window truth to fit on), so this gate cannot be read off the
review summary until that column is wired — an episode-level go call needs the causal
number from elsewhere, or it waits. DETECTION COVERAGE gates; onset LATENCY does not — the
alert feed is coincident-to-lagging by construction, so the model cannot lead it and
is not penalized for that, and the latency figures in §1 are reported for shape only.
Changepoint alignment (§3) is likewise reported, not gated. Tick-level Brier vs
persistence (Appendix) is a degenerate yardstick on the sticky severe-only truth —
persistence is near-optimal on the no-event mass — so it no longer drives the decision.
-->

# HMM shadow validation — go/no-go memo

- **Date:** &lt;YYYY-MM-DD&gt;
- **Decision:** &lt;GO | NO-GO | PARTIAL — which fields graduate&gt;
- **Params under review:** `trained_at = <summary.current_params.trained_at>` (must match the params version being graduated — if not, the segment is the wrong model)
- **Code:** `code_sha = <provenance>` · **Window:** `<window.start>..<window.end>` (`<days>` days)
- **Artifacts:** `reliability.png`, `confusion.png`, `changepoint_alignment.png`, `recovery_by_route.png`, `summary.json` (this dir)

## 0. Data sufficiency (gate before reading anything else)

| Check | Value | Enough? |
| --- | --- | --- |
| `current_params.n_predictions` (this model only) | &lt;n&gt; | &lt;y/n&gt; |
| `recovery.overall.n` (graded recovery ticks) | &lt;n&gt; | &lt;y/n&gt; |
| `recovery.per_regime.n` (distinct regimes) | &lt;n&gt; | &lt;y/n&gt; |
| `recovery_clearance.n_disruptions` | &lt;n&gt; | &lt;y/n&gt; |
| `recovery_independent` present? (~06-28) | &lt;yes/null&gt; | — |
| `episode_scorecard.n_truth_episodes` (incidents in window) | &lt;n&gt; | &lt;y/n&gt; |
| `episode_scorecard.recovery.n_scored` (uncensored, curve-backed) | &lt;n&gt; | &lt;y/n&gt; |

If these are thin, the rest is directional only — say so and don't graduate on it.

## 1. Event-based scorecard — the headline (`summary.episode_scorecard`)

Graded per incident episode (severe-only truth), not per tick. Report the event
count beside every number; a metric over a handful of episodes is directional only.

**Onset detection** (`episode_scorecard.onset_latency`)
- Detected `<n_detected>/<n_episodes>` (`<detection_rate>`), missed `<n_missed>`
- Median onset latency `<median_latency_min>` min (sign: + = model lags the truth onset); mean `<mean_latency_min>`
- Alerts are coincident-to-lagging by construction, so latency is reported, not gated.

**Recovery as a distribution** (`episode_scorecard.recovery`)
- Scored `<n_scored>` uncensored episodes (excluded: `<n_censored_excluded>` censored, `<n_no_curve>` no dwell curve)
- CRPS `<report.mean_crps>` min
- Skill vs a CAUSAL climatology (fitted before the graded window) — the gate:
  `<report.causal_skill>` per-tick / `<report.per_regime.causal_skill>` per-episode,
  baseline CRPS `<report.causal_baseline_crps>`. Null means no pre-window fit was
  supplied — record it as null, never substitute the oracle column.
- Skill vs the HINDSIGHT baseline (this window's own duration CDF) — reported, never
  decisive: `<report.oracle_skill>` / `<report.per_regime.oracle_skill>`, baseline CRPS
  `<report.oracle_baseline_crps>`. Kept for comparability with pre-2026-09 numbers.
- PIT mean `<report.mean_pit>` (&lt;0.5 pessimistic, &gt;0.5 optimistic); verdict `<verdict.label>` — `<verdict.detail>`
- **Key question:** positive CRPS skill vs the CAUSAL climatology on enough episodes?
  If `causal_skill` is null here, the answer is "not established", not "no".

**False alarms** (`episode_scorecard.false_alarms`)
- `<n_false_alarm>/<n_model_episodes>` model episodes had no truth counterpart (`<false_alarm_rate>`)
- Movement contradicts `<movement_contradicted>` (genuine over-calls), confirms `<movement_confirmed>` (alert-truth gaps), unjudgeable `<movement_unjudgeable>`

## 2. Regime confusion (`confusion.png`, `summary.confusion`)

HMM `condition` vs MTA-derived state (row-normalized):

- normal precision/recall: &lt;…&gt;
- disrupted precision/recall: &lt;…&gt;
- suspended precision/recall: &lt;…&gt;
- Worst confusion cell and whether it matters for a consumer: &lt;…&gt;

## 3. Changepoint alignment (`changepoint_alignment.png`, `summary.changepoint_alignment`)

Reported, not gated.

- Matched within ±30 min: `<n_matched>/<n_total>` (`<pct>`)
- Median |delta|: &lt;…&gt; min · mean delta: &lt;…&gt; min (sign = lead/lag)
- Reverse detection recall — severe-truth changes with a nearby filter transition: `<n_truth_matched>/<n_truth_changes>`
- **How to read it:** severe-truth changepoints are an order of magnitude sparser
  than filter transitions, so a low forward match rate is largely arithmetic, and
  widening the window barely moves it. The diagnosed levers are severity-matching
  the truth and applying the presence mask to it, not window or hysteresis tuning.
  Read the reverse recall line, and treat a forward rate near zero as a property of
  the metric rather than evidence against the filter.

## 4. Recovery accuracy — three truths side by side

Supporting detail for §1. Grade `recovery_minutes` against each independent-ness tier;
the graduation decision itself rides on the per-episode CRPS/PIT read in §1.

| Truth source | n | MAE (min) | RMSE (min) | IQR coverage (target ≈0.5) |
| --- | --- | --- | --- | --- |
| Argmax / self (`recovery`, per-tick) | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |
| Argmax / self (`recovery.per_regime`) | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |
| **Alert-feed clearance** (`recovery_clearance`) | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |
| **Trip-updates service** (`recovery_independent`) | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |

- Do the independent truths agree with the self metric, or is the self metric flattered by grading against its own argmax? &lt;…&gt;
- IQR coverage near 0.5 = honest intervals; far below = overconfident. &lt;…&gt;
- Per-route / per-alert-type outliers worth flagging (`recovery.by_route`, `by_alert_type`): &lt;…&gt;

## 5. Decision

**&lt;GO | NO-GO | PARTIAL&gt;**

Graduate (move from shadow `inference` to a stable published surface):
- `condition`: &lt;graduate / hold&gt; — because &lt;…&gt;
- `recovery_minutes` (+ low/high band): &lt;graduate / hold&gt; — because &lt;…&gt;
- `p_normal_in_H`: shadow-only by standing decision — do not re-litigate here. The
  `normal_now` backtest stratum settled it (Brier 0.877 geom / 0.886 KM-residual vs
  0.0009 persistence at H30, n=21,071; same shape at H60/H120), and re-scoping needs a
  different primitive cleared against its own gate, not another review pass.

Gating rationale tying back to §1–4: &lt;…&gt;

## 6. Caveats & follow-ups

- Known data-coverage limits (e.g. route 7 only ~8h/day in trip-updates; thin routes): &lt;…&gt;
- trip-updates truth matures ~2026-06-28 — re-read §4 then if it was null here.
- Follow-ups filed: &lt;…&gt;

## Appendix: tick-level calibration (secondary — the headline is §1)

Retained for continuity. On the sticky severe-only truth, tick Brier vs persistence
is a degenerate yardstick (persistence is near-optimal on the no-event mass), so it
no longer drives the decision. Read `summary.calibration` / `reliability.png`:

| Horizon | Brier | BSS vs persistence | BSS vs climatology |
| --- | --- | --- | --- |
| 30 min | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |
| 60 min | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |
| 120 min | &lt;…&gt; | &lt;…&gt; | &lt;…&gt; |
