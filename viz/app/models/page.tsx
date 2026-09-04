"use client";

import { useCallback, useEffect, useState } from "react";
import Nav from "../Nav";
import {
  ReliabilityChart,
  RecoveryScatter,
  ErrorByLine,
  ErrorByElapsed,
  RecoverySummary,
  ResumeChurnPanel,
  AdherencePanel,
  DetectionLatencyPanel,
  DriftPanel,
  Swimlane,
  RegimeBandChart,
  TransitionHeatmaps,
  MovementConfusion,
  AdvanceBaselineChart,
  RecoveryDistCurve,
  RecoveryScoreCard,
  type ReliabilityResult,
  type RecoveryResult,
  type AggregateRecovery,
  type CurrentParamsRecovery,
  type ResumeChurnResult,
  type AdherenceResult,
  type DetectionLatencyResult,
  type DriftResult,
  type TimelineDTO,
  type MovementConfusionResult,
  type RouteBaselineDTO,
  type RecoveryDistResult,
} from "./charts";
import type { RegimeBands } from "@/lib/regime_band";
import { ChartMetaProvider, ChartErrorBoundary } from "./ChartFrame";
import type { GradingResponse, HeatmapEntry } from "@/lib/types";
import type { EpisodeSupport } from "@/lib/calibrationFeed";

interface MovementResponse {
  configured: boolean;
  error?: string;
  counts?: {
    vehicleTicks: number;
    predictionRecords: number;
    judgeableTicks: number;
  };
  confusion?: MovementConfusionResult;
  baselines?: RouteBaselineDTO[];
}

const DAY_OPTIONS = [1, 3, 7, 14];

export default function ModelsPage() {
  const [days, setDays] = useState(3);
  const [route, setRoute] = useState("");
  const [data, setData] = useState<GradingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [movement, setMovement] = useState<MovementResponse | null>(null);
  const [movLoading, setMovLoading] = useState(false);
  // "static" (default) reads the prebuilt feed — fast, line-filterable, works
  // with no R2. "streams" is the opt-in credentialed recompute that adds the
  // per-point drilldowns (scatter, swimlane, detection, movement).
  const [source, setSource] = useState<"static" | "streams">("static");

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const qs = new URLSearchParams({ days: String(days) });
      if (route) qs.set("route", route);
      if (source === "streams") qs.set("source", "streams");
      const res = await fetch(`/api/grading?${qs}`);
      const json = (await res.json()) as GradingResponse;
      setData(json);
      if (json.error) setErr(json.error);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [days, route, source]);

  const loadMovement = useCallback(async () => {
    setMovLoading(true);
    try {
      const qs = new URLSearchParams({ days: String(days) });
      if (route) qs.set("route", route);
      const res = await fetch(`/api/movement?${qs}`);
      setMovement((await res.json()) as MovementResponse);
    } catch (e) {
      setMovement({ configured: true, error: (e as Error).message });
    } finally {
      setMovLoading(false);
    }
  }, [days, route]);

  useEffect(() => {
    load();
    // Movement is a credentialed per-tick drilldown, not part of the fast
    // static view — only fetch it once the user opts into the R2 recompute.
    if (source === "streams") loadMovement();
    else setMovement(null);
  }, [load, loadMovement, source]);

  const aggregate = data?.source === "calibration";
  const rel = (data?.reliability ?? []) as ReliabilityResult[];
  const rec = data?.recovery as RecoveryResult | undefined;
  const recAgg = data?.recovery as AggregateRecovery | undefined;
  const churn = data?.resumeChurn as ResumeChurnResult | undefined;
  const adher = data?.adherence as AdherenceResult | undefined;
  const detection = data?.detectionLatency as DetectionLatencyResult | undefined;
  const drift = data?.drift as DriftResult | undefined;
  const timelines = (data?.timelines ?? []) as TimelineDTO[];
  const regimeBands = data?.regimeBands as RegimeBands | undefined;
  const recoveryDist = data?.recoveryDist as RecoveryDistResult | undefined;
  const heatmap = (data?.heatmap ?? []) as HeatmapEntry[];
  const states = data?.states ?? ["normal", "disrupted", "suspended"];
  const support = data?.episodeSupport as EpisodeSupport | undefined;
  const currentParams = data?.currentParams as CurrentParamsRecovery | undefined;

  return (
    <div className="wrap">
      <div className="topbar">
        <h1>Momentarily</h1>
        <Nav />
      </div>
      <div className="sub">
        How well the model&apos;s calls hold up. Panels graded against the model&apos;s
        own published-condition stream are self-consistency, not independent ground
        truth. Train movement is now the independent arm the recovery axis is graded
        against — its false-alarm rate (calling a moving line stuck) is bounded and
        monitored, but its miss rate can&apos;t be measured, because no archived
        movement truth exists to catch what it lets through. Each chart&apos;s footer
        tags its truth source.
      </div>

      <div className="controls">
        <label>
          Window
          <select value={days} onChange={(e) => setDays(Number(e.target.value))}>
            {DAY_OPTIONS.map((d) => (
              <option key={d} value={d}>
                {d}d
              </option>
            ))}
          </select>
        </label>
        <label>
          Line
          <select
            value={route}
            onChange={(e) => setRoute(e.target.value)}
            disabled={(data?.routes ?? []).length === 0}
          >
            <option value="">all</option>
            {(data?.routes ?? []).map((r) => (
              <option key={r} value={r}>
                {r}
              </option>
            ))}
          </select>
        </label>
        <button onClick={load} disabled={loading}>
          {loading ? "loading…" : "refresh"}
        </button>
        {source === "static" ? (
          <button
            onClick={() => setSource("streams")}
            disabled={loading}
            title="Recompute the per-point drilldowns (recovery scatter, regime swimlane, detection latency, movement) live from R2 — slower, needs credentials."
          >
            load drilldowns
          </button>
        ) : (
          <button onClick={() => setSource("static")} disabled={loading}>
            fast view
          </button>
        )}
        {data?.counts && (
          <span className="counts">
            {data.counts.predictionRecords.toLocaleString()} predictions ·{" "}
            {data.counts.transitionRecords.toLocaleString()} transitions
            {support && (
              <span
                className="support"
                title={
                  `${support.n_episodes.toLocaleString()} independent incidents on the ` +
                  `${support.graded_arm} arm. Every count to the left is 5-minute ticks, and ` +
                  `ticks inside one regime say almost the same thing — a route disrupted for two ` +
                  `hours is 24 rows and one incident. Six consecutive weeks of near-identical ` +
                  `tick counts carried 5 to 89 incidents, so this is the number that says how ` +
                  `much the window can actually settle.` +
                  (support.excluded_pre_arm_rows
                    ? ` ${support.excluded_pre_arm_rows.toLocaleString()} rows predate this arm and are excluded.`
                    : "")
                }
              >
                {" · "}
                <strong>{support.n_episodes.toLocaleString()} incidents</strong>{" "}
                of real support
              </span>
            )}
            {aggregate ? (
              " · public aggregate feed"
            ) : (
              <>
                {" "}
                · {data.counts.alertVersions.toLocaleString()} planned alerts ·{" "}
                {data.counts.predictionFiles +
                  data.counts.transitionFiles +
                  data.counts.alertFiles}{" "}
                files
                {data.counts.pointsCapped && " · scatter downsampled"}
                {data.counts.alertsCapped && " · alert archive capped"}
              </>
            )}
          </span>
        )}
      </div>

      {data && !data.configured && (
        <div className="warnbox" style={{ maxWidth: 640 }}>
          <strong>R2 credentials not available.</strong> Phase B reads the
          prediction/transition history from R2 using the R2_* secrets in the
          project&apos;s murk vault. Launch with <code>npm run dev</code>, which
          wraps Next in <code>murk exec</code>. In a fresh worktree run{" "}
          <code>murk-wt link</code> once — murk keys on the vault&apos;s path, so
          each worktree needs its own link. If a stale <code>MURK_KEY_FILE</code>
          {" "}is exported in your shell it overrides that link; unset it.
        </div>
      )}

      {err && data?.configured && <div className="error">Error: {err}</div>}

      {aggregate && !err && (
        <div className="warnbox" style={{ maxWidth: 640 }}>
          <strong>Fast view</strong> — prebuilt <code>v1/calibration.json</code>:
          reliability, recovery, transition, and drift, filterable by line, no R2
          recompute. The per-point drilldowns (recovery scatter, regime swimlane,
          detection latency, schedule reliability, movement) aren&apos;t in the
          feed — hit <em>load drilldowns</em> to recompute them from R2.
        </div>
      )}

      {data?.configured && !err && (
        <ChartErrorBoundary key={`${route}:${days}`} label="the charts">
        <ChartMetaProvider
          value={{
            feed: aggregate ? "public" : "credentialed",
            // The response's window, not the selector: on the static feed the
            // selector doesn't choose the data, the publisher already did.
            window: `${data.window.days}d`,
            generatedAt: data.generatedAt ?? null,
          }}
        >
          {recoveryDist && recoveryDist.n > 0 ? (
            <>
              <h3 className="grp">How good are the recovery time estimates?</h3>
              <p className="grp-note">
                Does the model guess recovery times well? On the left, the green
                line is how quickly lines returned to normal in our published status
                stream and the blue line is what the model expected ahead of time —
                when they sit on top of each other, it&apos;s nailing it. (Both come from
                the model&apos;s own condition calls, so this is self-consistency, not an
                independent check — see the footer tag.) The card on the right sums
                that up: one accuracy score (lower is better), compared against a
                dead-simple yardstick that just guesses the average recovery time.
                On this view that yardstick is built from the same window&apos;s
                realized recoveries — hindsight the model never had — so it is a
                reference point, not a bar a forecast can fairly clear.
              </p>
              <div className="charts-row">
                <RecoveryDistCurve result={recoveryDist} />
                <RecoveryScoreCard result={recoveryDist} />
              </div>
            </>
          ) : (
            <>
              <h3 className="grp">How good are the recovery time estimates?</h3>
              <p className="grp-note">
                When the model gave a line an x% chance of being back within a set
                time, did it actually come back that often? Dots on the dashed line
                are spot-on; bigger dots mean more cases.
              </p>
              <p className="grp-note">
                Each panel grades the <em>same</em> forecast twice, because
                &ldquo;did it come back?&rdquo; has two answers here. Green checks it
                against train movement — the thing this forecast is actually about.
                Amber checks it against the alert-driven filter&apos;s own label,
                which is a self-consistency check, not a verdict. They disagree
                sharply, so read each arm&apos;s own sample count and coverage chip
                before you believe either — disrupted cases are rare in any window,
                and movement can&apos;t judge every tick.
              </p>
              <p className="grp-note">
                One thing to read correctly: skill-vs-persistence runs sharply
                negative here by construction. The outcome is ~99%
                &ldquo;normal,&rdquo; so &ldquo;assume it stays as it is&rdquo; is a
                near-unbeatable baseline on Brier — a large negative is the metric,
                not a broken model. The AUC chip (does the forecast rank the right
                cases higher?) is the more informative read at this base rate.
                Nothing on this chart decides whether a field ships: that call is
                made per incident episode, against a typical-duration baseline built
                from history before the episodes it grades, not per tick against
                persistence.
              </p>
              <div className="charts-grid-3">
                {rel.map((r) => (
                  <ReliabilityChart key={r.horizonMin} result={r} />
                ))}
              </div>
            </>
          )}

          {drift && (
            <>
              <h3 className="grp">Is the feed starting to look unfamiliar?</h3>
              <p className="grp-note">
                An early-warning light. New alert wordings the model has never seen
                show up here first — usually before the forecasts visibly slip — and
                so do lines whose day-to-day pattern has drifted from what the model
                learned on. Catch these and you can fix things before they break.
              </p>
              <DriftPanel result={drift} trainedAt={data.paramsTrainedAt} />
            </>
          )}

          {aggregate ? (
            recAgg && (
              <>
                <h3 className="grp">Recovery accuracy</h3>
                <p className="grp-note">
                  Retrains land weekly and this window is four weeks wide, so the
                  pooled row averages three or four different models — a change in
                  it can be which models were in the mix rather than anything one
                  of them learned. The second row is the model running right now,
                  on its own predictions.
                </p>
                <RecoverySummary result={recAgg} currentParams={currentParams} />
              </>
            )
          ) : (
            <>
              <h3 className="grp">Every forecast vs. how it really went</h3>
              <p className="grp-note">
                One dot per forecast: where the model guessed (across) against how
                long recovery took in our published status (up). Close to the dashed
                line is good. It&apos;s a busy plot — the cleaner take on this same
                question is the green-and-blue view up top.
              </p>
              {rec && <RecoveryScatter result={rec} capped={data.counts?.pointsCapped} />}

              <h3 className="grp">Where do the misses come from?</h3>
              <p className="grp-note">
                The scatter above shows every miss at once; these split them two
                ways. On the left, which lines the model is worst at — a handful
                drag the average down. On the right, whether the guess tightens up
                the longer a line has already been stuck. One quirk worth knowing:
                when the model calls a very long outage (several hours), the line
                often comes back far sooner — those are the over-long forecasts
                piled at the right edge of the scatter.
              </p>
              {rec && (
                <div className="charts-row">
                  <ErrorByLine result={rec} />
                  <ErrorByElapsed result={rec} />
                </div>
              )}

              <h3 className="grp">How fast does the model notice?</h3>
              <p className="grp-note">
                The minutes between a real alert showing up and the line&apos;s status
                flipping to disrupted or suspended.
              </p>
              {detection && <DetectionLatencyPanel result={detection} />}

              <h3 className="grp">Does planned work run on schedule?</h3>
              <p className="grp-note">
                Planned work (think scheduled track maintenance) comes with an
                announced end time, so the model doesn&apos;t forecast it — we just check
                the schedule itself: do the announced windows hold, and do lines come
                back when promised?
              </p>
              <div className="charts-row">
                {churn && <ResumeChurnPanel result={churn} />}
                {adher && <AdherencePanel result={adher} />}
              </div>

              <h3 className="grp">Each line&apos;s status over time</h3>
              <p className="grp-note">
                How the model saw each line through the window. Showing the 14 lines
                that spent the most time away from normal.
              </p>
              <Swimlane timelines={timelines} />

              <h3 className="grp">How sure was it, minute to minute?</h3>
              <p className="grp-note">
                The same lines, but instead of the one label the model settled on,
                this is the confidence behind it: the three colours are the chances
                it gave normal, disrupted and suspended, stacked to fill each row.
                Solid green is a confident, quiet line. Where the colours mix, the
                model was hedging — and a band that shifts well before the block
                above changes colour is it seeing trouble coming. Breaks in a row
                are gaps in the archive, not calm.
              </p>
              {regimeBands && <RegimeBandChart result={regimeBands} />}
            </>
          )}

          <h3 className="grp">
            The independent arm: trains vs. alerts
            {movement?.counts &&
              ` · ${movement.counts.judgeableTicks.toLocaleString()} judgeable ticks · ${movement.counts.vehicleTicks.toLocaleString()} vehicle snapshots`}
          </h3>
          <p className="grp-note">
            Train movement is now the independent arm the recovery axis is graded
            against — no longer a side check. The status we publish now follows
            train movement; the alerts feed is the cross-reference this panel
            compares it against. The two are different signals, so a disagreement
            is the signal, not
            noise. One honest asymmetry: movement&apos;s false-alarm rate — calling a
            moving line stuck — is bounded and monitored, but how often it{" "}
            <em>misses</em> a stuck line the trains don&apos;t reveal is unmeasurable
            here, because no archived movement truth exists to grade misses against.
          </p>
          {source === "static" && (
            <div className="muted">
              Not in the fast feed — hit <em>load drilldowns</em> to compare the
              published status against live train movement.
            </div>
          )}
          {movLoading && !movement && <div className="muted">loading movement archive…</div>}
          {movement && !movement.configured && (
            <div className="warnbox" style={{ maxWidth: 640 }}>
              <strong>Movement archive needs R2 credentials.</strong> Launch with{" "}
              <code>npm run dev</code> so the vehicle archive
              (<code>archive/vehicles/…</code>) is readable.
            </div>
          )}
          {movement?.error && (
            <div className="error">Movement: {movement.error}</div>
          )}
          {movement?.configured && !movement.error && (
            <>
              {movement.confusion && movement.confusion.total > 0 ? (
                <div className="charts-row">
                  <MovementConfusion result={movement.confusion} />
                </div>
              ) : (
                <div className="muted">
                  No overlapping movement + prediction ticks in this window yet.
                </div>
              )}
              {movement.baselines && movement.baselines.length > 0 && (
                <AdvanceBaselineChart routes={movement.baselines} />
              )}
            </>
          )}

          <h3 className="grp">
            What the model learned about each line
            {data.paramsTrainedAt
              ? ` · trained ${new Date(data.paramsTrainedAt * 1000).toLocaleDateString()}`
              : ""}
          </h3>
          <p className="grp-note">
            How likely each line is to stay where it is or switch between normal,
            disrupted, and suspended — learned from its own history, read against the
            system-average line.
          </p>
          <TransitionHeatmaps
            entries={heatmap}
            states={states}
            trainedAt={data.paramsTrainedAt}
            selfLoopCap={data.paramsSelfLoopCap}
          />

          <h3 className="grp">The supply axis</h3>
          <p className="grp-note">
            Everything above grades the flow/condition model. Supply is the second,
            now load-bearing axis — it catches service collapses (missing trains)
            that the flow signal is structurally blind to. Its derivation and
            scorecard are reserved here; the numbers land with the next review
            regeneration so they come from the fresh scorecard rather than a stale
            one.
          </p>
          <div className="chart-reserved" role="note">
            <div className="chart-reserved-tag">panel reserved · lands next review</div>
            <div className="chart-reserved-title">
              Supply: assigned trains vs. the line&apos;s own baseline
            </div>
            <p>
              The derivation this panel will show, end to end:{" "}
              <code>assigned_n</code> → the median for that{" "}
              <code>(route, schedule_bin)</code> cell → their ratio →{" "}
              <strong>degrade</strong> under 0.5× / <strong>recover</strong> over
              0.8×, with a 2-tick debounce so a single thin scan doesn&apos;t flip the
              state.
            </p>
            <p>
              And the honesty rule that makes it trustworthy: a cell with too few
              nights of history <strong>abstains</strong> — it reads &ldquo;no
              reading&rdquo; rather than guessing — so a thin weekend cell can&apos;t
              manufacture a supply collapse. The monitored thresholds and per-line
              support counts render once the regenerated review supplies them.
            </p>
          </div>
        </ChartMetaProvider>
        </ChartErrorBoundary>
      )}
    </div>
  );
}
