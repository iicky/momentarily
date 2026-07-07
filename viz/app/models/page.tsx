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
  TransitionHeatmaps,
  MovementConfusion,
  AdvanceBaselineChart,
  RecoveryDistCurve,
  RecoveryScoreCard,
  type ReliabilityResult,
  type RecoveryResult,
  type AggregateRecovery,
  type ResumeChurnResult,
  type AdherenceResult,
  type DetectionLatencyResult,
  type DriftResult,
  type TimelineDTO,
  type MovementConfusionResult,
  type RouteBaselineDTO,
  type RecoveryDistResult,
} from "./charts";
import { ChartMetaProvider } from "./ChartFrame";
import type { GradingResponse, HeatmapEntry } from "@/lib/types";

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

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const qs = new URLSearchParams({ days: String(days) });
      if (route) qs.set("route", route);
      const res = await fetch(`/api/grading?${qs}`);
      const json = (await res.json()) as GradingResponse;
      setData(json);
      if (json.error) setErr(json.error);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [days, route]);

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
    loadMovement();
  }, [load, loadMovement]);

  const aggregate = data?.source === "calibration";
  const rel = (data?.reliability ?? []) as ReliabilityResult[];
  const rec = data?.recovery as RecoveryResult | undefined;
  const recAgg = data?.recovery as AggregateRecovery | undefined;
  const churn = data?.resumeChurn as ResumeChurnResult | undefined;
  const adher = data?.adherence as AdherenceResult | undefined;
  const detection = data?.detectionLatency as DetectionLatencyResult | undefined;
  const drift = data?.drift as DriftResult | undefined;
  const timelines = (data?.timelines ?? []) as TimelineDTO[];
  const recoveryDist = data?.recoveryDist as RecoveryDistResult | undefined;
  const heatmap = (data?.heatmap ?? []) as HeatmapEntry[];
  const states = data?.states ?? ["normal", "disrupted", "suspended"];

  return (
    <div className="wrap">
      <div className="topbar">
        <h1>Momentarily</h1>
        <Nav />
      </div>
      <div className="sub">
        How well the model&apos;s calls hold up. Most panels grade against the
        model&apos;s own published-condition stream (self-consistency, not independent
        ground truth); the train-movement panels are the independent cross-check.
        Each chart&apos;s footer tags its truth source.
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
            disabled={aggregate}
            title={aggregate ? "Per-line filtering needs R2 credentials" : undefined}
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
        {data?.counts && (
          <span className="counts">
            {data.counts.predictionRecords.toLocaleString()} predictions ·{" "}
            {data.counts.transitionRecords.toLocaleString()} transitions
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
          project&apos;s murk vault. Launch with <code>npm run dev</code> (it
          sources <code>../.env</code> and wraps Next in <code>murk exec</code>).
          If you still see this, make sure your murk key is set —{" "}
          <code>source .env</code> at the repo root (sets{" "}
          <code>MURK_KEY_FILE</code>), or run <code>direnv allow</code>.
        </div>
      )}

      {err && data?.configured && <div className="error">Error: {err}</div>}

      {aggregate && !err && (
        <div className="warnbox" style={{ maxWidth: 640 }}>
          <strong>Public aggregate feed.</strong> Reading{" "}
          <code>v1/calibration.json</code> — no R2 credentials needed. The
          window-aggregate reliability, recovery, and transition charts are
          shown below. Per-point drilldowns (recovery scatter, detection
          latency, schedule reliability, regime swimlane, per-line filtering)
          need the credentialed stream history — launch with{" "}
          <code>npm run dev</code> to see them.
        </div>
      )}

      {data?.configured && !err && (
        <ChartMetaProvider
          value={{
            feed: aggregate ? "public" : "credentialed",
            window: `${days}d`,
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
                dead-simple baseline that just guesses the average every time.
                Beating that baseline is the bar to clear.
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
                are spot-on; bigger dots mean more cases. (For the richer
                curve-and-score view, launch with <code>npm run dev</code>.)
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
                <RecoverySummary result={recAgg} />
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
            </>
          )}

          <h3 className="grp">
            A second opinion: trains vs. alerts
            {movement?.counts &&
              ` · ${movement.counts.judgeableTicks.toLocaleString()} judgeable ticks · ${movement.counts.vehicleTicks.toLocaleString()} vehicle snapshots`}
          </h3>
          <p className="grp-note">
            A second opinion from the trains themselves. The status we publish comes
            from the alerts feed; this compares it to where trains are actually
            moving. They&apos;re different signals, so the spots where they disagree are
            the interesting ones — and movement is the read we&apos;re hoping to lean on
            for &ldquo;is this line stuck right now?&rdquo;
          </p>
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
          />
        </ChartMetaProvider>
      )}
    </div>
  );
}
