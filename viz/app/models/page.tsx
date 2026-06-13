"use client";

import { useCallback, useEffect, useState } from "react";
import Nav from "../Nav";
import {
  ReliabilityChart,
  RecoveryScatter,
  Swimlane,
  TransitionHeatmap,
  type ReliabilityResult,
  type RecoveryResult,
  type TimelineDTO,
} from "./charts";
import type { GradingResponse, HeatmapEntry } from "@/lib/types";

const DAY_OPTIONS = [1, 3, 7, 14];

export default function ModelsPage() {
  const [days, setDays] = useState(3);
  const [route, setRoute] = useState("");
  const [data, setData] = useState<GradingResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

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

  useEffect(() => {
    load();
  }, [load]);

  const rel = (data?.reliability ?? []) as ReliabilityResult[];
  const rec = data?.recovery as RecoveryResult | undefined;
  const timelines = (data?.timelines ?? []) as TimelineDTO[];
  const heatmap = (data?.heatmap ?? []) as HeatmapEntry[];
  const states = data?.states ?? ["normal", "disrupted", "suspended"];

  return (
    <div className="wrap">
      <div className="topbar">
        <h1>Momentarily</h1>
        <Nav />
      </div>
      <div className="sub">
        Model trust &amp; calibration · ground truth from the regime-transition
        stream (when lines actually recovered)
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
          <select value={route} onChange={(e) => setRoute(e.target.value)}>
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
            {data.counts.transitionRecords.toLocaleString()} transitions ·{" "}
            {data.counts.predictionFiles + data.counts.transitionFiles} files
            {data.counts.pointsCapped && " · scatter downsampled"}
          </span>
        )}
      </div>

      {data && !data.configured && (
        <div className="warnbox" style={{ maxWidth: 640 }}>
          <strong>R2 credentials not configured.</strong> Phase B reads the
          prediction/transition history directly from R2. Create{" "}
          <code>viz/.env.local</code> with:
          <pre style={{ margin: "8px 0 0", whiteSpace: "pre-wrap" }}>
{`R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...`}
          </pre>
          (Mint an R2 API token in the Cloudflare dashboard → R2 → Manage API
          Tokens. The Workers deploy token can&apos;t read objects.)
        </div>
      )}

      {err && data?.configured && <div className="error">Error: {err}</div>}

      {data?.configured && !err && (
        <>
          <h3 className="grp">Recovery-forecast reliability</h3>
          <p className="grp-note">
            When the model said “P(normal within H minutes) = x”, the line
            actually recovered that fast in the plotted fraction of cases. Points
            on the dashed diagonal are perfectly calibrated; size ∝ sample count.
          </p>
          <div className="charts-row">
            {rel.map((r) => (
              <ReliabilityChart key={r.horizonMin} result={r} />
            ))}
          </div>

          <h3 className="grp">Recovery time: predicted vs actual</h3>
          {rec && <RecoveryScatter result={rec} />}

          <h3 className="grp">Regime timeline vs reality</h3>
          <p className="grp-note">
            Each line&apos;s inferred regime over the window. Top 14 lines by
            non-normal time.
          </p>
          <Swimlane timelines={timelines} />

          <h3 className="grp">
            Learned transition matrices
            {data.paramsTrainedAt
              ? ` · trained ${new Date(data.paramsTrainedAt * 1000).toLocaleDateString()}`
              : ""}
          </h3>
          {heatmap.length === 0 ? (
            <div className="muted">No trained params available yet.</div>
          ) : (
            <div className="small-multiples">
              {heatmap.map((h) => (
                <TransitionHeatmap
                  key={h.route}
                  route={h.route}
                  transition={h.transition}
                  states={states}
                />
              ))}
            </div>
          )}
        </>
      )}
    </div>
  );
}
