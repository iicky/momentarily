"use client";

import { useState } from "react";
import {
  ChartFrame,
  useTooltip,
  FORECAST,
  REALIZED,
  AxisTitle,
  AxisTicks,
  GridLines,
  SkillChip,
} from "./ChartFrame";
import { recoveryVerdict } from "@/lib/recovery_dist";

// Hand-rolled SVG charts — these are bespoke scientific plots (calibration
// scatter on a diagonal, regime swimlane, transition heatmap) where a generic
// charting lib would fight us more than help.

export interface ReliabilityResult {
  horizonMin: number;
  bins: { p: number; predictedMean: number; observedFreq: number; n: number }[];
  brier: number;
  n: number;
  excludedSchedule: number;
  // Brier skill vs baselines, and the persistence decomposition by current
  // state — present on the public aggregate feed, absent on the credentialed
  // recompute (the reliability chart only renders on the feed path anyway).
  skillPersistence?: number | null;
  skillClimatology?: number | null;
  decomp?: {
    normalNow?: { n: number; bss: number | null };
    notNormalNow?: { n: number; bss: number | null };
  };
}

export interface RecoveryResult {
  points: {
    route: string;
    predictedMin: number;
    actualMin: number;
    inIqr: boolean;
    elapsedMin: number;
  }[];
  coverage: number;
  n: number;
  medianAbsErrorMin: number;
  excludedSchedule: number;
}

type RecoveryPoint = RecoveryResult["points"][number];

export interface ResumeChurnResult {
  windows: number;
  pushed: number;
  pulled: number;
  stable: number;
  pushedPct: number;
  pulledPct: number;
  pushMagnitudesMin: number[];
  byRoute: { route: string; windows: number; pushed: number }[];
  byAlertType: { alertType: string; windows: number; pushed: number }[];
}

export interface AdherenceResult {
  points: { route: string; resumeAt: number; actualNormalAt: number; errorMin: number }[];
  n: number;
  medianErrorMin: number;
  overrunPct: number;
  onTimePct: number;
  censored: number;
}

export interface DetectionLatencyResult {
  points: { route: string; alertType: string; latencyMin: number }[];
  n: number;
  missed: number;
  medianLatencyMin: number;
  byAlertType: { alertType: string; n: number; medianLatencyMin: number; missed: number }[];
}

export interface TimelineDTO {
  route: string;
  segments: { state: string; start: number; end: number }[];
  observedUntil: number;
}

export function stateColor(state: string): string {
  if (state === "normal") return "var(--normal)";
  if (state === "disrupted") return "var(--disrupted)";
  if (state === "suspended") return "var(--suspended)";
  return "var(--unknown)";
}

function quantile(sorted: number[], q: number): number {
  if (!sorted.length) return 0;
  const i = Math.min(sorted.length - 1, Math.floor(q * sorted.length));
  return sorted[i];
}

// Quantile of an unsorted array (sorts a copy). Returns NaN on empty input so
// callers can drop thin buckets rather than plotting a fabricated zero.
function quant(xs: number[], q: number): number {
  if (!xs.length) return NaN;
  const s = [...xs].sort((a, b) => a - b);
  const i = Math.min(s.length - 1, Math.max(0, Math.round(q * (s.length - 1))));
  return s[i];
}

interface Bin {
  lo: number;
  hi: number;
  center: number;
  pts: RecoveryPoint[];
}

// Equal-width buckets over [0, domainMax]. Points past domainMax are dropped (the
// caller sizes the domain to a high quantile), so a bin's range is exactly what
// it claims — no tail piled into the last bucket.
function binBy(
  points: RecoveryPoint[],
  value: (p: RecoveryPoint) => number,
  domainMax: number,
  nBins: number,
): Bin[] {
  const w = domainMax / nBins;
  const bins: Bin[] = Array.from({ length: nBins }, (_, i) => ({
    lo: i * w,
    hi: (i + 1) * w,
    center: (i + 0.5) * w,
    pts: [],
  }));
  for (const p of points) {
    const v = value(p);
    if (v < 0 || v > domainMax) continue;
    bins[Math.min(nBins - 1, Math.floor(v / w))].pts.push(p);
  }
  return bins;
}

// --- Reliability diagram ---

export function ReliabilityChart({ result }: { result: ReliabilityResult }) {
  const S = 220;
  const pad = 28;
  const sc = (v: number) => pad + v * (S - 2 * pad);
  const scY = (v: number) => S - pad - v * (S - 2 * pad);
  const maxN = Math.max(1, ...result.bins.map((b) => b.n));
  const hasSkill =
    result.skillPersistence !== undefined || result.skillClimatology !== undefined;
  const xn = result.decomp?.notNormalNow;
  const nn = result.decomp?.normalNow;

  return (
    <ChartFrame
      title={`P(normal within ${result.horizonMin}m)`}
      titleMeta={
        <>
          Brier {Number.isNaN(result.brier) ? "—" : result.brier.toFixed(3)}
        </>
      }
      meta={{
        source: "the model's own status stream",
        independent: false,
        unit: "per-forecast",
        n: result.n,
        excluded: result.excludedSchedule,
        extra: hasSkill ? (
          <>
            <SkillChip label="vs persistence" bss={result.skillPersistence} />
            <SkillChip label="vs climatology" bss={result.skillClimatology} />
          </>
        ) : undefined,
      }}
    >
      <svg viewBox={`0 0 ${S} ${S}`} width="100%" style={{ maxWidth: 260 }}>
        <rect x={pad} y={pad} width={S - 2 * pad} height={S - 2 * pad} fill="none" stroke="var(--border)" />
        {/* perfect-calibration diagonal */}
        <line x1={sc(0)} y1={scY(0)} x2={sc(1)} y2={scY(1)} stroke="var(--muted)" strokeDasharray="3 3" />
        {result.bins
          .filter((b) => b.n > 0)
          .map((b, i) => (
            <circle
              key={i}
              cx={sc(b.predictedMean)}
              cy={scY(b.observedFreq)}
              r={3 + 6 * Math.sqrt(b.n / maxN)}
              fill="var(--accent)"
              fillOpacity={0.75}
              stroke="var(--accent)"
            >
              <title>
                predicted {(b.predictedMean * 100).toFixed(0)}% · observed{" "}
                {(b.observedFreq * 100).toFixed(0)}% · n={b.n}
              </title>
            </circle>
          ))}
        <AxisTitle x={pad} y={S - 6} anchor="start">predicted →</AxisTitle>
        <AxisTitle x={6} y={pad + 4} anchor="start">observed ↑</AxisTitle>
      </svg>
      {(nn || xn) && (
        <div className="grp-note" style={{ marginTop: 6 }}>
          Skill by state at forecast time:{" "}
          {nn && (
            <>
              normal-now {fmtBss(nn.bss)} (n={nn.n})
            </>
          )}
          {nn && xn && " · "}
          {xn && (
            <>
              recovery {fmtBss(xn.bss)} (n={xn.n})
            </>
          )}
        </div>
      )}
    </ChartFrame>
  );
}

function fmtBss(bss: number | null): string {
  if (bss == null || Number.isNaN(bss)) return "—";
  return `${bss >= 0 ? "+" : ""}${(bss * 100).toFixed(0)}%`;
}

// --- Recovery scatter: predicted vs actual time-to-normal ---

export function RecoveryScatter({
  result,
  capped,
}: {
  result: RecoveryResult;
  capped?: boolean;
}) {
  const W = 460;
  const H = 300;
  const pad = 40;
  const vals = result.points.flatMap((p) => [p.predictedMin, p.actualMin]).sort((a, b) => a - b);
  const domainMax = Math.max(10, Math.ceil(quantile(vals, 0.97) / 10) * 10);
  const sx = (v: number) => pad + Math.min(1, v / domainMax) * (W - 2 * pad);
  const sy = (v: number) => H - pad - Math.min(1, v / domainMax) * (H - 2 * pad);

  return (
    <ChartFrame
      title="Predicted vs actual recovery"
      titleMeta={
        <>
          IQR coverage {Number.isNaN(result.coverage) ? "—" : (result.coverage * 100).toFixed(0)}%
          (target ~50%) · median abs err{" "}
          {Number.isNaN(result.medianAbsErrorMin) ? "—" : Math.round(result.medianAbsErrorMin)}m
        </>
      }
      meta={{
        source: "the model's own status stream",
        independent: false,
        unit: "per-forecast",
        n: result.n,
        excluded: result.excludedSchedule,
        note: capped ? "scatter downsampled" : undefined,
      }}
      legend={[
        { color: "var(--normal)", label: "inside IQR", shape: "dot" },
        { color: "var(--suspended)", label: "outside IQR", shape: "dot" },
      ]}
    >
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 520 }}>
        <rect x={pad} y={pad} width={W - 2 * pad} height={H - 2 * pad} fill="none" stroke="var(--border)" />
        <line x1={sx(0)} y1={sy(0)} x2={sx(domainMax)} y2={sy(domainMax)} stroke="var(--muted)" strokeDasharray="3 3" />
        {result.points.map((p, i) => (
          <circle
            key={i}
            cx={sx(p.predictedMin)}
            cy={sy(p.actualMin)}
            r={2.5}
            fill={p.inIqr ? "var(--normal)" : "var(--suspended)"}
            fillOpacity={0.5}
          >
            <title>
              {p.route}: predicted {Math.round(p.predictedMin)}m · actual{" "}
              {Math.round(p.actualMin)}m
            </title>
          </circle>
        ))}
        <AxisTitle x={W - pad} y={H - 8} anchor="end">predicted minutes →</AxisTitle>
        <AxisTitle x={8} y={pad + 4} anchor="start">actual ↑</AxisTitle>
        <AxisTicks axis="x" scale={sx} ticks={[domainMax]} at={sy(0) + 14} anchor="end" format={(v) => `${v}m`} />
      </svg>
    </ChartFrame>
  );
}

// --- Two conditional reads of the recovery scatter ---
// The CDF + PIT above answer "is it honest overall?" These keep a conditioning
// axis the marginal views throw away — which line, and how long it's been down —
// to show where the misses actually concentrate.

const recoveryMeta = (result: RecoveryResult) => ({
  source: "the model's own status stream",
  independent: false as const,
  unit: "per-forecast",
  n: result.n,
  excluded: result.excludedSchedule,
});

// 1 — Per-line error ranking: which lines drag the aggregate down? The one
// actionable cut — it names where to go look.
export function ErrorByLine({ result }: { result: RecoveryResult }) {
  const byRoute = new Map<string, RecoveryPoint[]>();
  for (const p of result.points) {
    const arr = byRoute.get(p.route);
    if (arr) arr.push(p);
    else byRoute.set(p.route, [p]);
  }
  const rows = [...byRoute.entries()]
    .map(([route, pts]) => ({
      route,
      n: pts.length,
      mae: quant(pts.map((p) => Math.abs(p.actualMin - p.predictedMin)), 0.5),
      cov: pts.filter((p) => p.inIqr).length / pts.length,
    }))
    .filter((r) => r.n >= 5)
    .sort((a, b) => b.mae - a.mae)
    .slice(0, 12);

  if (!rows.length)
    return (
      <ChartFrame title="Error by line" empty emptyText="No line has ≥5 graded forecasts in this window." />
    );

  const labelW = 36;
  const rightPad = 64;
  const W = 460;
  const plotW = W - labelW - rightPad;
  const rh = 22;
  const top = 8;
  const H = top + rows.length * rh + 8;
  const xMax = Math.max(15, Math.ceil(Math.max(...rows.map((r) => r.mae)) / 15) * 15);
  const x = (v: number) => labelW + (v / xMax) * plotW;

  return (
    <ChartFrame
      title="Which lines is it worst at?"
      titleMeta="typical miss, worst first"
      note={
        <>
          How far off the recovery guess typically is, per line — longest bar is
          the worst. <span style={{ color: "var(--suspended)" }}>Red</span> lines
          are the ones the model is both wrong about and overconfident on; a few
          lines drag the whole average down.
        </>
      }
      meta={recoveryMeta(result)}
    >
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 520 }}>
        <GridLines axis="x" scale={x} ticks={[0, xMax / 2, xMax]} from={top} to={top + rows.length * rh} />
        <AxisTicks axis="x" scale={x} ticks={[0, xMax / 2, xMax]} at={H - 1} format={(v) => `${Math.round(v)}m`} />
        {rows.map((r, i) => {
          const cy = top + i * rh;
          return (
            <g key={r.route}>
              <text x={labelW - 6} y={cy + rh / 2 + 3} fill="var(--text)" fontSize="11" textAnchor="end">
                {r.route}
              </text>
              <rect
                x={labelW}
                y={cy + 3}
                width={Math.max(1, x(r.mae) - labelW)}
                height={rh - 8}
                rx={2}
                fill={r.cov >= 0.45 ? "var(--normal)" : "var(--suspended)"}
                fillOpacity={0.7}
              >
                <title>
                  {r.route}: median |err| {Math.round(r.mae)}m · IQR coverage {(r.cov * 100).toFixed(0)}% · n={r.n}
                </title>
              </rect>
              <text x={x(r.mae) + 5} y={cy + rh / 2 + 3} fill="var(--muted)" fontSize="10">
                {Math.round(r.mae)}m · {(r.cov * 100).toFixed(0)}%
              </text>
            </g>
          );
        })}
      </svg>
    </ChartFrame>
  );
}

// 2 — Error vs. elapsed: does the forecast sharpen as the line nears recovery?
// Median |error| through a p25–p75 band, over minutes-into-the-disruption.
export function ErrorByElapsed({ result }: { result: RecoveryResult }) {
  const W = 460;
  const H = 240;
  const padL = 44;
  const padR = 16;
  const padT = 18;
  const padB = 42;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const absErr = (p: RecoveryPoint) => Math.abs(p.actualMin - p.predictedMin);
  const elapsed = result.points.map((p) => p.elapsedMin).sort((a, b) => a - b);
  const domainMax = Math.max(30, Math.ceil(quantile(elapsed, 0.95) / 15) * 15);
  const bins = binBy(result.points, (p) => p.elapsedMin, domainMax, 8).filter(
    (b) => b.pts.length >= 5,
  );
  const stats = bins.map((b) => {
    const es = b.pts.map(absErr);
    return { b, med: quant(es, 0.5), lo: quant(es, 0.25), hi: quant(es, 0.75) };
  });
  const yMax = Math.max(15, Math.ceil(Math.max(0, ...stats.map((s) => s.hi)) / 15) * 15);
  const x = (v: number) => padL + (v / domainMax) * plotW;
  const y = (e: number) => padT + (1 - Math.min(1, e / yMax)) * plotH;
  const xTicks = [0, domainMax / 2, domainMax].map((v) => Math.round(v));
  const yTicks = [0, yMax / 2, yMax];

  return (
    <ChartFrame
      title="Does the guess get better the longer a line's down?"
      titleMeta="typical miss vs. time already disrupted"
      note={
        <>
          How far off the recovery guess is, against how long the line had already
          been down when the model made it. The line sloping{" "}
          <strong>downward</strong> means the longer a line&apos;s been stuck, the
          better the model pins when it&apos;ll be back.
        </>
      }
      meta={recoveryMeta(result)}
    >
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 520 }}>
        <GridLines axis="y" scale={y} ticks={yTicks} from={padL} to={padL + plotW} />
        <AxisTicks axis="y" scale={(e) => y(e) + 3} ticks={yTicks} at={padL - 6} format={(e) => `${Math.round(e)}m`} />
        <AxisTicks axis="x" scale={x} ticks={xTicks} at={H - padB + 16} format={(v) => `${v}m`} />
        <AxisTitle x={padL + plotW / 2} y={H - 6}>minutes already disrupted</AxisTitle>
        {/* band */}
        <path
          d={
            stats.map((s, i) => `${i === 0 ? "M" : "L"}${x(s.b.center).toFixed(1)} ${y(s.hi).toFixed(1)}`).join(" ") +
            " " +
            stats.slice().reverse().map((s) => `L${x(s.b.center).toFixed(1)} ${y(s.lo).toFixed(1)}`).join(" ") +
            " Z"
          }
          fill="var(--realized)"
          fillOpacity={0.15}
          stroke="none"
        />
        {/* median line */}
        <path
          d={stats.map((s, i) => `${i === 0 ? "M" : "L"}${x(s.b.center).toFixed(1)} ${y(s.med).toFixed(1)}`).join(" ")}
          fill="none"
          stroke="var(--realized)"
          strokeWidth={2}
        />
        {stats.map((s, i) => (
          <circle key={i} cx={x(s.b.center)} cy={y(s.med)} r={2.5} fill="var(--realized)">
            <title>
              {Math.round(s.b.lo)}–{Math.round(s.b.hi)}m in: median |err| {Math.round(s.med)}m · n={s.b.pts.length}
            </title>
          </circle>
        ))}
      </svg>
    </ChartFrame>
  );
}

// --- Recovery summary (aggregate feed, no per-point scatter) ---

export interface AggregateRecoveryStats {
  n: number;
  mae_min: number | null;
  rmse_min: number | null;
  iqr_coverage: number | null;
}

export interface AggregateRecovery {
  overall: AggregateRecoveryStats;
  per_regime: AggregateRecoveryStats;
}

export function RecoverySummary({ result }: { result: AggregateRecovery }) {
  const fmtPct = (x: number | null) =>
    x == null ? "—" : `${(x * 100).toFixed(0)}%`;
  const fmtMin = (x: number | null) => (x == null ? "—" : `${Math.round(x)}m`);
  const rows: { label: string; s: AggregateRecoveryStats; note: string }[] = [
    { label: "Per-tick", s: result.overall, note: "every prediction tick weighted equally" },
    { label: "Per-regime", s: result.per_regime, note: "each disruption weighted equally" },
  ];

  return (
    <ChartFrame
      title="Recovery accuracy"
      titleMeta="IQR coverage target ~50% · MAE/RMSE of recovery_minutes vs argmax return"
      meta={{
        source: "the model's own status stream",
        independent: false,
        unit: "5 min / per incident",
      }}
    >
      <table className="mini-table">
        <thead>
          <tr>
            <th>weighting</th>
            <th>n</th>
            <th>IQR coverage</th>
            <th>MAE</th>
            <th>RMSE</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.label}>
              <td title={r.note}>{r.label}</td>
              <td>{r.s.n}</td>
              <td>{fmtPct(r.s.iqr_coverage)}</td>
              <td>{fmtMin(r.s.mae_min)}</td>
              <td>{fmtMin(r.s.rmse_min)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </ChartFrame>
  );
}

// --- Regime swimlane ---

export function Swimlane({ timelines }: { timelines: TimelineDTO[] }) {
  const nonNormalDur = (t: TimelineDTO) =>
    t.segments.filter((s) => s.state !== "normal").reduce((a, s) => a + (s.end - s.start), 0);
  const rows = timelines
    .filter((t) => nonNormalDur(t) > 0)
    .sort((a, b) => nonNormalDur(b) - nonNormalDur(a))
    .slice(0, 14);

  if (!rows.length)
    return (
      <ChartFrame
        empty
        emptyText="No non-normal regimes observed in this window."
      />
    );

  const t0 = Math.min(...rows.flatMap((t) => t.segments.map((s) => s.start)));
  const t1 = Math.max(...rows.flatMap((t) => t.segments.map((s) => s.end)));
  const W = 820;
  const labelW = 44;
  const rowH = 22;
  const gap = 4;
  const span = Math.max(1, t1 - t0);
  const sx = (ts: number) => labelW + ((ts - t0) / span) * (W - labelW - 8);
  const H = rows.length * (rowH + gap) + 24;

  const ticks = 5;
  const tickVals = Array.from({ length: ticks + 1 }, (_, i) => t0 + (span * i) / ticks);
  const fmtTick = (ts: number) =>
    new Date(ts * 1000).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
    });

  return (
    <ChartFrame
      legend={[
        { color: "var(--normal)", label: "normal" },
        { color: "var(--disrupted)", label: "disrupted" },
        { color: "var(--suspended)", label: "suspended" },
      ]}
      meta={{
        source: "the model's own status stream",
        unit: "published condition · top 14 by non-normal time",
        n: rows.length,
      }}
    >
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        <GridLines axis="x" scale={sx} ticks={tickVals} from={16} to={H - 8} />
        <AxisTicks axis="x" scale={sx} ticks={tickVals} at={11} format={fmtTick} />
        {rows.map((t, ri) => {
          const y = 20 + ri * (rowH + gap);
          return (
            <g key={t.route}>
              <text x={4} y={y + rowH / 2 + 4} fill="var(--text)" fontSize="11" fontWeight={600}>
                {t.route}
              </text>
              {t.segments.map((s, si) => (
                <rect
                  key={si}
                  x={sx(s.start)}
                  y={y}
                  width={Math.max(1, sx(s.end) - sx(s.start))}
                  height={rowH}
                  fill={stateColor(s.state)}
                  fillOpacity={s.state === "normal" ? 0.18 : 0.85}
                >
                  <title>
                    {t.route} {s.state}: {fmtTick(s.start)} → {fmtTick(s.end)}
                  </title>
                </rect>
              ))}
            </g>
          );
        })}
      </svg>
    </ChartFrame>
  );
}

// --- Schedule reliability: resume-churn ---

function pct(x: number): string {
  return Number.isNaN(x) ? "—" : `${(x * 100).toFixed(1)}%`;
}

/** Simple signed-value histogram into `bins` buckets over [lo, hi]. */
function histogram(values: number[], lo: number, hi: number, bins: number): number[] {
  const counts = new Array(bins).fill(0) as number[];
  const span = hi - lo || 1;
  for (const v of values) {
    const idx = Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / span) * bins)));
    counts[idx] += 1;
  }
  return counts;
}

export function ResumeChurnPanel({ result }: { result: ResumeChurnResult }) {
  if (result.windows === 0)
    return (
      <ChartFrame
        title="Resume churn"
        empty
        emptyText="No planned-work windows with ≥2 archived versions in this window."
      />
    );

  const mags = result.pushMagnitudesMin;
  const hiMag = Math.max(60, Math.ceil(quantile(mags, 0.95) / 30) * 30);
  const bins = histogram(mags, 0, hiMag, 12);
  const maxBin = Math.max(1, ...bins);
  const W = 460;
  const H = 160;
  const pad = 28;
  const bw = (W - 2 * pad) / bins.length;

  return (
    <ChartFrame
      title="Resume churn"
      titleMeta={
        <>
          {result.windows.toLocaleString()} windows · pushed {pct(result.pushedPct)} · pulled{" "}
          {pct(result.pulledPct)} · stable{" "}
          {pct(result.stable / result.windows)}
        </>
      }
      note={
        <>
          How far announced resume times moved later across an alert&apos;s versions
          (pushed windows only). The MTA rarely pulls a resume earlier, so a push is
          the main way &ldquo;it&apos;s back&rdquo; gets announced too soon.
        </>
      }
      meta={{
        source: "the announced alert schedule",
        independent: true,
        unit: "per-window",
        n: result.windows,
      }}
    >
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 520 }}>
        <rect x={pad} y={8} width={W - 2 * pad} height={H - pad - 8} fill="none" stroke="var(--border)" />
        {bins.map((c, i) => {
          const h = (c / maxBin) * (H - pad - 8);
          return (
            <rect
              key={i}
              x={pad + i * bw + 1}
              y={H - pad - h}
              width={bw - 2}
              height={h}
              fill="var(--suspended)"
              fillOpacity={0.7}
            >
              <title>
                {Math.round((i / bins.length) * hiMag)}–
                {Math.round(((i + 1) / bins.length) * hiMag)}m · {c}
              </title>
            </rect>
          );
        })}
        <AxisTitle x={pad} y={H - 8} anchor="start">0m</AxisTitle>
        <AxisTitle x={W - pad} y={H - 8} anchor="end">{hiMag}m push →</AxisTitle>
      </svg>
      {result.byAlertType.length > 0 && (
        <table className="mini-table">
          <thead>
            <tr>
              <th>alert type</th>
              <th>windows</th>
              <th>pushed</th>
            </tr>
          </thead>
          <tbody>
            {result.byAlertType.slice(0, 6).map((r) => (
              <tr key={r.alertType}>
                <td>{r.alertType}</td>
                <td>{r.windows}</td>
                <td>
                  {r.pushed} ({pct(r.windows ? r.pushed / r.windows : NaN)})
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </ChartFrame>
  );
}

// --- Schedule reliability: adherence ---

export function AdherencePanel({ result }: { result: AdherenceResult }) {
  if (result.n === 0)
    return (
      <ChartFrame
        title="Schedule adherence"
        empty
        emptyText={`No schedule resumes with an observed return-to-normal yet${
          result.censored > 0 ? ` (${result.censored} censored)` : ""
        }.`}
      />
    );

  const errs = result.points.map((p) => p.errorMin);
  const bound = Math.max(30, Math.ceil(quantile(errs.map(Math.abs).sort((a, b) => a - b), 0.95) / 15) * 15);
  const bins = 16;
  const counts = histogram(errs, -bound, bound, bins);
  const maxBin = Math.max(1, ...counts);
  const W = 460;
  const H = 180;
  const pad = 28;
  const bw = (W - 2 * pad) / bins;
  const zeroX = pad + ((0 + bound) / (2 * bound)) * (W - 2 * pad);

  return (
    <ChartFrame
      title="Schedule adherence"
      titleMeta={
        <>
          median{" "}
          {Number.isNaN(result.medianErrorMin) ? "—" : `${result.medianErrorMin > 0 ? "+" : ""}${Math.round(result.medianErrorMin)}m`}{" "}
          · overran {pct(result.overrunPct)} · on-time {pct(result.onTimePct)}
        </>
      }
      note={
        <>
          Announced resume vs when the line actually returned to normal. Right of
          the dashed line = back later than promised (overrun); the only read on the
          silent-overrun case.
        </>
      }
      meta={{
        source: "announced schedule vs the model's own return-to-normal",
        independent: false,
        unit: "per-resume",
        n: result.n,
        censored: result.censored,
      }}
    >
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 520 }}>
        <rect x={pad} y={8} width={W - 2 * pad} height={H - pad - 8} fill="none" stroke="var(--border)" />
        <line x1={zeroX} y1={8} x2={zeroX} y2={H - pad} stroke="var(--muted)" strokeDasharray="3 3" />
        {counts.map((c, i) => {
          const h = (c / maxBin) * (H - pad - 8);
          const center = -bound + ((i + 0.5) / bins) * 2 * bound;
          return (
            <rect
              key={i}
              x={pad + i * bw + 1}
              y={H - pad - h}
              width={bw - 2}
              height={h}
              fill={center > 10 ? "var(--suspended)" : center < -10 ? "var(--disrupted)" : "var(--normal)"}
              fillOpacity={0.7}
            >
              <title>
                {Math.round(center)}m · {c}
              </title>
            </rect>
          );
        })}
        <AxisTitle x={pad} y={H - 8} anchor="start">−{bound}m early</AxisTitle>
        <AxisTitle x={W - pad} y={H - 8} anchor="end">+{bound}m late →</AxisTitle>
      </svg>
    </ChartFrame>
  );
}

// --- Detection latency ---

export function DetectionLatencyPanel({ result }: { result: DetectionLatencyResult }) {
  if (result.n === 0)
    return (
      <ChartFrame
        title="Detection latency"
        empty
        emptyText={`No alert→disruption detections in this window${
          result.missed > 0 ? ` (${result.missed} onsets never flipped)` : ""
        }.`}
      />
    );

  const lats = result.points.map((p) => p.latencyMin);
  const hi = Math.max(15, Math.ceil(quantile(lats.slice().sort((a, b) => a - b), 0.95) / 5) * 5);
  const nbins = Math.max(4, Math.min(20, Math.round(hi / 5)));
  const counts = histogram(lats, 0, hi, nbins);
  const maxBin = Math.max(1, ...counts);
  const W = 460;
  const H = 170;
  const pad = 28;
  const bw = (W - 2 * pad) / nbins;

  return (
    <ChartFrame
      title="Detection latency"
      titleMeta={
        <>
          median{" "}
          {Number.isNaN(result.medianLatencyMin) ? "—" : Math.round(result.medianLatencyMin)}m
          {result.missed > 0 && ` · ${result.missed} missed`}
        </>
      }
      note={
        <>
          Minutes from a real alert appearing to the HMM flipping to
          disrupted/suspended. Resolution is one prediction tick (~5m). Breakdown by
          cause below.
        </>
      }
      meta={{
        source: "the alert feed",
        independent: true,
        unit: "per-onset",
        n: result.n,
      }}
    >
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 520 }}>
        <rect x={pad} y={8} width={W - 2 * pad} height={H - pad - 8} fill="none" stroke="var(--border)" />
        {counts.map((c, i) => {
          const h = (c / maxBin) * (H - pad - 8);
          return (
            <rect
              key={i}
              x={pad + i * bw + 1}
              y={H - pad - h}
              width={bw - 2}
              height={h}
              fill="var(--disrupted)"
              fillOpacity={0.7}
            >
              <title>
                {Math.round((i / nbins) * hi)}–{Math.round(((i + 1) / nbins) * hi)}m · {c}
              </title>
            </rect>
          );
        })}
        <AxisTitle x={pad} y={H - 8} anchor="start">0m</AxisTitle>
        <AxisTitle x={W - pad} y={H - 8} anchor="end">{hi}m latency →</AxisTitle>
      </svg>
      {result.byAlertType.length > 0 && (
        <table className="mini-table">
          <thead>
            <tr>
              <th>cause (primary_alert_type)</th>
              <th>detections</th>
              <th>median</th>
              <th>missed</th>
            </tr>
          </thead>
          <tbody>
            {result.byAlertType.slice(0, 8).map((r) => (
              <tr key={r.alertType}>
                <td>{r.alertType}</td>
                <td>{r.n}</td>
                <td>{Number.isNaN(r.medianLatencyMin) ? "—" : `${Math.round(r.medianLatencyMin)}m`}</td>
                <td>{r.missed}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </ChartFrame>
  );
}

// --- Input drift ---

export interface DriftResult {
  unmapped_alert_type: {
    n_typed_ticks: number;
    unmapped_rate: number;
    unmapped_types: Record<string, number>;
    by_route: Record<string, number>;
  };
  emission_channels: {
    available: boolean;
    cells_scored?: number;
    cells_skipped_thin?: number;
    psi_threshold?: number;
    routes_drifted?: string[];
    by_route?: Record<
      string,
      {
        max_alert_count_psi: number;
        max_flag_delta: number;
        max_flag_delta_channel: string | null;
        n_cells: number;
        significant: boolean;
      }
    >;
  };
}

export function DriftPanel({
  result,
  trainedAt,
}: {
  result: DriftResult;
  trainedAt?: number | null;
}) {
  const u = result.unmapped_alert_type;
  const e = result.emission_channels;
  const offenders = Object.entries(u.unmapped_types ?? {}).sort(
    (a, b) => b[1] - a[1],
  );
  const emRoutes = Object.entries(e.by_route ?? {}).sort(
    (a, b) => b[1].max_alert_count_psi - a[1].max_alert_count_psi,
  );
  const rate = u.unmapped_rate;
  const scoredRoutes = emRoutes.length;
  const driftedCount = (e.routes_drifted ?? []).length;
  const driftFrac = scoredRoutes ? driftedCount / scoredRoutes : 0;
  // When most scored lines trip at once, the signal is the feed itself, not any
  // one line — call that out instead of listing every route as an offender.
  const feedWide = e.available && scoredRoutes >= 3 && driftFrac >= 0.6;
  const trainedStr = trainedAt
    ? new Date(trainedAt * 1000).toLocaleDateString()
    : null;

  let verdict: string;
  let tone: "warn" | "good";
  if (feedWide) {
    verdict = `Feed-wide shift — ${driftedCount} of ${scoredRoutes} lines drifted from the training profile at once. That points upstream (a feed change), not to any one line. Consider a retrain${trainedStr ? ` (last trained ${trainedStr})` : ""}.`;
    tone = "warn";
  } else if (rate > 0 && driftedCount > 0) {
    verdict = `New alert wording (${(rate * 100).toFixed(2)}% of typed ticks) and ${driftedCount} line${driftedCount === 1 ? "" : "s"} drifting from the training profile.`;
    tone = "warn";
  } else if (rate > 0) {
    verdict = `New alert wording the model has never seen — ${(rate * 100).toFixed(2)}% of typed ticks. Map the types below before forecasts slip.`;
    tone = "warn";
  } else if (driftedCount > 0) {
    verdict = `${driftedCount} line${driftedCount === 1 ? "" : "s"} drifting from the training profile; the rest look familiar.`;
    tone = "warn";
  } else {
    verdict = "Feed looks familiar — no new alert vocabulary, no significant route drift.";
    tone = "good";
  }

  // Route-specific drift list: significant lines first, capped — not a wall of
  // near-zero PSIs. (Suppressed entirely when the shift is feed-wide.)
  const shown = feedWide ? [] : emRoutes.filter(([, v]) => v.significant).slice(0, 8);

  return (
    <>
      <div className={tone === "warn" ? "warnbox" : "note"} style={{ maxWidth: 720 }}>
        {verdict}
      </div>
      <div className="charts-row">
        <ChartFrame
          title="Global: unfamiliar alert vocabulary"
          titleMeta={
            <>
              share of {u.n_typed_ticks.toLocaleString()} typed ticks with an
              alert_type we don&apos;t map
            </>
          }
          meta={{ source: "alert feed", unit: "5 min", n: u.n_typed_ticks }}
        >
          <div
            style={{
              fontSize: 28,
              fontWeight: 600,
              margin: "4px 0 8px",
              color: rate > 0 ? "var(--disrupted)" : "inherit",
            }}
          >
            {(rate * 100).toFixed(2)}%
          </div>
          {offenders.length === 0 ? (
            <div className="muted">Every observed alert_type is mapped.</div>
          ) : (
            <table className="mini-table">
              <thead>
                <tr>
                  <th>unmapped type (add to mapping)</th>
                  <th>ticks</th>
                </tr>
              </thead>
              <tbody>
                {offenders.slice(0, 8).map(([t, n]) => (
                  <tr key={t}>
                    <td>{t}</td>
                    <td>{n}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </ChartFrame>

        <ChartFrame
          title="By line: emission-channel drift"
          titleMeta={
            <>
              alert_count PSI vs the training profile (≥{" "}
              {e.psi_threshold ?? 0.25} significant)
            </>
          }
          meta={{
            source: "alert_count vs training profile",
            unit: "per (line × time-of-day) cell",
            n: e.cells_scored,
            note: e.cells_skipped_thin
              ? `${e.cells_skipped_thin} thin skipped`
              : undefined,
            extra: trainedStr ? (
              <span className="chart-chip chart-chip-muted">trained {trainedStr}</span>
            ) : undefined,
          }}
        >
          {!e.available ? (
            <div className="muted">
              Not yet available — activates once a retrain stores the reference
              profile in params.json.
            </div>
          ) : feedWide ? (
            <div className="muted">
              {driftedCount} of {scoredRoutes} lines tripped — see the feed-wide
              warning above. Per-line ranking is suppressed when nearly everything
              drifts at once.
            </div>
          ) : shown.length === 0 ? (
            <div className="muted">
              No line crossed the {e.psi_threshold ?? 0.25} PSI threshold
              {scoredRoutes ? ` (of ${scoredRoutes} scored)` : ""}
              {e.cells_skipped_thin ? ` · ${e.cells_skipped_thin} cells too thin` : ""}
              .
            </div>
          ) : (
            <table className="mini-table">
              <thead>
                <tr>
                  <th>line</th>
                  <th>max PSI</th>
                  <th>worst flag Δ</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {shown.map(([r, v]) => (
                  <tr key={r}>
                    <td>{r}</td>
                    <td style={{ color: "var(--disrupted)" }}>
                      {v.max_alert_count_psi.toFixed(2)}
                    </td>
                    <td>
                      {v.max_flag_delta_channel
                        ? `${v.max_flag_delta_channel} ${(v.max_flag_delta * 100).toFixed(0)}pp`
                        : "—"}
                    </td>
                    <td>⚠</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </ChartFrame>
      </div>
    </>
  );
}

// --- Transition matrix heatmaps: system-average baseline + per-route deltas ---

export function TransitionHeatmaps({
  entries,
  states,
  trainedAt,
}: {
  entries: { route: string; transition: number[][] }[];
  states: string[];
  trainedAt?: number | null;
}) {
  const { ref, show, hide, overlay } = useTooltip();
  const k = states.length;
  if (!entries.length || k === 0)
    return <ChartFrame empty emptyText="No trained params available yet." />;

  // System-average transition matrix — the "typical line" every route is read
  // against. Rows are P(next state | current), so each sums to ~1.
  const baseline = Array.from({ length: k }, (_, r) =>
    Array.from({ length: k }, (_, c) =>
      entries.reduce((a, e) => a + (e.transition[r]?.[c] ?? 0), 0) / entries.length,
    ),
  );
  // One shared ruler for off-diagonal deltas so every route card is comparable.
  let maxAbsDelta = 0;
  for (const e of entries)
    for (let r = 0; r < k; r++)
      for (let c = 0; c < k; c++)
        if (r !== c)
          maxAbsDelta = Math.max(
            maxAbsDelta,
            Math.abs((e.transition[r]?.[c] ?? 0) - baseline[r][c]),
          );
  maxAbsDelta = maxAbsDelta || 1;

  const cell = 42;
  const labelW = 30;
  const top = 16;
  const W = labelW + k * cell;
  const H = top + k * cell + 2;

  const grid = (transition: number[][], mode: "baseline" | "delta", label: string) => (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: W }}>
      {states.map((st, c) => (
        <text key={`c${c}`} x={labelW + c * cell + cell / 2} y={top - 4} fill="var(--muted)" fontSize="8" textAnchor="middle">
          {st.slice(0, 4)}
        </text>
      ))}
      {states.map((st, r) => (
        <text key={`r${r}`} x={labelW - 4} y={top + r * cell + cell / 2 + 3} fill="var(--muted)" fontSize="8" textAnchor="end">
          {st.slice(0, 4)}
        </text>
      ))}
      {Array.from({ length: k }, (_, r) =>
        Array.from({ length: k }, (_, c) => {
          const v = transition[r]?.[c] ?? 0;
          const b = baseline[r][c];
          const d = v - b;
          const diag = r === c;
          let fill = "var(--panel-2)";
          let op = 1;
          if (mode === "baseline") {
            fill = stateColor(states[c]);
            op = 0.1 + 0.8 * v;
          } else if (!diag) {
            // Off-diagonal only: diverging vs the system average.
            fill = d >= 0 ? "var(--suspended)" : "var(--accent)";
            op = 0.12 + 0.7 * (Math.abs(d) / maxAbsDelta);
          }
          return (
            <g
              key={`${r}-${c}`}
              onMouseMove={(e) =>
                show(
                  e,
                  <>
                    <strong>{label}</strong> {states[r]} → {states[c]}
                    <br />
                    <span className="muted">
                      P {v.toFixed(2)} · system avg {b.toFixed(2)} · Δ {d >= 0 ? "+" : ""}
                      {d.toFixed(2)}
                    </span>
                    <br />
                    <span className="muted">row-normalized: P(next | {states[r]} now)</span>
                  </>,
                )
              }
              onMouseLeave={hide}
              style={{ cursor: "default" }}
            >
              <rect
                x={labelW + c * cell}
                y={top + r * cell}
                width={cell - 2}
                height={cell - 2}
                rx={3}
                fill={diag && mode === "delta" ? "transparent" : fill}
                fillOpacity={diag && mode === "delta" ? 1 : op}
                stroke={diag ? "var(--border)" : "none"}
              />
              <text
                x={labelW + c * cell + (cell - 2) / 2}
                y={top + r * cell + (cell - 2) / 2 + 3}
                fill={diag && mode === "delta" ? "var(--muted)" : "var(--text)"}
                fontSize="10"
                textAnchor="middle"
              >
                {v.toFixed(2)}
              </text>
            </g>
          );
        }),
      )}
    </svg>
  );

  const miniCard = {
    background: "var(--panel-2)",
    border: "1px solid var(--border)",
    borderRadius: 8,
    padding: 8,
  } as const;

  return (
    <ChartFrame
      containerRef={ref}
      overlay={overlay}
      note={
        <>
          Each line&apos;s learned transition odds, row-normalized (P of the next state
          given the current one). The <strong>system average</strong> card anchors
          the absolute rates; each line card colors only its{" "}
          <strong>off-diagonal</strong> cells by deviation from that average —{" "}
          <span style={{ color: "var(--suspended)" }}>more</span> /{" "}
          <span style={{ color: "var(--accent)" }}>less</span> likely to switch than
          typical — so self-loop stickiness on the diagonal doesn&apos;t drown out
          where a line actually differs.
        </>
      }
      meta={{
        source: "trained params",
        n: entries.length,
        unit: "per-line transition matrix",
        note: trainedAt
          ? `trained ${new Date(trainedAt * 1000).toLocaleDateString()}`
          : undefined,
      }}
    >
      <div className="small-multiples">
        <div style={miniCard}>
          <div style={{ fontSize: 11, fontWeight: 600, marginBottom: 4, color: "var(--muted)" }}>
            system average
          </div>
          {grid(baseline, "baseline", "system avg")}
        </div>
        {entries.map((e) => (
          <div key={e.route} style={miniCard}>
            <div style={{ fontSize: 11, fontWeight: 700, marginBottom: 4 }}>{e.route}</div>
            {grid(e.transition, "delta", e.route)}
          </div>
        ))}
      </div>
    </ChartFrame>
  );
}

// --- Movement vs. alerts: confusion + advance baselines ---

const MOVE_THRESHOLD = 0.25; // FROZEN_ADVANCE_FRAC — the old global state threshold

export interface MovementConfusionResult {
  states: string[];
  matrix: number[][]; // [hmmRow][moveCol] counts
  rowTotals: number[];
  total: number;
  agreement: number;
  perRoute: { route: string; n: number; agree: number; agreePct: number }[];
  disagreements: {
    route: string;
    hmm: string;
    move: string;
    count: number;
    rate: number;
    kind: "false-normal" | "false-disrupted" | "state-mismatch";
  }[];
  coverage: { judged: number; unjudged: number; suspendedUnjudged: number };
}

const DISAGREEMENT_LABEL: Record<string, string> = {
  "false-normal": "published normal, trains weren't",
  "false-disrupted": "published disruption, trains fine",
  "state-mismatch": "disrupted ↔ suspended",
};

export interface RouteBaselineDTO {
  route: string;
  p0: number;
  n: number;
  disruptedShare: number;
  fracs: number[];
  north: { p0: number; n: number } | null;
  south: { p0: number; n: number } | null;
}

export function MovementConfusion({ result }: { result: MovementConfusionResult }) {
  const { ref, show, hide, overlay } = useTooltip();
  const { states, matrix, rowTotals, total } = result;
  const cell = 78;
  const labelW = 92;
  const axisPad = 26; // room for the rotated y-axis title + column header
  const W = labelW + states.length * cell + 14;
  const H = axisPad + labelW - 18 + states.length * cell + 14;
  const gridTop = axisPad + labelW - 18;

  return (
    <ChartFrame
      containerRef={ref}
      overlay={overlay}
      maxWidth={480}
      title="Published condition vs. movement state"
      titleMeta={
        total
          ? `${(result.agreement * 100).toFixed(0)}% agree · n=${total.toLocaleString()} ticks`
          : "no judgeable ticks in window"
      }
      note={
        <>
          Down the side: the status we published, from alerts. Across the top: what
          the trains&apos; movement actually showed. Boxes down the diagonal are the two
          agreeing; anything off it is a disagreement. Hover a box.
        </>
      }
      meta={{
        source: "train movement",
        independent: true,
        unit: "5 min",
        n: total,
      }}
    >
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 360 }}>
        {/* axis titles */}
        <AxisTitle x={labelW + (states.length * cell) / 2} y={axisPad - 12}>
          movement state (independent) →
        </AxisTitle>
        <AxisTitle x={14} y={gridTop + (states.length * cell) / 2} rotate={-90}>
          published condition →
        </AxisTitle>
        {/* column headers */}
        {states.map((st, c) => (
          <text key={`c${c}`} x={labelW + c * cell + cell / 2} y={gridTop - 7} fill="var(--text)" fontSize="11" textAnchor="middle">
            {st}
          </text>
        ))}
        {/* row headers */}
        {states.map((st, r) => (
          <text key={`r${r}`} x={labelW - 9} y={gridTop + r * cell + cell / 2 + 4} fill="var(--text)" fontSize="11" textAnchor="end">
            {st}
          </text>
        ))}
        {matrix.map((row, r) =>
          row.map((v, c) => {
            const frac = rowTotals[r] ? v / rowTotals[r] : 0;
            return (
              <g
                key={`${r}-${c}`}
                onMouseMove={(e) =>
                  show(
                    e,
                    <>
                      <strong>{(frac * 100).toFixed(0)}%</strong> of{" "}
                      <span style={{ color: stateColor(states[r]) }}>{states[r]}</span>{" "}
                      ticks read{" "}
                      <span style={{ color: stateColor(states[c]) }}>{states[c]}</span>{" "}
                      on movement
                      <br />
                      <span className="muted">
                        {v.toLocaleString()} of {rowTotals[r].toLocaleString()} ticks ·{" "}
                        {r === c ? "agree" : "disagree"}
                      </span>
                    </>,
                  )
                }
                onMouseLeave={hide}
              >
                <rect
                  x={labelW + c * cell}
                  y={gridTop + r * cell}
                  width={cell - 3}
                  height={cell - 3}
                  rx={5}
                  fill={stateColor(states[c])}
                  fillOpacity={0.08 + 0.85 * frac}
                  stroke={r === c ? "var(--text)" : "var(--border)"}
                  strokeOpacity={r === c ? 0.4 : 0.6}
                  style={{ cursor: "default" }}
                />
                <text x={labelW + c * cell + (cell - 3) / 2} y={gridTop + r * cell + (cell - 3) / 2 - 1} fill="var(--text)" fontSize="14" textAnchor="middle" fontWeight={r === c ? 700 : 400}>
                  {(frac * 100).toFixed(0)}%
                </text>
                <text x={labelW + c * cell + (cell - 3) / 2} y={gridTop + r * cell + (cell - 3) / 2 + 14} fill="var(--muted)" fontSize="9" textAnchor="middle">
                  {v.toLocaleString()}
                </text>
              </g>
            );
          }),
        )}
      </svg>
      {(result.coverage.unjudged > 0 || result.total > 0) && (
        <div className="grp-note" style={{ marginTop: 8 }}>
          Coverage: judged {result.coverage.judged.toLocaleString()} ticks ·{" "}
          {result.coverage.unjudged.toLocaleString()} unjudged (no movement read
          {result.coverage.suspendedUnjudged > 0
            ? `, ${result.coverage.suspendedUnjudged.toLocaleString()} suspended w/ no vehicles`
            : ""}
          ).
        </div>
      )}
      {result.disagreements.length > 0 && (
        <table className="mini-table">
          <thead>
            <tr>
              <th>line</th>
              <th>disagreement</th>
              <th>what happened</th>
              <th>ticks</th>
              <th>rate</th>
            </tr>
          </thead>
          <tbody>
            {result.disagreements.slice(0, 8).map((d, i) => (
              <tr key={i}>
                <td style={{ fontWeight: 600 }}>{d.route}</td>
                <td>
                  <span style={{ color: stateColor(d.hmm) }}>{d.hmm}</span> →{" "}
                  <span style={{ color: stateColor(d.move) }}>{d.move}</span>
                </td>
                <td className="muted">{DISAGREEMENT_LABEL[d.kind]}</td>
                <td>{d.count.toLocaleString()}</td>
                <td>{(d.rate * 100).toFixed(0)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </ChartFrame>
  );
}

type DirSel = "combined" | "north" | "south";
type SortSel = "p0" | "disrupted" | "route";

export function AdvanceBaselineChart({ routes }: { routes: RouteBaselineDTO[] }) {
  const { ref, show, hide, overlay } = useTooltip();
  const [dir, setDir] = useState<DirSel>("combined");
  const [sort, setSort] = useState<SortSel>("p0");

  const p0For = (r: RouteBaselineDTO): number | null =>
    dir === "combined" ? r.p0 : dir === "north" ? (r.north?.p0 ?? null) : (r.south?.p0 ?? null);

  const sorted = [...routes].sort((a, b) => {
    if (sort === "route") return a.route.localeCompare(b.route);
    if (sort === "disrupted") return b.disruptedShare - a.disruptedShare;
    const pa = p0For(a),
      pb = p0For(b);
    return (pa ?? 1.1) - (pb ?? 1.1);
  });

  const labelW = 48;
  const rightPad = 16;
  const W = 580;
  const plotW = W - labelW - rightPad;
  const rh = 24;
  const top = 30; // headroom for the threshold label
  const axisH = 44; // x-axis ticks + title
  const plotBottom = top + sorted.length * rh;
  const H = plotBottom + axisH;
  const x = (frac: number) => labelW + frac * plotW;
  const jitter = (i: number) => ((((i * 2654435761) >>> 0) % 1000) / 1000 - 0.5) * 12;
  const ticks = [0, 0.25, 0.5, 0.75, 1];

  if (!routes.length)
    return (
      <ChartFrame
        title="How much each line normally moves"
        empty
        emptyText="No advance-rate baselines in this window."
      />
    );

  const tipFor = (r: RouteBaselineDTO) => {
    const p0 = p0For(r);
    return (
      <>
        <strong>{r.route}</strong> · {dir} baseline {p0 !== null ? p0.toFixed(2) : "—"}
        <br />
        <span className="muted">
          {r.n.toLocaleString()} ticks · {(r.disruptedShare * 100).toFixed(0)}% below 0.25
        </span>
        <br />
        <span className="muted">
          north {r.north ? r.north.p0.toFixed(2) : "—"} · south{" "}
          {r.south ? r.south.p0.toFixed(2) : "—"}
        </span>
      </>
    );
  };

  return (
    <ChartFrame
      containerRef={ref}
      overlay={overlay}
      maxWidth={660}
      title="How much each line normally moves"
      titleMeta="typical share of trains advancing a stop between updates"
      meta={{
        source: "the train-movement archive",
        independent: true,
        unit: "advance-rate · per line",
        n: routes.length,
      }}
    >
      <div className="charts-row" style={{ gap: 14, margin: "4px 0 8px" }}>
        <label style={{ display: "flex", gap: 6, alignItems: "center", color: "var(--muted)", fontSize: 12 }}>
          direction
          <select value={dir} onChange={(e) => setDir(e.target.value as DirSel)}>
            <option value="combined">combined</option>
            <option value="north">north</option>
            <option value="south">south</option>
          </select>
        </label>
        <label style={{ display: "flex", gap: 6, alignItems: "center", color: "var(--muted)", fontSize: 12 }}>
          sort
          <select value={sort} onChange={(e) => setSort(e.target.value as SortSel)}>
            <option value="p0">baseline</option>
            <option value="disrupted">% below threshold</option>
            <option value="route">line</option>
          </select>
        </label>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        {/* vertical gridlines */}
        <GridLines axis="x" scale={x} ticks={ticks} from={top - 8} to={plotBottom} strong={(t) => t === 0 || t === 1} />
        {/* zebra rows */}
        {sorted.map((r, i) =>
          i % 2 === 1 ? (
            <rect key={`z${r.route}`} x={labelW} y={top + i * rh} width={plotW} height={rh} fill="var(--panel-2)" fillOpacity={0.5} />
          ) : null,
        )}
        {/* threshold line */}
        <line x1={x(MOVE_THRESHOLD)} x2={x(MOVE_THRESHOLD)} y1={top - 8} y2={plotBottom} stroke="var(--suspended)" strokeDasharray="3 3" strokeOpacity={0.8} />
        <text x={x(MOVE_THRESHOLD)} y={top - 13} fill="var(--suspended)" fontSize="9.5" textAnchor="middle">
          0.25 global threshold
        </text>
        {/* x-axis ticks + title */}
        <AxisTicks axis="x" scale={x} ticks={ticks} at={plotBottom + 16} format={(t) => t.toFixed(2)} />
        <AxisTitle x={labelW + plotW / 2} y={plotBottom + 36}>
          advance fraction — share of matched trips advancing a stop per tick
        </AxisTitle>
        {sorted.map((r, i) => {
          const cy = top + i * rh + rh / 2;
          const p0 = p0For(r);
          const mColor = (f: number) => stateColor(f <= MOVE_THRESHOLD ? "disrupted" : "normal");
          return (
            <g
              key={r.route}
              onMouseMove={(e) => show(e, tipFor(r))}
              onMouseLeave={hide}
              style={{ cursor: "default" }}
            >
              <text x={labelW - 9} y={cy + 4} fill="var(--text)" fontSize="11" textAnchor="end">
                {r.route}
              </text>
              {/* distribution strip (combined fractions) */}
              {r.fracs.map((f, j) => (
                <circle key={j} cx={x(f)} cy={cy + jitter(j)} r={1.6} fill={mColor(f)} fillOpacity={0.32} />
              ))}
              {/* baseline p0 marker */}
              {p0 !== null && (
                <g>
                  <line x1={x(p0)} x2={x(p0)} y1={cy - 8} y2={cy + 8} stroke={mColor(p0)} strokeWidth={2.5} />
                  <circle cx={x(p0)} cy={cy} r={3} fill={mColor(p0)} />
                </g>
              )}
              {/* full-row hover target */}
              <rect x={0} y={cy - rh / 2} width={W} height={rh} fill="transparent" />
            </g>
          );
        })}
      </svg>
      <div className="grp-note" style={{ marginTop: 4 }}>
        Each dot is one reading; the bar is the line&apos;s typical rate. The dashed line
        is a single &ldquo;stuck&rdquo; cutoff applied to every line — and you can see why
        that&apos;s unfair: slow lines like the shuttles sit left of it even on a good
        day, so one fixed cutoff would call them{" "}
        <span style={{ color: "var(--disrupted)" }}>disrupted</span> around the clock.
        The fix is to judge each line against its own normal.
      </div>
    </ChartFrame>
  );
}

// --- Recovery as a distribution: predicted-vs-realized CDF + PIT report card ---

export interface RecoveryWeighting {
  n: number;
  meanCrps: number;
  baselineCrps: number;
  skill: number;
  meanPit: number;
}

export interface RecoveryDistResult {
  n: number;
  meanCrps: number;
  baselineCrps: number;
  skill: number;
  meanPit: number;
  perTick: RecoveryWeighting;
  perRegime: RecoveryWeighting;
  pit: number[];
  grid: number[];
  predictedCurve: number[];
  empiricalCurve: number[];
  horizons: { h: number; predicted: number; observed: number }[];
}

const CURVE_PRED = FORECAST;
const CURVE_OBS = REALIZED;

export function RecoveryDistCurve({ result }: { result: RecoveryDistResult }) {
  const { ref, show, hide, overlay } = useTooltip();
  const [hi, setHi] = useState<number | null>(null);
  const { grid, predictedCurve, empiricalCurve } = result;
  const tMax = grid[grid.length - 1] || 240;

  const W = 580;
  const H = 300;
  const padL = 44;
  const padR = 16;
  const padT = 16;
  const padB = 42;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const x = (t: number) => padL + (t / tMax) * plotW;
  const y = (f: number) => padT + (1 - f) * plotH;
  const path = (ys: number[]) =>
    ys.map((f, i) => `${i === 0 ? "M" : "L"}${x(grid[i]).toFixed(1)} ${y(f).toFixed(1)}`).join(" ");

  return (
    <ChartFrame
      containerRef={ref}
      overlay={overlay}
      maxWidth={640}
      title="How long until a line's back"
      titleMeta="chance of being normal again within t minutes"
      meta={{
        source: "the model's own status stream",
        independent: false,
        unit: "5 min",
        n: result.n,
      }}
      note={
        <>
          The full picture, not just a few checkpoints.{" "}
          <span style={{ color: CURVE_OBS }}>Green</span> is how fast lines really came
          back; <span style={{ color: CURVE_PRED }}>blue</span> is what the model
          expected. Blue sitting below green means it&apos;s being too pessimistic.
        </>
      }
    >
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        {/* y gridlines + labels */}
        <GridLines axis="y" scale={y} ticks={[0, 0.25, 0.5, 0.75, 1]} from={padL} to={padL + plotW} />
        <AxisTicks axis="y" scale={(f) => y(f) + 3} ticks={[0, 0.25, 0.5, 0.75, 1]} at={padL - 6} format={(f) => f.toFixed(2)} />
        {/* x ticks at 30/60/120/240 */}
        <AxisTicks axis="x" scale={x} ticks={[0, 30, 60, 120, 180, 240].filter((t) => t <= tMax)} at={H - padB + 16} />
        <AxisTitle x={padL + plotW / 2} y={H - 6}>minutes since now</AxisTitle>
        {/* hover guide */}
        {hi != null && (
          <line x1={x(grid[hi])} x2={x(grid[hi])} y1={padT} y2={padT + plotH} stroke="var(--text)" strokeOpacity={0.25} />
        )}
        {/* curves */}
        <path d={path(empiricalCurve)} fill="none" stroke={CURVE_OBS} strokeWidth={2} />
        <path d={path(predictedCurve)} fill="none" stroke={CURVE_PRED} strokeWidth={2} />
        {/* hover capture */}
        <rect
          x={padL}
          y={padT}
          width={plotW}
          height={plotH}
          fill="transparent"
          onMouseMove={(e) => {
            const box = (e.currentTarget.ownerSVGElement as SVGSVGElement).getBoundingClientRect();
            const px = ((e.clientX - box.left) / box.width) * W;
            const t = ((px - padL) / plotW) * tMax;
            const idx = Math.min(grid.length - 1, Math.max(0, Math.round(t / (grid[1] - grid[0]))));
            setHi(idx);
            show(
              e,
              <>
                <strong>within {grid[idx]} min</strong>
                <br />
                <span style={{ color: CURVE_OBS }}>
                  {(empiricalCurve[idx] * 100).toFixed(0)}% really had recovered
                </span>
                <br />
                <span style={{ color: CURVE_PRED }}>
                  model expected {(predictedCurve[idx] * 100).toFixed(0)}%
                </span>
              </>,
            );
          }}
          onMouseLeave={() => {
            setHi(null);
            hide();
          }}
          style={{ cursor: "crosshair" }}
        />
        {/* legend */}
        <g transform={`translate(${padL + plotW - 150}, ${padT + 8})`}>
          <rect x={0} y={-8} width={10} height={3} fill={CURVE_OBS} />
          <text x={14} y={-3} fill="var(--muted)" fontSize="10">actual recovery</text>
          <rect x={0} y={6} width={10} height={3} fill={CURVE_PRED} />
          <text x={14} y={11} fill="var(--muted)" fontSize="10">model forecast</text>
        </g>
      </svg>
    </ChartFrame>
  );
}

export function RecoveryScoreCard({ result }: { result: RecoveryDistResult }) {
  const { ref, show, hide, overlay } = useTooltip();
  const fmt = (x: number) => (Number.isNaN(x) ? "—" : Math.round(x).toString());
  const { verdict, explain, tone, warning } = recoveryVerdict(result);
  const verdictColor =
    tone === "good"
      ? "var(--normal)"
      : tone === "warn"
        ? "var(--disrupted)"
        : "var(--muted)";
  const skillStr = (skill: number) =>
    Number.isNaN(skill)
      ? "—"
      : skill >= 0
        ? `beats the simple baseline by ${(skill * 100).toFixed(0)}%`
        : `${Math.abs(skill * 100).toFixed(0)}% behind the simple baseline`;
  const perTick = result.perTick;
  const perRegime = result.perRegime;

  // PIT histogram geometry
  const { pit } = result;
  const total = pit.reduce((a, b) => a + b, 0);
  const expected = total / pit.length; // flat = honest
  const maxC = Math.max(1, ...pit, expected);
  const W = 380;
  const H = 168;
  const padL = 12;
  const padR = 12;
  const padT = 12;
  const padB = 30;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;
  const bw = plotW / pit.length;
  const y = (c: number) => padT + (1 - c / maxC) * plotH;

  return (
    <ChartFrame
      containerRef={ref}
      overlay={overlay}
      maxWidth={420}
      title="How honest are the odds?"
      meta={{
        source: "the model's own status stream",
        independent: false,
        unit: "5 min",
        n: result.n,
      }}
    >
      <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginTop: 8 }}>
        <span style={{ width: 9, height: 9, borderRadius: "50%", background: verdictColor, display: "inline-block" }} />
        <span style={{ fontSize: 18, fontWeight: 600 }}>{verdict}</span>
      </div>
      <p className="grp-note" style={{ marginTop: 6 }}>{explain}</p>
      {warning && (
        <div className="warnbox" style={{ margin: "0 0 6px" }}>
          {warning}
        </div>
      )}

      {/* PIT histogram: each recovery scored on the model's own curve; flat = honest */}
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        <line x1={padL} x2={padL + plotW} y1={y(expected)} y2={y(expected)} stroke="var(--text)" strokeDasharray="3 3" strokeOpacity={0.45} />
        <text x={padL + plotW} y={y(expected) - 4} fill="var(--muted)" fontSize="9" textAnchor="end">
          flat = honest
        </text>
        {pit.map((c, i) => (
          <g
            key={i}
            onMouseMove={(e) =>
              show(
                e,
                <>
                  <strong>{c} of {total}</strong> forecast ticks
                  <br />
                  <span className="muted">
                    {i < 5 ? "came back sooner than forecast" : "took longer than forecast"}
                  </span>
                </>,
              )
            }
            onMouseLeave={hide}
          >
            <rect
              x={padL + i * bw + 1}
              y={y(c)}
              width={bw - 2}
              height={padT + plotH - y(c)}
              fill={i < 5 ? "var(--disrupted)" : "var(--accent)"}
              fillOpacity={0.75}
              rx={2}
            />
          </g>
        ))}
        {/* x axis */}
        <line x1={padL} x2={padL + plotW} y1={padT + plotH} y2={padT + plotH} stroke="var(--border)" />
        <AxisTitle x={padL} y={H - padB + 16} anchor="start">← recovered sooner</AxisTitle>
        <AxisTitle x={padL + plotW} y={H - padB + 16} anchor="end">took longer →</AxisTitle>
      </svg>

      <p className="grp-note" style={{ marginTop: 2 }}>
        Each forecast tick scored on the model&apos;s own curve. <strong>Flat</strong> bars =
        honest. Piled <strong>left</strong> = recoveries beat the forecast (cautious),
        <strong> right</strong> = ran long (rosy); a <strong>U</strong> means it&apos;s
        overconfident, a <strong>hump</strong> underconfident.
      </p>

      <div className="grp-note" style={{ marginTop: 8, borderTop: "1px solid var(--border)", paddingTop: 10 }}>
        <div>
          <strong>Per-incident</strong> (each disruption weighted equally) ·{" "}
          {fmt(perRegime.meanCrps)} min · {skillStr(perRegime.skill)} ·{" "}
          {perRegime.n.toLocaleString()} incidents
        </div>
        <div style={{ marginTop: 4, opacity: 0.8 }}>
          <strong>Per-tick</strong> (every forecast tick; long incidents dominate) ·{" "}
          {fmt(perTick.meanCrps)} min · {skillStr(perTick.skill)} ·{" "}
          {perTick.n.toLocaleString()} ticks
        </div>
      </div>
    </ChartFrame>
  );
}
