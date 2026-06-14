"use client";

// Hand-rolled SVG charts — these are bespoke scientific plots (calibration
// scatter on a diagonal, regime swimlane, transition heatmap) where a generic
// charting lib would fight us more than help.

export interface ReliabilityResult {
  horizonMin: number;
  bins: { p: number; predictedMean: number; observedFreq: number; n: number }[];
  brier: number;
  n: number;
  excludedSchedule: number;
}

export interface RecoveryResult {
  points: { route: string; predictedMin: number; actualMin: number; inIqr: boolean }[];
  coverage: number;
  n: number;
  medianAbsErrorMin: number;
  excludedSchedule: number;
}

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

// --- Reliability diagram ---

export function ReliabilityChart({ result }: { result: ReliabilityResult }) {
  const S = 220;
  const pad = 28;
  const sc = (v: number) => pad + v * (S - 2 * pad);
  const scY = (v: number) => S - pad - v * (S - 2 * pad);
  const maxN = Math.max(1, ...result.bins.map((b) => b.n));

  return (
    <div className="chart">
      <div className="chart-title">
        P(normal within {result.horizonMin}m) ·{" "}
        <span className="muted">
          Brier {Number.isNaN(result.brier) ? "—" : result.brier.toFixed(3)} · n=
          {result.n}
          {result.excludedSchedule > 0 &&
            ` · ${result.excludedSchedule} schedule excl.`}
        </span>
      </div>
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
        <text x={pad} y={S - 6} fill="var(--muted)" fontSize="9">
          predicted →
        </text>
        <text x={6} y={pad + 4} fill="var(--muted)" fontSize="9">
          observed ↑
        </text>
      </svg>
    </div>
  );
}

// --- Recovery scatter: predicted vs actual time-to-normal ---

export function RecoveryScatter({ result }: { result: RecoveryResult }) {
  const W = 460;
  const H = 300;
  const pad = 40;
  const vals = result.points.flatMap((p) => [p.predictedMin, p.actualMin]).sort((a, b) => a - b);
  const domainMax = Math.max(10, Math.ceil(quantile(vals, 0.97) / 10) * 10);
  const sx = (v: number) => pad + Math.min(1, v / domainMax) * (W - 2 * pad);
  const sy = (v: number) => H - pad - Math.min(1, v / domainMax) * (H - 2 * pad);

  return (
    <div className="chart">
      <div className="chart-title">
        Predicted vs actual recovery ·{" "}
        <span className="muted">
          IQR coverage {Number.isNaN(result.coverage) ? "—" : (result.coverage * 100).toFixed(0)}%
          (target ~50%) · median abs err{" "}
          {Number.isNaN(result.medianAbsErrorMin) ? "—" : Math.round(result.medianAbsErrorMin)}m ·
          n={result.n}
          {result.excludedSchedule > 0 &&
            ` · ${result.excludedSchedule} schedule excl.`}
        </span>
      </div>
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
        <text x={W - pad} y={H - 8} fill="var(--muted)" fontSize="10" textAnchor="end">
          predicted minutes →
        </text>
        <text x={8} y={pad + 4} fill="var(--muted)" fontSize="10">
          actual ↑
        </text>
        <text x={sx(domainMax)} y={sy(0) + 14} fill="var(--muted)" fontSize="9" textAnchor="end">
          {domainMax}m
        </text>
      </svg>
      <div className="legend">
        <span><i style={{ background: "var(--normal)" }} /> inside IQR</span>
        <span><i style={{ background: "var(--suspended)" }} /> outside IQR</span>
      </div>
    </div>
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
    <div className="chart">
      <div className="chart-title">
        Recovery accuracy ·{" "}
        <span className="muted">
          IQR coverage target ~50% · MAE/RMSE of recovery_minutes vs actual
        </span>
      </div>
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
    </div>
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
    return <div className="muted">No non-normal regimes observed in this window.</div>;

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
  const fmtTick = (ts: number) =>
    new Date(ts * 1000).toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "numeric",
    });

  return (
    <div className="chart">
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        {Array.from({ length: ticks + 1 }, (_, i) => {
          const ts = t0 + (span * i) / ticks;
          return (
            <g key={i}>
              <line x1={sx(ts)} y1={16} x2={sx(ts)} y2={H - 8} stroke="var(--border)" />
              <text x={sx(ts)} y={11} fill="var(--muted)" fontSize="9" textAnchor="middle">
                {fmtTick(ts)}
              </text>
            </g>
          );
        })}
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
      <div className="legend">
        <span><i style={{ background: "var(--normal)", opacity: 0.4 }} /> normal</span>
        <span><i style={{ background: "var(--disrupted)" }} /> disrupted</span>
        <span><i style={{ background: "var(--suspended)" }} /> suspended</span>
      </div>
    </div>
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
      <div className="muted">
        No planned-work windows with ≥2 archived versions in this window.
      </div>
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
    <div className="chart">
      <div className="chart-title">
        Resume churn ·{" "}
        <span className="muted">
          {result.windows.toLocaleString()} windows · pushed {pct(result.pushedPct)} · pulled{" "}
          {pct(result.pulledPct)} · stable{" "}
          {pct(result.stable / result.windows)}
        </span>
      </div>
      <p className="grp-note" style={{ margin: "2px 0 8px" }}>
        How far announced resume times moved later across an alert&apos;s versions
        (pushed windows only). The MTA rarely pulls a resume earlier, so a push is
        the main way &ldquo;it&apos;s back&rdquo; gets announced too soon.
      </p>
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
        <text x={pad} y={H - 8} fill="var(--muted)" fontSize="9">
          0m
        </text>
        <text x={W - pad} y={H - 8} fill="var(--muted)" fontSize="9" textAnchor="end">
          {hiMag}m push →
        </text>
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
    </div>
  );
}

// --- Schedule reliability: adherence ---

export function AdherencePanel({ result }: { result: AdherenceResult }) {
  if (result.n === 0)
    return (
      <div className="muted">
        No schedule resumes with an observed return-to-normal yet
        {result.censored > 0 && ` (${result.censored} censored)`}.
      </div>
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
    <div className="chart">
      <div className="chart-title">
        Schedule adherence ·{" "}
        <span className="muted">
          n={result.n} · median{" "}
          {Number.isNaN(result.medianErrorMin) ? "—" : `${result.medianErrorMin > 0 ? "+" : ""}${Math.round(result.medianErrorMin)}m`}{" "}
          · overran {pct(result.overrunPct)} · on-time {pct(result.onTimePct)}
          {result.censored > 0 && ` · ${result.censored} censored`}
        </span>
      </div>
      <p className="grp-note" style={{ margin: "2px 0 8px" }}>
        Announced resume vs when the line actually returned to normal. Right of
        the dashed line = back later than promised (overrun); the only read on the
        silent-overrun case.
      </p>
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
        <text x={pad} y={H - 8} fill="var(--muted)" fontSize="9">
          −{bound}m early
        </text>
        <text x={W - pad} y={H - 8} fill="var(--muted)" fontSize="9" textAnchor="end">
          +{bound}m late →
        </text>
      </svg>
    </div>
  );
}

// --- Detection latency ---

export function DetectionLatencyPanel({ result }: { result: DetectionLatencyResult }) {
  if (result.n === 0)
    return (
      <div className="muted">
        No alert→disruption detections in this window
        {result.missed > 0 && ` (${result.missed} onsets never flipped)`}.
      </div>
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
    <div className="chart">
      <div className="chart-title">
        Detection latency ·{" "}
        <span className="muted">
          median{" "}
          {Number.isNaN(result.medianLatencyMin) ? "—" : Math.round(result.medianLatencyMin)}m
          · n={result.n}
          {result.missed > 0 && ` · ${result.missed} missed`}
        </span>
      </div>
      <p className="grp-note" style={{ margin: "2px 0 8px" }}>
        Minutes from a real alert appearing to the HMM flipping to
        disrupted/suspended. Resolution is one prediction tick (~5m). Breakdown by
        cause below.
      </p>
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
        <text x={pad} y={H - 8} fill="var(--muted)" fontSize="9">
          0m
        </text>
        <text x={W - pad} y={H - 8} fill="var(--muted)" fontSize="9" textAnchor="end">
          {hi}m latency →
        </text>
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
    </div>
  );
}

// --- Transition matrix heatmap ---

export function TransitionHeatmap({
  route,
  transition,
  states,
}: {
  route: string;
  transition: number[][];
  states: string[];
}) {
  const cell = 64;
  const labelW = 70;
  const W = labelW + states.length * cell + 8;
  const H = labelW + states.length * cell + 8;
  return (
    <div className="chart">
      <div className="chart-title">
        Transition matrix · <span className="muted">{route} (from row → to col)</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 320 }}>
        {states.map((st, c) => (
          <text
            key={`c${c}`}
            x={labelW + c * cell + cell / 2}
            y={labelW - 6}
            fill="var(--muted)"
            fontSize="10"
            textAnchor="middle"
          >
            {st.slice(0, 4)}
          </text>
        ))}
        {states.map((st, r) => (
          <text key={`r${r}`} x={labelW - 6} y={labelW + r * cell + cell / 2 + 4} fill="var(--muted)" fontSize="10" textAnchor="end">
            {st.slice(0, 4)}
          </text>
        ))}
        {transition.map((row, r) =>
          row.map((v, c) => (
            <g key={`${r}-${c}`}>
              <rect
                x={labelW + c * cell}
                y={labelW + r * cell}
                width={cell - 2}
                height={cell - 2}
                fill={stateColor(states[c])}
                fillOpacity={0.12 + 0.8 * v}
              />
              <text
                x={labelW + c * cell + cell / 2}
                y={labelW + r * cell + cell / 2 + 4}
                fill="var(--text)"
                fontSize="12"
                textAnchor="middle"
              >
                {v.toFixed(2)}
              </text>
            </g>
          )),
        )}
      </svg>
    </div>
  );
}
