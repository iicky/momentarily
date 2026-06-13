"use client";

// Hand-rolled SVG charts — these are bespoke scientific plots (calibration
// scatter on a diagonal, regime swimlane, transition heatmap) where a generic
// charting lib would fight us more than help.

export interface ReliabilityResult {
  horizonMin: number;
  bins: { p: number; predictedMean: number; observedFreq: number; n: number }[];
  brier: number;
  n: number;
}

export interface RecoveryResult {
  points: { route: string; predictedMin: number; actualMin: number; inIqr: boolean }[];
  coverage: number;
  n: number;
  medianAbsErrorMin: number;
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
