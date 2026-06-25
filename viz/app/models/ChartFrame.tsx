"use client";

import {
  createContext,
  useContext,
  useRef,
  useState,
  type ReactNode,
  type Ref,
} from "react";

// Shared chart chrome for the Models page: one frame, one metadata footer, one
// legend, one tooltip, one empty-state — so the diagnostics read as a single
// evaluation product instead of a pile of bespoke plots.

// Color roles. State colors (--normal/--disrupted/--suspended) mean a line's
// condition and nothing else; forecast/realized are reserved for
// predicted-vs-observed overlays so the two never collide in a reader's head.
export const FORECAST = "var(--forecast)";
export const REALIZED = "var(--realized)";

export interface ChartMeta {
  // Truth the panel grades against, e.g. "argmax transition", "alert
  // clearance", "movement". Phase 1 (truth-source honesty) fills these in.
  source?: string;
  // Whether that truth is independent of the HMM. false → self-consistency,
  // which the footer flags so readers don't mistake it for real-world recovery.
  independent?: boolean;
  unit?: string; // sample unit: "per-tick", "per-regime", "per-forecast"…
  n?: number;
  excluded?: number; // dropped from scoring (e.g. planned schedule)
  censored?: number; // right-censored / never resolved in window
  note?: string; // caps / downsampling, e.g. "scatter downsampled"
  feed?: "credentialed" | "public";
  extra?: ReactNode; // chart-specific chips (e.g. skill vs baseline)
}

// Page-level facts shared by every footer (the data window and which feed the
// page is reading), so individual charts don't each thread them through props.
interface ChartContext {
  feed?: "credentialed" | "public";
  window?: string; // e.g. "3d"
}
const ChartCtx = createContext<ChartContext>({});
export function ChartMetaProvider({
  value,
  children,
}: {
  value: ChartContext;
  children: ReactNode;
}) {
  return <ChartCtx.Provider value={value}>{children}</ChartCtx.Provider>;
}

export interface LegendItem {
  color: string;
  label: string;
  // Second encoding so hue isn't the only signal for a critical comparison.
  shape?: "box" | "line" | "dot";
}

export function Chip({
  children,
  tone,
}: {
  children: ReactNode;
  tone?: "warn" | "muted" | "good";
}) {
  return (
    <span className={`chart-chip${tone ? ` chart-chip-${tone}` : ""}`}>
      {children}
    </span>
  );
}

// Brier skill score as a chip. Positive (beats the baseline) reads green;
// negative is flagged — success styling is reserved for genuine skill.
export function SkillChip({
  label,
  bss,
}: {
  label: string;
  bss: number | null | undefined;
}) {
  if (bss == null || Number.isNaN(bss))
    return (
      <Chip tone="muted">
        {label} —
      </Chip>
    );
  const tone = bss >= 0 ? "good" : "warn";
  return (
    <Chip tone={tone}>
      {bss < 0 ? "⚠ " : ""}
      {label} {bss >= 0 ? "+" : ""}
      {(bss * 100).toFixed(0)}%
    </Chip>
  );
}

export function MetaFooter({ meta }: { meta: ChartMeta }) {
  const ctx = useContext(ChartCtx);
  const feed = meta.feed ?? ctx.feed;
  const chips: ReactNode[] = [];
  if (ctx.window) chips.push(<Chip key="win">window: {ctx.window}</Chip>);
  if (meta.source) {
    const dependent = meta.independent === false;
    chips.push(
      <Chip key="src" tone={dependent ? "warn" : undefined}>
        {dependent ? "⚠ " : ""}source: {meta.source}
        {meta.independent === false
          ? " · not independent of HMM"
          : meta.independent === true
            ? " · independent"
            : ""}
      </Chip>,
    );
  }
  if (meta.unit) chips.push(<Chip key="unit">unit: {meta.unit}</Chip>);
  if (meta.n != null)
    chips.push(<Chip key="n">n={meta.n.toLocaleString()}</Chip>);
  if (meta.excluded)
    chips.push(
      <Chip key="ex" tone="muted">
        {meta.excluded.toLocaleString()} excl.
      </Chip>,
    );
  if (meta.censored)
    chips.push(
      <Chip key="cen" tone="muted">
        {meta.censored.toLocaleString()} censored
      </Chip>,
    );
  if (meta.note)
    chips.push(
      <Chip key="note" tone="muted">
        {meta.note}
      </Chip>,
    );
  if (meta.extra) chips.push(<span key="extra" style={{ display: "contents" }}>{meta.extra}</span>);
  if (feed)
    chips.push(
      <Chip key="feed" tone="muted">
        {feed}
      </Chip>,
    );
  if (!chips.length) return null;
  return <div className="chart-meta">{chips}</div>;
}

export function Legend({ items }: { items: LegendItem[] }) {
  return (
    <div className="legend">
      {items.map((it, i) => (
        <span key={i}>
          <i
            className={`lg-${it.shape ?? "box"}`}
            style={{
              background: it.shape === "line" ? it.color : it.color,
              borderColor: it.color,
            }}
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}

export interface ChartFrameProps {
  // Optional: panels grouped under a page-level <h3> may carry no in-card title.
  title?: ReactNode;
  titleMeta?: ReactNode; // muted inline suffix after the title
  note?: ReactNode; // one-line explanation under the title
  children?: ReactNode;
  legend?: LegendItem[];
  meta?: ChartMeta;
  empty?: boolean; // render emptyText instead of children
  emptyText?: ReactNode;
  maxWidth?: number;
  className?: string;
  // From useTooltip(): makes the frame the positioning context + draws overlay.
  containerRef?: Ref<HTMLDivElement>;
  overlay?: ReactNode;
}

export function ChartFrame({
  title,
  titleMeta,
  note,
  children,
  legend,
  meta,
  empty,
  emptyText,
  maxWidth,
  className,
  containerRef,
  overlay,
}: ChartFrameProps) {
  return (
    <div
      className={`chart${className ? ` ${className}` : ""}`}
      ref={containerRef}
      style={{
        position: overlay !== undefined ? "relative" : undefined,
        maxWidth,
      }}
    >
      {(title != null || titleMeta != null) && (
        <div className="chart-title">
          {title}
          {titleMeta != null && (
            <>
              {title != null ? " · " : ""}
              <span className="muted">{titleMeta}</span>
            </>
          )}
        </div>
      )}
      {note != null && (
        <div className="grp-note" style={{ marginTop: 0 }}>
          {note}
        </div>
      )}
      {empty ? (
        <div className="chart-empty muted">
          {emptyText ?? "No data in this window."}
        </div>
      ) : (
        children
      )}
      {!empty && legend && legend.length > 0 && <Legend items={legend} />}
      {meta && <MetaFooter meta={meta} />}
      {overlay}
    </div>
  );
}

// --- Shared axis primitives ---
// One styling vocabulary for every plot's axes: muted ticks at one size, titles
// at a slightly larger one, gridlines at one opacity. Category-axis charts
// (transition matrix, confusion, swimlane) keep their own row/col labels; these
// cover the numeric scales (CDF, scatter, histograms, the baseline strip).

const AXIS_TICK_FS = 9;
const AXIS_TITLE_FS = 10;

export function AxisTitle({
  x,
  y,
  children,
  rotate,
  anchor = "middle",
}: {
  x: number;
  y: number;
  children: ReactNode;
  rotate?: number;
  anchor?: "start" | "middle" | "end";
}) {
  return (
    <text
      x={x}
      y={y}
      fill="var(--muted)"
      fontSize={AXIS_TITLE_FS}
      textAnchor={anchor}
      transform={rotate ? `rotate(${rotate} ${x} ${y})` : undefined}
    >
      {children}
    </text>
  );
}

// Tick labels along one axis. `scale` maps a tick value to its pixel position on
// that axis; `at` is the fixed perpendicular coordinate (the row's y for an
// x-axis, the column's x for a y-axis).
export function AxisTicks({
  axis,
  scale,
  ticks,
  at,
  format = String,
  anchor,
}: {
  axis: "x" | "y";
  scale: (v: number) => number;
  ticks: number[];
  at: number;
  format?: (v: number) => string;
  anchor?: "start" | "middle" | "end";
}) {
  return (
    <>
      {ticks.map((t) => {
        const p = scale(t);
        return (
          <text
            key={t}
            x={axis === "x" ? p : at}
            y={axis === "x" ? at : p}
            fill="var(--muted)"
            fontSize={AXIS_TICK_FS}
            textAnchor={anchor ?? (axis === "x" ? "middle" : "end")}
          >
            {format(t)}
          </text>
        );
      })}
    </>
  );
}

// Gridlines perpendicular to `axis`, spanning [from, to] pixels on the other.
export function GridLines({
  axis,
  scale,
  ticks,
  from,
  to,
  strong,
}: {
  axis: "x" | "y";
  scale: (v: number) => number;
  ticks: number[];
  from: number;
  to: number;
  strong?: (v: number) => boolean;
}) {
  return (
    <>
      {ticks.map((t) => {
        const p = scale(t);
        const op = strong?.(t) ? 0.6 : axis === "x" ? 0.3 : 0.4;
        return axis === "x" ? (
          <line key={t} x1={p} x2={p} y1={from} y2={to} stroke="var(--border)" strokeOpacity={op} />
        ) : (
          <line key={t} x1={from} x2={to} y1={p} y2={p} stroke="var(--border)" strokeOpacity={op} />
        );
      })}
    </>
  );
}

// Shared floating tooltip: a container ref + a positioned overlay. Coordinates
// are measured relative to the container so it works the same for any scaled
// SVG. Pair with ChartFrame's containerRef/overlay props.
export function useTooltip() {
  const ref = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; node: ReactNode } | null>(
    null,
  );
  const show = (e: { clientX: number; clientY: number }, node: ReactNode) => {
    const box = ref.current?.getBoundingClientRect();
    if (!box) return;
    setTip({ x: e.clientX - box.left, y: e.clientY - box.top, node });
  };
  const hide = () => setTip(null);
  const overlay = tip ? (
    <div
      style={{
        position: "absolute",
        left: tip.x + 14,
        top: tip.y + 14,
        // flip left when near the right edge so it never clips out of the card
        transform:
          ref.current && tip.x > ref.current.offsetWidth - 150
            ? "translateX(-100%) translateX(-28px)"
            : undefined,
        background: "var(--panel-2)",
        border: "1px solid var(--border)",
        borderRadius: 6,
        padding: "7px 10px",
        fontSize: 11,
        lineHeight: 1.5,
        color: "var(--text)",
        pointerEvents: "none",
        whiteSpace: "nowrap",
        boxShadow: "0 4px 14px rgba(0,0,0,0.35)",
        zIndex: 10,
      }}
    >
      {tip.node}
    </div>
  ) : null;
  return { ref, show, hide, overlay };
}
