// Two-car brand mark. Gap and bar height encode route state — geometry from
// docs/brand/assets/mark-*.svg. Colour is inherited via currentColor from
// the .mark.* CSS classes when kind is passed, or from the aria-label prop
// for standalone use without a sibling text label.
export type MarkKind = "normal" | "disrupted" | "suspended" | "muted" | "logo";

// [x, y, height] per bar; width=5 rx=2.5 fixed on a 24-unit grid.
// Three service-state silhouettes are shape-distinct in monochrome:
//   normal    — h=18 y=3  nominal gap=4
//   disrupted — h=18 y=3  gap≈8 (wider footprint)
//   suspended — h=10 y=7  gap=11 saturated (short bars, max spread)
// muted/logo use the nominal geometry; no-reading state is carried by the
// text label, not a fourth shape — the three service states exhaust the
// shape encoding the SVG assets define.
const MARK_BARS: Record<MarkKind, [number, number, number][]> = {
  normal:    [[5, 3, 18], [14, 3, 18]],
  disrupted: [[3.02, 3, 18], [15.98, 3, 18]],
  suspended: [[1.5, 7, 10], [17.5, 7, 10]],
  muted:     [[5, 3, 18], [14, 3, 18]],
  logo:      [[5, 3, 18], [14, 3, 18]],
};

export function StateMark({
  kind,
  size = 20,
  label,
}: {
  kind: MarkKind;
  size?: number;
  /** Accessible label. Omit when a sibling text node carries the state name. */
  label?: string;
}) {
  return (
    <svg
      className={`mark ${kind}`}
      viewBox="0 0 24 24"
      width={size}
      height={size}
      fill="currentColor"
      aria-hidden={label ? undefined : "true"}
      aria-label={label}
      role={label ? "img" : undefined}
    >
      {MARK_BARS[kind].map(([x, y, h], i) => (
        <rect key={i} x={x} y={y} width={5} height={h} rx={2.5} />
      ))}
    </svg>
  );
}

/** Map a .cond CSS class string (normal/disrupted/suspended/quiet/unknown) to MarkKind. */
export function clsToMarkKind(cls: string): MarkKind {
  if (cls === "disrupted") return "disrupted";
  if (cls === "suspended") return "suspended";
  if (cls === "normal") return "normal";
  return "muted";
}
