// The Gauge: a per-route "how bad is it right now vs usual" dial. It renders
// route_status.service_percentile — a COARSE placement of this line's current
// train count against its own same-daypart baseline, computed by the Worker
// (movement_state.servicePercentile) from the only shape the baseline carries:
// the cell's p10, median, and p90.
//
// What the number honestly means, and its hard limits:
//   - It is EXACT at three points — p10 -> 10, median -> 50, p90 -> 90 — and a
//     straight-line interpolation between them. It is NOT an empirical percentile
//     drawn from the full distribution (the feed does not carry one), so it
//     cannot resolve a bimodal cell finer than those anchors.
//   - Below the median it interpolates down to the one true floor: zero trains at
//     the 0th percentile. That is where the resolution lives, and it is the "bad"
//     direction the dial exists to show.
//   - It SATURATES at 90. The baseline has no anchor above p90, so any count at
//     or above this cell's own p90 reads exactly 90 — "top decile", no finer. The
//     Worker never emits 91-100; the arc's last tenth is a labelled ceiling, not a
//     reachable value.
//   - It is a percentile of the baseline, NOT a forecast. It says where the
//     present sits against history for this time of week, and predicts nothing.
//
// Working name pending Mickey's call (candidates: Misery Index, The Gauge,
// Commute Weather, Rough Ride Index).

// Reuse the supply tones the card/meter already use, so the dial never disagrees
// with the rest of a route's surface about what its supply reading means.
export type GaugeTone = "low" | "thin" | "normal" | "high" | "unknown";

// Above this the reading saturates: the baseline has no p95/p99 to place against,
// so the Worker caps servicePercentile here and the arc past it is a ceiling.
const SATURATION = 90;

const CX = 60;
const CY = 58;
const R = 46;

// A point on the upper semicircle for a 0-100 reading: 0 -> left (180deg),
// 100 -> right (0deg). SVG y grows downward, so the arc bows up over the centre.
function pointFor(value: number): { x: number; y: number } {
  const theta = (Math.PI * (100 - value)) / 100; // radians, 180deg at 0
  return { x: CX + R * Math.cos(theta), y: CY - R * Math.sin(theta) };
}

// SVG arc path from reading `from` to `to` along the upper semicircle. Always the
// short way (largeArc=0), sweeping clockwise as the reading rises (sweep=1).
function arcPath(from: number, to: number): string {
  const a = pointFor(from);
  const b = pointFor(to);
  return `M ${a.x.toFixed(2)} ${a.y.toFixed(2)} A ${R} ${R} 0 0 1 ${b.x.toFixed(2)} ${b.y.toFixed(2)}`;
}

function caption(pct: number, saturated: boolean): string {
  if (saturated) return "in the top tenth of trains for this time of week";
  if (pct < 40) return "fewer trains than usual for this time of week";
  if (pct <= 70) return "about the usual number of trains for this time of week";
  return "more trains than usual for this time of week";
}

/**
 * The dial. `percentile` is route_status.service_percentile (0-90; the Worker
 * saturates at 90). `tone` is the shared supply tone. Render nothing upstream
 * when the percentile is null — there is no honest dial without a reading.
 */
export function Gauge({
  percentile,
  tone,
  size = 132,
}: {
  percentile: number;
  tone: GaugeTone;
  size?: number;
}) {
  // Defensive clamp; the Worker already bounds this to [0, 90].
  const pct = Math.max(0, Math.min(SATURATION, Math.round(percentile)));
  const saturated = pct >= SATURATION;
  const end = pointFor(pct);
  const median = pointFor(50);
  const satStart = pointFor(SATURATION);
  const height = Math.round(size * 0.62);
  return (
    <figure className={`gauge ${tone}${saturated ? " sat" : ""}`} style={{ width: size }}>
      <svg
        viewBox="0 0 120 74"
        width={size}
        height={height}
        role="img"
        aria-label={
          saturated
            ? "Trains in the top tenth for this time of week (90th percentile or higher)"
            : `Trains at roughly the ${pct}th percentile for this time of week`
        }
      >
        {/* Resolvable track (0-90), then the labelled ceiling (90-100) the reading
            can never enter, then the fill painted up to the reading. */}
        <path className="gauge-track" d={arcPath(0, SATURATION)} fill="none" />
        <path className="gauge-ceiling" d={arcPath(SATURATION, 100)} fill="none" />
        {pct > 0 && <path className="gauge-fill" d={arcPath(0, pct)} fill="none" />}
        {/* Median mark — the 50th percentile, "a usual day for this hour". */}
        <line
          className="gauge-median"
          x1={median.x}
          y1={median.y - 3}
          x2={median.x}
          y2={median.y + 6}
        />
        <circle className="gauge-dot" cx={end.x} cy={end.y} r={4} />
        <text className="gauge-value" x={CX} y={CY - 6} textAnchor="middle">
          {saturated ? "90+" : pct}
        </text>
        <text className="gauge-unit" x={CX} y={CY + 8} textAnchor="middle">
          percentile
        </text>
        {/* End caps so the scale's meaning is legible in a screenshot. */}
        <text className="gauge-end" x={pointFor(0).x} y={CY + 12} textAnchor="start">
          few
        </text>
        <text className="gauge-end" x={satStart.x} y={CY + 12} textAnchor="end">
          many
        </text>
      </svg>
      <figcaption className="gauge-cap">
        {caption(pct, saturated)}
        <span className="gauge-fine"> · percentile of baseline, not a forecast</span>
      </figcaption>
    </figure>
  );
}
