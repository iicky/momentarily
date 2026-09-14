/**
 * Map an MTA alert_type string to a coarse status label.
 *
 * Mirrors src/momentarily/mapping.py. Used by the compat view so existing
 * HA installs see stable status strings. Unknown alert_types fall through as
 * their raw label rather than being dropped.
 */

export const NO_ALERTS_FALLBACK = 'Good Service';

const SUBSTRING_TO_STATUS: ReadonlyArray<readonly [string, string]> = [
  // Order matters — first match wins. Put the most specific patterns first.
  ['Planned -', 'Planned Work'],
  ['Suspend', 'Suspended'],
  ['No Trains', 'Suspended'],
  ['No Scheduled Service', 'Suspended'],
  ['Severe Delays', 'Delays'],
  ['Delays', 'Delays'],
  ['Reroute', 'Service Change'],
  ['Trains Rerouted', 'Service Change'],
  ['Stops Skipped', 'Service Change'],
  ['Express to Local', 'Service Change'],
  ['Local to Express', 'Service Change'],
  ['Service Change', 'Service Change'],
  ['Boarding Change', 'Service Change'],
  ['Slow Speeds', 'Slow Speeds'],
  ['Station Notice', 'Information'],
  ['Special Schedule', 'Information'],
  ['Information', 'Information'],
];

export function coarseStatus(alertType: string | null | undefined): string {
  if (!alertType) return NO_ALERTS_FALLBACK;
  for (const [needle, status] of SUBSTRING_TO_STATUS) {
    if (alertType.includes(needle)) return status;
  }
  return alertType;
}

/**
 * The `category` axis — the cause/kind of disruption, in our own stable
 * vocabulary. Orthogonal to `condition` (severity). Derived from the coarse
 * label so there's one mapping table to maintain, not two.
 */
export type AlertCategory =
  | 'none'
  | 'planned_work'
  | 'delays'
  | 'service_change'
  | 'service_suspension'
  | 'slow_speeds'
  | 'information'
  | 'other';

const LABEL_TO_CATEGORY: Readonly<Record<string, AlertCategory>> = {
  'Good Service': 'none',
  'Planned Work': 'planned_work',
  Delays: 'delays',
  'Service Change': 'service_change',
  Suspended: 'service_suspension',
  'Slow Speeds': 'slow_speeds',
  Information: 'information',
};

export function categoryForLabel(label: string): AlertCategory {
  return LABEL_TO_CATEGORY[label] ?? 'other';
}

/**
 * The canonical severe-only truth floor. A route-tick grades `disrupted` only
 * when an active alert reaches this tier. Mirrors
 * src/momentarily/mapping.py CANONICAL_SEVERITY_FLOOR (truth_version 2): at
 * floor 2, only Severe Delays (tier 2) and suspensions (tier 3) count, so
 * chronic minor delays and reroutes (tier 1) and planned work (tier 0) read
 * normal. This is the SAME definition training/review.py grades the published
 * condition against.
 */
export const CANONICAL_SEVERITY_FLOOR = 2;

// Exact-match severity mapping — a faithful port of the buckets in
// src/momentarily/mapping.py coarse_status's ALERT_TYPE_TO_STATUS table,
// restricted to the three buckets severity cares about (Suspended / Delays /
// Service Change). Deliberately NOT the substring coarseStatus above: that one
// serves the label axis and diverges here (it maps 'Slow Speeds' to its own
// bucket and 'Part Suspended' to Suspended, where the Python truth grades them
// Delays and Service Change). The truth-grade path mirrors Python's exact-match
// rule so the derive_graded_mta_state parity fixture holds.
const ALERT_TYPE_SEVERITY_BUCKET: Readonly<
  Record<string, 'Suspended' | 'Delays' | 'Service Change'>
> = {
  Delays: 'Delays',
  'Some Delays': 'Delays',
  'Severe Delays': 'Delays',
  'Slow Speeds': 'Delays',
  'Service Change': 'Service Change',
  'Part Suspended': 'Service Change',
  'Trains Rerouted': 'Service Change',
  Reroute: 'Service Change',
  'Stations Skipped': 'Service Change',
  'Stops Skipped': 'Service Change',
  'Local to Express': 'Service Change',
  'Express to Local': 'Service Change',
  'Reduced Service': 'Service Change',
  'Boarding Change': 'Service Change',
  Suspended: 'Suspended',
  'No Trains': 'Suspended',
  Cancellations: 'Service Change',
  'Track Change': 'Service Change',
  Weather: 'Service Change',
};

/**
 * Severity rank for an MTA alert_type, for grading ground truth:
 *   3 — service suspension (Suspended / No Trains / No <Direction> Service)
 *   2 — major degradation (Severe Delays)
 *   1 — minor/moderate (ordinary Delays, Slow Speeds, reroutes, skips)
 *   0 — non-disruptive (Planned work, Information, Good Service, unknown)
 *
 * A 1:1 port of src/momentarily/mapping.py severity_tier, pinned by
 * tests/fixtures/parity_graded_state.json.
 */
export function severityTier(alertType: string | null): number {
  if (alertType === null) return 0;
  if (alertType.startsWith('Planned')) return 0;
  let bucket: 'Suspended' | 'Delays' | 'Service Change' | undefined =
    ALERT_TYPE_SEVERITY_BUCKET[alertType];
  if (bucket === undefined && alertType.startsWith('No ') && alertType.includes('Service')) {
    bucket = 'Suspended';
  }
  if (bucket === 'Suspended') return 3;
  if (bucket === 'Delays' || bucket === 'Service Change') {
    return alertType.includes('Severe') ? 2 : 1;
  }
  return 0;
}

/**
 * Severity-graded ground-truth state from a route-tick's active alert_types.
 * suspended if any suspension alert (tier 3); disrupted if any alert reaches
 * `floor`; otherwise normal — so sub-floor alerts (minor delays, routine
 * reroutes) read normal. floor=1 reproduces the breadth truth; floor=2 is
 * severe-only. A 1:1 port of training/recovery_baseline.py derive_graded_mta_state, and
 * the SAME definition the review uses as canonical truth. Pinned by
 * tests/fixtures/parity_graded_state.json.
 */
export function deriveGradedMtaState(
  alertTypes: readonly string[],
  floor: number,
): 'normal' | 'disrupted' | 'suspended' {
  const tiers = alertTypes.map(severityTier);
  if (tiers.some((t) => t === 3)) return 'suspended';
  if (tiers.some((t) => t >= floor)) return 'disrupted';
  return 'normal';
}
