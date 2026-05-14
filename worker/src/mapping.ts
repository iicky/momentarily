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
