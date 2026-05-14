/**
 * Render and publish the public snapshot.
 *
 * Shape matches src/momentarily/schema.py's Snapshot (subset for v0):
 *   - schema_version, generated_at, attribution, supported_modes, freshness
 *   - route_status: per-route alerts, primary type, label, per-direction,
 *     HMM-derived condition + recovery_minutes
 *
 * Stations/equipment/compat/system rollup land in the next iteration.
 *
 * Output is publicly readable at https://feed.momentarily.nyc/v1/snapshot.json
 * via the R2 custom domain. Cache headers per ADR (max-age=60, s-maxage=300).
 */

import type { RouteRoll } from './alpha';
import type { RouteSnapshot } from './derive';
import { N_STATES, STATES, projectForward } from './hmm';
import { coarseStatus, NO_ALERTS_FALLBACK } from './mapping';
import type { TrainedParams } from './params';
import { paramsForRoute } from './params';

const SNAPSHOT_KEY = 'v1/snapshot.json';

export const SCHEMA_VERSION = '1';

export const ATTRIBUTION =
  'Snapshot built from MTA GTFS-RT feeds via api.mta.info. '
  + 'Published by Momentarily (https://feed.momentarily.nyc). '
  + 'Not affiliated with the MTA.';

interface DirectionStatus {
  primary_alert_type: string | null;
  label: string | null;
}

interface Inference {
  condition: string;
  recovery_minutes: number;
  is_disrupted: boolean;
  p_normal: number;
  p_disrupted: number;
  p_suspended: number;
  regime_entered_at: number;
  regime_age_seconds: number;
  recovery_minutes_low: number;
  recovery_minutes_high: number;
  p_normal_in_30min: number;
  p_normal_in_60min: number;
  p_normal_in_120min: number;
}

interface RouteStatusOut {
  route_id: string;
  alerts: string[];
  primary_alert_type: string | null;
  label: string;
  by_direction: {
    northbound: DirectionStatus;
    southbound: DirectionStatus;
  };
  inference: Inference | null;
}

interface Snapshot {
  schema_version: string;
  generated_at: number;
  attribution: string;
  supported_modes: string[];
  freshness: {
    subway_alerts: number;
  };
  route_status: Record<string, RouteStatusOut>;
}

export function buildSnapshot(args: {
  generatedAt: number;
  alertsFreshness: number;
  routeSnapshots: Map<string, RouteSnapshot>;
  rolls: Record<string, RouteRoll>;
  trainedParams: TrainedParams | null;
  tickSeconds: number;
}): Snapshot {
  const route_status: Record<string, RouteStatusOut> = {};

  for (const [routeId, snap] of args.routeSnapshots) {
    const roll = args.rolls[routeId];
    let inference: Inference | null = null;
    if (roll) {
      inference = buildInference(roll, args.generatedAt, args.tickSeconds, routeId, args.trainedParams);
    }

    route_status[routeId] = {
      route_id: routeId,
      alerts: snap.active_alert_ids,
      primary_alert_type: snap.primary_alert_type,
      label: snap.coarse_label,
      by_direction: {
        northbound: {
          primary_alert_type: snap.by_direction.northbound,
          label: snap.by_direction.northbound
            ? coarseStatus(snap.by_direction.northbound)
            : null,
        },
        southbound: {
          primary_alert_type: snap.by_direction.southbound,
          label: snap.by_direction.southbound
            ? coarseStatus(snap.by_direction.southbound)
            : null,
        },
      },
      inference,
    };
  }

  // Also include routes we have an alpha for but no current alerts — they
  // exist and are presumably "normal." Skip these for v0 to keep output tight;
  // they can be added when we have a real route registry.

  return {
    schema_version: SCHEMA_VERSION,
    generated_at: args.generatedAt,
    attribution: ATTRIBUTION,
    supported_modes: ['subway'],
    freshness: { subway_alerts: args.alertsFreshness },
    route_status,
  };
}

function buildInference(
  roll: RouteRoll,
  now: number,
  tickSeconds: number,
  routeId: string,
  trained: TrainedParams | null,
): Inference {
  const probs = roll.filter.probabilities;
  const params = paramsForRoute(trained, routeId);

  // p_normal_in_X — project forward to find marginal P(normal in k min)
  const ticksFor = (minutes: number): number =>
    Math.max(1, Math.round((minutes * 60) / tickSeconds));
  const p30 = projectForward(roll.filter, params, ticksFor(30));
  const p60 = projectForward(roll.filter, params, ticksFor(60));
  const p120 = projectForward(roll.filter, params, ticksFor(120));

  // Dwell math, using the same percentile geometry as Python's expected_dwell_ticks
  const argmaxIdx = argmaxOf(probs);
  const selfLoop = params.transition[argmaxIdx]![argmaxIdx]!;
  const dwellTicks = dwellQuantiles(selfLoop);
  const dwellToMinutes = (t: number): number => Math.round((t * tickSeconds) / 60);

  // Use the published label (hysteresis-stable) for `condition`. If still
  // "unknown" from a feed gap, fall back to argmax — Inference itself isn't
  // gated on hysteresis, that's the publish layer's job.
  const condition =
    roll.published.label === 'unknown' ? STATES[argmaxIdx]! : roll.published.label;

  return {
    condition,
    recovery_minutes: dwellToMinutes(dwellTicks.median),
    is_disrupted: probs[1] + probs[2] > 0.7,
    p_normal: probs[0],
    p_disrupted: probs[1],
    p_suspended: probs[2],
    regime_entered_at: roll.filter.regime_entered_at,
    regime_age_seconds: Math.max(0, now - roll.filter.regime_entered_at),
    recovery_minutes_low: dwellToMinutes(dwellTicks.q25),
    recovery_minutes_high: dwellToMinutes(dwellTicks.q75),
    p_normal_in_30min: p30[0],
    p_normal_in_60min: p60[0],
    p_normal_in_120min: p120[0],
  };
}

function argmaxOf(v: readonly [number, number, number]): 0 | 1 | 2 {
  if (v[0] >= v[1] && v[0] >= v[2]) return 0;
  if (v[1] >= v[2]) return 1;
  return 2;
}

function dwellQuantiles(selfLoop: number): {
  median: number;
  q25: number;
  q75: number;
} {
  const LARGE = 10_000;
  if (selfLoop >= 1.0) return { median: LARGE, q25: LARGE, q75: LARGE };
  if (selfLoop <= 0) return { median: 1, q25: 1, q75: 1 };
  const logSelf = Math.log(selfLoop);
  const q = (qv: number): number => {
    const target = 1 - qv;
    return target <= 0 ? LARGE : Math.max(1, Math.ceil(Math.log(target) / logSelf));
  };
  return { median: q(0.5), q25: q(0.25), q75: q(0.75) };
}

export async function publishSnapshot(
  bucket: R2Bucket,
  snapshot: Snapshot,
): Promise<void> {
  await bucket.put(SNAPSHOT_KEY, JSON.stringify(snapshot), {
    httpMetadata: {
      contentType: 'application/json',
      cacheControl: 'public, max-age=60, s-maxage=300',
    },
  });
}

// Re-export for the entrypoint to use, no need to import N_STATES directly.
export { N_STATES };
export const TICK_SECONDS = 300;
export const NO_ALERTS = NO_ALERTS_FALLBACK;
