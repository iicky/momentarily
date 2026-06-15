/**
 * Derive a compact per-route "is service actually running" metric from the
 * decoded GTFS-RT trip-updates feeds. This is archived for OFFLINE validation
 * only (an independent recovery-truth signal — see the training tooling); it is
 * NOT published to snapshot.json in this phase.
 *
 * The headline channel is `assigned_n`: NYCT marks a trip `is_assigned` when a
 * physical train is dispatched to run it, so this counts trains actually moving
 * on a route — orthogonal to both the alerts feed and the HMM's own argmax. A
 * suspension drives it toward 0; degraded service depresses it below the
 * route's time-of-day baseline (the baseline is computed downstream in Python
 * from this same archived series).
 */

import type { TripLite } from './gtfsrt';

export interface ServiceRow {
  assigned_n: number; // dispatched, running trains on this route
  trips_n: number; // all trips referencing this route (assigned or not)
  with_movement_n: number; // assigned trips with >=1 remaining stop (going somewhere)
  dir_n: number; // assigned, northbound
  dir_s: number; // assigned, southbound
}

/** Express variants (6X, 7X, FX) fold to their base route, matching derive.ts. */
function baseRoute(routeId: string): string {
  return routeId.replace(/X$/, '');
}

/** NYCT direction: the extension enum (1=N, 3=S) when present, else the
 * direction char after `..` in the trip_id (e.g. `..N` / `..S`). */
function directionOf(t: TripLite): 'N' | 'S' | null {
  if (t.direction === 1) return 'N';
  if (t.direction === 3) return 'S';
  const i = t.tripId.indexOf('..');
  if (i >= 0) {
    const c = t.tripId[i + 2];
    if (c === 'N') return 'N';
    if (c === 'S') return 'S';
  }
  return null;
}

function emptyRow(): ServiceRow {
  return { assigned_n: 0, trips_n: 0, with_movement_n: 0, dir_n: 0, dir_s: 0 };
}

/**
 * Group decoded trips (across all fetched feeds) into per-route service rows.
 * Trips with no route id are already dropped by the decoder.
 */
export function deriveRouteServiceMetric(
  trips: TripLite[],
): Map<string, ServiceRow> {
  const out = new Map<string, ServiceRow>();
  for (const t of trips) {
    const route = baseRoute(t.routeId);
    let row = out.get(route);
    if (!row) {
      row = emptyRow();
      out.set(route, row);
    }
    row.trips_n += 1;
    if (t.isAssigned) {
      row.assigned_n += 1;
      if (t.stopCount > 0) row.with_movement_n += 1;
      const dir = directionOf(t);
      if (dir === 'N') row.dir_n += 1;
      else if (dir === 'S') row.dir_s += 1;
    }
  }
  return out;
}
