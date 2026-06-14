/**
 * not_scheduled condition + schedule-based recovery.
 *
 * Covers the deterministic planned-work path: derive surfaces the planned-vs-
 * realtime namespace split and the current-window resume time; the snapshot
 * applies condition precedence, picks schedule vs HMM recovery, and keeps the
 * planned non-disruption out of the disruption rollups.
 */

import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import type { RouteSnapshot } from '../src/derive';
import { deriveRouteSnapshots } from '../src/derive';
import { TICK_SECONDS, buildSnapshot } from '../src/snapshot';

const NOW = 1_700_000_000;

/** Build a single GTFS-RT alert entity for one route. */
function entity(opts: {
  id: string;
  alertType: string;
  route: string;
  sortOrder?: number;
  periods: Array<{ start: number; end?: number }>;
}): unknown {
  return {
    id: opts.id,
    alert: {
      active_period: opts.periods,
      informed_entity: [
        {
          agency_id: 'MTASBWY',
          route_id: opts.route,
          'transit_realtime.mercury_entity_selector': {
            sort_order: `MTASBWY:${opts.route}:${opts.sortOrder ?? 10}`,
          },
        },
      ],
      header_text: { translation: [{ text: `${opts.alertType} on ${opts.route}`, language: 'en' }] },
      'transit_realtime.mercury_alert': { alert_type: opts.alertType },
    },
  };
}

function payload(...entities: unknown[]): unknown {
  return { entity: entities };
}

/** A roll whose filter sits in `state` with high confidence. */
function roll(state: 'normal' | 'disrupted' | 'suspended', enteredAt: number): RouteRoll {
  const probs: [number, number, number] =
    state === 'normal' ? [0.95, 0.04, 0.01] : state === 'disrupted' ? [0.04, 0.95, 0.01] : [0.02, 0.03, 0.95];
  return {
    filter: { probabilities: probs, regime_entered_at: enteredAt, last_updated_at: NOW },
    published: {
      label: state,
      pending_state: state,
      pending_streak: 5,
      last_updated_at: NOW,
    },
    alert_type_at_entry: null,
  };
}

describe('derive: namespace split + scheduled resume', () => {
  test('no-service alert → quiet observation, is_not_scheduled, current-window resume', () => {
    // Z runs rush-hours only: a recurring No Scheduled Service alert with the
    // current off-peak gap plus future gaps weeks out.
    const snaps = deriveRouteSnapshots(
      payload(
        entity({
          id: 'lmm:planned_work:19829',
          alertType: 'No Scheduled Service',
          route: 'Z',
          sortOrder: 20,
          periods: [
            { start: NOW - 3600, end: NOW + 1800 }, // current gap, ends in 30 min
            { start: NOW + 604_800, end: NOW + 604_800 + 7200 }, // next week
          ],
        }),
      ),
      NOW,
    );
    const z = snaps.get('Z')!;
    // Featureless observation: dropped from `counted` like Extra Service.
    expect(z.observation.alert_count).toBe(0);
    expect(z.observation.has_suspended_alert).toBe(false);
    expect(z.is_not_scheduled).toBe(true);
    expect(z.has_realtime_alert).toBe(false);
    // Current window end, NOT max(end) across the recurring alert.
    expect(z.scheduled_resume_at).toBe(NOW + 1800);
    // Still surfaced for display.
    expect(z.active_alert_ids).toContain('lmm:planned_work:19829');
  });

  test('real-time + planned alerts → has_realtime_alert set', () => {
    const snaps = deriveRouteSnapshots(
      payload(
        entity({
          id: 'lmm:alert:535417',
          alertType: 'Delays',
          route: 'N',
          sortOrder: 32,
          periods: [{ start: NOW - 600 }], // realtime: no resume end
        }),
        entity({
          id: 'lmm:planned_work:20534',
          alertType: 'Planned - Part Suspended',
          route: 'N',
          sortOrder: 25,
          periods: [{ start: NOW - 3600, end: NOW + 5400 }],
        }),
      ),
      NOW,
    );
    const n = snaps.get('N')!;
    expect(n.has_realtime_alert).toBe(true);
    expect(n.is_not_scheduled).toBe(false);
    expect(n.scheduled_resume_at).toBe(NOW + 5400);
    // The Delays + Part Suspended both count toward the HMM observation.
    expect(n.observation.alert_count).toBe(2);
  });
});

/** Hand-built RouteSnapshot with the new fields, for snapshot-layer control. */
function routeSnap(overrides: Partial<RouteSnapshot> & { route_id: string }): RouteSnapshot {
  return {
    observation: {
      alert_count: 0,
      severity_sum: 0,
      has_suspended_alert: false,
      has_delays: false,
      has_service_change: false,
      has_planned: false,
      tod_bin: 0,
    },
    active_alert_ids: [],
    alerts: [],
    severity_max: 0,
    primary_alert_type: null,
    coarse_label: 'Good Service',
    by_direction: {
      northbound: { alerts: [], primary_alert_type: null },
      southbound: { alerts: [], primary_alert_type: null },
    },
    has_realtime_alert: false,
    is_not_scheduled: false,
    scheduled_resume_at: null,
    ...overrides,
  };
}

function build(routeSnapshots: Map<string, RouteSnapshot>, rolls: Record<string, RouteRoll>) {
  return buildSnapshot({
    generatedAt: NOW,
    alertsFreshness: NOW,
    routeSnapshots,
    rolls,
    trainedParams: null,
    tickSeconds: TICK_SECONDS,
  });
}

describe('snapshot: not_scheduled condition + schedule recovery', () => {
  test('no-service route: not_scheduled, finite schedule recovery, out of rollups, compat', () => {
    const snaps = new Map<string, RouteSnapshot>([
      [
        'Z',
        routeSnap({
          route_id: 'Z',
          active_alert_ids: ['lmm:planned_work:19829'],
          severity_max: 20,
          primary_alert_type: 'No Scheduled Service',
          coarse_label: 'No Scheduled Service',
          is_not_scheduled: true,
          scheduled_resume_at: NOW + 1800,
        }),
      ],
    ]);
    const snap = build(snaps, { Z: roll('normal', NOW - 3600) });
    const rs = snap.route_status['Z']!;
    expect(rs.condition).toBe('not_scheduled');
    const inf = rs.inference!;
    expect(inf.condition).toBe('not_scheduled');
    expect(inf.recovery_source).toBe('schedule');
    expect(inf.resumes_at).toBe(NOW + 1800);
    expect(inf.recovery_minutes).toBe(30);
    expect(inf.overdue).toBe(false);
    expect(inf.is_disrupted).toBe(false);
    expect(inf.recovery_indeterminate).toBe(false);
    expect(inf.p_normal_in_60min).toBe(1); // resume within 60 min
    // Excluded from disruption rollups.
    expect(snap.system.lines_disrupted_count).toBe(0);
    expect(snap.system.by_mode.subway!.severity_max).toBe(0);
    expect(snap.system.most_degraded_line).toBeNull();
    // Compat renders the gap, not an unknown status.
    expect(snap.compat.subwaynow_routes['Z']!.status).toBe('Not Scheduled');
    expect(snap.compat.subwaynow_routes['Z']!.scheduled).toBe(false);
  });

  test('planned Part Suspended: HMM disrupted condition, schedule recovery (no indeterminate)', () => {
    const snaps = new Map<string, RouteSnapshot>([
      [
        'A',
        routeSnap({
          route_id: 'A',
          observation: {
            alert_count: 1,
            severity_sum: 25,
            has_suspended_alert: false,
            has_delays: false,
            has_service_change: false,
            has_planned: true,
            tod_bin: 0,
          },
          active_alert_ids: ['lmm:planned_work:20534'],
          severity_max: 25,
          primary_alert_type: 'Planned - Part Suspended',
          coarse_label: 'Part Suspended',
          scheduled_resume_at: NOW + 2700, // 45 min
        }),
      ],
    ]);
    const snap = build(snaps, { A: roll('disrupted', NOW - 7200) });
    const inf = snap.route_status['A']!.inference!;
    expect(snap.route_status['A']!.condition).toBe('disrupted');
    expect(inf.recovery_source).toBe('schedule');
    expect(inf.resumes_at).toBe(NOW + 2700);
    expect(inf.recovery_minutes).toBe(45);
    expect(inf.recovery_indeterminate).toBe(false);
    // A planned suspension is still a disruption — it counts.
    expect(inf.is_disrupted).toBe(true);
    expect(snap.system.lines_disrupted_count).toBe(1);
  });

  test('real-time alert wins precedence: HMM condition + HMM recovery even if planned also active', () => {
    const snaps = new Map<string, RouteSnapshot>([
      [
        'N',
        routeSnap({
          route_id: 'N',
          observation: {
            alert_count: 2,
            severity_sum: 50,
            has_suspended_alert: false,
            has_delays: true,
            has_service_change: false,
            has_planned: true,
            tod_bin: 0,
          },
          active_alert_ids: ['lmm:alert:535417', 'lmm:planned_work:20534'],
          severity_max: 32,
          primary_alert_type: 'Delays',
          coarse_label: 'Delays',
          has_realtime_alert: true,
          is_not_scheduled: false,
          scheduled_resume_at: NOW + 5400,
        }),
      ],
    ]);
    const snap = build(snaps, { N: roll('disrupted', NOW - 3600) });
    const inf = snap.route_status['N']!.inference!;
    expect(snap.route_status['N']!.condition).toBe('disrupted');
    expect(inf.recovery_source).toBe('hmm');
    expect(inf.resumes_at).toBeNull();
  });

  test('overdue: resume already passed but alert still active → recovery clamped to 0', () => {
    const snaps = new Map<string, RouteSnapshot>([
      [
        'Z',
        routeSnap({
          route_id: 'Z',
          active_alert_ids: ['lmm:planned_work:19829'],
          primary_alert_type: 'No Scheduled Service',
          coarse_label: 'No Scheduled Service',
          is_not_scheduled: true,
          scheduled_resume_at: NOW - 600, // announced resume 10 min ago
        }),
      ],
    ]);
    const inf = build(snaps, { Z: roll('normal', NOW - 7200) }).route_status['Z']!.inference!;
    expect(inf.condition).toBe('not_scheduled');
    expect(inf.recovery_source).toBe('schedule');
    expect(inf.overdue).toBe(true);
    expect(inf.recovery_minutes).toBe(0);
  });
});
