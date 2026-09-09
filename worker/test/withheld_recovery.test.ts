/**
 * The published snapshot withholds curve-fitted recovery; the grading stream
 * does not.
 *
 * Decision 2026-09-08: the 2026-09-04 review
 * (docs/review/2026-09-04-shadow-hmm/memo.md) graded recovery_minutes wrong —
 * causal skill -1.70 against a pre-window duration climatology, PIT 0.17, IQR
 * coverage 0.03-0.06 — so an estimate off a fitted dwell curve is not
 * published. What IS published is the deterministic schedule countdown, and
 * the fact that something was withheld.
 *
 * These tests pin the observable contract on both surfaces of one tick: the
 * document a consumer fetches, and the JSONL row the grader reads.
 */

import Ajv2020 from 'ajv/dist/2020';
import { describe, expect, test } from 'vitest';

import schema from '../../schema/snapshot.schema.json';
import type { RouteRoll } from '../src/alpha';
import { deriveRouteSnapshots } from '../src/derive';
import { buildPredictionRows, writePredictions } from '../src/grading';
import type { TrainedParams } from '../src/params';
import { parseTrainedParams } from '../src/params';
import type { Inference, Snapshot } from '../src/snapshot';
import { TICK_SECONDS, buildSnapshot, projectInference } from '../src/snapshot';

const ajv = new Ajv2020({ allErrors: true, strict: false });
const validate = ajv.compile(schema);

const NOW = 1_700_000_000;
const MIN = 60;
const HOUR = 3600;

// Heavy-tailed movement dwell fixtures, as in movement_recovery.test.ts.
const DISRUPTED_CURVE = Array.from({ length: 21 }, (_, i) => Math.round(120 * 1.35 ** i));
const NORMAL_CURVE = Array.from({ length: 21 }, (_, i) => Math.round(600 * 1.55 ** i));

function trained(routeIds: string[]): TrainedParams | null {
  const routes: Record<string, unknown> = {};
  const dwell_movement: Record<string, unknown> = {};
  for (const id of routeIds) {
    routes[id] = {
      transition: [
        [0.95, 0.04, 0.01],
        [0.08, 0.9, 0.02],
        [0.02, 0.1, 0.88],
      ],
      initial: [0.9, 0.08, 0.02],
      emissions: {
        poisson_lambda: [0.3, 4.0, 12.0],
        gamma_alpha: [1.0, 3.0, 6.0],
        gamma_beta: [2.0, 0.4, 0.2],
        bernoulli_p: [0.001, 0.05, 0.95],
        bernoulli_p_delays: [0.02, 0.6, 0.35],
        bernoulli_p_service_change: [0.02, 0.6, 0.4],
        bernoulli_p_planned: [0.05, 0.6, 0.35],
      },
    };
    dwell_movement[id] = {
      normal: { n: 40, q25_sec: 5368, median_sec: 48025, q75_sec: 429662, curve_sec: NORMAL_CURVE },
      disrupted: { n: 30, q25_sec: 538, median_sec: 2413, q75_sec: 10819, curve_sec: DISRUPTED_CURVE },
    };
  }
  return parseTrainedParams({
    schema_version: '1',
    trained_at: NOW,
    routes,
    dwell_movement,
  });
}

function alertEntity(opts: {
  id: string;
  alertType: string;
  route: string;
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
          'transit_realtime.mercury_entity_selector': { sort_order: `MTASBWY:${opts.route}:10` },
        },
      ],
      header_text: { translation: [{ text: `${opts.alertType} on ${opts.route}`, language: 'en' }] },
      'transit_realtime.mercury_alert': { alert_type: opts.alertType },
    },
  };
}

function disruptedRoll(): RouteRoll {
  return {
    filter: {
      probabilities: [0.000004, 0.999992, 0.000004],
      regime_entered_at: NOW - 3 * HOUR,
      last_updated_at: NOW,
    },
    published: { label: 'disrupted', pending_state: 'disrupted', pending_streak: 5, last_updated_at: NOW },
    alert_type_at_entry: 'Delays',
  };
}

/**
 * One tick carrying both arms at once: J is movement-sourced (a fitted curve)
 * and M is schedule-sourced (a real-time alert plus an announced resume 20
 * minutes out, the shape scheduleRecovery keys off).
 */
function tick(): { snapshot: Snapshot; full: Map<string, Inference> } {
  const full = new Map<string, Inference>();
  const snapshot = buildSnapshot({
    generatedAt: NOW,
    alertsFreshness: NOW,
    routeSnapshots: deriveRouteSnapshots(
      {
        entity: [
          alertEntity({
            id: 'lmm:alert:535417',
            alertType: 'Delays',
            route: 'J',
            periods: [{ start: NOW - 2 * HOUR }],
          }),
          alertEntity({
            id: 'lmm:planned_work:19830',
            alertType: 'Planned - Part Suspended',
            route: 'M',
            periods: [{ start: NOW - 3 * HOUR, end: NOW + 20 * MIN }],
          }),
        ],
      },
      NOW,
    ),
    rolls: { J: disruptedRoll(), M: disruptedRoll() },
    trainedParams: trained(['J', 'M']),
    tickSeconds: TICK_SECONDS,
    movementStates: {
      observed_at: NOW - 300,
      regimes: {
        J: { state: 'disrupted', entered_at: NOW - 30 * MIN },
        M: { state: 'disrupted', entered_at: NOW - 40 * MIN },
      },
    },
    vehicleFreshFeeds: [],
    vehicleExpectedFeeds: [],
    fullInferences: full,
  });
  return { snapshot, full };
}

describe('published snapshot withholds fitted recovery, grading stream keeps it', () => {
  test('a movement-sourced inference publishes no recovery numbers, and says so', () => {
    const { snapshot, full } = tick();
    const inf = snapshot.route_status.J!.inference!;

    expect(inf.recovery_source).toBe('movement');
    expect(inf.recovery_minutes).toBeNull();
    expect(inf.recovery_minutes_low).toBeNull();
    expect(inf.recovery_minutes_high).toBeNull();
    expect(inf.p_normal_in_30min).toBeNull();
    expect(inf.p_normal_in_60min).toBeNull();
    expect(inf.p_normal_in_120min).toBeNull();
    expect(inf.recovery_withheld).toBe('pending_validation');

    // Withheld, not absent: the estimate existed this tick.
    const graded = full.get('J')!;
    expect(graded.recovery_minutes).toBeGreaterThan(0);

    // Everything else on the block is untouched.
    expect(inf.condition).toBe(graded.condition);
    expect(inf.p_disrupted).toBe(graded.p_disrupted);
    expect(inf.regime_entered_at).toBe(graded.regime_entered_at);

    // And the document is still a valid published snapshot.
    expect(validate(snapshot), JSON.stringify(validate.errors, null, 2)).toBe(true);
  });

  test('a schedule-sourced inference is untouched: a countdown carries no fit', () => {
    const { snapshot, full } = tick();
    const inf = snapshot.route_status.M!.inference!;
    const graded = full.get('M')!;

    expect(inf.recovery_source).toBe('schedule');
    expect(inf.recovery_withheld).toBeNull();
    expect(inf.recovery_minutes).toBe(20);
    expect(inf.recovery_minutes_low).toBe(graded.recovery_minutes_low);
    expect(inf.recovery_minutes_high).toBe(graded.recovery_minutes_high);
    expect(inf.resumes_at).toBe(NOW + 20 * MIN);
  });

  test('the same tick grades on the full numbers: they reach the JSONL row', async () => {
    const { snapshot, full } = tick();
    const rows = buildPredictionRows({
      ts: NOW,
      routeStatuses: snapshot.route_status,
      inferences: full,
      paramsVersion: NOW,
      movementRegimes: { J: { entered_at: NOW - 30 * MIN } },
      movementCounts: new Map(),
    });

    const puts: Array<{ key: string; body: string }> = [];
    const bucket = {
      put: async (key: string, body: string) => {
        puts.push({ key, body });
      },
    } as unknown as R2Bucket;
    await writePredictions(bucket, NOW, rows);

    expect(puts).toHaveLength(1);
    const written = puts[0]!.body
      .split('\n')
      .map((line) => JSON.parse(line) as Record<string, unknown>);
    const jRow = written.find((r) => r.route === 'J')!;
    const jGraded = full.get('J')!;

    // The grader sees the numbers the public block withheld — this is the whole
    // point of the projection being a projection.
    expect(jRow.recovery_source).toBe('movement');
    expect(jRow.recovery_minutes).toBe(jGraded.recovery_minutes);
    expect(jRow.recovery_minutes_low).toBe(jGraded.recovery_minutes_low);
    expect(jRow.recovery_minutes_high).toBe(jGraded.recovery_minutes_high);
    expect(typeof jRow.recovery_minutes).toBe('number');
    expect(jRow.p_normal_in_30min).toBe(jGraded.p_normal_in_30min);
    // No withheld marker on the grading contract: the row is never partial.
    expect('recovery_withheld' in jRow).toBe(false);
    // The published context it is graded against comes from the same tick.
    expect(jRow.published_condition).toBe(snapshot.route_status.J!.condition);
  });

  test('graduation restores the previous document byte for byte', () => {
    const { snapshot, full } = tick();
    const graded = full.get('J')!;
    // What buildSnapshot would attach with PUBLISH_FITTED_RECOVERY flipped.
    const graduated = projectInference(graded, true);

    // Not "the same values plus a marker" — the same serialized bytes as the
    // object the Worker published before the gate existed. That is what makes
    // graduation a one-line flip with no contract negotiation.
    expect(JSON.stringify(graduated)).toBe(JSON.stringify(graded));
    expect('recovery_withheld' in graduated).toBe(false);

    // Withholding is the only difference between the two projections.
    const withheld = projectInference(graded, false);
    expect(withheld).not.toEqual(graduated);
    expect(withheld.condition).toBe(graduated.condition);

    // A schedule row keeps every number either way — it had no fit to withhold.
    // The gate only decides whether it carries the marker saying so.
    const scheduleGraded = full.get('M')!;
    const scheduleWithheld = projectInference(scheduleGraded, false);
    expect(JSON.stringify(projectInference(scheduleGraded, true))).toBe(
      JSON.stringify(scheduleGraded),
    );
    expect(scheduleWithheld.recovery_minutes).toBe(scheduleGraded.recovery_minutes);
    expect(scheduleWithheld.recovery_withheld).toBeNull();
    expect(snapshot.route_status.M!.inference!.recovery_withheld).toBeNull();
  });

  test('a scrubbed route is not graded: the publish path refused those numbers', () => {
    const { snapshot, full } = tick();
    // What publishSnapshot does to a route carrying a non-finite posterior
    // (scrubCorruptInferences), which runs before the grading write.
    snapshot.route_status.J!.inference = null;
    const rows = buildPredictionRows({
      ts: NOW,
      routeStatuses: snapshot.route_status,
      inferences: full,
      paramsVersion: NOW,
      movementRegimes: undefined,
      movementCounts: new Map(),
    });
    expect(rows.map((r) => r.route)).not.toContain('J');
    expect(rows.map((r) => r.route)).toContain('M');
  });
});
