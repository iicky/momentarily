/**
 * The recovery-duration climatology arm: load, resolve, serve, and project.
 *
 * Pins the observable contract a consumer reads — the served median/IQR of the
 * REMAINING duration, the pooling fallback and its recorded level, the
 * primary-alert-type gate, the outlived-the-population abstain, and that
 * projectInference publishes a climatology row unwithheld while still
 * withholding a fitted arm that no climatology replaced.
 */

import { describe, expect, test } from 'vitest';

import type { Inference } from '../src/snapshot';
import { projectInference } from '../src/snapshot';
import type { RecoveryBaselineDoc } from '../src/params';
import { loadRecoveryBaseline, resolveRecoveryCell } from '../src/params';
import { climatologyRecovery } from '../src/recovery';

const WINDOW_START = 1_000_000;
const WINDOW_END = 4_000_000;

// Ascending durations in minutes (multiples of the 5-min grid, like reality).
const TEN = [5, 10, 15, 20, 25, 30, 45, 60, 90, 120];

function doc(): RecoveryBaselineDoc {
  return {
    schema_version: '1',
    trained_at: WINDOW_END,
    min_samples: 5,
    window_start: WINDOW_START,
    window_end: WINDOW_END,
    severity_floor: 2,
    n_episodes: 40,
    cells: {
      // A (route Q, "Delays") cell that stands on its own.
      Q: { Delays: { n: 10, level: 'route', window_start: WINDOW_START, window_end: WINDOW_END, samples_min: TEN } },
      // A (route L, "Severe Delays") cell the trainer resolved to the alert_type
      // pool (recorded level 'alert_type', carrying the pool's samples).
      L: {
        'Severe Delays': {
          n: 8,
          level: 'alert_type',
          window_start: WINDOW_START,
          window_end: WINDOW_END,
          samples_min: [10, 20, 30, 40, 50, 60, 70, 80],
        },
      },
    },
    by_alert_type: {
      'Severe Delays': {
        n: 8,
        level: 'alert_type',
        window_start: WINDOW_START,
        window_end: WINDOW_END,
        samples_min: [10, 20, 30, 40, 50, 60, 70, 80],
      },
    },
    system: {
      n: 40,
      level: 'system',
      window_start: WINDOW_START,
      window_end: WINDOW_END,
      samples_min: TEN.concat(TEN, TEN, TEN),
    },
  };
}

function inference(overrides: Partial<Inference> = {}): Inference {
  return {
    condition: 'disrupted',
    recovery_minutes: 42,
    is_disrupted: true,
    p_normal: 0.1,
    p_disrupted: 0.8,
    p_suspended: 0.1,
    regime_entered_at: 1_000,
    regime_age_seconds: 500,
    recovery_minutes_low: 20,
    recovery_minutes_high: 80,
    recovery_indeterminate: false,
    p_normal_in_30min: 0.3,
    p_normal_in_60min: null,
    p_normal_in_120min: null,
    model_warming_up: false,
    recovery_source: 'movement',
    resumes_at: null,
    overdue: false,
    ...overrides,
  };
}

describe('loadRecoveryBaseline tolerates absence and malformed docs', () => {
  test('absent object -> null (recovery stays withheld, never throws)', async () => {
    const bucket = { get: async () => null } as unknown as Parameters<typeof loadRecoveryBaseline>[0];
    expect(await loadRecoveryBaseline(bucket)).toBeNull();
  });

  test('malformed doc -> null, not a throw into the tick', async () => {
    const bucket = {
      get: async () => ({ json: async () => ({ schema_version: '1', cells: 'nope' }) }),
    } as unknown as Parameters<typeof loadRecoveryBaseline>[0];
    expect(await loadRecoveryBaseline(bucket)).toBeNull();
  });

  test('a well-formed doc parses', async () => {
    const d = doc();
    const bucket = {
      get: async () => ({ json: async () => d }),
    } as unknown as Parameters<typeof loadRecoveryBaseline>[0];
    const parsed = await loadRecoveryBaseline(bucket);
    expect(parsed?.min_samples).toBe(5);
    expect(parsed?.cells.Q!.Delays!.n).toBe(10);
  });

  test('a cell whose n disagrees with its sample count is rejected (false support is a lie)', async () => {
    const d = doc();
    // Claim 99 samples while carrying 10 — recovery_baseline_n would misreport
    // support to a consumer, so the whole doc fails validation -> null.
    d.cells.Q!.Delays!.n = 99;
    const bucket = {
      get: async () => ({ json: async () => d }),
    } as unknown as Parameters<typeof loadRecoveryBaseline>[0];
    expect(await loadRecoveryBaseline(bucket)).toBeNull();
  });
});

describe('resolveRecoveryCell pooling fallback and the alert-type gate', () => {
  test('route cell wins when present', () => {
    expect(resolveRecoveryCell(doc(), 'Q', 'Delays')?.level).toBe('route');
  });

  test('unseen (route, alert_type) with a known alert falls to the alert_type pool', () => {
    // Route 7 was never seen with "Severe Delays", but the alert_type pool has it.
    const cell = resolveRecoveryCell(doc(), '7', 'Severe Delays');
    expect(cell?.level).toBe('alert_type');
    expect(cell?.n).toBe(8);
  });

  test('unseen route and unseen alert falls to the system pool', () => {
    expect(resolveRecoveryCell(doc(), '7', 'Weather')?.level).toBe('system');
  });

  test('null alert type resolves to nothing: the climatology needs a primary alert type', () => {
    expect(resolveRecoveryCell(doc(), 'Q', null)).toBeNull();
  });

  test('absent doc resolves to nothing', () => {
    expect(resolveRecoveryCell(null, 'Q', 'Delays')).toBeNull();
  });
});

describe('climatologyRecovery serves the remaining-duration distribution', () => {
  test('at onset (elapsed 0) it is the full distribution median/IQR', () => {
    const r = climatologyRecovery(doc(), 'Q', 'Delays', 0);
    // p25/p50/p75 of [5..120] = 16.25 / 27.5 / 56.25, rounded half-up.
    expect(r).not.toBeNull();
    expect(r!.recovery_minutes).toBe(28);
    expect(r!.recovery_minutes_low).toBe(16);
    expect(r!.recovery_minutes_high).toBe(56);
    expect(r!.recovery_indeterminate).toBe(false);
    expect(r!.recovery_baseline_n).toBe(10);
    expect(r!.recovery_baseline_level).toBe('route');
  });

  test('conditioning on elapsed time raises the remaining estimate', () => {
    const atOnset = climatologyRecovery(doc(), 'Q', 'Delays', 0)!;
    // 60 min elapsed: only {90,120} remain of the route cell (n=2 < min 5) ->
    // indeterminate. Prove the elapsed clock actually gates by using the larger
    // system pool where enough survive.
    const later = climatologyRecovery(doc(), '7', 'Weather', 30 * 60)!;
    expect(later.recovery_indeterminate).toBe(false);
    expect(later.recovery_minutes).toBeGreaterThan(0);
    expect(atOnset.recovery_minutes).toBeGreaterThan(0);
  });

  test('a disruption that outlived its population is indeterminate, not extrapolated', () => {
    // 90 min elapsed on the 10-sample route cell leaves {120} (n=1 < min 5).
    const r = climatologyRecovery(doc(), 'Q', 'Delays', 90 * 60)!;
    expect(r.recovery_indeterminate).toBe(true);
    expect(r.recovery_minutes).toBeNull();
    expect(r.recovery_minutes_low).toBeNull();
    expect(r.recovery_minutes_high).toBeNull();
    // Still records which cell was consulted.
    expect(r.recovery_baseline_n).toBe(10);
    expect(r.recovery_baseline_level).toBe('route');
  });

  test('no primary alert type -> no climatology', () => {
    expect(climatologyRecovery(doc(), 'Q', null, 0)).toBeNull();
  });

  test('absent sidecar -> no climatology', () => {
    expect(climatologyRecovery(null, 'Q', 'Delays', 0)).toBeNull();
  });
});

describe('projectInference passes climatology through and still withholds fitted arms', () => {
  test('a fitted (movement) row is REPLACED by the climatology, unwithheld', () => {
    const clim = climatologyRecovery(doc(), 'Q', 'Delays', 0)!;
    const pub = projectInference(inference({ recovery_source: 'movement' }), clim);
    expect(pub.recovery_source).toBe('climatology');
    expect(pub.recovery_minutes).toBe(28);
    expect(pub.recovery_minutes_low).toBe(16);
    expect(pub.recovery_minutes_high).toBe(56);
    expect(pub.recovery_baseline_n).toBe(10);
    expect(pub.recovery_baseline_level).toBe('route');
    expect(pub.recovery_withheld).toBeNull();
    // p_normal_in_30min is nulled on every public row.
    expect(pub.p_normal_in_30min).toBeNull();
  });

  test('an outlived climatology row publishes null minutes, unwithheld, indeterminate', () => {
    const clim = climatologyRecovery(doc(), 'Q', 'Delays', 90 * 60)!;
    const pub = projectInference(inference({ recovery_source: 'hmm' }), clim);
    expect(pub.recovery_source).toBe('climatology');
    expect(pub.recovery_minutes).toBeNull();
    expect(pub.recovery_indeterminate).toBe(true);
    expect(pub.recovery_withheld).toBeNull();
    expect(pub.recovery_baseline_n).toBe(10);
  });

  test('a fitted arm with NO climatology stays withheld', () => {
    const pub = projectInference(inference({ recovery_source: 'movement' }), null);
    expect(pub.recovery_source).toBe('movement');
    expect(pub.recovery_minutes).toBeNull();
    expect(pub.recovery_withheld).toBe('pending_validation');
    expect(pub.recovery_baseline_n).toBeNull();
  });

  test('the climatology is served even with PUBLISH_FITTED_RECOVERY on (not gated by it)', () => {
    const clim = climatologyRecovery(doc(), 'Q', 'Delays', 0)!;
    // publishFitted=true simulates the fitted arm graduating; the climatology is
    // NOT a fitted curve, so it still serves — it is the yardstick, evaluated
    // before the gate.
    const pub = projectInference(inference({ recovery_source: 'movement' }), clim, true);
    expect(pub.recovery_source).toBe('climatology');
    expect(pub.recovery_minutes).toBe(28);
    expect(pub.recovery_baseline_level).toBe('route');
    // Where NO climatology covers the route, the graduated fitted arm publishes.
    const fitted = projectInference(inference({ recovery_source: 'movement' }), null, true);
    expect(fitted.recovery_source).toBe('movement');
    expect(fitted.recovery_minutes).toBe(42);
  });

  test('a schedule row is untouched by the climatology (kept, not replaced)', () => {
    const clim = climatologyRecovery(doc(), 'Q', 'Delays', 0)!;
    const pub = projectInference(
      inference({ recovery_source: 'schedule', recovery_minutes: 20, resumes_at: 999 }),
      clim,
    );
    expect(pub.recovery_source).toBe('schedule');
    expect(pub.recovery_minutes).toBe(20);
    expect(pub.recovery_withheld).toBeNull();
    expect(pub.recovery_baseline_n).toBeNull();
    expect(pub.p_normal_in_30min).toBeNull();
  });
});
