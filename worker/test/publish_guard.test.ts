/**
 * Pre-publish snapshot guard. Two scopes, deliberately separated:
 *  - document-level corruption (no timestamp, no version) BLOCKS the publish;
 *  - a single route's corrupt (non-finite) inference is SCRUBBED to null and
 *    the rest of the feed still publishes — one bad route must never black out
 *    the whole feed (regression: a range check on a 1.0000001 float once did,
 *    and separately, treating a deliberately-withheld null horizon as
 *    corruption once would have too — see the regression tests below).
 */

import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import {
  TICK_SECONDS,
  buildSnapshot,
  scrubCorruptInferences,
  snapshotFatalViolations,
} from '../src/snapshot';

const INFERRED_ROLL: RouteRoll = {
  filter: {
    probabilities: [0.05, 0.94, 0.01],
    regime_entered_at: 1_699_990_000,
    last_updated_at: 1_700_000_000,
  },
  published: {
    label: 'disrupted',
    pending_state: 'disrupted',
    pending_streak: 5,
    last_updated_at: 1_700_000_000,
  },
  alert_type_at_entry: 'Delays',
};

function buildInferred() {
  return buildSnapshot({
    generatedAt: 1_700_000_000,
    alertsFreshness: 1_700_000_000,
    routeSnapshots: new Map(),
    rolls: { '1': INFERRED_ROLL },
    trainedParams: null,
    tickSeconds: TICK_SECONDS,
  });
}

describe('snapshotFatalViolations', () => {
  test('a freshly built snapshot has none', () => {
    expect(snapshotFatalViolations(buildInferred())).toEqual([]);
  });

  test('flags a non-positive generated_at (blocks publish)', () => {
    const s = buildInferred();
    s.generated_at = 0;
    expect(snapshotFatalViolations(s).some((v) => v.includes('generated_at'))).toBe(
      true,
    );
  });

  test('does NOT treat a per-route inference issue as fatal', () => {
    const s = buildInferred();
    s.route_status['1']!.inference!.p_normal = Number.NaN;
    // Document is structurally fine; the bad route is handled by scrubbing.
    expect(snapshotFatalViolations(s)).toEqual([]);
  });
});

describe('scrubCorruptInferences', () => {
  test('leaves a clean snapshot untouched', () => {
    const s = buildInferred();
    expect(scrubCorruptInferences(s)).toEqual([]);
    expect(s.route_status['1']!.inference).not.toBeNull();
    // 60/120min are withheld by design on this fitted-curve (recovery_source
    // 'hmm') row — null is their valid published value, not something the
    // guard should treat as missing/corrupt.
    expect(s.route_status['1']!.inference!.p_normal_in_60min).toBeNull();
    expect(s.route_status['1']!.inference!.p_normal_in_120min).toBeNull();
  });

  test('nulls a route whose inference has a NaN, keeps the field for others', () => {
    const s = buildInferred();
    s.route_status['1']!.inference!.recovery_minutes = Number.NaN;
    expect(scrubCorruptInferences(s)).toEqual(['1']);
    expect(s.route_status['1']!.inference).toBeNull();
  });

  test('nulls a route whose inference has an Infinity', () => {
    const s = buildInferred();
    s.route_status['1']!.inference!.p_disrupted = Number.POSITIVE_INFINITY;
    expect(scrubCorruptInferences(s)).toEqual(['1']);
    expect(s.route_status['1']!.inference).toBeNull();
  });

  test('REGRESSION: a marginal float just over 1.0 is finite — NOT scrubbed', () => {
    // The outage: a strict p>1 range check stalled the entire feed on a
    // 1.0000001 rounding artifact. Finite values must ship as-is.
    const s = buildInferred();
    s.route_status['1']!.inference!.p_normal = 1.0000001;
    expect(scrubCorruptInferences(s)).toEqual([]);
    expect(s.route_status['1']!.inference).not.toBeNull();
    // The withheld (null) 60/120min horizons on this row are untouched by
    // the marginal-float fix — finite-OR-withheld, not "finite only".
    expect(s.route_status['1']!.inference!.p_normal_in_60min).toBeNull();
    expect(s.route_status['1']!.inference!.p_normal_in_120min).toBeNull();
  });

  test('REGRESSION: a withheld (null) horizon is not treated as corrupt', () => {
    // The outage this guards against: every fitted-curve row nulls
    // p_normal_in_60min/120min by design (see the Inference interface in
    // snapshot.ts) — reading that deliberate withholding as "missing/corrupt
    // data" would scrub the inference block off every route on every tick,
    // silently emptying the feed of exactly the data it exists to carry.
    const s = buildInferred();
    expect(s.route_status['1']!.inference!.p_normal_in_60min).toBeNull();
    expect(s.route_status['1']!.inference!.p_normal_in_120min).toBeNull();
    expect(scrubCorruptInferences(s)).toEqual([]);
    expect(s.route_status['1']!.inference).not.toBeNull();
  });
});
