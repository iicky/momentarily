/**
 * Pre-publish snapshot invariants. publishSnapshot refuses to ship a malformed
 * snapshot, so a derivation bug (NaN probability, negative recovery, bad
 * condition) can't reach consumers — the CDN keeps serving the last good one.
 */

import { describe, expect, test } from 'vitest';

import type { RouteRoll } from '../src/alpha';
import { TICK_SECONDS, buildSnapshot, snapshotViolations } from '../src/snapshot';

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

describe('snapshotViolations', () => {
  test('a freshly built snapshot is clean', () => {
    expect(snapshotViolations(buildInferred())).toEqual([]);
  });

  test('flags a non-positive generated_at', () => {
    const s = buildInferred();
    s.generated_at = 0;
    expect(snapshotViolations(s).some((v) => v.includes('generated_at'))).toBe(true);
  });

  test('flags a NaN probability in an inference', () => {
    const s = buildInferred();
    s.route_status['1']!.inference!.p_normal = Number.NaN;
    expect(snapshotViolations(s).some((v) => v.includes('probability'))).toBe(true);
  });

  test('flags a negative recovery_minutes', () => {
    const s = buildInferred();
    s.route_status['1']!.inference!.recovery_minutes = -5;
    expect(snapshotViolations(s).some((v) => v.includes('recovery_minutes'))).toBe(
      true,
    );
  });

  test('flags an unrecognized condition', () => {
    const s = buildInferred();
    s.route_status['1']!.inference!.condition = 'meltdown';
    expect(snapshotViolations(s).some((v) => v.includes('condition'))).toBe(true);
  });
});
